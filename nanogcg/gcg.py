# Vendored from nanoGCG.
#
#   Original work:  GraySwanAI/nanoGCG  —  https://github.com/GraySwanAI/nanoGCG
#   License:        MIT, Copyright (c) 2024 Gray Swan AI  (see ./LICENSE)
#
# This is a MODIFIED copy taken from the fork scdivad/nanoGCG at commit
# f631a51 (2026-06-11), which diverges from GraySwanAI/main in this file by
# roughly +357/-42 lines. The fork's changes to gcg.py include:
#   - transformers-version compatibility for the prefix KV cache
#     (DynamicCache snapshot/rebuild across >=4.40 / >=4.50)
#   - a prefix-cache mutation fix (was producing fake-low losses)
#   - crash-resilient sweeping (retry filter_ids, per-prompt error catch, resume)
#   - a periodic verifier callback (used here to stop on a prefill-scored flip)
#   - ACG / I-GCG multi-coordinate update presets
# It is bundled here so this repo runs GCG without an external nanoGCG install.
# The MIT license terms above apply to this file.

import copy
import gc
import logging
import queue
import threading

from dataclasses import dataclass
from tqdm import tqdm
from typing import Callable, List, Optional, Tuple, Union

import torch
import transformers
from torch import Tensor
from transformers import set_seed
from scipy.stats import spearmanr

from nanogcg.utils import (
    INIT_CHARS,
    configure_pad_token,
    find_executable_batch_size,
    get_nonascii_toks,
    get_special_toks,
    mellowmax,
)

logger = logging.getLogger("nanogcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class ProbeSamplingConfig:
    draft_model: transformers.PreTrainedModel
    draft_tokenizer: transformers.PreTrainedTokenizer
    r: int = 8
    sampling_factor: int = 16


@dataclass
class GCGConfig:
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
    # MULTI-REGION (N independent suffixes) GCG. When the prompt template
    # contains the ordered placeholders `{optim_0}`, `{optim_1}`, ... `{optim_{K-1}}`
    # (instead of the usual single `{optim_str}`), K independent adversarial
    # strings are optimized JOINTLY: `optim_str_inits[i]` initializes region i.
    # They are concatenated into one optimization vector and updated together;
    # only buffer_size in {0, 1} is supported in this mode. Leave None for the
    # ordinary single-region behavior (`{optim_str}` / append-at-end).
    optim_str_inits: Optional[List[str]] = None
    # Extra kwargs forwarded to tokenizer.apply_chat_template (e.g.
    # {"enable_thinking": False} for Qwen3, which must match the inference-time
    # rendering or the optimized suffix won't transfer). None => no extra kwargs.
    chat_template_kwargs: Optional[dict] = None
    # Compute logits only at the target span (via logits_to_keep = n_target + 1)
    # instead of over the whole sequence. The target sits at the END of the input,
    # so only the last n_target+1 positions' logits are needed for the CE — this
    # skips the lm_head projection over the (huge) vocab at every other position,
    # a ~100x reduction in the dominant cost for long prompts / big vocabularies.
    # Numerically identical to the full-logits path (same logits at those
    # positions); verified against a full-sequence forward. Applies to the
    # serial (per-bundle) gradient and candidate-loss paths.
    target_logits_only: bool = True
    search_width: int = 512
    batch_size: int = None
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: int = None
    verbosity: str = "INFO"
    probe_sampling_config: Optional[ProbeSamplingConfig] = None
    # I-GCG (Jia et al. 2024, arXiv:2405.21018): after scoring all single-token
    # candidates, cumulatively merge the top-p lowest-loss candidates' diffs
    # against the current suffix and re-score those p merged variants, picking
    # the best. Assumes `n_replace=1`.
    use_i_gcg: bool = False
    i_gcg_top_p: int = 7
    # String-space early stop. Every `early_stop_check_every` steps, call this
    # callback with (step, current_best_optim_ids, current_loss); if it returns
    # True the run stops. Intended for decoding the current best suffix to a
    # string, running the real inference pipeline, and checking whether the
    # attack actually works end-to-end (rather than just winning in token
    # space, which can spuriously fire via `early_stop` when the chat-template
    # boundary re-tokenizes differently at inference time).
    early_stop_callback: Optional[Callable[[int, Tensor, float], bool]] = None
    early_stop_check_every: int = 5


@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]
    # Multi-region runs only: the raw token ids of the best step (all regions
    # concatenated in order) and each region's token length. best_string is the
    # decoded concatenation (not directly usable as a prompt); callers should
    # split best_ids by cumulative sums of optim_region_lens to recover the K
    # suffixes. None for single-region runs.
    best_ids: Optional[Tensor] = None
    optim_region_lens: Optional[List[int]] = None


@dataclass
class PromptBundle:
    """Per-prompt state used by GCG. In single-prompt runs there's one bundle;
    in universal (multi-prompt) runs there's one per prompt and losses/gradients
    are averaged across them.

    prefix_cache is set only when config.use_prefix_cache is True. When set,
    the model call omits before_embeds and passes prefix_cache/prefix_legacy;
    otherwise before_embeds is prepended to the input embeddings.
    """
    messages: Union[str, List[dict]]
    target: str
    before_ids: Tensor          # (1, n_before) — used only for filter_ids boundary
    before_str: str             # decoded prefix string for filter_ids boundary
    target_ids: Tensor          # (1, n_target)
    before_embeds: Tensor       # (1, n_before, d)
    after_embeds: Tensor        # (1, n_after, d)
    target_embeds: Tensor       # (1, n_target, d)
    prefix_cache: object = None       # DynamicCache (kept truthy for gates)
    prefix_legacy: object = None      # tuple[(k, v), …] snapshot for _fresh_prefix_cache
    # after_str/after_ids are stashed for boundary filtering. Populated for all
    # bundles; used by the multi-region filter (and harmless otherwise).
    after_str: str = None
    after_ids: Tensor = None
    # MULTI-REGION only: the K-1 fixed text segments sitting BETWEEN consecutive
    # optimized regions (mids[i] is between region i and region i+1). None for
    # single-region.
    region_mids_str: list = None      # List[str],   len K-1
    region_mids_ids: list = None      # List[Tensor], len K-1
    region_mids_embeds: list = None   # List[Tensor (1, n_mid_i, d)], len K-1


class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = []  # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]

    def log_buffer(self, tokenizer):
        message = "buffer:"
        for loss, ids in self.buffer:
            optim_str = tokenizer.batch_decode(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            message += f"\nloss: {loss}" + f" | string: {optim_str}"
        logger.info(message)


def sample_ids_from_grad(
    ids: Tensor,
    grad: Tensor,
    search_width: int,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor = False,
):
    """Returns `search_width` combinations of token ids based on the token gradient.

    Args:
        ids : Tensor, shape = (n_optim_ids)
            the sequence of token ids that are being optimized
        grad : Tensor, shape = (n_optim_ids, vocab_size)
            the gradient of the GCG loss computed with respect to the one-hot token embeddings
        search_width : int
            the number of candidate sequences to return
        topk : int
            the topk to be used when sampling from the gradient
        n_replace : int
            the number of token positions to update per sequence
        not_allowed_ids : Tensor, shape = (n_ids)
            the token ids that should not be used in optimization

    Returns:
        sampled_ids : Tensor, shape = (search_width, n_optim_ids)
            sampled token ids
    """
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices

    sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device),
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

    return new_ids


def filter_ids(
    ids: Tensor,
    tokenizer: transformers.PreTrainedTokenizer,
    raise_on_empty: bool = True,
    before_str: str = None,
    before_ids: Tensor = None,
    boundary_ctx: Optional[List[Tuple[str, Tensor]]] = None,
):
    """Filters out sequeneces of token ids that change after retokenization.

    Args:
        ids : Tensor, shape = (search_width, n_optim_ids)
            token ids
        tokenizer : ~transformers.PreTrainedTokenizer
            the model's tokenizer
        raise_on_empty : bool
            if True (default) and no rows survive, raise RuntimeError;
            if False, return a zero-row tensor instead. Callers that are
            using filtering as a best-effort pass (e.g. pruning randomly
            generated buffer entries at init) should set this to False.
        before_str, before_ids : optional prompt-prefix context. When given,
            a candidate survives only if `before_str + decode(optim)` re-tokenizes
            to exactly `before_ids ++ optim` — i.e. the suffix does not bleed
            across the prompt/suffix boundary. This is the realistic objective:
            an attacker submits the suffix as TEXT appended to the prompt, so the
            in-context tokenization (not the isolated one) is what the model sees.
            Without this, GCG can win in token-space with a suffix that the model
            never actually tokenizes that way at inference.
        boundary_ctx : optional List[(before_str, before_ids)] for UNIVERSAL GCG.
            A candidate survives only if it round-trips against EVERY boundary
            (intersection). Ignored when `before_str`/`before_ids` are set;
            single-prompt callers should keep using those. Universal callers
            pass this list and leave before_str/before_ids as None.

    Returns:
        filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
            all token ids that are the same after retokenization
    """
    # Universal-boundary path: intersect across all prompt prefixes. We do this
    # by recursively narrowing `ids` through the single-boundary path once per
    # prompt. Each pass is cheap (batch-tokenize of ~search_width strings).
    if boundary_ctx is not None and before_str is None and before_ids is None:
        surviving = ids
        for b_str, b_ids in boundary_ctx:
            if surviving.shape[0] == 0:
                break
            surviving = filter_ids(
                surviving, tokenizer, raise_on_empty=False,
                before_str=b_str, before_ids=b_ids,
            )
        if surviving.shape[0] == 0 and raise_on_empty:
            raise RuntimeError(
                "No token sequences survived multi-prompt boundary filter_ids. "
                "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
            )
        return surviving

    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []

    boundary = before_str is not None and before_ids is not None
    n_before = before_ids.shape[0] if boundary else 0

    if boundary:
        # Batch-tokenize all `before_str + suffix` strings in ONE call (much
        # faster than per-candidate). A candidate survives iff the prefix region
        # is unchanged AND the suffix region re-tokenizes to exactly this
        # candidate — i.e. no token bleed across the prompt/suffix boundary, so
        # GCG's token-space loss equals the loss the model sees on the text.
        L = n_before + ids.shape[1]
        before_list = before_ids.tolist()
        encoded = tokenizer([before_str + s for s in ids_decoded],
                            padding=False, add_special_tokens=True)["input_ids"]
        for i, toks in enumerate(encoded):
            if (len(toks) == L
                    and toks[:n_before] == before_list
                    and toks[n_before:] == ids[i].tolist()):
                filtered_ids.append(ids[i])
    else:
        # Retokenize the decoded token ids in isolation (original behavior,
        # per-candidate to stay compatible with tokenizer wrappers).
        for i in range(len(ids_decoded)):
            ids_encoded = tokenizer(ids_decoded[i], return_tensors="pt", add_special_tokens=False).to(ids.device)["input_ids"][0]
            if torch.equal(ids[i], ids_encoded):
                filtered_ids.append(ids[i])

    if not filtered_ids:
        if raise_on_empty:
            # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
            raise RuntimeError(
                "No token sequences are the same after decoding and re-encoding. "
                "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
            )
        return ids.new_empty((0, ids.shape[1]))

    return torch.stack(filtered_ids)


def filter_ids_multi_region(
    ids: Tensor,
    tokenizer: transformers.PreTrainedTokenizer,
    region_ctx: List[Tuple[str, list, list, str, list]],
    region_lens: List[int],
    raise_on_empty: bool = True,
):
    """Boundary filter for MULTI-REGION (K independent suffixes) GCG.

    Each candidate row is `region_0 ++ region_1 ++ ... ++ region_{K-1}`, with
    `region_lens[i]` tokens in region i. At inference the suffixes are spliced
    into the prompt as TEXT:
        before + d(r0) + mid_0 + d(r1) + mid_1 + ... + d(r_{K-1}) + after
    A candidate survives iff EVERY optimized region's tokens are preserved in
    their actual left-context — i.e. for each region i, re-tokenizing the text
    up to and including region i extends the region's left-context tokenization
    by EXACTLY region i's ids (no bleed at region i's left boundary, and its
    tokens are not disturbed by the fixed text that precedes it). This is the
    per-region generalization of the single-region `filter_ids` boundary check
    and prevents token-space wins that don't transfer to the real prompt.

    `region_ctx` is one tuple per bundle: (before_str, before_ids_list,
    mids_str_list [len K-1], after_str, after_ids_list). A candidate must survive
    against EVERY bundle (intersection), since the fixed text (and thus the bleed
    pattern) differs per prompt.
    """
    K = len(region_lens)
    # cumulative region start offsets within a candidate row
    starts = [0]
    for L in region_lens:
        starts.append(starts[-1] + L)

    surviving = ids
    for (before_str, before_list, mids_str, after_str, after_list) in region_ctx:
        if surviving.shape[0] == 0:
            break
        S = surviving.shape[0]
        # decode each region for every surviving candidate
        dec = [tokenizer.batch_decode(surviving[:, starts[i]:starts[i + 1]]) for i in range(K)]
        # left_context[c] accumulates before + d(r0) + mid_0 + ... as we walk regions
        left_context = [before_str] * S
        ok = [True] * S
        nb = len(before_list)
        for i in range(K):
            # tokenize left-context and left-context+region_i in two batch calls
            base = tokenizer([left_context[c] for c in range(S)], padding=False, add_special_tokens=True)["input_ids"]
            ext = tokenizer([left_context[c] + dec[i][c] for c in range(S)], padding=False, add_special_tokens=True)["input_ids"]
            for c in range(S):
                if not ok[c]:
                    continue
                reg_ids = surviving[c, starts[i]:starts[i + 1]].tolist()
                # region i's tokens must be exactly appended to its left-context tokenization
                if not (ext[c][:len(base[c])] == base[c] and ext[c][len(base[c]):] == reg_ids):
                    ok[c] = False
                    continue
                # region 0 must also start exactly after `before` (left anchor)
                if i == 0 and base[c][:nb] != before_list:
                    ok[c] = False
            # advance left-context by this region's text + the following mid (if any)
            for c in range(S):
                left_context[c] = left_context[c] + dec[i][c] + (mids_str[i] if i < K - 1 else "")
        # final: full string's tail must equal after (right anchor)
        if after_list:
            na = len(after_list)
            full = tokenizer([left_context[c] + after_str for c in range(S)], padding=False, add_special_tokens=True)["input_ids"]
            for c in range(S):
                if ok[c] and full[c][len(full[c]) - na:] != after_list:
                    ok[c] = False
        keep = [surviving[c] for c in range(S) if ok[c]]
        surviving = torch.stack(keep) if keep else ids.new_empty((0, surviving.shape[1]))

    if surviving.shape[0] == 0 and raise_on_empty:
        raise RuntimeError(
            "No token sequences survived the multi-region boundary filter. "
            "Consider setting `filter_ids=False` or different inits."
        )
    return surviving


class GCG:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: GCGConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.embedding_layer = model.get_input_embeddings()
        # Special/added tokens are ALWAYS disallowed in the optimized string
        # (even with allow_non_ascii) — they break boundary tokenization and
        # behave as control tokens. Union with non-ascii unless allowed.
        special_ids = get_special_toks(tokenizer, device=model.device)
        if config.allow_non_ascii:
            self.not_allowed_ids = special_ids
        else:
            nonascii = get_nonascii_toks(tokenizer, device=model.device)
            self.not_allowed_ids = torch.unique(torch.cat([nonascii, special_ids]))
        self.prefix_cache = None
        self.draft_prefix_cache = None

        # Multi-region (K-suffix) state. optim_region_lens is the per-region
        # token length list; when not None the run is in multi-region mode.
        self.optim_region_lens = None
        self._region_ctx = None

        self.stop_flag = False

        self.draft_model = None
        self.draft_tokenizer = None
        self.draft_embedding_layer = None
        if self.config.probe_sampling_config:
            if self.config.use_i_gcg:
                raise ValueError("`use_i_gcg` is not compatible with `probe_sampling_config`.")
            self.draft_model = self.config.probe_sampling_config.draft_model
            self.draft_tokenizer = self.config.probe_sampling_config.draft_tokenizer
            self.draft_embedding_layer = self.draft_model.get_input_embeddings()
            if self.draft_tokenizer.pad_token is None:
                configure_pad_token(self.draft_tokenizer)

        if model.dtype in (torch.float32, torch.float64):
            logger.warning(f"Model is in {model.dtype}. Use a lower precision data type, if possible, for much faster optimization.")

        if model.device == torch.device("cpu"):
            logger.warning("Model is on the CPU. Use a hardware accelerator for faster optimization.")

        if not tokenizer.chat_template:
            logger.warning("Tokenizer does not have a chat template. Assuming base model and setting chat template to empty.")
            tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

    def run(
        self,
        messages: Union[str, List[dict]],
        target: str,
    ) -> GCGResult:
        """Optimize a single suffix against one (messages, target) pair.

        Bit-identical to the single-prompt implementation prior to universal-GCG
        support: builds one PromptBundle and delegates to `_run_bundles`.
        """
        bundle = self._prepare_bundle(messages, target)
        return self._run_bundles([bundle])

    def run_universal(
        self,
        prompt_targets: List[Tuple[Union[str, List[dict]], str]],
    ) -> GCGResult:
        """Optimize a single suffix that jointly minimizes GCG loss across all
        (messages, target) pairs (a.k.a. universal / multi-prompt GCG,
        Zou et al. 2023 §3.3). Loss and gradient are the MEAN across prompts.

        The optimized string is the same shared suffix for all prompts — it
        gets inserted at each prompt's `{optim_str}` slot independently. Chat
        template, before/after/target token spans, and prefix KV cache are
        computed per prompt.

        Requires all prompts to share the same tokenizer (single model).
        Not compatible with `probe_sampling_config` (single-prompt only).
        """
        if not prompt_targets:
            raise ValueError("prompt_targets must be non-empty")
        if self.config.probe_sampling_config is not None and len(prompt_targets) > 1:
            raise ValueError("probe_sampling_config is not supported with universal (multi-prompt) GCG")
        bundles = [self._prepare_bundle(m, t) for (m, t) in prompt_targets]
        return self._run_bundles(bundles)

    def _prepare_bundle(
        self,
        messages: Union[str, List[dict]],
        target: str,
    ) -> PromptBundle:
        """Tokenize / embed / KV-cache a single (messages, target) into a
        PromptBundle. Extracted from the pre-universal `run()` body verbatim."""
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)

        # Multi-region mode: template carries ordered `{optim_0}`..`{optim_{K-1}}`.
        joined = "".join(d["content"] for d in messages)
        n_regions = 0
        while f"{{optim_{n_regions}}}" in joined:
            n_regions += 1
        multi_region = n_regions > 0

        if not multi_region:
            # Append the GCG string at the end of the prompt if location not specified
            if not any(["{optim_str}" in d["content"] for d in messages]):
                messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

        ct_kwargs = config.chat_template_kwargs or {}
        template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **ct_kwargs)
        # Remove the BOS token -- this will get added when tokenizing, if necessary
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "")

        target = " " + target if config.add_space_before_target else target
        embedding_layer = self.embedding_layer

        region_mids_str = region_mids_ids = region_mids_embeds = None
        if multi_region:
            if config.optim_str_inits is None or len(config.optim_str_inits) != n_regions:
                raise ValueError(
                    f"multi-region GCG found {n_regions} `{{optim_i}}` placeholders but "
                    f"config.optim_str_inits has {None if config.optim_str_inits is None else len(config.optim_str_inits)} entries"
                )
            # Split the template on the ordered placeholders into
            # [before, mid_0, mid_1, ..., mid_{K-2}, after].
            before_str, rest = template.split("{optim_0}")
            region_mids_str = []
            for i in range(1, n_regions):
                seg, rest = rest.split(f"{{optim_{i}}}")
                region_mids_str.append(seg)   # text between region i-1 and region i
            after_str = rest
            region_mids_ids = [tokenizer([s], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
                               for s in region_mids_str]
            region_mids_embeds = [embedding_layer(x) for x in region_mids_ids]
        else:
            before_str, after_str = template.split("{optim_str}")

        # Tokenize everything that doesn't get optimized
        before_ids = tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        target_ids = tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)

        # Embed everything that doesn't get optimized
        before_embeds, after_embeds, target_embeds = [embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)]

        prefix_cache = None
        prefix_legacy = None
        # Compute the KV Cache for tokens that appear before the optimized tokens.
        # IMPORTANT (transformers>=4.40): the model mutates whatever Cache object
        # it's handed during the forward pass — torch.cat'ing new KVs into the
        # cache's internal lists. Reusing prefix_cache across iterations
        # therefore pollutes it with previous-step optim/after/target KVs, which
        # silently boosts P(target) on subsequent steps and reports artificially
        # low loss values that don't transfer to inference. Snapshot the legacy
        # tuple form (which holds references to the original "before"-only K/V
        # tensors) and rebuild a fresh DynamicCache from it on every model call.
        if config.use_prefix_cache:
            with torch.no_grad():
                output = model(inputs_embeds=before_embeds, use_cache=True)
                cache = output.past_key_values
                # transformers compat: the DynamicCache API has shifted across
                # versions (key_cache/value_cache lists → layers → ...).
                if hasattr(cache, "to_legacy_cache"):
                    prefix_legacy = cache.to_legacy_cache()
                elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
                    # transformers ~4.40–4.49
                    prefix_legacy = tuple(
                        (cache.key_cache[i], cache.value_cache[i])
                        for i in range(len(cache.key_cache))
                    )
                elif hasattr(cache, "layers"):
                    # transformers 4.55+: cache.layers[i].keys / .values
                    prefix_legacy = tuple(
                        (layer.keys, layer.values) for layer in cache.layers
                    )
                else:
                    # Last resort: iterate. DynamicCache.__iter__ yields
                    # (key, value) tuples per layer.
                    prefix_legacy = tuple((k, v) for k, v in cache)
                # Keep prefix_cache truthy for the `if bundle.prefix_cache:`
                # gates; never pass it to model() directly.
                prefix_cache = cache

        return PromptBundle(
            messages=messages,
            target=target,
            before_ids=before_ids,
            before_str=before_str,
            target_ids=target_ids,
            before_embeds=before_embeds,
            after_embeds=after_embeds,
            target_embeds=target_embeds,
            prefix_cache=prefix_cache,
            prefix_legacy=prefix_legacy,
            after_str=after_str,
            after_ids=after_ids,
            region_mids_str=region_mids_str,
            region_mids_ids=region_mids_ids,
            region_mids_embeds=region_mids_embeds,
        )

    def _run_bundles(self, bundles: List[PromptBundle]) -> GCGResult:
        """Shared optimization loop. Single-prompt path passes a 1-element list
        and the semantics are bit-identical to the pre-universal code path."""
        model = self.model
        tokenizer = self.tokenizer
        config = self.config
        embedding_layer = self.embedding_layer

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        self.bundles = bundles
        # For probe-sampling back-compat: still uses self.target_ids etc. from
        # the primary bundle. Universal + probe-sampling is rejected upstream.
        primary = bundles[0]
        self.target_ids = primary.target_ids
        self.before_embeds = primary.before_embeds
        self.after_embeds = primary.after_embeds
        self.target_embeds = primary.target_embeds
        self.prefix_cache = primary.prefix_cache
        self.prefix_legacy = primary.prefix_legacy
        # Multi-prompt filter_ids intersection context. Empty list => single-prompt
        # boundary check via _filter_before_str / _filter_before_ids (unchanged).
        if len(bundles) > 1:
            self._filter_boundary_ctx = [(b.before_str, b.before_ids[0]) for b in bundles]
            self._filter_before_str = None
            self._filter_before_ids = None
        else:
            self._filter_boundary_ctx = None
            self._filter_before_str = primary.before_str
            self._filter_before_ids = primary.before_ids[0]

        # Multi-region (K-suffix) setup. optim_region_lens = per-region token
        # lengths (from the inits). Build the per-bundle boundary context used by
        # the multi-region filter. The cross-bundle fast path is NOT compatible
        # with multi-region (there is fixed text between the adv slots), so it is
        # disabled below.
        if primary.region_mids_embeds is not None:
            self.optim_region_lens = [
                tokenizer([s], add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1]
                for s in config.optim_str_inits
            ]
            # filter needs, per bundle: before, before_ids, the K-1 mid strings, after, after_ids
            self._region_ctx = [
                (b.before_str, b.before_ids[0].tolist(), list(b.region_mids_str),
                 b.after_str, b.after_ids[0].tolist())
                for b in bundles
            ]
        else:
            self.optim_region_lens = None
            self._region_ctx = None

        # Universal fast path: cross-bundle batching. Available only when every
        # bundle shares the same before_ids (typical for adv-prefix placement
        # with a single system prompt). One forward per chunk covers all
        # bundles' candidate evaluations, so per-step forward count drops from
        # N × chunks_per_bundle to just chunks_per_bundle. See
        # `_setup_universal_shared_prefix` for what gets precomputed.
        self._univ_shared = False
        if len(bundles) > 1 and config.use_prefix_cache and self.optim_region_lens is None:
            self._setup_universal_shared_prefix()

        # Initialize components for probe sampling, if enabled (single-prompt only).
        if config.probe_sampling_config:
            assert self.draft_model and self.draft_tokenizer and self.draft_embedding_layer, "Draft model wasn't properly set up."

            draft_before_ids = self.draft_tokenizer([primary.before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
            # Recover after_str from primary bundle's messages by re-splitting
            # its template. (We don't stash after_str on the bundle since only
            # probe sampling needs it, and it's easy to recompute here.)
            _tmpl_msgs = copy.deepcopy(primary.messages)
            _tmpl = tokenizer.apply_chat_template(_tmpl_msgs, tokenize=False, add_generation_prompt=True,
                                                  **(config.chat_template_kwargs or {}))
            if tokenizer.bos_token and _tmpl.startswith(tokenizer.bos_token):
                _tmpl = _tmpl.replace(tokenizer.bos_token, "")
            _, _after_str = _tmpl.split("{optim_str}")
            draft_after_ids = self.draft_tokenizer([_after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
            self.draft_target_ids = self.draft_tokenizer([primary.target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)

            (
                self.draft_before_embeds,
                self.draft_after_embeds,
                self.draft_target_embeds,
            ) = [
                self.draft_embedding_layer(ids)
                for ids in (
                    draft_before_ids,
                    draft_after_ids,
                    self.draft_target_ids,
                )
            ]

            if config.use_prefix_cache:
                with torch.no_grad():
                    output = self.draft_model(inputs_embeds=self.draft_before_embeds, use_cache=True)
                    self.draft_prefix_cache = output.past_key_values

        # Initialize the attack buffer (per-prompt losses averaged across bundles)
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []
        optim_ids_history = []   # best optim_ids per step, index-aligned with `losses`

        for step in tqdm(range(config.num_steps)):
            # Compute the token gradient (averaged across bundles)
            optim_ids_onehot_grad = self.compute_token_gradient(optim_ids)

            with torch.no_grad():

                # Sample candidate token sequences based on the token gradient.
                # If filter_ids is on and nothing survives round-trip, retry a
                # few times — the sampling is stochastic, so a fresh draw
                # usually surfaces at least some surviving candidates. Only if
                # all retries fail do we skip this step (keep current optim_ids).
                filter_attempts = 4
                sampled_ids = None
                for attempt in range(filter_attempts):
                    candidates = sample_ids_from_grad(
                        optim_ids.squeeze(0),
                        optim_ids_onehot_grad.squeeze(0),
                        config.search_width,
                        config.topk,
                        config.n_replace,
                        not_allowed_ids=self.not_allowed_ids,
                    )
                    if config.filter_ids:
                        if self.optim_region_lens is not None:
                            candidates = filter_ids_multi_region(
                                candidates, tokenizer, self._region_ctx,
                                self.optim_region_lens, raise_on_empty=False,
                            )
                        elif self._filter_boundary_ctx is not None:
                            candidates = filter_ids(
                                candidates, tokenizer, raise_on_empty=False,
                                boundary_ctx=self._filter_boundary_ctx,
                            )
                        else:
                            candidates = filter_ids(
                                candidates, tokenizer, raise_on_empty=False,
                                before_str=self._filter_before_str,
                                before_ids=self._filter_before_ids,
                            )
                    if candidates.shape[0] > 0:
                        sampled_ids = candidates
                        break
                if sampled_ids is None:
                    logger.warning(
                        f"Step {step + 1}: no sampled candidates survived `filter_ids` "
                        f"after {filter_attempts} retries; skipping this step."
                    )
                    losses.append(buffer.get_lowest_loss())
                    optim_str = tokenizer.batch_decode(buffer.get_best_ids())[0]
                    optim_strings.append(optim_str)
                    optim_ids_history.append(buffer.get_best_ids().clone())
                    continue

                new_search_width = sampled_ids.shape[0]
                batch_size = new_search_width if config.batch_size is None else config.batch_size

                if self.config.probe_sampling_config is None:
                    # Universal-capable path: iterate bundles, mean per-candidate loss.
                    loss = self._compute_candidates_loss_all_bundles(
                        sampled_ids, batch_size,
                    )
                    if config.use_i_gcg:
                        current_loss, optim_ids = self._i_gcg_merge_step(
                            sampled_ids=sampled_ids,
                            single_token_losses=loss,
                            current_optim_ids=optim_ids,
                            batch_size=batch_size,
                        )
                    else:
                        current_loss = loss.min().item()
                        optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)
                else:
                    # Probe sampling path (single-prompt only) — unchanged.
                    if primary.prefix_cache:
                        input_embeds = torch.cat([
                            embedding_layer(sampled_ids),
                            primary.after_embeds.repeat(new_search_width, 1, 1),
                            primary.target_embeds.repeat(new_search_width, 1, 1),
                        ], dim=1)
                    else:
                        input_embeds = torch.cat([
                            primary.before_embeds.repeat(new_search_width, 1, 1),
                            embedding_layer(sampled_ids),
                            primary.after_embeds.repeat(new_search_width, 1, 1),
                            primary.target_embeds.repeat(new_search_width, 1, 1),
                        ], dim=1)
                    current_loss, optim_ids = find_executable_batch_size(self._compute_candidates_loss_probe_sampling, batch_size)(
                        input_embeds, sampled_ids,
                    )

                # Update the buffer based on the loss
                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)
            optim_ids_history.append(optim_ids.clone())

            buffer.log_buffer(tokenizer)

            # String-space early-stop: periodically invoke the user-supplied
            # callback with the current best suffix. Callback gets to decode
            # the suffix and run the real inference pipeline.
            if (
                config.early_stop_callback is not None
                and (step + 1) % max(1, config.early_stop_check_every) == 0
            ):
                try:
                    if config.early_stop_callback(step, optim_ids, current_loss):
                        logger.info(f"Early stopping at step {step + 1} via verifier callback.")
                        self.stop_flag = True
                except Exception as e:
                    logger.warning(f"early_stop_callback raised {type(e).__name__}: {e}; continuing.")

            if self.stop_flag:
                logger.info("Early stopping due to finding a perfect match.")
                break

        min_loss_index = losses.index(min(losses))

        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
            best_ids=(optim_ids_history[min_loss_index] if optim_ids_history else None),
            optim_region_lens=self.optim_region_lens,
        )

        return result

    def _setup_universal_shared_prefix(self):
        """Detect and precompute the cross-bundle batched fast path.

        Requirement: all bundles have IDENTICAL `before_ids`. When adv slot is
        at the very start of the user turn (adv_position="prefix") and every
        bundle uses the same system prompt, the chat template up to the
        `{optim_str}` split is bundle-independent — this condition holds. In
        that case we can:
          1. Reuse a SINGLE shared prefix KV cache across all bundles.
          2. Right-pad per-bundle `after_embeds` to the same length, likewise
             for `target_embeds`. Pack per-bundle tails into one tensor of
             shape [N, n_adv + max_after + max_target, d]. After adding a
             candidate's adv embedding this becomes a batched tail we can
             stack across N bundles × C candidates in one forward.
          3. Apply an attention mask that zeros out the per-bundle pad regions
             so the padded positions don't contribute (matches the un-padded
             loss numerically for real positions).

        Falls back to the serial per-bundle path when bundles' before_ids
        differ (e.g. mixed prompt-format universal runs).
        """
        b0 = self.bundles[0]
        for b in self.bundles[1:]:
            if b.before_ids.shape != b0.before_ids.shape:
                logger.info("Universal fast path disabled: bundles have different before_ids lengths.")
                return
            if not torch.equal(b.before_ids, b0.before_ids):
                logger.info("Universal fast path disabled: bundles have different before_ids content.")
                return

        self._univ_shared = True
        N = len(self.bundles)
        after_lens = [b.after_embeds.shape[1] for b in self.bundles]
        target_lens = [b.target_embeds.shape[1] for b in self.bundles]
        max_after = max(after_lens)
        max_target = max(target_lens)
        d = b0.after_embeds.shape[-1]
        dev = b0.after_embeds.device
        dtype = b0.after_embeds.dtype

        # [N, max_after + max_target, d] — right-pad each bundle's after span,
        # then place its target span RIGHT after the padded-after region so
        # target-token positions are IDENTICAL across bundles: target starts at
        # (n_adv + max_after) in the tail for every bundle. Only target length
        # varies; pad each bundle's target span on the right up to max_target.
        after_target_pad = torch.zeros(N, max_after + max_target, d, device=dev, dtype=dtype)
        # [N, max_after + max_target] valid-token mask (1 = real, 0 = pad)
        after_target_mask = torch.zeros(N, max_after + max_target, device=dev, dtype=torch.long)
        # [N, max_target] target ids, right-padded with 0 (masked out via
        # attention_mask + per-bundle target_len when computing CE)
        target_ids_pad = torch.zeros(N, max_target, device=dev, dtype=torch.long)
        for i, b in enumerate(self.bundles):
            al = after_lens[i]; tl = target_lens[i]
            after_target_pad[i, :al] = b.after_embeds[0]
            after_target_pad[i, max_after : max_after + tl] = b.target_embeds[0]
            after_target_mask[i, :al] = 1
            after_target_mask[i, max_after : max_after + tl] = 1
            target_ids_pad[i, :tl] = b.target_ids[0]

        self._univ_after_target_padded = after_target_pad          # [N, max_after+max_target, d]
        self._univ_after_target_mask = after_target_mask            # [N, max_after+max_target]
        self._univ_target_ids_padded = target_ids_pad               # [N, max_target]
        self._univ_target_lens = torch.tensor(target_lens, device=dev, dtype=torch.long)  # [N]
        self._univ_max_after = max_after
        self._univ_max_target = max_target
        self._univ_prefix_len = b0.before_ids.shape[1]
        # Shared prefix KV: reuse bundle 0's snapshot (identical across bundles
        # because before_ids match). Materialized fresh per-forward via
        # `_fresh_prefix_cache(expand_to=batch, bundle=b0)`.
        self._univ_prefix_bundle = b0
        logger.info(
            f"Universal fast path ON: N={N}, shared prefix_len={self._univ_prefix_len}, "
            f"max_after={max_after}, max_target={max_target}."
        )

    def _fresh_prefix_cache(self, expand_to: int = None, bundle: Optional[PromptBundle] = None):
        """Rebuild a fresh DynamicCache from the stable prefix-tensor snapshot.

        The model mutates whatever past_key_values it receives (appending new
        KVs via torch.cat). Every model() call that wants the prefix cached
        gets a fresh wrapper whose internal key_cache/value_cache lists are
        NEW (so the model's torch.cat-into-list-element mutates this throwaway
        wrapper, not the snapshot tensors).

        If expand_to is given, also expand each prefix tensor's batch dim to
        match the candidate batch size (used in candidate-loss eval). Expanding
        is a view, not a copy.

        `bundle` selects which prompt's prefix snapshot to use for universal
        GCG; None falls back to self.prefix_legacy (single-prompt / primary
        bundle) for backward compatibility.
        """
        from transformers.cache_utils import DynamicCache
        legacy = bundle.prefix_legacy if bundle is not None else self.prefix_legacy
        if expand_to is None or expand_to == 1:
            layer_kvs = legacy
        else:
            layer_kvs = tuple(
                (k.expand(expand_to, -1, -1, -1), v.expand(expand_to, -1, -1, -1))
                for (k, v) in legacy
            )
        # transformers compat: from_legacy_cache() classmethod was removed in
        # 4.55+. Fall back to incremental update() which is portable.
        if hasattr(DynamicCache, "from_legacy_cache"):
            return DynamicCache.from_legacy_cache(layer_kvs)
        cache = DynamicCache()
        for i, (k, v) in enumerate(layer_kvs):
            cache.update(k, v, layer_idx=i)
        return cache

    def init_buffer(self) -> AttackBuffer:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config.buffer_size)

        if self.optim_region_lens is not None:
            # Multi-region: concatenate all region inits into one vector. Only a
            # single buffer entry is supported (buffer_size in {0, 1}).
            if config.buffer_size > 1:
                raise NotImplementedError("multi-region GCG supports only buffer_size in {0, 1}")
            region_ids = [tokenizer(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
                          for s in config.optim_str_inits]
            assert [x.shape[1] for x in region_ids] == self.optim_region_lens, "region init length mismatch"
            init_buffer_ids = torch.cat(region_ids, dim=1)
        elif isinstance(config.optim_str_init, str):
            init_optim_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            if config.buffer_size > 1:
                init_buffer_chars = tokenizer(INIT_CHARS, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze().to(model.device)
                n_needed = config.buffer_size - 1
                n_optim_ids = init_optim_ids.shape[1]
                if config.filter_ids:
                    # Generate buffer entries by perturbing the user's init at
                    # a small random subset of positions with random INIT_CHARS
                    # tokens. Purely-random INIT_CHARS sequences essentially
                    # never survive decode/re-encode round-trip on Llama 3's
                    # BPE (empirically 0%); perturbing the user init preserves
                    # its spacing structure and round-trips ~35% of the time,
                    # so oversampling gets us enough valid entries. Oversample
                    # and drop ones that fail round-trip.
                    n_perturb = max(1, n_optim_ids // 4)
                    oversample_factor = 8
                    max_attempts = 4
                    valid_random = init_optim_ids.new_empty((0, n_optim_ids))
                    for _ in range(max_attempts):
                        if valid_random.shape[0] >= n_needed:
                            break
                        n_candidates = n_needed * oversample_factor
                        base = init_optim_ids.repeat(n_candidates, 1)
                        # Random distinct positions to perturb in each row.
                        positions = torch.stack([
                            torch.randperm(n_optim_ids, device=base.device)[:n_perturb]
                            for _ in range(n_candidates)
                        ])
                        values = init_buffer_chars[
                            torch.randint(0, init_buffer_chars.shape[0], (n_candidates, n_perturb), device=base.device)
                        ]
                        base = base.scatter(1, positions, values)
                        valid_random = torch.cat(
                            [valid_random, filter_ids(base, tokenizer, raise_on_empty=False)],
                            dim=0,
                        )
                    if valid_random.shape[0] < n_needed:
                        logger.warning(
                            f"Only {valid_random.shape[0]}/{n_needed} perturbed buffer entries "
                            f"survived round-trip tokenization after {max_attempts} attempts; "
                            f"buffer will be smaller than requested."
                        )
                    n_filled = min(valid_random.shape[0], n_needed)
                    init_buffer_ids = torch.cat([init_optim_ids, valid_random[:n_filled]], dim=0)
                else:
                    # Legacy behavior: pure-random INIT_CHARS sequences. Fine
                    # when filter_ids is off (caller isn't relying on round-trip).
                    init_indices = torch.randint(0, init_buffer_chars.shape[0], (n_needed, n_optim_ids))
                    init_buffer_ids = torch.cat([init_optim_ids, init_buffer_chars[init_indices]], dim=0)
            else:
                init_buffer_ids = init_optim_ids

        else:  # assume list
            if len(config.optim_str_init) != config.buffer_size:
                logger.warning(f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}")
            try:
                init_buffer_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            except ValueError:
                logger.error("Unable to create buffer. Ensure that all initializations tokenize to the same length.")

        # Buffer may be smaller than requested if random-entry filtering came up short.
        true_buffer_size = max(1, init_buffer_ids.shape[0])

        # Compute the loss on the initial buffer entries, averaged across all
        # bundles (single-prompt: just the primary bundle). Uses the same
        # helper as the main loop so the arithmetic matches exactly.
        init_buffer_losses = self._compute_candidates_loss_all_bundles(
            init_buffer_ids, true_buffer_size,
        )

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        buffer.log_buffer(tokenizer)

        logger.info("Initialized attack buffer.")

        return buffer

    def _optim_segments(self, optim_embeds_full: Tensor, bundle: PromptBundle) -> List[Tensor]:
        """Split the (batched) optim embeddings into the segments to splice into
        the prompt. Single-region: `[optim]`. Multi-region: interleave the K
        region slices with the K-1 fixed per-bundle mid segments (broadcast to
        the batch dim): `[r0, mid0, r1, mid1, ..., r_{K-1}]`. optim_embeds_full
        has shape [B, n_total, d].
        """
        if self.optim_region_lens is None:
            return [optim_embeds_full]
        B = optim_embeds_full.shape[0]
        segs = []
        idx = 0
        K = len(self.optim_region_lens)
        for i, L in enumerate(self.optim_region_lens):
            segs.append(optim_embeds_full[:, idx: idx + L, :])
            idx += L
            if i < K - 1:
                segs.append(bundle.region_mids_embeds[i].expand(B, -1, -1))
        return segs

    def compute_token_gradient(
        self,
        optim_ids: Tensor,
    ) -> Tensor:
        """Computes the gradient of the GCG loss w.r.t. the one-hot token matrix.

        Universal-capable: accumulates the mean cross-entropy across all bundles
        in `self.bundles`, backpropagates ONCE, returns a single gradient tensor
        shaped like the one-hot input. In the single-prompt case (len(bundles)
        == 1) this reduces to the pre-universal computation exactly.

        Args:
            optim_ids : Tensor, shape = (1, n_optim_ids)
                the sequence of token ids that are being optimized
        """
        model = self.model
        embedding_layer = self.embedding_layer

        optim_ids_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=embedding_layer.num_embeddings)
        optim_ids_onehot = optim_ids_onehot.to(model.device, model.dtype)
        optim_ids_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        # Fast path: shared-prefix cross-bundle batched gradient. One
        # forward+backward covers all N bundles instead of N serial pairs.
        # Skipped when mellowmax is on (its per-position aggregation is not
        # trivially masked across padded target positions — the slow path
        # already handles it correctly).
        if self._univ_shared and self.config.use_prefix_cache and not self.config.use_mellowmax:
            N = len(self.bundles)
            n_adv = optim_embeds.shape[1]
            max_after = self._univ_max_after
            max_target = self._univ_max_target
            prefix_len = self._univ_prefix_len
            tail_len = n_adv + max_after + max_target

            adv_expand = optim_embeds.expand(N, n_adv, -1)                              # [N, n_adv, d]
            at_expand = self._univ_after_target_padded                                   # [N, MA+MT, d]
            tail_embeds = torch.cat([adv_expand, at_expand], dim=1)                      # [N, tail_len, d]

            adv_mask = torch.ones(N, n_adv, device=tail_embeds.device, dtype=torch.long)
            at_mask = self._univ_after_target_mask                                       # [N, MA+MT]
            tail_mask = torch.cat([adv_mask, at_mask], dim=1)                            # [N, tail_len]
            prefix_mask = torch.ones(N, prefix_len, device=tail_embeds.device, dtype=torch.long)
            full_attn = torch.cat([prefix_mask, tail_mask], dim=1)

            position_ids = torch.arange(prefix_len, prefix_len + tail_len,
                                        device=tail_embeds.device).unsqueeze(0).expand(N, -1)

            prefix_cache = self._fresh_prefix_cache(expand_to=N, bundle=self._univ_prefix_bundle)
            output = model(
                inputs_embeds=tail_embeds,
                attention_mask=full_attn,
                position_ids=position_ids,
                past_key_values=prefix_cache,
                use_cache=True,
            )
            logits = output.logits                                                       # [N, tail_len, V]
            target_start = n_adv + max_after
            shift_logits = logits[:, target_start - 1 : target_start - 1 + max_target, :]
            V = shift_logits.size(-1)
            labels = self._univ_target_ids_padded                                        # [N, max_target]
            pos_range = torch.arange(max_target, device=tail_embeds.device).unsqueeze(0)
            per_bundle_valid = (pos_range < self._univ_target_lens.unsqueeze(1)).to(shift_logits.dtype)
            ce = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, V),
                labels.reshape(-1),
                reduction="none",
            ).view(N, max_target)
            per_row_loss = (ce * per_bundle_valid).sum(dim=1) / per_bundle_valid.sum(dim=1).clamp(min=1)
            loss = per_row_loss.mean()
            optim_ids_onehot_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]
            return optim_ids_onehot_grad

        # Slow path (per-bundle loop). Accumulate the gradient bundle-by-bundle
        # with a SEPARATE backward per bundle, freeing each bundle's autograd
        # graph immediately. To avoid freeing the SHARED `optim_embeds = onehot @ W`
        # node (which every bundle's forward reads), each backward targets
        # `optim_embeds` (not the one-hot): this traverses/free only that bundle's
        # forward and leaves the shared matmul intact. Since the embedding map is
        # linear, d loss/d onehot = (d loss/d optim_embeds) @ W^T exactly, done
        # once at the end. Result is identical to one backward over the mean loss,
        # but peak memory is ONE bundle's activations instead of N — the
        # single-backward version holds N x activations and crawls under memory
        # pressure (the dominant cost of universal GCG); this keeps it flat.
        bundles = self.bundles
        grad_embeds_accum = None
        for b in bundles:
            optim_seg = self._optim_segments(optim_embeds, b)   # [optim] or [optim_q, mid, optim_opt]
            nt = b.target_ids.shape[1]
            keep_kw = {"logits_to_keep": nt + 1} if self.config.target_logits_only else {}
            if b.prefix_cache:
                input_embeds = torch.cat([*optim_seg, b.after_embeds, b.target_embeds], dim=1)
                output = model(
                    inputs_embeds=input_embeds,
                    past_key_values=self._fresh_prefix_cache(bundle=b),
                    use_cache=True,
                    **keep_kw,
                )
            else:
                input_embeds = torch.cat(
                    [b.before_embeds, *optim_seg, b.after_embeds, b.target_embeds],
                    dim=1,
                )
                output = model(inputs_embeds=input_embeds, **keep_kw)

            logits = output.logits

            # Shift logits so token n-1 predicts token n. With target_logits_only,
            # logits already cover only the last nt+1 positions -> the first nt of
            # them predict the nt target tokens.
            if self.config.target_logits_only:
                shift_logits = logits[:, :-1, :].contiguous()
            else:
                shift = input_embeds.shape[1] - nt
                shift_logits = logits[..., shift - 1 : -1, :].contiguous()
            shift_labels = b.target_ids

            if self.config.use_mellowmax:
                label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                loss_b = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
            else:
                loss_b = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # backward for THIS bundle only, to the shared optim_embeds (not the
            # one-hot), then release this bundle's forward graph.
            ge = torch.autograd.grad(outputs=[loss_b], inputs=[optim_embeds], retain_graph=False)[0]
            grad_embeds_accum = ge if grad_embeds_accum is None else grad_embeds_accum + ge
            del output, logits, shift_logits, loss_b, ge

        # map accumulated grad wrt embeds back to the one-hot analytically:
        # optim_embeds = onehot @ W  =>  d loss/d onehot = (d loss/d optim_embeds) @ W^T.
        grad_embeds_accum = grad_embeds_accum / len(bundles)
        optim_ids_onehot_grad = grad_embeds_accum.to(embedding_layer.weight.dtype) @ embedding_layer.weight.t()

        return optim_ids_onehot_grad

    def _i_gcg_merge_step(
        self,
        sampled_ids: Tensor,
        single_token_losses: Tensor,
        current_optim_ids: Tensor,
        batch_size: int,
    ) -> Tuple[float, Tensor]:
        """I-GCG multi-coordinate update (Jia et al. 2024, Eq. 8 / Algo. 1).

        Sort the single-token candidates by loss, then cumulatively merge the
        top-p lowest-loss candidates' position-wise diffs into the current
        suffix. Re-score the p merged variants and return the best.

        Universal-capable: the re-scoring goes through
        `_compute_candidates_loss_all_bundles` so merged variants are evaluated
        against every bundle (averaged), identical to the primary optimization
        loss. In the single-prompt case this reduces to the pre-universal path.
        """
        p = min(self.config.i_gcg_top_p, sampled_ids.shape[0])
        # Indices of the p lowest-loss single-token candidates, sorted ascending.
        top_idx = torch.topk(single_token_losses, p, largest=False).indices
        top_idx = top_idx[torch.argsort(single_token_losses[top_idx])]

        original_ids = current_optim_ids.squeeze(0)  # (n_optim_ids,)
        merged_ids = original_ids.clone()
        merged_candidates = []
        for i in range(p):
            cand = sampled_ids[top_idx[i]]
            # Accept any position where this candidate differs from the
            # *original* suffix. Since GCG samples with n_replace=1, each
            # candidate changes exactly one position vs. the original, so this
            # cumulatively unions those changed positions into merged_ids.
            diff_mask = cand != original_ids
            merged_ids = torch.where(diff_mask, cand, merged_ids)
            merged_candidates.append(merged_ids.clone())
        merged_candidates = torch.stack(merged_candidates, dim=0)

        merged_losses = self._compute_candidates_loss_all_bundles(
            merged_candidates, min(batch_size, p),
        )
        best = merged_losses.argmin()
        return merged_losses[best].item(), merged_candidates[best].unsqueeze(0)

    def _compute_candidates_loss_all_bundles(
        self,
        sampled_ids: Tensor,
        batch_size: int,
    ) -> Tensor:
        """Mean per-candidate loss across all bundles.

        Fast path (self._univ_shared): cross-bundle batching. See
        `_setup_universal_shared_prefix`. Per chunk of C candidates we build a
        single [N*C, n_adv + max_after + max_target, d] embedding tensor,
        forward it once through the SHARED prefix cache (broadcast to N*C),
        then extract per-bundle-row logits at the target span (identical
        position across bundles by construction) and compute per-row CE.
        Result shape: [S]. Falls back to the serial per-bundle path when the
        universal fast path is not available (single-prompt runs, or bundles
        with differing before_ids).

        Slow path: iterates `self.bundles`, builds input embeddings per
        bundle, and sums the returned per-candidate loss vectors, then
        divides by the number of bundles. Returns a Tensor of shape [S]
        where S = sampled_ids.shape[0]. Single-prompt path
        (len(bundles) == 1) is bit-identical to the pre-universal
        `_compute_candidates_loss_original` pipeline modulo one extra
        division-by-1.
        """
        if self._univ_shared:
            return self._compute_candidates_loss_universal_fast(sampled_ids, batch_size)

        S = sampled_ids.shape[0]
        embedding_layer = self.embedding_layer
        optim_embeds = embedding_layer(sampled_ids)   # [S, n_total, d]
        total = None
        for b in self.bundles:
            optim_seg = self._optim_segments(optim_embeds, b)   # [optim] or [optim_q, mid, optim_opt]
            if b.prefix_cache:
                input_embeds = torch.cat([
                    *optim_seg,
                    b.after_embeds.repeat(S, 1, 1),
                    b.target_embeds.repeat(S, 1, 1),
                ], dim=1)
            else:
                input_embeds = torch.cat([
                    b.before_embeds.repeat(S, 1, 1),
                    *optim_seg,
                    b.after_embeds.repeat(S, 1, 1),
                    b.target_embeds.repeat(S, 1, 1),
                ], dim=1)
            losses = find_executable_batch_size(self._compute_candidates_loss_original, batch_size)(
                input_embeds, bundle=b,
            )
            total = losses if total is None else total + losses
        return total / len(self.bundles)

    def _compute_candidates_loss_universal_fast(
        self,
        sampled_ids: Tensor,
        batch_size: int,
    ) -> Tensor:
        """Cross-bundle batched candidate eval. Requires `_univ_shared=True`.

        Layout of one row in the batched tail (fixed per this call, shared
        across all rows since bundles share the prefix):
            [adv (n_adv), after_b (real, pad to max_after), target_b (real, pad to max_target)]
        - `after` padding sits at positions [n_adv + after_len_b, n_adv + max_after)
        - `target` padding sits at positions
            [n_adv + max_after + target_len_b, n_adv + max_after + max_target)
        Both pad regions are masked out via attention_mask. Because we pad
        `after` up to `max_after` BEFORE the target span, target tokens start
        at position `n_adv + max_after` in the tail for every bundle — so the
        CE slice on the returned logits is bundle-independent in START
        position and only differs in length.

        Returns per-candidate mean-across-bundles loss, shape [S].
        """
        N = len(self.bundles)
        S = sampled_ids.shape[0]
        embedding_layer = self.embedding_layer
        max_after = self._univ_max_after
        max_target = self._univ_max_target
        n_adv = sampled_ids.shape[1]
        prefix_len = self._univ_prefix_len
        tail_len = n_adv + max_after + max_target

        # Adv embeds (shared across all N × S rows for a given candidate c):
        # [S, n_adv, d]. Broadcast across N later.
        adv_embeds = embedding_layer(sampled_ids)                        # [S, n_adv, d]
        # After+target for each bundle (precomputed padded): [N, max_after+max_target, d].
        # Expand to [N, S, ...] then concat with per-candidate adv embeds.
        d = adv_embeds.shape[-1]
        # tail_embeds shape [N, S, tail_len, d]
        adv_expand = adv_embeds.unsqueeze(0).expand(N, S, n_adv, d)                   # [N, S, n_adv, d]
        at_expand = self._univ_after_target_padded.unsqueeze(1).expand(N, S, max_after + max_target, d)  # [N, S, MA+MT, d]
        tail_embeds = torch.cat([adv_expand, at_expand], dim=2)                       # [N, S, tail_len, d]
        tail_embeds = tail_embeds.reshape(N * S, tail_len, d)

        # Per-row attention mask over the tail:
        #   adv region (n_adv): all 1s (real)
        #   after+target region: from self._univ_after_target_mask expanded to [N, S, MA+MT]
        adv_mask = torch.ones(N, S, n_adv, device=tail_embeds.device, dtype=torch.long)
        at_mask = self._univ_after_target_mask.unsqueeze(1).expand(N, S, max_after + max_target)
        tail_mask = torch.cat([adv_mask, at_mask], dim=2).reshape(N * S, tail_len)    # [N*S, tail_len]
        # Full attention mask = prefix (all 1s) concat tail_mask
        prefix_mask = torch.ones(N * S, prefix_len, device=tail_embeds.device, dtype=torch.long)
        full_attn = torch.cat([prefix_mask, tail_mask], dim=1)                        # [N*S, prefix_len+tail_len]

        # Per-row position ids for the tail: [prefix_len, prefix_len+1, ..., prefix_len+tail_len-1].
        # SAME across rows because prefix length is shared and we use natural
        # (continuous) positions for the padded slots too (padded slots don't
        # affect anything because they're masked out of attention).
        position_ids = torch.arange(prefix_len, prefix_len + tail_len,
                                    device=tail_embeds.device).unsqueeze(0).expand(N * S, -1)

        # Target position span (in tail-local coordinates): starts at n_adv + max_after,
        # length target_lens[b] for row b. In the returned logits (which cover
        # only the tail positions), the logits predicting target token j come
        # from position (n_adv + max_after + j - 1). So the CE slice for
        # bundle b uses logits[row, n_adv + max_after - 1 : n_adv + max_after - 1 + target_len_b, :]
        # against target_ids_padded[b, :target_len_b].
        target_start_in_tail = n_adv + max_after
        target_ids_padded = self._univ_target_ids_padded                              # [N, max_target]
        target_lens = self._univ_target_lens                                          # [N]
        # Target token ids replicated per candidate: [N, S, max_target]
        target_ids_batched = target_ids_padded.unsqueeze(1).expand(N, S, max_target)  # [N, S, MT]

        # A per-row target-position mask: 1 for the first target_len_b positions, 0 after.
        # Used to zero-out CE on padded target positions.
        pos_range = torch.arange(max_target, device=tail_embeds.device).unsqueeze(0)  # [1, MT]
        per_bundle_valid = (pos_range < target_lens.unsqueeze(1)).to(torch.float32)   # [N, MT]
        valid_mask = per_bundle_valid.unsqueeze(1).expand(N, S, max_target).reshape(N * S, max_target)

        all_losses = []
        rows_total = N * S
        for start in range(0, rows_total, batch_size):
            end = min(start + batch_size, rows_total)
            cb = end - start
            with torch.no_grad():
                prefix_cache = self._fresh_prefix_cache(expand_to=cb, bundle=self._univ_prefix_bundle)
                outputs = self.model(
                    inputs_embeds=tail_embeds[start:end],
                    attention_mask=full_attn[start:end],
                    position_ids=position_ids[start:end],
                    past_key_values=prefix_cache,
                    use_cache=True,
                )
                logits = outputs.logits                                                 # [cb, tail_len, V]
                # Slice: predictions for target tokens 0..max_target-1
                shift_logits = logits[:, target_start_in_tail - 1 : target_start_in_tail - 1 + max_target, :]
                # shape [cb, max_target, V]
                V = shift_logits.size(-1)
                labels = target_ids_batched.reshape(N * S, max_target)[start:end]       # [cb, max_target]
                # CE per position (no reduction); then mask + mean over valid positions per row.
                ce = torch.nn.functional.cross_entropy(
                    shift_logits.reshape(-1, V),
                    labels.reshape(-1),
                    reduction="none",
                ).view(cb, max_target)                                                  # [cb, max_target]
                mask_row = valid_mask[start:end]                                        # [cb, max_target]
                per_row_loss = (ce * mask_row).sum(dim=1) / mask_row.sum(dim=1).clamp(min=1)
                all_losses.append(per_row_loss)
                del outputs, logits, shift_logits

        # Hoisted like the serial path: one sync per call, not per chunk.
        gc.collect()
        torch.cuda.empty_cache()

        per_row = torch.cat(all_losses, dim=0)                                        # [N*S]
        per_row = per_row.view(N, S)                                                  # [N, S]
        return per_row.mean(dim=0)                                                    # [S]

    def _compute_candidates_loss_original(
        self,
        search_batch_size: int,
        input_embeds: Tensor,
        bundle: Optional[PromptBundle] = None,
    ) -> Tensor:
        """Computes the GCG loss on all candidate token id sequences.

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            input_embeds : Tensor, shape = (search_width, seq_len, embd_dim)
                the embeddings of the `search_width` candidate sequences to evaluate
            bundle : optional PromptBundle
                if provided, use this bundle's prefix cache and target_ids
                instead of self.prefix_legacy / self.target_ids. Required for
                universal-GCG; None keeps the pre-universal (primary-bundle)
                behavior.
        """
        target_ids = bundle.target_ids if bundle is not None else self.target_ids
        has_prefix_cache = (bundle.prefix_cache if bundle is not None else self.prefix_cache)
        nt = target_ids.shape[1]
        # logits_to_keep: target is at the END, so only the last nt+1 positions'
        # logits are needed. Skips the lm_head over the vocab at every other
        # position (~100x on the dominant cost). Numerically identical.
        keep_kw = {"logits_to_keep": nt + 1} if self.config.target_logits_only else {}

        all_loss = []

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i + search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]

                if has_prefix_cache:
                    # Fresh cache per inner batch: the model mutates the passed-in
                    # DynamicCache by appending KVs during forward, so reusing
                    # across inner batches would feed earlier candidates' target
                    # tokens into the cache and silently boost P(target) on
                    # later candidates. Build a throwaway fresh wrapper around
                    # the immutable prefix-tensor snapshot every time.
                    prefix_cache_batch = self._fresh_prefix_cache(expand_to=current_batch_size, bundle=bundle)
                    outputs = self.model(inputs_embeds=input_embeds_batch, past_key_values=prefix_cache_batch, use_cache=True, **keep_kw)
                else:
                    outputs = self.model(inputs_embeds=input_embeds_batch, **keep_kw)

                logits = outputs.logits

                if self.config.target_logits_only:
                    # logits already cover only the last nt+1 positions
                    shift_logits = logits[:, :-1, :].contiguous()
                else:
                    tmp = input_embeds.shape[1] - nt
                    shift_logits = logits[..., tmp-1:-1, :].contiguous()
                shift_labels = target_ids.repeat(current_batch_size, 1)

                if self.config.use_mellowmax:
                    label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                    loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
                else:
                    loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")

                loss = loss.view(current_batch_size, -1).mean(dim=-1)
                all_loss.append(loss)

                if self.config.early_stop:
                    if torch.any(torch.all(torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1)).item():
                        self.stop_flag = True

                del outputs

        # Hoisted out of the per-inner-batch loop: gc.collect + empty_cache
        # each sync + release ~0.5s on MI210. In single-prompt runs the inner
        # loop is ~4 iterations so the old per-iter cost was negligible; in
        # universal-GCG with N bundles the outer loop calls this function
        # N times per step, and each inner iteration used to pay for a full
        # sync — the cumulative cost dominated wall time (100 syncs/step at
        # N=25, sw=256, bs=64). One sync per call is enough for memory
        # hygiene without paying for it every chunk.
        gc.collect()
        torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0)

    def _compute_candidates_loss_probe_sampling(
        self,
        search_batch_size: int,
        input_embeds: Tensor,
        sampled_ids: Tensor,
    ) -> Tuple[float, Tensor]:
        """Computes the GCG loss using probe sampling (https://arxiv.org/abs/2403.01251).

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            input_embeds : Tensor, shape = (search_width, seq_len, embd_dim)
                the embeddings of the `search_width` candidate sequences to evaluate
            sampled_ids: Tensor, all candidate token id sequences. Used for further sampling.

        Returns:
            A tuple of (min_loss: float, corresponding_sequence: Tensor)

        """
        probe_sampling_config = self.config.probe_sampling_config
        assert probe_sampling_config, "Probe sampling config wasn't set up properly."

        B = input_embeds.shape[0]
        probe_size = B // probe_sampling_config.sampling_factor
        probe_idxs = torch.randperm(B)[:probe_size].to(input_embeds.device)
        probe_embeds = input_embeds[probe_idxs]

        def _compute_probe_losses(result_queue: queue.Queue, search_batch_size: int, probe_embeds: Tensor) -> None:
            probe_losses = self._compute_candidates_loss_original(search_batch_size, probe_embeds)
            result_queue.put(("probe", probe_losses))

        def _compute_draft_losses(
            result_queue: queue.Queue,
            search_batch_size: int,
            draft_sampled_ids: Tensor,
        ) -> None:
            assert self.draft_model and self.draft_embedding_layer, "Draft model and embedding layer weren't initialized properly."

            draft_losses = []
            draft_prefix_cache_batch = None
            for i in range(0, B, search_batch_size):
                with torch.no_grad():
                    batch_size = min(search_batch_size, B - i)
                    draft_sampled_ids_batch = draft_sampled_ids[i : i + batch_size]

                    if self.draft_prefix_cache:
                        if not draft_prefix_cache_batch or batch_size != search_batch_size:
                            legacy = tuple(
                                tuple(x.expand(batch_size, -1, -1, -1) for x in self.draft_prefix_cache[i])
                                for i in range(len(self.draft_prefix_cache))
                            )
                            from transformers.cache_utils import DynamicCache
                            if hasattr(DynamicCache, "from_legacy_cache"):
                                draft_prefix_cache_batch = DynamicCache.from_legacy_cache(legacy)
                            else:
                                draft_prefix_cache_batch = DynamicCache()
                                for _li, (_k, _v) in enumerate(legacy):
                                    draft_prefix_cache_batch.update(_k, _v, layer_idx=_li)
                        draft_embeds = torch.cat(
                            [
                                self.draft_embedding_layer(draft_sampled_ids_batch),
                                self.draft_after_embeds.repeat(batch_size, 1, 1),
                                self.draft_target_embeds.repeat(batch_size, 1, 1),
                            ],
                            dim=1,
                        )
                        draft_output = self.draft_model(
                            inputs_embeds=draft_embeds,
                            past_key_values=draft_prefix_cache_batch,
                        )
                    else:
                        draft_embeds = torch.cat(
                            [
                                self.draft_before_embeds.repeat(batch_size, 1, 1),
                                self.draft_embedding_layer(draft_sampled_ids_batch),
                                self.draft_after_embeds.repeat(batch_size, 1, 1),
                                self.draft_target_embeds.repeat(batch_size, 1, 1),
                            ],
                            dim=1,
                        )
                        draft_output = self.draft_model(inputs_embeds=draft_embeds)

                    draft_logits = draft_output.logits
                    tmp = draft_embeds.shape[1] - self.draft_target_ids.shape[1]
                    shift_logits = draft_logits[..., tmp - 1 : -1, :].contiguous()
                    shift_labels = self.draft_target_ids.repeat(batch_size, 1)

                    if self.config.use_mellowmax:
                        label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                        loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
                    else:
                        loss = (
                            torch.nn.functional.cross_entropy(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                                reduction="none",
                            )
                            .view(batch_size, -1)
                            .mean(dim=-1)
                        )

                    draft_losses.append(loss)

            draft_losses = torch.cat(draft_losses)
            result_queue.put(("draft", draft_losses))

        def _convert_to_draft_tokens(token_ids: Tensor) -> Tensor:
            decoded_text_list = self.tokenizer.batch_decode(token_ids)
            assert self.draft_tokenizer, "Draft tokenizer wasn't properly initialized."
            return self.draft_tokenizer(
                decoded_text_list,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )[
                "input_ids"
            ].to(self.draft_model.device, torch.int64)

        result_queue = queue.Queue()
        draft_sampled_ids = _convert_to_draft_tokens(sampled_ids)

        # Step 1. Compute loss of all candidates using the draft model
        draft_thread = threading.Thread(
            target=_compute_draft_losses,
            args=(result_queue, search_batch_size, draft_sampled_ids),
        )

        # Step 2. In parallel to 1., compute loss of the probe set on target model
        probe_thread = threading.Thread(
            target=_compute_probe_losses,
            args=(result_queue, search_batch_size, probe_embeds),
        )

        draft_thread.start()
        probe_thread.start()

        draft_thread.join()
        probe_thread.join()

        results = {}
        while not result_queue.empty():
            key, value = result_queue.get()
            results[key] = value

        probe_losses = results["probe"]
        draft_losses = results["draft"]

        # Step 3. Calculate agreement score using Spearman correlation
        draft_probe_losses = draft_losses[probe_idxs]
        rank_correlation = spearmanr(
            probe_losses.cpu().type(torch.float32).numpy(),
            draft_probe_losses.cpu().type(torch.float32).numpy(),
        ).correlation
        # normalized from [-1, 1] to [0, 1]
        alpha = (1 + rank_correlation) / 2

        # Step 4. Calculate the filtered set and evaluate using the target model.
        R = probe_sampling_config.r
        filtered_size = int((1 - alpha) * B / R)
        filtered_size = max(1, min(filtered_size, B))

        _, top_indices = torch.topk(draft_losses, k=filtered_size, largest=False)

        filtered_embeds = input_embeds[top_indices]
        filtered_losses = self._compute_candidates_loss_original(search_batch_size, filtered_embeds)

        # Step 5. Return best loss between probe set and filtered set
        best_probe_loss = probe_losses.min().item()
        best_filtered_loss = filtered_losses.min().item()

        probe_ids = sampled_ids[probe_idxs]
        filtered_ids = sampled_ids[top_indices]
        return (
            (best_probe_loss, probe_ids[probe_losses.argmin()].unsqueeze(0))
            if best_probe_loss < best_filtered_loss
            else (
                best_filtered_loss,
                filtered_ids[filtered_losses.argmin()].unsqueeze(0),
            )
        )


# A wrapper around the GCG `run` method that provides a simple API
def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    target: str,
    config: Optional[GCGConfig] = None,
) -> GCGResult:
    """Generates a single optimized string using GCG.

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization.
        target: The target generation.
        config: The GCG configuration to use.

    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = GCGConfig()

    logger.setLevel(getattr(logging, config.verbosity))

    gcg = GCG(model, tokenizer, config)
    result = gcg.run(messages, target)
    return result


def run_universal(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    prompt_targets: List[Tuple[Union[str, List[dict]], str]],
    config: Optional[GCGConfig] = None,
) -> GCGResult:
    """Universal (multi-prompt) GCG: optimize a single suffix that jointly
    minimizes GCG loss across many (messages, target) pairs.

    Follows Zou et al. 2023 §3.3: at each step, the loss and gradient used to
    drive candidate sampling are the MEAN across the given prompts (each
    prompt independently splits `messages` on its `{optim_str}` slot, or
    appends it, using the model's chat template).

    All prompts share this tokenizer/model. `probe_sampling_config` is
    incompatible with universal GCG and will raise. All other GCGConfig
    fields (I-GCG merge, prefix cache, filter_ids boundary intersection,
    early-stop callback, buffer, etc.) work unchanged.

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        prompt_targets: List of (messages, target) pairs. Non-empty.
        config: The GCG configuration.

    Returns:
        A GCGResult where `best_loss` is the min mean-across-prompts loss
        observed, and `best_string` is the corresponding decoded suffix.
    """
    if config is None:
        config = GCGConfig()

    logger.setLevel(getattr(logging, config.verbosity))

    gcg = GCG(model, tokenizer, config)
    return gcg.run_universal(prompt_targets)
