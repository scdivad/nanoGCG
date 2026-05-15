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

    Returns:
        filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
            all token ids that are the same after retokenization
    """
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []

    for i in range(len(ids_decoded)):
        # Retokenize the decoded token ids
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
        self.not_allowed_ids = None if config.allow_non_ascii else get_nonascii_toks(tokenizer, device=model.device)
        self.prefix_cache = None
        self.draft_prefix_cache = None

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
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)

        # Append the GCG string at the end of the prompt if location not specified
        if not any(["{optim_str}" in d["content"] for d in messages]):
            messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

        template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Remove the BOS token -- this will get added when tokenizing, if necessary
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "")
        before_str, after_str = template.split("{optim_str}")

        target = " " + target if config.add_space_before_target else target

        # Tokenize everything that doesn't get optimized
        before_ids = tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        target_ids = tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)

        # Embed everything that doesn't get optimized
        embedding_layer = self.embedding_layer
        before_embeds, after_embeds, target_embeds = [embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)]

        # Compute the KV Cache for tokens that appear before the optimized tokens.
        # IMPORTANT (transformers>=4.40): the model mutates whatever Cache object
        # it's handed during the forward pass — torch.cat'ing new KVs into the
        # cache's internal lists. Reusing self.prefix_cache across iterations
        # therefore pollutes it with previous-step optim/after/target KVs, which
        # silently boosts P(target) on subsequent steps (because attention now
        # has the target token sitting in its own past) and reports artificially
        # low loss values that don't transfer to inference. Snapshot the legacy
        # tuple form (which holds references to the original "before"-only K/V
        # tensors) and rebuild a fresh DynamicCache from it on every model call.
        if config.use_prefix_cache:
            with torch.no_grad():
                output = model(inputs_embeds=before_embeds, use_cache=True)
                cache = output.past_key_values
                # Snapshot. to_legacy_cache() returns a tuple of (k, v) tuples
                # referencing the current key_cache[i] / value_cache[i] tensors;
                # those tensors stay alive as long as we hold the snapshot, and
                # they won't be mutated since we never pass self.prefix_legacy
                # directly to model().
                self.prefix_legacy = cache.to_legacy_cache()
                # Keep self.prefix_cache truthy for the `if self.prefix_cache:`
                # gates throughout the codebase, but never pass it to model().
                self.prefix_cache = cache

        self.target_ids = target_ids
        self.before_embeds = before_embeds
        self.after_embeds = after_embeds
        self.target_embeds = target_embeds

        # Initialize components for probe sampling, if enabled.
        if config.probe_sampling_config:
            assert self.draft_model and self.draft_tokenizer and self.draft_embedding_layer, "Draft model wasn't properly set up."

            # Tokenize everything that doesn't get optimized for the draft model
            draft_before_ids = self.draft_tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
            draft_after_ids = self.draft_tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
            self.draft_target_ids = self.draft_tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)

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

        # Initialize the attack buffer
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []

        for step in tqdm(range(config.num_steps)):
            # Compute the token gradient
            optim_ids_onehot_grad = self.compute_token_gradient(optim_ids)

            with torch.no_grad():

                # Sample candidate token sequences based on the token gradient.
                # If filter_ids is on and nothing survives round-trip, retry a
                # few times — the sampling is stochastic (random positions +
                # random topk picks), so a fresh draw usually surfaces at
                # least some surviving candidates. Only if all retries fail do
                # we skip this step (keep current optim_ids, move on).
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
                        candidates = filter_ids(candidates, tokenizer, raise_on_empty=False)
                    if candidates.shape[0] > 0:
                        sampled_ids = candidates
                        break
                if sampled_ids is None:
                    logger.warning(
                        f"Step {step + 1}: no sampled candidates survived `filter_ids` "
                        f"after {filter_attempts} retries; skipping this step."
                    )
                    # Record the unchanged loss so the step is still reflected in
                    # the losses list (otherwise len(losses) != num_steps).
                    losses.append(buffer.get_lowest_loss())
                    optim_str = tokenizer.batch_decode(buffer.get_best_ids())[0]
                    optim_strings.append(optim_str)
                    continue

                new_search_width = sampled_ids.shape[0]

                # Compute loss on all candidate sequences
                batch_size = new_search_width if config.batch_size is None else config.batch_size
                if self.prefix_cache:
                    input_embeds = torch.cat([
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)
                else:
                    input_embeds = torch.cat([
                        before_embeds.repeat(new_search_width, 1, 1),
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)

                if self.config.probe_sampling_config is None:
                    loss = find_executable_batch_size(self._compute_candidates_loss_original, batch_size)(input_embeds)
                    if config.use_i_gcg:
                        current_loss, optim_ids = self._i_gcg_merge_step(
                            sampled_ids=sampled_ids,
                            single_token_losses=loss,
                            current_optim_ids=optim_ids,
                            after_embeds=after_embeds,
                            target_embeds=target_embeds,
                            before_embeds=before_embeds,
                            batch_size=batch_size,
                        )
                    else:
                        current_loss = loss.min().item()
                        optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)
                else:
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

            buffer.log_buffer(tokenizer)

            # String-space early-stop: periodically invoke the user-supplied
            # callback with the current best suffix. Unlike the token-space
            # `early_stop` flag (which checks argmax logits against target_ids
            # and fires spuriously when the chat-template boundary
            # re-tokenizes differently at inference time), the callback gets
            # to decode the suffix and run the real inference pipeline. Use
            # `(step + 1) % K == 0` so step indexing starts from 1 and we
            # don't waste a check at step 0 (before any optimization).
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
        )

        return result

    def _fresh_prefix_cache(self, expand_to: int = None):
        """Rebuild a fresh DynamicCache from the stable prefix-tensor snapshot.

        The model mutates whatever past_key_values it receives (appending new
        KVs via torch.cat). To keep self.prefix_legacy clean across iterations,
        every model() call that wants the prefix cached gets a fresh wrapper
        whose internal key_cache/value_cache lists are NEW (so the model's
        torch.cat-into-list-element mutates this throwaway wrapper, not the
        snapshot tensors).

        If expand_to is given, also expand each prefix tensor's batch dim to
        match the candidate batch size (used in candidate-loss eval). Expanding
        is a view, not a copy.
        """
        from transformers.cache_utils import DynamicCache
        if expand_to is None or expand_to == 1:
            return DynamicCache.from_legacy_cache(self.prefix_legacy)
        legacy = tuple(
            tuple(x.expand(expand_to, -1, -1, -1) for x in self.prefix_legacy[i])
            for i in range(len(self.prefix_legacy))
        )
        return DynamicCache.from_legacy_cache(legacy)

    def init_buffer(self) -> AttackBuffer:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config.buffer_size)

        if isinstance(config.optim_str_init, str):
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

        # Compute the loss on the initial buffer entries
        if self.prefix_cache:
            init_buffer_embeds = torch.cat([
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.repeat(true_buffer_size, 1, 1),
                self.target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)
        else:
            init_buffer_embeds = torch.cat([
                self.before_embeds.repeat(true_buffer_size, 1, 1),
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.repeat(true_buffer_size, 1, 1),
                self.target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)

        init_buffer_losses = find_executable_batch_size(self._compute_candidates_loss_original, true_buffer_size)(init_buffer_embeds)

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        buffer.log_buffer(tokenizer)

        logger.info("Initialized attack buffer.")

        return buffer

    def compute_token_gradient(
        self,
        optim_ids: Tensor,
    ) -> Tensor:
        """Computes the gradient of the GCG loss w.r.t the one-hot token matrix.

        Args:
            optim_ids : Tensor, shape = (1, n_optim_ids)
                the sequence of token ids that are being optimized
        """
        model = self.model
        embedding_layer = self.embedding_layer

        # Create the one-hot encoding matrix of our optimized token ids
        optim_ids_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=embedding_layer.num_embeddings)
        optim_ids_onehot = optim_ids_onehot.to(model.device, model.dtype)
        optim_ids_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        if self.prefix_cache:
            input_embeds = torch.cat([optim_embeds, self.after_embeds, self.target_embeds], dim=1)
            output = model(
                inputs_embeds=input_embeds,
                past_key_values=self._fresh_prefix_cache(),
                use_cache=True,
            )
        else:
            input_embeds = torch.cat(
                [
                    self.before_embeds,
                    optim_embeds,
                    self.after_embeds,
                    self.target_embeds,
                ],
                dim=1,
            )
            output = model(inputs_embeds=input_embeds)

        logits = output.logits

        # Shift logits so token n-1 predicts token n
        shift = input_embeds.shape[1] - self.target_ids.shape[1]
        shift_logits = logits[..., shift - 1 : -1, :].contiguous()  # (1, num_target_ids, vocab_size)
        shift_labels = self.target_ids

        if self.config.use_mellowmax:
            label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
            loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
        else:
            loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        optim_ids_onehot_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

        return optim_ids_onehot_grad

    def _i_gcg_merge_step(
        self,
        sampled_ids: Tensor,
        single_token_losses: Tensor,
        current_optim_ids: Tensor,
        after_embeds: Tensor,
        target_embeds: Tensor,
        before_embeds: Tensor,
        batch_size: int,
    ) -> Tuple[float, Tensor]:
        """I-GCG multi-coordinate update (Jia et al. 2024, Eq. 8 / Algo. 1).

        Sort the single-token candidates by loss, then cumulatively merge the
        top-p lowest-loss candidates' position-wise diffs into the current
        suffix. Re-score the p merged variants and return the best.
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

        embedding_layer = self.embedding_layer
        if self.prefix_cache:
            merged_embeds = torch.cat(
                [
                    embedding_layer(merged_candidates),
                    after_embeds.repeat(p, 1, 1),
                    target_embeds.repeat(p, 1, 1),
                ],
                dim=1,
            )
        else:
            merged_embeds = torch.cat(
                [
                    before_embeds.repeat(p, 1, 1),
                    embedding_layer(merged_candidates),
                    after_embeds.repeat(p, 1, 1),
                    target_embeds.repeat(p, 1, 1),
                ],
                dim=1,
            )

        merged_losses = find_executable_batch_size(self._compute_candidates_loss_original, min(batch_size, p))(merged_embeds)
        best = merged_losses.argmin()
        return merged_losses[best].item(), merged_candidates[best].unsqueeze(0)

    def _compute_candidates_loss_original(
        self,
        search_batch_size: int,
        input_embeds: Tensor,
    ) -> Tensor:
        """Computes the GCG loss on all candidate token id sequences.

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            input_embeds : Tensor, shape = (search_width, seq_len, embd_dim)
                the embeddings of the `search_width` candidate sequences to evaluate
        """
        all_loss = []

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i + search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]

                if self.prefix_cache:
                    # Fresh cache per inner batch: the model mutates the passed-in
                    # DynamicCache by appending KVs during forward, so reusing
                    # across inner batches would feed earlier candidates' target
                    # tokens into the cache and silently boost P(target) on
                    # later candidates. Build a throwaway fresh wrapper around
                    # the immutable prefix-tensor snapshot every time.
                    prefix_cache_batch = self._fresh_prefix_cache(expand_to=current_batch_size)
                    outputs = self.model(inputs_embeds=input_embeds_batch, past_key_values=prefix_cache_batch, use_cache=True)
                else:
                    outputs = self.model(inputs_embeds=input_embeds_batch)

                logits = outputs.logits

                tmp = input_embeds.shape[1] - self.target_ids.shape[1]
                shift_logits = logits[..., tmp-1:-1, :].contiguous()
                shift_labels = self.target_ids.repeat(current_batch_size, 1)

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
                            draft_prefix_cache_batch = DynamicCache.from_legacy_cache(legacy)
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
