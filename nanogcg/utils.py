import functools
import gc
import inspect
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

INIT_CHARS = [
    ".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}",
    "@", "#", "$", "%", "&", "*",
    "w", "x", "y", "z",
]

def get_nonascii_toks(tokenizer, device="cpu"):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    nonascii_toks = []
    for i in range(tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            nonascii_toks.append(i)
    
    if tokenizer.bos_token_id is not None:
        nonascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        nonascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        nonascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        nonascii_toks.append(tokenizer.unk_token_id)

    return torch.tensor(nonascii_toks, device=device)


def get_special_toks(tokenizer, device="cpu"):
    """All special / added-vocab token ids (bos/eos/pad/unk plus every entry in
    added_tokens_decoder, e.g. Llama-3's <|...|> control + reserved tokens).

    These must never appear in an optimized GCG string: they decode to ASCII
    strings (so get_nonascii_toks does NOT catch them), but the model treats
    them as single control tokens, and as literal text in a suffix they
    re-tokenize inconsistently at the prompt boundary — producing a fake
    low loss that doesn't transfer to inference.
    """
    ids = set()
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        v = getattr(tokenizer, attr, None)
        if v is not None:
            ids.add(int(v))
    for v in (getattr(tokenizer, "all_special_ids", None) or []):
        ids.add(int(v))
    # added_tokens_decoder: {id: AddedToken} — covers all reserved/control toks.
    for k in (getattr(tokenizer, "added_tokens_decoder", None) or {}):
        ids.add(int(k))
    return torch.tensor(sorted(ids), device=device, dtype=torch.int64)

def mellowmax(t: Tensor, alpha=1.0, dim=-1):
   return 1.0 / alpha * (torch.logsumexp(alpha * t, dim=dim) - torch.log(torch.tensor(t.shape[-1], dtype=t.dtype, device=t.device)))

# borrowed from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L69
def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False

# Persistent cache of resolved batch sizes per (function, id) so that the
# OOM-halving dance only happens once instead of every iteration when the
# decorator is recreated on each call. Keyed by the wrapped function so that
# different callsites (e.g. main-loop candidate-loss vs init-buffer loss) don't
# share a single global value.
_RESOLVED_BATCH_SIZES: dict = {}


# modified from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L87
def find_executable_batch_size(function: callable = None, starting_batch_size: int = 128):
    """
    A basic decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory or
    CUDNN, the batch size is cut in half and passed to `function`

    `function` must take in a `batch_size` parameter as its first argument.

    Persists the resolved batch_size across decorator-creations via a
    module-level cache keyed by the wrapped function, so callers that
    re-create the decorator on every loop iteration (as nanoGCG does)
    don't pay the OOM-halving cost every step.

    Args:
        function (`callable`, *optional*):
            A function to wrap
        starting_batch_size (`int`, *optional*):
            The batch size to try and fit into memory. If a smaller value
            has already been resolved for this function in a previous call,
            we use that instead (capped by this starting_batch_size).

    Example:

    ```python
    >>> from utils import find_executable_batch_size


    >>> @find_executable_batch_size(starting_batch_size=128)
    ... def train(batch_size, model, optimizer):
    ...     ...


    >>> train(model, optimizer)
    ```
    """
    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)

    # Start from the persisted batch_size (capped by the requested starting size)
    # rather than always starting at `starting_batch_size`. This avoids re-doing
    # the OOM-halving every time the decorator is recreated.
    #
    # Use the function itself as the dict key. For bound methods (e.g.
    # `self._compute_candidates_loss_original`) Python creates a fresh bound-
    # method object on every attribute access, so id() is unstable — but
    # bound-method hashing is based on (self, __func__), so dict lookup with
    # the bound method as key matches across accesses.
    cached = _RESOLVED_BATCH_SIZES.get(function)
    batch_size = min(cached, starting_batch_size) if cached is not None else starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())
        # Guard against user error
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                result = function(batch_size, *args, **kwargs)
                _RESOLVED_BATCH_SIZES[function] = batch_size
                return result
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    print(f"Decreasing batch size to: {batch_size}")
                else:
                    raise

    return decorator

def configure_pad_token(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Checks if the (Hugging Face) tokenizer has a padding token and sets it if not present.

    Borrowed from https://github.com/EleutherAI/lm-evaluation-harness/blob/5c006ed417a2f4d01248d487bcbd493ebe3e5edd/lm_eval/models/utils.py#L624
    """
    if tokenizer.pad_token:
        return tokenizer

    if tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer
