"""
Simple, heavily commented, single-layer M3 evaluation for nanoGPT.

What this file does:
1. Load a normal nanoGPT checkpoint into the normal GPT model.
2. Evaluate baseline loss / perplexity.
3. Pick ONE existing nn.Linear layer by path.
4. Replace only that layer's forward pass with a simple M3 approximation.
5. Evaluate the patched model again.
6. Optionally compare logits on one batch to see how much the outputs changed.

Why this file exists:
- It is intentionally simpler than the previous full-model M3 experiment.
- It does NOT create a new model class.
- It does NOT change checkpoint structure.
- It does NOT try to implement the paper's compressed storage pipeline.
- It only simulates the forward-pass approximation on one chosen layer.

This makes it easier to answer a first question:
"If I approximate one trained GPT layer with M3-style forward logic,
 how much does perplexity change?"

Important limitations:
- This is inference-time evaluation only.
- This is not training with M3.
- This is not a true memory-compressed implementation.
- This is a practical experiment, not a claim that the full paper pipeline
  has been reproduced exactly.
"""

import math
import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

from model import GPT, GPTConfig


# -----------------------------------------------------------------------------
# User-editable settings
# -----------------------------------------------------------------------------

# Checkpoint to evaluate.
CKPT_PATH = os.path.join("out-shakespeare-char", "ckpt.pt")

# Dataset directory to evaluate on.
#
# We set this explicitly to Shakespeare-char because that is the dataset you
# are working with here. This is simpler and less error-prone than relying on
# whatever dataset name may or may not be stored in the checkpoint metadata.
DATA_DIR = os.path.join("data", "shakespeare_char")

# Which dataset split to evaluate.
# Valid choices: "train", "val", "test"
SPLIT = "test"

# Full-split evaluation settings.
EVAL_ITERS = 100
BATCH_SIZE = 64
STRIDE = 128
FULL_SPLIT = True

# Reproducibility.
SEED = 1337

# Device / dtype settings.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
COMPILE = False

# -------------------------------------------------------------------------
# M3 experiment settings
# -------------------------------------------------------------------------

# This is the single layer that will be patched.
#
# Recommended first target:
# - transformer.h.0.mlp.c_proj
#
# Why:
# - It is a normal linear layer inside the MLP.
# - It is usually easier to experiment on than attention projections.
# - It keeps the experiment narrow and interpretable.
#
# Other possible examples:
# - transformer.h.0.mlp.c_fc
# - transformer.h.0.attn.c_proj
# - transformer.h.0.attn.c_attn
TARGET_LAYER_PATH = "transformer.h.0.attn.c_attn"

# Number of random hyperplanes used by the M3 approximation.
# Larger k usually gives a better approximation but costs more compute.
M3_K = 2048

# Whether to also compute a one-batch logits MSE diagnostic.
RUN_LOGITS_MSE_CHECK = True

# Allow command line / configurator overrides, same style as the rest of nanoGPT.
exec(open("configurator.py").read())


# -----------------------------------------------------------------------------
# Utility functions: reproducibility and mixed precision context
# -----------------------------------------------------------------------------

if SPLIT not in {"train", "val", "test"}:
    raise ValueError(f"Invalid split '{SPLIT}'. Expected one of: train, val, test.")

if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE_TYPE = "cuda" if "cuda" in DEVICE else "cpu"
PTDTYPE = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[DTYPE]

CTX = nullcontext() if DEVICE_TYPE == "cpu" else torch.amp.autocast(device_type=DEVICE_TYPE, dtype=PTDTYPE)


# -----------------------------------------------------------------------------
# Step 1: checkpoint loading
# -----------------------------------------------------------------------------

print(f"Loading checkpoint: {CKPT_PATH}")
checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)

gptconf = GPTConfig(**checkpoint["model_args"])


def strip_unwanted_prefix(state_dict):
    """
    Remove the '_orig_mod.' prefix that can appear in checkpoints produced from
    compiled models.

    Why this exists:
    - nanoGPT sometimes saves checkpoints where parameter names begin with
      '_orig_mod.'.
    - The normal GPT model expects names without that prefix.
    """
    state_dict = dict(state_dict)
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    return state_dict


model = GPT(gptconf)
state_dict = strip_unwanted_prefix(checkpoint["model"])
model.load_state_dict(state_dict)
model.eval()
model.to(DEVICE)

if COMPILE:
    model = torch.compile(model)


# -----------------------------------------------------------------------------
# Step 2: dataset loading
# -----------------------------------------------------------------------------

# We intentionally do NOT infer the dataset from checkpoint["config"] here.
# Your goal is to evaluate on the Shakespeare-char files you already have:
# - train.bin
# - val.bin
# - test.bin
# - GPTQ_data.bin
#
# For perplexity evaluation, test.bin is the clean default choice.
data_path = os.path.join(DATA_DIR, f"{SPLIT}.bin")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data split file not found: {data_path}")

block_size = gptconf.block_size
data = np.memmap(data_path, dtype=np.uint16, mode="r")

if len(data) <= block_size:
    raise ValueError(
        f"{data_path} length ({len(data)}) must be > block_size ({block_size})."
    )


def get_batch_from_starts(starts):
    """
    Build a batch from explicit starting positions.

    Input:
    - starts: sequence of token offsets in the memmap file

    Output:
    - x: model inputs of shape [batch, block_size]
    - y: next-token targets of shape [batch, block_size]
    """
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in starts]
    )
    y = torch.stack(
        [torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in starts]
    )

    if DEVICE_TYPE == "cuda":
        x = x.pin_memory().to(DEVICE, non_blocking=True)
        y = y.pin_memory().to(DEVICE, non_blocking=True)
    else:
        x, y = x.to(DEVICE), y.to(DEVICE)

    return x, y


@torch.no_grad()
def evaluate_loss(current_model):
    """
    Compute mean next-token loss on the chosen split.

    This is copied in spirit from calculate_PPL.py, but written out more
    explicitly to keep the script self-contained and easy to inspect.
    """
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError("Not enough tokens for evaluation.")

    local_stride = block_size if STRIDE is None else int(STRIDE)
    if local_stride <= 0:
        raise ValueError(f"STRIDE must be > 0, got {local_stride}")

    starts = torch.arange(0, max_start + 1, local_stride, dtype=torch.long)

    if not FULL_SPLIT:
        max_examples = EVAL_ITERS * BATCH_SIZE
        starts = starts[:max_examples]

    num_examples = len(starts)
    if num_examples == 0:
        raise ValueError("No evaluation windows were generated.")

    total_loss = 0.0
    total_examples = 0

    for i in range(0, num_examples, BATCH_SIZE):
        batch_starts = starts[i : i + BATCH_SIZE]
        x, y = get_batch_from_starts(batch_starts.tolist())

        with CTX:
            _, loss = current_model(x, y)

        bs = len(batch_starts)
        total_loss += loss.item() * bs
        total_examples += bs

    return total_loss / total_examples


def print_loss_and_ppl(label, mean_loss):
    """
    Small helper to print loss and perplexity together.
    """
    ppl = math.exp(mean_loss)
    print(f"{label} loss: {mean_loss:.6f}")
    print(f"{label} ppl : {ppl:.6f}")


# -----------------------------------------------------------------------------
# Step 3: locating a single target layer by string path
# -----------------------------------------------------------------------------

def get_module_by_path(root_module, module_path):
    """
    Resolve a module path like:
        transformer.h.0.mlp.c_proj

    Rules:
    - split the path by dots
    - if a path component is an integer, index into ModuleList / list-like
    - otherwise access it as an attribute

    Example:
    - "transformer" -> root_module.transformer
    - "h"           -> .h
    - "0"           -> [0]
    - "mlp"         -> .mlp
    - "c_proj"      -> .c_proj
    """
    current = root_module
    for part in module_path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


target_layer = get_module_by_path(model, TARGET_LAYER_PATH)

if not isinstance(target_layer, nn.Linear):
    raise TypeError(
        f"Target layer '{TARGET_LAYER_PATH}' is {type(target_layer)}, not nn.Linear."
    )

print("\nTarget layer")
print("------------")
print(f"Path   : {TARGET_LAYER_PATH}")
print(f"Type   : {type(target_layer).__name__}")
print(f"Weight : {tuple(target_layer.weight.shape)}")
if target_layer.bias is not None:
    print(f"Bias   : {tuple(target_layer.bias.shape)}")
else:
    print("Bias   : None")


# -----------------------------------------------------------------------------
# Step 4: simple M3 approximation function
# -----------------------------------------------------------------------------

def m3_approx_linear(x, weight, bias, E, k):
    """
    Approximate a linear layer output using M3-style forward logic.

    Inputs:
    - x:
        Usually a 3D activation tensor for GPT, shape [B, T, C_in]
        but this function also supports 2D shape [N, C_in].
    - weight:
        Standard linear weight matrix, shape [C_out, C_in]
    - bias:
        Standard linear bias, shape [C_out] or None
    - E:
        Random Gaussian projection matrix, shape [C_in, k]
    - k:
        Number of random hyperplanes.

    Output:
    - Tensor with the same leading dimensions as x, but last dimension replaced
      by C_out.

    Important design choice:
    - We do NOT modify the weight tensor structure.
    - We use the existing trained weight values exactly as they are.
    - We only change how the forward output is computed.

    This makes the experiment easier to understand:
    - same checkpoint
    - same model
    - same layer object
    - different forward math only
    """

    # Save the original input shape so we can reshape back later.
    original_shape = x.shape

    # Convert everything to a 2D matrix where each row is one vector whose
    # dot products we want to approximate.
    #
    # Example:
    # - GPT activations often have shape [B, T, C]
    # - here we flatten that to [B*T, C]
    x_flat = x.reshape(-1, original_shape[-1])

    # ------------------------------------------------------------------
    # Step A: vector norms
    # ------------------------------------------------------------------
    # exact dot product can be written as:
    #   x dot w = ||x|| * ||w|| * cos(theta)
    #
    # M3 estimates cos(theta) from random hyperplane sign comparisons.
    norm_x = torch.norm(x_flat, p=2, dim=1, keepdim=True)
    norm_w = torch.norm(weight, p=2, dim=1, keepdim=True)

    # ------------------------------------------------------------------
    # Step B: random projections onto hyperplanes
    # ------------------------------------------------------------------
    # Both the activations and the layer weights are projected with the same E.
    #
    # Shapes:
    # - x_flat  : [N, C_in]
    # - weight  : [C_out, C_in]
    # - E       : [C_in, k]
    # - x_proj  : [N, k]
    # - w_proj  : [C_out, k]
    x_proj = x_flat @ E
    w_proj = weight @ E

    # ------------------------------------------------------------------
    # Step C: keep only the sign of each projection
    # ------------------------------------------------------------------
    # We convert each projected value into either +1 or -1.
    #
    # Example:
    #   [ 0.8, -1.2, 0.0 ] -> [ +1, -1, +1 ]
    #
    # This is where the method starts behaving more like a "binary sketch" of
    # the vectors rather than full-precision linear algebra.
    x_sign = torch.where(x_proj >= 0, 1.0, -1.0)
    w_sign = torch.where(w_proj >= 0, 1.0, -1.0)

    # ------------------------------------------------------------------
    # Step D: compare the sign patterns
    # ------------------------------------------------------------------
    # Agreement score S:
    # - if signs match on all k hyperplanes -> S = +k
    # - if signs differ on all k hyperplanes -> S = -k
    # - if about half match -> S is near 0
    S = x_sign @ w_sign.t()

    # Convert agreement score into Hamming distance:
    #   H = number of mismatched sign positions
    H = (k - S) / 2.0

    # ------------------------------------------------------------------
    # Step E: convert mismatch fraction into angle estimate
    # ------------------------------------------------------------------
    theta = (math.pi * H) / k
    cos_theta = torch.cos(theta)

    # ------------------------------------------------------------------
    # Step F: reconstruct approximate dot products
    # ------------------------------------------------------------------
    y_flat = cos_theta * (norm_x @ norm_w.t())

    if bias is not None:
        y_flat = y_flat + bias

    # Restore the original leading dimensions.
    y = y_flat.reshape(*original_shape[:-1], weight.shape[0])
    return y


# -----------------------------------------------------------------------------
# Step 5: patch only the chosen layer
# -----------------------------------------------------------------------------

def patch_linear_layer_with_m3(layer, k, seed):
    """
    Replace one nn.Linear layer's forward pass with M3 logic.

    What stays the same:
    - layer.weight
    - layer.bias
    - parameter names
    - checkpoint format outside this runtime session

    What changes:
    - only the mathematical computation used during forward(...)

    Why this is useful:
    - it isolates the experiment to one layer
    - it is easy to undo by restoring the original forward method
    - it avoids introducing a new custom model class
    """

    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(layer)}")

    # Make the random projection matrix reproducible.
    generator = torch.Generator(device=layer.weight.device)
    generator.manual_seed(seed)

    in_features = layer.weight.shape[1]

    # Keep E in the same dtype/device family as the layer weights.
    #
    # Note:
    # - We use full random Gaussian entries here.
    # - We are not implementing compressed storage or a special structured E.
    E = torch.randn(
        in_features,
        k,
        generator=generator,
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    )

    # Save the original forward so the caller can restore it later if desired.
    original_forward = layer.forward

    def m3_forward(x):
        return m3_approx_linear(
            x=x,
            weight=layer.weight,
            bias=layer.bias,
            E=E,
            k=k,
        )

    # Attach some attributes for inspection / debugging.
    layer._m3_original_forward = original_forward
    layer._m3_E = E
    layer._m3_k = k

    # Monkey-patch the layer.
    layer.forward = m3_forward

    return original_forward


def restore_original_forward(layer):
    """
    Restore the layer's original forward if it was previously patched.
    """
    if hasattr(layer, "_m3_original_forward"):
        layer.forward = layer._m3_original_forward


# -----------------------------------------------------------------------------
# Step 6: baseline evaluation
# -----------------------------------------------------------------------------

print("\nEvaluating baseline model...")
baseline_loss = evaluate_loss(model)

print("\nBaseline results")
print("----------------")
print_loss_and_ppl("Baseline", baseline_loss)


# -----------------------------------------------------------------------------
# Step 7: optional one-batch logits MSE before / after patch
# -----------------------------------------------------------------------------

@torch.no_grad()
def get_one_batch_for_logits_check():
    """
    Use the first evaluation batch as a simple, deterministic comparison batch.
    """
    max_start = len(data) - block_size - 1
    local_stride = block_size if STRIDE is None else int(STRIDE)
    starts = torch.arange(0, max_start + 1, local_stride, dtype=torch.long)
    batch_starts = starts[:BATCH_SIZE]
    return get_batch_from_starts(batch_starts.tolist())


baseline_logits = None
batch_x = None
batch_y = None

if RUN_LOGITS_MSE_CHECK:
    batch_x, batch_y = get_one_batch_for_logits_check()
    with torch.no_grad():
        with CTX:
            baseline_logits, _ = model(batch_x, batch_y)


# -----------------------------------------------------------------------------
# Step 8: patch the chosen layer and evaluate again
# -----------------------------------------------------------------------------

print("\nPatching one layer with M3...")
patch_linear_layer_with_m3(target_layer, k=M3_K, seed=SEED)

patched_loss = evaluate_loss(model)

print("\nPatched results")
print("---------------")
print_loss_and_ppl("Patched", patched_loss)

print("\nDelta")
print("-----")
print(f"Loss delta: {patched_loss - baseline_loss:.6f}")
print(f"PPL ratio : {math.exp(patched_loss) / math.exp(baseline_loss):.6f}")


# -----------------------------------------------------------------------------
# Step 9: optional logits MSE on one batch
# -----------------------------------------------------------------------------

if RUN_LOGITS_MSE_CHECK:
    with torch.no_grad():
        with CTX:
            patched_logits, _ = model(batch_x, batch_y)

    logits_mse = torch.nn.functional.mse_loss(
        patched_logits.float(),
        baseline_logits.float(),
    ).item()

    print("\nOne-batch logits comparison")
    print("---------------------------")
    print(f"Logits MSE: {logits_mse:.6f}")


# -----------------------------------------------------------------------------
# Step 10: final notes for interpretation
# -----------------------------------------------------------------------------

print("\nInterpretation notes")
print("--------------------")
print("1. This script changes only one layer's forward computation.")
print("2. The checkpoint itself is unchanged.")
print("3. A large loss increase means that even one-layer M3 approximation is disruptive.")
print("4. If the result is promising, the next step is usually to try a different layer or larger k.")
print("5. If the result is poor, that does not automatically mean the code is wrong; it may mean the approximation is too destructive for that layer.")
print(f"6. Evaluation split used here: {data_path}")
