"""
Simple, heavily commented, multi-layer M3 evaluation for nanoGPT.

This file is intentionally very close to M3_single_layer_eval.py.

Main difference:
- instead of patching exactly one layer,
- this script patches a list of existing nn.Linear layers.

This lets you test questions like:
- "What happens if I apply M3 to the whole first transformer block?"
- "What happens if I patch only the MLP layers in block 0?"
- "What happens if I patch several specific layers at once?"

Important design choice:
- We still do NOT change the GPT model class.
- We still do NOT change the checkpoint format.
- We still do NOT restructure the stored weights.
- We only replace the runtime forward() computation of selected layers.

So this remains a simple inference-time simulation experiment.
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

CKPT_PATH = os.path.join("out-shakespeare-char", "ckpt.pt")
DATA_DIR = os.path.join("data", "shakespeare_char")
SPLIT = "test"

EVAL_ITERS = 100
BATCH_SIZE = 64
STRIDE = 128
FULL_SPLIT = True

SEED = 1337

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
COMPILE = False

# -------------------------------------------------------------------------
# M3 experiment settings
# -------------------------------------------------------------------------

# Paste any layer paths you want here.
#
# Default below = the entire first transformer block (block 0) in standard
# nanoGPT naming:
# - attention input projection
# - attention output projection
# - MLP expansion
# - MLP projection back down
#
# You can remove lines or add others.
# Example smaller subsets:
# - only MLP in block 0:
#   "transformer.h.0.mlp.c_fc",
#   "transformer.h.0.mlp.c_proj",
#
# - only attention in block 0:
#   "transformer.h.0.attn.c_attn",
#   "transformer.h.0.attn.c_proj",

# All layers
'''
"transformer.h.0.attn.c_attn"
"transformer.h.0.attn.c_proj"
"transformer.h.0.mlp.c_fc"
"transformer.h.0.mlp.c_proj"
"transformer.h.1.attn.c_attn"
"transformer.h.1.attn.c_proj"
"transformer.h.1.mlp.c_fc"
"transformer.h.1.mlp.c_proj"
"transformer.h.2.attn.c_attn"
"transformer.h.2.attn.c_proj"
"transformer.h.2.mlp.c_fc"
"transformer.h.2.mlp.c_proj"
"transformer.h.3.attn.c_attn"
"transformer.h.3.attn.c_proj"
"transformer.h.3.mlp.c_fc"
"transformer.h.3.mlp.c_proj"
"transformer.h.4.attn.c_attn"
"transformer.h.4.attn.c_proj"
"transformer.h.4.mlp.c_fc"
"transformer.h.4.mlp.c_proj"
"transformer.h.5.attn.c_attn"
"transformer.h.5.attn.c_proj"
"transformer.h.5.mlp.c_fc"
"transformer.h.5.mlp.c_proj"
'''
#ALL
TARGET_LAYER_PATHS = [
    "transformer.h.0.attn.c_attn",
    "transformer.h.0.attn.c_proj",
    "transformer.h.0.mlp.c_fc",
    #"transformer.h.0.mlp.c_proj",
    "transformer.h.1.attn.c_attn",
    "transformer.h.1.attn.c_proj",
    "transformer.h.1.mlp.c_fc",
    #"transformer.h.1.mlp.c_proj",
    "transformer.h.2.attn.c_attn",
    "transformer.h.2.attn.c_proj",
    "transformer.h.2.mlp.c_fc",
    #"transformer.h.2.mlp.c_proj",
    "transformer.h.3.attn.c_attn",
    "transformer.h.3.attn.c_proj",
    "transformer.h.3.mlp.c_fc",
    #"transformer.h.3.mlp.c_proj",
    "transformer.h.4.attn.c_attn",
    "transformer.h.4.attn.c_proj",
    "transformer.h.4.mlp.c_fc",
    #"transformer.h.4.mlp.c_proj",
    "transformer.h.5.attn.c_attn",
    "transformer.h.5.attn.c_proj",
    #"transformer.h.5.mlp.c_fc",
    #"transformer.h.5.mlp.c_proj",
]

# #MLP
# TARGET_LAYER_PATHS = [
#     "transformer.h.0.mlp.c_fc",
#     "transformer.h.0.mlp.c_proj",
#     "transformer.h.1.mlp.c_fc",
#     "transformer.h.1.mlp.c_proj",
#     "transformer.h.2.mlp.c_fc",
#     "transformer.h.2.mlp.c_proj",
#     "transformer.h.3.mlp.c_fc",
#     "transformer.h.3.mlp.c_proj",
#     "transformer.h.4.mlp.c_fc",
#     "transformer.h.4.mlp.c_proj",
#     "transformer.h.5.mlp.c_fc",
#     "transformer.h.5.mlp.c_proj",
# ]

#Only attention
# TARGET_LAYER_PATHS = [
#     "transformer.h.0.attn.c_attn",
#     "transformer.h.0.attn.c_proj",
#     "transformer.h.1.attn.c_attn",
#     "transformer.h.1.attn.c_proj",
#     "transformer.h.2.attn.c_attn",
#     "transformer.h.2.attn.c_proj",
#     "transformer.h.3.attn.c_attn",
#     "transformer.h.3.attn.c_proj",
#     "transformer.h.4.attn.c_attn",
#     "transformer.h.4.attn.c_proj",
#     "transformer.h.5.attn.c_attn",
#     "transformer.h.5.attn.c_proj",
# ]



# One shared k for all selected layers.
#
# This keeps the script simple.
# If later you want different k per layer, that can be added, but this version
# keeps one global setting for easier interpretation.
M3_K = 512

# Optional per-layer override for k.
#
# If a layer path is present here, that value is used instead of the global M3_K.
# This is the smallest code change that lets you try, for example:
# - attention layers at k = 4096
# - MLP layers at k = 8192
#
# Example:
#M3_K_BY_LAYER = {}
M3_K_BY_LAYER = {
    "transformer.h.0.attn.c_attn": 4096,
    "transformer.h.0.attn.c_proj": 2048,
    "transformer.h.0.mlp.c_fc": 4096,
    "transformer.h.0.mlp.c_proj": 4096,
    "transformer.h.1.attn.c_attn": 4096,
    "transformer.h.1.attn.c_proj": 2048,
    "transformer.h.1.mlp.c_fc": 4096,
    "transformer.h.1.mlp.c_proj": 4096,
    "transformer.h.2.attn.c_attn": 4096,
    "transformer.h.2.attn.c_proj": 2048,
    "transformer.h.2.mlp.c_fc": 4096,
    "transformer.h.2.mlp.c_proj": 4096,
    "transformer.h.3.attn.c_attn": 4096,
    "transformer.h.3.attn.c_proj": 2048,
    "transformer.h.3.mlp.c_fc": 4096,
    "transformer.h.3.mlp.c_proj": 4096,
    "transformer.h.4.attn.c_attn": 4096,
    "transformer.h.4.attn.c_proj": 2048,
    "transformer.h.4.mlp.c_fc": 4096,
    "transformer.h.4.mlp.c_proj": 4096,
    "transformer.h.5.attn.c_attn": 4096,
    "transformer.h.5.attn.c_proj": 2048,
    "transformer.h.5.mlp.c_fc": 4096,
    "transformer.h.5.mlp.c_proj": 4096,
  }
RUN_LOGITS_MSE_CHECK = True

exec(open("configurator.py").read())


# -----------------------------------------------------------------------------
# Reproducibility and mixed precision context
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
    Remove the '_orig_mod.' prefix that can appear in compiled checkpoints.
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
    ppl = math.exp(mean_loss)
    print(f"{label} loss: {mean_loss:.6f}")
    print(f"{label} ppl : {ppl:.6f}")


# -----------------------------------------------------------------------------
# Step 3: module path resolution
# -----------------------------------------------------------------------------

def get_module_by_path(root_module, module_path):
    """
    Resolve a string path like:
        transformer.h.0.mlp.c_proj

    Rules:
    - integer path components index into ModuleList-style containers
    - non-integer path components access attributes
    """
    current = root_module
    for part in module_path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


target_layers = []
for layer_path in TARGET_LAYER_PATHS:
    layer = get_module_by_path(model, layer_path)
    if not isinstance(layer, nn.Linear):
        raise TypeError(
            f"Target layer '{layer_path}' is {type(layer)}, not nn.Linear."
        )
    target_layers.append((layer_path, layer))

print("\nTarget layers")
print("-------------")
for layer_path, layer in target_layers:
    print(f"Path   : {layer_path}")
    print(f"Type   : {type(layer).__name__}")
    print(f"Weight : {tuple(layer.weight.shape)}")
    if layer.bias is not None:
        print(f"Bias   : {tuple(layer.bias.shape)}")
    else:
        print("Bias   : None")
    print("")


# -----------------------------------------------------------------------------
# Step 4: simple M3 approximation
# -----------------------------------------------------------------------------

def m3_approx_linear(x, weight, bias, E, k):
    """
    Approximate a linear layer output using M3-style forward logic.

    This function is intentionally written in a straightforward way rather than
    a highly optimized way. The goal is readability and easy modification.
    """
    original_shape = x.shape
    x_flat = x.reshape(-1, original_shape[-1])

    norm_x = torch.norm(x_flat, p=2, dim=1, keepdim=True)
    norm_w = torch.norm(weight, p=2, dim=1, keepdim=True)

    x_proj = x_flat @ E
    w_proj = weight @ E

    x_sign = torch.where(x_proj >= 0, 1.0, -1.0)
    w_sign = torch.where(w_proj >= 0, 1.0, -1.0)

    S = x_sign @ w_sign.t()
    H = (k - S) / 2.0
    theta = (math.pi * H) / k
    cos_theta = torch.cos(theta)

    y_flat = cos_theta * (norm_x @ norm_w.t())
    if bias is not None:
        y_flat = y_flat + bias

    y = y_flat.reshape(*original_shape[:-1], weight.shape[0])
    return y


# -----------------------------------------------------------------------------
# Step 5: patch many selected layers
# -----------------------------------------------------------------------------

def patch_linear_layer_with_m3(layer, k, seed_offset):
    """
    Patch one nn.Linear layer.

    seed_offset exists so that each chosen layer gets a different random E,
    while still keeping the whole experiment reproducible from one base seed.
    """
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(layer)}")

    generator = torch.Generator(device=layer.weight.device)
    generator.manual_seed(SEED + seed_offset)

    in_features = layer.weight.shape[1]
    E = torch.randn(
        in_features,
        k,
        generator=generator,
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    )

    original_forward = layer.forward

    def m3_forward(x):
        return m3_approx_linear(
            x=x,
            weight=layer.weight,
            bias=layer.bias,
            E=E,
            k=k,
        )

    layer._m3_original_forward = original_forward
    layer._m3_E = E
    layer._m3_k = k
    layer.forward = m3_forward

    return original_forward


def restore_original_forward(layer):
    """
    Restore a previously patched layer.
    """
    if hasattr(layer, "_m3_original_forward"):
        layer.forward = layer._m3_original_forward


def patch_selected_layers(layer_tuples, k):
    """
    Patch every selected layer in order.

    layer_tuples:
    - list of (layer_path, layer_module)
    """
    for index, (layer_path, layer) in enumerate(layer_tuples):
        layer_k = M3_K_BY_LAYER.get(layer_path, k)
        patch_linear_layer_with_m3(layer, k=layer_k, seed_offset=index)
        print(f"Patched with M3: {layer_path} (k={layer_k})")


# -----------------------------------------------------------------------------
# Step 6: baseline evaluation
# -----------------------------------------------------------------------------

print("\nEvaluating baseline model...")
baseline_loss = evaluate_loss(model)

print("\nBaseline results")
print("----------------")
print_loss_and_ppl("Baseline", baseline_loss)


# -----------------------------------------------------------------------------
# Step 7: optional one-batch logits MSE
# -----------------------------------------------------------------------------

@torch.no_grad()
def get_one_batch_for_logits_check():
    """
    Use the first evaluation batch as a deterministic comparison batch.
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
# Step 8: patch selected layers and evaluate again
# -----------------------------------------------------------------------------

print("\nPatching selected layers with M3...")
patch_selected_layers(target_layers, k=M3_K)

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
# Step 10: final notes
# -----------------------------------------------------------------------------

print("\nInterpretation notes")
print("--------------------")
print("1. This script changes only the selected layers' forward computations.")
print("2. The checkpoint itself is unchanged.")
print("3. A large loss increase means the chosen multi-layer patch is disruptive.")
print("4. If results are promising, try adding or removing layers one group at a time.")
print("5. If results are poor, it may mean the approximation is too destructive for this combination.")
print(f"6. Evaluation split used here: {data_path}")
print(f"7. Number of patched layers: {len(target_layers)}")
