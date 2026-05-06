"""
Layer-by-layer M3 grid search for nanoGPT.

What this script does:
1. Load the normal nanoGPT checkpoint.
2. Evaluate the baseline model once.
3. Iterate through all 6 transformer blocks and the 4 linear layers in each:
   - attn.c_attn
   - attn.c_proj
   - mlp.c_fc
   - mlp.c_proj
4. For each layer, try the requested M3 k values:
   - 256, 512, 1024, 2048, 4096, 8192, 12288
5. Measure perplexity for each single-layer patch.
6. Save one markdown table with every layer and every tested k.

Important scope:
- This is still the simple runtime forward-patch experiment.
- The checkpoint is not modified.
- The model class is not modified.
- Only one chosen layer is patched at a time for each run.
"""

import math
import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

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

# The requested k sweep.
M3_K_VALUES = [256, 512, 1024, 2048, 4096, 8192, 12288]

# Output files.
RESULTS_DIR = "m3_gridsearch_results"
RESULTS_MD_PATH = os.path.join(RESULTS_DIR, "M3_layerwise_gridsearch.md")

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
# Checkpoint loading
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
model.load_state_dict(strip_unwanted_prefix(checkpoint["model"]))
model.eval()
model.to(DEVICE)

if COMPILE:
    model = torch.compile(model)


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------

data_path = os.path.join(DATA_DIR, f"{SPLIT}.bin")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data split file not found: {data_path}")

block_size = gptconf.block_size
data = np.memmap(data_path, dtype=np.uint16, mode="r")
if len(data) <= block_size:
    raise ValueError(f"{data_path} length ({len(data)}) must be > block_size ({block_size}).")


def get_batch_from_starts(starts):
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

    total_loss = 0.0
    total_examples = 0

    for i in range(0, len(starts), BATCH_SIZE):
        batch_starts = starts[i : i + BATCH_SIZE]
        x, y = get_batch_from_starts(batch_starts.tolist())
        with CTX:
            _, loss = current_model(x, y)
        bs = len(batch_starts)
        total_loss += loss.item() * bs
        total_examples += bs

    return total_loss / total_examples


# -----------------------------------------------------------------------------
# M3 helpers
# -----------------------------------------------------------------------------

def get_module_by_path(root_module, module_path):
    current = root_module
    for part in module_path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def m3_approx_linear(x, weight, bias, E, k):
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

    return y_flat.reshape(*original_shape[:-1], weight.shape[0])


def patch_linear_layer_with_m3(layer, k, seed):
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(layer)}")

    generator = torch.Generator(device=layer.weight.device)
    generator.manual_seed(seed)

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


def restore_original_forward(layer):
    if hasattr(layer, "_m3_original_forward"):
        layer.forward = layer._m3_original_forward


# -----------------------------------------------------------------------------
# Layer list
# -----------------------------------------------------------------------------

LAYER_NAMES_PER_BLOCK = [
    "attn.c_attn",
    "attn.c_proj",
    "mlp.c_fc",
    "mlp.c_proj",
]

TARGET_LAYER_PATHS = [
    f"transformer.h.{block_idx}.{layer_name}"
    for block_idx in range(gptconf.n_layer)
    for layer_name in LAYER_NAMES_PER_BLOCK
]


# -----------------------------------------------------------------------------
# Markdown writer
# -----------------------------------------------------------------------------

def format_float(value):
    return f"{value:.4f}"


def write_results_markdown(output_path, baseline_ppl, results):
    lines = []
    lines.append("# M3 Layerwise Grid Search")
    lines.append("")
    lines.append("This file contains a single-layer-at-a-time M3 grid search over all 6 transformer blocks and the 4 linear layers in each block.")
    lines.append("")
    lines.append("Layers tested:")
    lines.append("- `attn.c_attn`")
    lines.append("- `attn.c_proj`")
    lines.append("- `mlp.c_fc`")
    lines.append("- `mlp.c_proj`")
    lines.append("")
    lines.append(f"Baseline PPL: **{format_float(baseline_ppl)}**")
    lines.append("")

    header = ["Layer", "Baseline PPL"] + [f"k={k}" for k in M3_K_VALUES]
    sep = ["---", "---:"] + ["---:"] * len(M3_K_VALUES)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep) + " |")

    for layer_path in TARGET_LAYER_PATHS:
        row = [f"`{layer_path}`", format_float(baseline_ppl)]
        for k in M3_K_VALUES:
            value = results.get(layer_path, {}).get(k, None)
            row.append("" if value is None else format_float(value))
        lines.append("| " + " | ".join(row) + " |")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# Main search
# -----------------------------------------------------------------------------

print("\nEvaluating baseline model once...")
baseline_loss = evaluate_loss(model)
baseline_ppl = math.exp(baseline_loss)
print(f"Baseline loss: {baseline_loss:.6f}")
print(f"Baseline ppl : {baseline_ppl:.6f}")

results = {}

for layer_index, layer_path in enumerate(tqdm(TARGET_LAYER_PATHS, desc="Layers", unit="layer")):
    layer = get_module_by_path(model, layer_path)
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Target layer '{layer_path}' is {type(layer)}, not nn.Linear.")

    print(f"\n=== Layer {layer_index + 1}/{len(TARGET_LAYER_PATHS)}: {layer_path} ===")
    results[layer_path] = {}

    for k in tqdm(M3_K_VALUES, desc=f"k sweep for {layer_path}", unit="k", leave=False):
        print(f"Testing k={k} ...")
        patch_linear_layer_with_m3(layer, k=k, seed=SEED + layer_index)
        patched_loss = evaluate_loss(model)
        patched_ppl = math.exp(patched_loss)
        restore_original_forward(layer)

        results[layer_path][k] = patched_ppl
        print(f"Patched loss: {patched_loss:.6f}")
        print(f"Patched ppl : {patched_ppl:.6f}")
        write_results_markdown(RESULTS_MD_PATH, baseline_ppl, results)

# The previous in-loop writes use partial rows; now write the final complete file.
for layer_path in TARGET_LAYER_PATHS:
    results.setdefault(layer_path, {})

write_results_markdown(RESULTS_MD_PATH, baseline_ppl, results)

print("\nSaved results to:")
print(RESULTS_MD_PATH)
