"""
Grouped M3 search for nanoGPT.

This script implements the search plan described in:
    m3_gridsearch_results/M3_grouped_search_plan.md

It is intentionally separate from:
- M3_single_layer_eval.py
- M3_multi_layer_eval.py
- M3_layerwise_gridsearch.py

Purpose:
- test a manageable shortlist of grouped M3 configurations
- avoid brute-forcing all layer-by-layer combinations
- record whether a configuration stays under the chosen acceptable PPL threshold

Current acceptable threshold:
- PPL <= 6.5

Important scope:
- this is still the simple runtime forward-patching experiment
- the checkpoint is not modified
- the model structure is not modified
- only selected layers are patched during evaluation
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

ACCEPTABLE_PPL = 6.5

RESULTS_DIR = "m3_gridsearch_results"
RESULTS_MD_PATH = os.path.join(RESULTS_DIR, "M3_grouped_search_results.md")

exec(open("configurator.py").read())


# -----------------------------------------------------------------------------
# Group definitions
# -----------------------------------------------------------------------------

GROUP_TO_LAYERS = {
    "attn.c_attn": [
        f"transformer.h.{i}.attn.c_attn" for i in range(6)
    ],
    "attn.c_proj": [
        f"transformer.h.{i}.attn.c_proj" for i in range(6)
    ],
    "mlp.c_fc early": [
        f"transformer.h.{i}.mlp.c_fc" for i in range(4)
    ],
    "mlp.c_fc b4": [
        "transformer.h.4.mlp.c_fc"
    ],
    "mlp.c_fc b5": [
        "transformer.h.5.mlp.c_fc"
    ],
    "mlp.c_proj b0": [
        "transformer.h.0.mlp.c_proj"
    ],
    "mlp.c_proj b1": [
        "transformer.h.1.mlp.c_proj"
    ],
    "mlp.c_proj b2": [
        "transformer.h.2.mlp.c_proj"
    ],
    "mlp.c_proj b3": [
        "transformer.h.3.mlp.c_proj"
    ],
    "mlp.c_proj b4": [
        "transformer.h.4.mlp.c_proj"
    ],
    "mlp.c_proj b5": [
        "transformer.h.5.mlp.c_proj"
    ],
}


# Search around the current best non-mlp.c_proj backbone.
# `none` means "leave these layers exact".
BASE_CONFIG = {
    "attn.c_attn": 2048,
    "attn.c_proj": 2048,
    "mlp.c_fc early": 4096,
    "mlp.c_fc b4": 4096,
    "mlp.c_fc b5": "none",
    "mlp.c_proj b0": "none",
    "mlp.c_proj b1": "none",
    "mlp.c_proj b2": "none",
    "mlp.c_proj b3": "none",
    "mlp.c_proj b4": "none",
    "mlp.c_proj b5": "none",
}


def make_config(config_id, note, **overrides):
    cfg = dict(BASE_CONFIG)
    cfg.update(overrides)
    cfg["id"] = config_id
    cfg["note"] = note
    return cfg


SEARCH_CONFIGS = [
    make_config("B0", "Current best backbone: no mlp.c_proj approximated."),
    make_config("B1", "Raise attn.c_attn to 4096.", **{"attn.c_attn": 4096}),
    make_config("B2", "Raise attn.c_proj to 4096.", **{"attn.c_proj": 4096}),
    make_config("B3", "Raise both attention groups to 4096.", **{"attn.c_attn": 4096, "attn.c_proj": 4096}),
    make_config("B4", "Lower mlp.c_fc early to 2048.", **{"mlp.c_fc early": 2048}),
    make_config("B5", "Lower mlp.c_fc b4 to 2048.", **{"mlp.c_fc b4": 2048}),
    make_config("B6", "Turn on mlp.c_fc b5 at 4096.", **{"mlp.c_fc b5": 4096}),

    make_config("S1_2048", "Single block trial: h1.mlp.c_proj = 2048.", **{"mlp.c_proj b1": 2048}),
    make_config("S1_4096", "Single block trial: h1.mlp.c_proj = 4096.", **{"mlp.c_proj b1": 4096}),
    make_config("S1_8192", "Single block trial: h1.mlp.c_proj = 8192.", **{"mlp.c_proj b1": 8192}),
    make_config("S3_2048", "Single block trial: h3.mlp.c_proj = 2048.", **{"mlp.c_proj b3": 2048}),
    make_config("S3_4096", "Single block trial: h3.mlp.c_proj = 4096.", **{"mlp.c_proj b3": 4096}),
    make_config("S3_8192", "Single block trial: h3.mlp.c_proj = 8192.", **{"mlp.c_proj b3": 8192}),
    make_config("S0_2048", "Single block trial: h0.mlp.c_proj = 2048.", **{"mlp.c_proj b0": 2048}),
    make_config("S0_4096", "Single block trial: h0.mlp.c_proj = 4096.", **{"mlp.c_proj b0": 4096}),
    make_config("S0_8192", "Single block trial: h0.mlp.c_proj = 8192.", **{"mlp.c_proj b0": 8192}),
    make_config("S2_2048", "Single block trial: h2.mlp.c_proj = 2048.", **{"mlp.c_proj b2": 2048}),
    make_config("S2_4096", "Single block trial: h2.mlp.c_proj = 4096.", **{"mlp.c_proj b2": 4096}),
    make_config("S2_8192", "Single block trial: h2.mlp.c_proj = 8192.", **{"mlp.c_proj b2": 8192}),

    make_config("P13_2048", "Pair trial: h1+h3.mlp.c_proj at 2048.", **{"mlp.c_proj b1": 2048, "mlp.c_proj b3": 2048}),
    make_config("P13_4096", "Pair trial: h1+h3.mlp.c_proj at 4096.", **{"mlp.c_proj b1": 4096, "mlp.c_proj b3": 4096}),
    make_config("P13_8192", "Pair trial: h1+h3.mlp.c_proj at 8192.", **{"mlp.c_proj b1": 8192, "mlp.c_proj b3": 8192}),
    make_config("P13_mixA", "Pair trial: h1=4096, h3=8192.", **{"mlp.c_proj b1": 4096, "mlp.c_proj b3": 8192}),
    make_config("P13_mixB", "Pair trial: h1=8192, h3=4096.", **{"mlp.c_proj b1": 8192, "mlp.c_proj b3": 4096}),

    make_config("Q130_2048", "Add h0 on top of h1+h3 at 4096.", **{"mlp.c_proj b1": 4096, "mlp.c_proj b3": 4096, "mlp.c_proj b0": 2048}),
    make_config("Q130_4096", "Add h0 on top of h1+h3 at 4096.", **{"mlp.c_proj b1": 4096, "mlp.c_proj b3": 4096, "mlp.c_proj b0": 4096}),
    make_config("Q132_2048", "Add h2 on top of h1+h3 at 4096.", **{"mlp.c_proj b1": 4096, "mlp.c_proj b3": 4096, "mlp.c_proj b2": 2048}),
    make_config("Q132_4096", "Add h2 on top of h1+h3 at 4096.", **{"mlp.c_proj b1": 4096, "mlp.c_proj b3": 4096, "mlp.c_proj b2": 4096}),
    make_config("Q1302_2048", "Add h0+h2 at 2048 on top of h1+h3 at 4096.", **{"mlp.c_proj b1": 4096, "mlp.c_proj b3": 4096, "mlp.c_proj b0": 2048, "mlp.c_proj b2": 2048}),
    make_config("Q1302_4096", "Add h0+h2 at 4096 on top of h1+h3 at 4096.", **{"mlp.c_proj b1": 4096, "mlp.c_proj b3": 4096, "mlp.c_proj b0": 4096, "mlp.c_proj b2": 4096}),
    make_config("Q13hi_2048", "Stronger pair h1+h3 at 8192, add h0+h2 at 2048.", **{"mlp.c_proj b1": 8192, "mlp.c_proj b3": 8192, "mlp.c_proj b0": 2048, "mlp.c_proj b2": 2048}),
    make_config("Q13hi_4096", "Stronger pair h1+h3 at 8192, add h0+h2 at 4096.", **{"mlp.c_proj b1": 8192, "mlp.c_proj b3": 8192, "mlp.c_proj b0": 4096, "mlp.c_proj b2": 4096}),

    make_config("T4_4096", "Turn on h4.mlp.c_proj only at 4096 on top of base.", **{"mlp.c_proj b4": 4096}),
    make_config("T4_8192", "Turn on h4.mlp.c_proj only at 8192 on top of base.", **{"mlp.c_proj b4": 8192}),
    make_config("T5_8192", "Turn on h5.mlp.c_proj only at 8192 on top of base.", **{"mlp.c_proj b5": 8192}),
    make_config("T13_4", "Best early pair h1+h3 at 4096 plus h4 at 4096.", **{"mlp.c_proj b1": 4096, "mlp.c_proj b3": 4096, "mlp.c_proj b4": 4096}),
    make_config("T13_5", "Best early pair h1+h3 at 4096 plus h5 at 8192.", **{"mlp.c_proj b1": 4096, "mlp.c_proj b3": 4096, "mlp.c_proj b5": 8192}),
    make_config("T13_45", "Best early pair h1+h3 at 4096 plus h4=4096 and h5=8192.", **{"mlp.c_proj b1": 4096, "mlp.c_proj b3": 4096, "mlp.c_proj b4": 4096, "mlp.c_proj b5": 8192}),

    make_config("Ufull_lo", "All early mlp.c_proj on: h1,h3=4096 and h0,h2=2048.", **{"mlp.c_proj b0": 2048, "mlp.c_proj b1": 4096, "mlp.c_proj b2": 2048, "mlp.c_proj b3": 4096}),
    make_config("Ufull_mid", "All early mlp.c_proj on: h0-h3 all at 4096.", **{"mlp.c_proj b0": 4096, "mlp.c_proj b1": 4096, "mlp.c_proj b2": 4096, "mlp.c_proj b3": 4096}),
    make_config("Ufull_hi", "All early mlp.c_proj on: h1,h3=8192 and h0,h2=4096.", **{"mlp.c_proj b0": 4096, "mlp.c_proj b1": 8192, "mlp.c_proj b2": 4096, "mlp.c_proj b3": 8192}),

    make_config("V1", "All early mlp.c_proj on plus h4 at 4096.", **{"mlp.c_proj b0": 2048, "mlp.c_proj b1": 4096, "mlp.c_proj b2": 2048, "mlp.c_proj b3": 4096, "mlp.c_proj b4": 4096}),
    make_config("V2", "All early mlp.c_proj on plus h5 at 8192.", **{"mlp.c_proj b0": 2048, "mlp.c_proj b1": 4096, "mlp.c_proj b2": 2048, "mlp.c_proj b3": 4096, "mlp.c_proj b5": 8192}),
    make_config("V3", "All early mlp.c_proj on plus h4=4096 and h5=8192.", **{"mlp.c_proj b0": 2048, "mlp.c_proj b1": 4096, "mlp.c_proj b2": 2048, "mlp.c_proj b3": 4096, "mlp.c_proj b4": 4096, "mlp.c_proj b5": 8192}),
]


# -----------------------------------------------------------------------------
# Runtime setup
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
# Model and dataset loading
# -----------------------------------------------------------------------------

print(f"Loading checkpoint: {CKPT_PATH}")
checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
gptconf = GPTConfig(**checkpoint["model_args"])


def strip_unwanted_prefix(state_dict):
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
# M3 patching helpers
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


def patch_linear_layer_with_m3(layer, k, seed_offset):
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


def restore_original_forward(layer):
    if hasattr(layer, "_m3_original_forward"):
        layer.forward = layer._m3_original_forward


def build_layer_to_k_mapping(config_entry):
    """
    Expand one grouped config into a per-layer mapping.

    Layers assigned 'none' are left exact and therefore omitted from the map.
    """
    mapping = {}
    for group_name, layer_paths in GROUP_TO_LAYERS.items():
        value = config_entry[group_name]
        if value == "none":
            continue
        for layer_path in layer_paths:
            mapping[layer_path] = value
    return mapping


def patch_config(current_model, layer_to_k):
    patched_layers = []
    for index, (layer_path, layer_k) in enumerate(layer_to_k.items()):
        layer = get_module_by_path(current_model, layer_path)
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"Target layer '{layer_path}' is {type(layer)}, not nn.Linear.")
        patch_linear_layer_with_m3(layer, k=layer_k, seed_offset=index)
        patched_layers.append(layer)
    return patched_layers


def restore_layers(patched_layers):
    for layer in patched_layers:
        restore_original_forward(layer)


def compute_proxy_cost(layer_to_k):
    """
    Simple cost proxy: sum of k over all approximated layers.

    This is not exact hardware cost. It is only a rough ranking helper.
    """
    return sum(layer_to_k.values())


# -----------------------------------------------------------------------------
# Markdown writer
# -----------------------------------------------------------------------------

def format_value(x):
    return f"{x:.4f}"


def write_results_markdown(output_path, baseline_ppl, rows):
    lines = []
    lines.append("# M3 Grouped Search Results")
    lines.append("")
    lines.append(f"Baseline PPL: **{format_value(baseline_ppl)}**")
    lines.append(f"Acceptable threshold: **PPL <= {format_value(ACCEPTABLE_PPL)}**")
    lines.append("")
    lines.append("Group meanings:")
    lines.append("- `attn.c_attn` = all 6 attention input projections")
    lines.append("- `attn.c_proj` = all 6 attention output projections")
    lines.append("- `mlp.c_fc early` = blocks 0-3")
    lines.append("- `mlp.c_fc b4` = block 4")
    lines.append("- `mlp.c_fc b5` = block 5")
    lines.append("- `mlp.c_proj b0` = block 0")
    lines.append("- `mlp.c_proj b1` = block 1")
    lines.append("- `mlp.c_proj b2` = block 2")
    lines.append("- `mlp.c_proj b3` = block 3")
    lines.append("- `mlp.c_proj b4` = block 4")
    lines.append("- `mlp.c_proj b5` = block 5")
    lines.append("")

    header = [
        "ID",
        "attn.c_attn",
        "attn.c_proj",
        "mlp.c_fc early",
        "mlp.c_fc b4",
        "mlp.c_fc b5",
        "mlp.c_proj b0",
        "mlp.c_proj b1",
        "mlp.c_proj b2",
        "mlp.c_proj b3",
        "mlp.c_proj b4",
        "mlp.c_proj b5",
        "PPL",
        "Acceptable",
        "Proxy Cost",
        "Note",
    ]
    sep = ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---", "---:", "---"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep) + " |")

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["id"],
                    str(row["attn.c_attn"]),
                    str(row["attn.c_proj"]),
                    str(row["mlp.c_fc early"]),
                    str(row["mlp.c_fc b4"]),
                    str(row["mlp.c_fc b5"]),
                    str(row["mlp.c_proj b0"]),
                    str(row["mlp.c_proj b1"]),
                    str(row["mlp.c_proj b2"]),
                    str(row["mlp.c_proj b3"]),
                    str(row["mlp.c_proj b4"]),
                    str(row["mlp.c_proj b5"]),
                    format_value(row["ppl"]),
                    "yes" if row["acceptable"] else "no",
                    str(row["proxy_cost"]),
                    row["note"],
                ]
            )
            + " |"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

print("\nEvaluating baseline model once...")
baseline_loss = evaluate_loss(model)
baseline_ppl = math.exp(baseline_loss)
print(f"Baseline loss: {baseline_loss:.6f}")
print(f"Baseline ppl : {baseline_ppl:.6f}")

results_rows = []

for config_entry in tqdm(SEARCH_CONFIGS, desc="Grouped configs", unit="cfg"):
    config_id = config_entry["id"]
    print(f"\n=== Testing config {config_id} ===")

    layer_to_k = build_layer_to_k_mapping(config_entry)
    patched_layers = patch_config(model, layer_to_k)

    patched_loss = evaluate_loss(model)
    patched_ppl = math.exp(patched_loss)
    restore_layers(patched_layers)

    acceptable = patched_ppl <= ACCEPTABLE_PPL
    proxy_cost = compute_proxy_cost(layer_to_k)

    print(f"PPL: {patched_ppl:.6f}")
    print(f"Acceptable: {acceptable}")

    row = {
        "id": config_id,
        "attn.c_attn": config_entry["attn.c_attn"],
        "attn.c_proj": config_entry["attn.c_proj"],
        "mlp.c_fc early": config_entry["mlp.c_fc early"],
        "mlp.c_fc b4": config_entry["mlp.c_fc b4"],
        "mlp.c_fc b5": config_entry["mlp.c_fc b5"],
        "mlp.c_proj b0": config_entry["mlp.c_proj b0"],
        "mlp.c_proj b1": config_entry["mlp.c_proj b1"],
        "mlp.c_proj b2": config_entry["mlp.c_proj b2"],
        "mlp.c_proj b3": config_entry["mlp.c_proj b3"],
        "mlp.c_proj b4": config_entry["mlp.c_proj b4"],
        "mlp.c_proj b5": config_entry["mlp.c_proj b5"],
        "ppl": patched_ppl,
        "acceptable": acceptable,
        "proxy_cost": proxy_cost,
        "note": config_entry["note"],
    }
    results_rows.append(row)
    write_results_markdown(RESULTS_MD_PATH, baseline_ppl, results_rows)

write_results_markdown(RESULTS_MD_PATH, baseline_ppl, results_rows)

print("\nSaved grouped search results to:")
print(RESULTS_MD_PATH)
