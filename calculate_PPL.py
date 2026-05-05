"""
Evaluate perplexity for a saved nanoGPT checkpoint.
"""
import math
import os
from contextlib import nullcontext

import numpy as np
import torch

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# defaults (can be overridden via configurator.py or CLI, e.g. --device=cpu)
ckpt_path = os.path.join("out-shakespeare-char", "ckpt_M3_k1024_mlp-c_proj".pt)
split = "test"  # 'train', 'val', or 'test'
eval_iters = 100
batch_size = 64
stride = 128  # None -> block_size (no overlap). Smaller value enables sliding-window overlap.
full_split = True  # If True, evaluate whole split deterministically; if False, cap by eval_iters*batch_size.
seed = 1337
device = "cuda"  # 'cpu', 'cuda', 'cuda:0', ...
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile = False
exec(open("configurator.py").read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

if split not in {"train", "val", "test"}:
    raise ValueError(f"Invalid split '{split}'. Expected one of: train, val, test.")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

print(f"Loading checkpoint: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device)

gptconf = GPTConfig(**checkpoint["model_args"])
model = GPT(gptconf)
state_dict = checkpoint["model"]
unwanted_prefix = "_orig_mod."
for k in list(state_dict.keys()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)


if compile:
    model = torch.compile(model)

dataset = checkpoint.get("config", {}).get("dataset", "openwebtext")
data_path = os.path.join("data", dataset, f"{split}.bin")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data split file not found: {data_path}")

block_size = gptconf.block_size
data = np.memmap(data_path, dtype=np.uint16, mode="r")
if len(data) <= block_size:
    raise ValueError(
        f"{data_path} length ({len(data)}) must be > block_size ({block_size})."
    )

def get_batch_from_starts(starts):
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in starts])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in starts])
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def evaluate_loss():
    # Deterministic sliding-window evaluation:
    # use fixed sequential starts with configurable stride.
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError("Not enough tokens for deterministic evaluation.")

    local_stride = block_size if stride is None else int(stride)
    if local_stride <= 0:
        raise ValueError(f"stride must be > 0, got {local_stride}")

    starts = torch.arange(0, max_start + 1, local_stride, dtype=torch.long)

    if not full_split:
        max_examples = eval_iters * batch_size
        starts = starts[:max_examples]

    num_examples = len(starts)
    if num_examples == 0:
        raise ValueError("No evaluation windows were generated.")

    total_loss = 0.0
    total_examples = 0
    for i in range(0, num_examples, batch_size):
        batch_starts = starts[i:i + batch_size]
        x, y = get_batch_from_starts(batch_starts.tolist())
        with ctx:
            _, loss = model(x, y)
        bs = len(batch_starts)
        total_loss += loss.item() * bs
        total_examples += bs
    return total_loss / total_examples

mean_loss = evaluate_loss()
ppl = math.exp(mean_loss)

print("\n" + "=" * 40)
print(f"Dataset: {dataset}")
print(f"Split: {split}")
print(f"Full split eval: {full_split}")
print(f"Stride: {block_size if stride is None else stride}")
if not full_split:
    print(f"Eval iters cap: {eval_iters}")
print(f"Batch size: {batch_size}")
print(f"Block size: {block_size}")
print(f"{split.capitalize()} Loss: {mean_loss:.4f}")
print(f"{split.capitalize()} Perplexity: {ppl:.4f}")
print("=" * 40)
