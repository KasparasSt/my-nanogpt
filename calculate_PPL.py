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
ckpt_path = os.path.join("out-shakespeare-char", "ckpt.pt")
split = "test"  # 'train', 'val', or 'test'
eval_iters = 200
batch_size = 64
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

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def evaluate_loss():
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch()
        with ctx:
            _, loss = model(x, y)
        losses[k] = loss.item()
    return losses.mean().item()

mean_loss = evaluate_loss()
ppl = math.exp(mean_loss)

print("\n" + "=" * 40)
print(f"Dataset: {dataset}")
print(f"Split: {split}")
print(f"Eval iters: {eval_iters}")
print(f"Batch size: {batch_size}")
print(f"Block size: {block_size}")
print(f"{split.capitalize()} Loss: {mean_loss:.4f}")
print(f"{split.capitalize()} Perplexity: {ppl:.4f}")
print("=" * 40)
