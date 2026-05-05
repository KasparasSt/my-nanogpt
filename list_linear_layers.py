"""
Print all nn.Linear layers in a nanoGPT checkpoint, along with their weight
shapes.

Output format example:
    transformer.h.0.attn.c_attn (384, 128)
    transformer.h.0.attn.c_proj (128, 128)
    transformer.h.0.mlp.c_fc (512, 128)
    transformer.h.0.mlp.c_proj (128, 512)

This is intentionally simple.
"""

import os

import torch
import torch.nn as nn

from model import GPT, GPTConfig


CKPT_PATH = os.path.join("out-shakespeare-char", "ckpt.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def strip_unwanted_prefix(state_dict):
    state_dict = dict(state_dict)
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    return state_dict


if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
config = GPTConfig(**checkpoint["model_args"])
model = GPT(config)
model.load_state_dict(strip_unwanted_prefix(checkpoint["model"]))
model.eval()


for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(f"{name} {tuple(module.weight.shape)}")
