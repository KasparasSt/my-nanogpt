"""
Validate a quantized checkpoint against a baseline checkpoint.

What this script does:
1) Loads `ckpt.pt` (baseline) and `ckpt_GPTQ_all.pt` (quantized).
2) Compares matching tensors key-by-key.
3) Prints unique-weight statistics and which tensors changed.
"""

import os
import torch


# Edit these if your paths are different
BASELINE_CKPT = os.path.join("out-shakespeare-char", "ckpt.pt")
QUANT_CKPT = os.path.join("out-shakespeare-char", "ckpt_GPTQ_all.pt")


def load_model_state_dict(ckpt_path):
    """Load checkpoint and return normalized model state_dict."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)

    # Normalize potential compiled-model prefix so keys match better.
    normalized = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            normalized[k[len("_orig_mod."):]] = v
        else:
            normalized[k] = v
    return normalized


def tensor_unique_count(tensor):
    """Return count of unique values in a tensor."""
    return torch.unique(tensor.detach().cpu()).numel()


def state_dict_tensor_bytes(state_dict):
    """Approximate bytes used by tensors in a state_dict."""
    total = 0
    for v in state_dict.values():
        if isinstance(v, torch.Tensor):
            total += v.numel() * v.element_size()
    return total


def pct_shrink(old_value, new_value):
    """Percent shrink from old -> new. Positive means smaller."""
    if old_value == 0:
        return 0.0
    return (old_value - new_value) / old_value * 100.0


def main():
    if not os.path.exists(BASELINE_CKPT):
        raise FileNotFoundError(f"Baseline checkpoint not found: {BASELINE_CKPT}")
    if not os.path.exists(QUANT_CKPT):
        raise FileNotFoundError(f"Quantized checkpoint not found: {QUANT_CKPT}")

    base_sd = load_model_state_dict(BASELINE_CKPT)
    quant_sd = load_model_state_dict(QUANT_CKPT)

    base_keys = set(base_sd.keys())
    quant_keys = set(quant_sd.keys())
    common_keys = sorted(base_keys & quant_keys)
    only_base = sorted(base_keys - quant_keys)
    only_quant = sorted(quant_keys - base_keys)

    print("=== Checkpoint Key Summary ===")
    print(f"Baseline keys: {len(base_keys)}")
    print(f"Quantized keys: {len(quant_keys)}")
    print(f"Common keys: {len(common_keys)}")
    print(f"Only in baseline: {len(only_base)}")
    print(f"Only in quantized: {len(only_quant)}")

    if only_base:
        print("\nSample keys only in baseline:")
        for k in only_base[:10]:
            print(f"  - {k}")
    if only_quant:
        print("\nSample keys only in quantized:")
        for k in only_quant[:10]:
            print(f"  - {k}")

    # Size comparison
    baseline_file_bytes = os.path.getsize(BASELINE_CKPT)
    quant_file_bytes = os.path.getsize(QUANT_CKPT)
    file_shrink_pct = pct_shrink(baseline_file_bytes, quant_file_bytes)

    baseline_tensor_bytes = state_dict_tensor_bytes(base_sd)
    quant_tensor_bytes = state_dict_tensor_bytes(quant_sd)
    tensor_shrink_pct = pct_shrink(baseline_tensor_bytes, quant_tensor_bytes)

    print("\n=== Size Comparison ===")
    print(
        f"Checkpoint file size (bytes): {baseline_file_bytes} -> {quant_file_bytes} "
        f"(shrink: {file_shrink_pct:.2f}%)"
    )
    print(
        f"State_dict tensor bytes: {baseline_tensor_bytes} -> {quant_tensor_bytes} "
        f"(shrink: {tensor_shrink_pct:.2f}%)"
    )

    print("\n=== Tensor Comparison (common keys) ===")
    changed = []
    unchanged = 0

    for k in common_keys:
        a = base_sd[k]
        b = quant_sd[k]

        # Skip non-tensor entries just in case.
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            continue

        if a.shape != b.shape:
            print(f"[SHAPE MISMATCH] {k}: {tuple(a.shape)} vs {tuple(b.shape)}")
            continue

        same = torch.equal(a, b)
        if same:
            unchanged += 1
            continue

        u_base = tensor_unique_count(a)
        u_quant = tensor_unique_count(b)
        changed.append((k, u_base, u_quant))

    print(f"Unchanged tensors: {unchanged}")
    print(f"Changed tensors: {len(changed)}")

    if not changed:
        print("\nNo changed tensors found.")
        return

    print("\nChanged tensor unique counts (baseline -> quantized):")
    for k, u_base, u_quant in changed:
        print(f"  - {k}: {u_base} -> {u_quant}")

    # Aggregate unique count over all changed weight tensors.
    changed_weight_keys = [k for k, _, _ in changed if k.endswith(".weight")]
    changed_weights = [
        quant_sd[k].detach().cpu().reshape(-1)
        for k, _, _ in changed
        if k.endswith(".weight")
    ]
    if changed_weights:
        base_changed_weights = torch.cat(
            [base_sd[k].detach().cpu().reshape(-1) for k in changed_weight_keys]
        )
        all_changed_weights = torch.cat(changed_weights)
        base_unique = torch.unique(base_changed_weights)
        all_unique = torch.unique(all_changed_weights)
        unique_shrink_pct = pct_shrink(base_unique.numel(), all_unique.numel())

        print("\n=== Unique-Value Compression ===")
        print(
            "Aggregate unique values across changed .weight tensors: "
            f"{base_unique.numel()} -> {all_unique.numel()}"
        )
        print(f"Unique-value shrink: {unique_shrink_pct:.2f}%")

        print("\nAggregate unique values across ALL changed .weight tensors in quantized ckpt:")
        print(f"  Unique count: {all_unique.numel()}")
        if all_unique.numel() <= 20:
            print(f"  Values: {all_unique.tolist()}")
        else:
            print(f"  Min/Max: {all_unique.min().item():.6f} / {all_unique.max().item():.6f}")


if __name__ == "__main__":
    main()
