import torch
import numpy as np


CKPT_PATH = r"C:\Users\kaspa\projects\nanogpt\out-shakespeare-char\ckpt.pt"


def main() -> None:
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)

    print("Checkpoint keys:")
    for key in ckpt.keys():
        print(f"  - {key}")
    # this checkpoint uses "_orig_mod.transformer.wte.weight"
    # but we also support "transformer.wte.weight" just in case
    wte = state_dict.get("_orig_mod.transformer.wte.weight")
    if wte is None:
        wte = state_dict.get("transformer.wte.weight")

    state_dict = ckpt.get("model", ckpt)
    print(f"\nNumber of tensors in state dict: {len(state_dict)}")
    print("\nTensor shapes and dtypes:")
    for name, tensor in state_dict.items():
        if torch.is_tensor(tensor):
            print(f"{name:50s} shape={tuple(tensor.shape)!s:20s} dtype={tensor.dtype}")
    if wte is None:
        print("Could not find wte.weight in checkpoint.")
        return
    
    min_weight = wte.min().item()
    max_weight = wte.max().item()

    # Here we fin the smallest and largest weights
    print("wte.weight min:", min_weight)
    print("wte.weight max:", max_weight)


    #Here we will devide our weighs into thirds
    threshold = max_weight / 3
    print(f"\nUsing threshold: {threshold:.4f}")

    #Here we create a copy of wte and apply the thresholds
    wte_quantized = torch.zeros_like(wte)
    wte_quantized[wte > threshold] = max_weight
    wte_quantized[wte < -threshold] = min_weight


    total_elements = wte.numel()
    n_min = (wte_quantized == min_weight).sum().item()
    n_max = (wte_quantized == max_weight).sum().item()
    n_zero = (wte_quantized == 0).sum().item()



    print(f"Quantization Summary:")
    print(f"  - Set to Min ({min_weight:.4f}): {n_min} ({100*n_min/total_elements:.1f}%)")
    print(f"  - Set to Zero (0.0000): {n_zero} ({100*n_zero/total_elements:.1f}%)")
    print(f"  - Set to Max ({max_weight:.4f}): {n_max} ({100*n_max/total_elements:.1f}%)")

    # We save our new weights
    state_dict["_orig_mod.transformer.wte.weight"] = wte_quantized
    # Also update the non-compiled key if it exists
    if "transformer.wte.weight" in state_dict:
        state_dict["transformer.wte.weight"] = wte_quantized

    # fixing lm head #ask ReJ
    if "_orig_mod.lm_head.weight" in state_dict:
        state_dict["_orig_mod.lm_head.weight"] = wte_quantized
    if "lm_head.weight" in state_dict:
        state_dict["lm_head.weight"] = wte_quantized

    new_ckpt_path = CKPT_PATH.replace("ckpt.pt", "ckpt_quantized_3.pt")
    torch.save(ckpt, new_ckpt_path)
    print(f"\nQuantized checkpoint saved to: {new_ckpt_path}")





if __name__ == "__main__":
    main()
