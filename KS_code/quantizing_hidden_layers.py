import torch
import os

# --- CONFIGURATION ---
CKPT_PATH = r"C:\Users\kaspa\projects\nanogpt\out-shakespeare-char\ckpt.pt"

# Target the first transformer attention block
TARGET_KEY = "transformer.h.0.attn.c_attn.weight" 
# ---------------------

def main():
    if not os.path.exists(CKPT_PATH):
        print(f"❌ Error: Could not find {CKPT_PATH}")
        return
    #load the checkpoint
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)

    
    # Handling prefix naming
    # If the key isn't there, try adding the '_orig_mod.' prefix
    actual_key = TARGET_KEY
    if actual_key not in state_dict:
        actual_key = "_orig_mod." + TARGET_KEY

    if actual_key not in state_dict:
        print(f"❌ Error: Could not find '{TARGET_KEY}' in checkpoint.")
        print("Available keys start with:", list(state_dict.keys())[:5])
        return

    # Obtain the weights
    weights = state_dict[actual_key]
    print(f"Quantizing the matrix: {actual_key} ---")
    print(f"Initial shape: {tuple(weights.shape)}")

    #Calculating the threshold
    min_val = weights.min().item()
    max_val = weights.max().item()
    threshold = max_val / 6

    print(f"Min: {min_val:.4f} | Max: {max_val:.4f}")
    print(f"hreshold: {threshold:.4f}")

    # Create masks and quantize into discrete steps
    q_weights = torch.zeros_like(weights)
    q_weights[weights > threshold] = max_val
    q_weights[weights < -threshold] = min_val

    # Statistics
    total = q_weights.numel()
    n_zero = (q_weights == 0).sum().item()
    n_min = (q_weights == min_val).sum().item()
    n_max = (q_weights == max_val).sum().item()

    print(f"Quantization Summary for this matrix:")
    print(f"  - Zeros: {n_zero} ({100*n_zero/total:.1f}%)")
    print(f"  - Mins:  {n_min} ({100*n_min/total:.1f}%)")
    print(f"  - Maxs:  {n_max} ({100*n_max/total:.1f}%)")

    # 7. Overwrite ONLY this matrix in the state_dict
    state_dict[actual_key] = q_weights

    # 8. Save to a new checkpoint file
    new_path = CKPT_PATH.replace("ckpt.pt", "ckpt_h0_attn_only_thr_6.pt")
    torch.save(ckpt, new_path)


if __name__ == "__main__":
    main()