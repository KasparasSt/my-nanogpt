import os
import numpy as np
import torch
from model import GPT, GPTConfig


def get_calibration_batch(data_dir, block_size, batch_size, seed):
    """
    Build one GPTQ calibration batch from test.bin.

    Inputs:
    - data_dir: folder that contains test.bin
    - block_size: number of tokens per sample
    - batch_size: number of samples in the batch
    - seed: random seed for reproducibility

    Output:
    - x: tensor of shape [batch_size, block_size], dtype torch.int64
    """

    # For reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load test.bin
    test_data = np.memmap(
        os.path.join(data_dir, "test.bin"),
        dtype=np.uint16,
        mode="r",
    )

    # Random starting positions for each sample
    ix = torch.randint(len(test_data) - block_size, (batch_size,))

    # Slice token blocks and stack into one batch tensor
    x = torch.stack(
        [
            torch.from_numpy((test_data[i : i + block_size]).astype(np.int64))
            for i in ix
        ]
    )

    return x


def get_activations(model, x_tokens, layer_index=0):
    activations = []

    # catching the data
    def hook(module, input, output):
        activations.append(input[layer_index].detach())

    # attaching the hook to the target layers
    handle = model.transformer.h[layer_index].attn.c_attn.register_forward_hook(hook)

    # Run the model
    with torch.no_grad():
        model(x_tokens)

    # Removing the hook
    handle.remove()

    return torch.cat(activations, dim=0)


def get_layer_weights(model, layer_index=0):
    # We target the layer - transformer.h[0].attn.c_attn
    target_layer = model.transformer.h[layer_index].attn.c_attn

    # .detach() creates a copy
    # .clone() ensures we do not modify the original weights
    W = target_layer.weight.detach().clone()

    # Extract bias
    b = None
    if target_layer.bias is not None:
        b = target_layer.bias.detach().clone()

    return W, b


def quantize_with_hessian_per_row(W, H_inv, threshold_multiplier=0.5):
    """
    W: [1152, 384] - Original Weights
    H_inv: [384, 384] - Inverse Hessian

    Quantization behavior:
    - threshold is controlled by threshold_multiplier (searched by grid search)
    - scale is recomputed from remaining relevant weights:
      s(row) = mean(abs(weights > threshold)) with mean(abs(row)) fallback
    """
    W_quant = W.clone().float()
    n_out, n_in = W.shape

    row_means = W.abs().mean(dim=1, keepdim=True)
    thresholds = row_means * threshold_multiplier

    print("Weight compensation running")

    for i in range(n_in):
        # Extracting column by column
        w_col = W_quant[:, i].clone()

        # Per-row thresholds
        t_vec = thresholds.view(-1)

        # Scale from remaining weights in the current working matrix.
        # "Relevant" means abs(weight) > threshold for that row.
        remaining_abs = W_quant.abs()
        relevant_mask = remaining_abs > thresholds
        relevant_sum = (remaining_abs * relevant_mask).sum(dim=1)
        relevant_count = relevant_mask.sum(dim=1)
        fallback_mean = remaining_abs.mean(dim=1)
        s_vec = torch.where(
            relevant_count > 0,
            relevant_sum / relevant_count.clamp_min(1),
            fallback_mean,
        )

        # Ternary quantization using dynamic per-row scales.
        w_q = torch.zeros_like(w_col)
        w_q[w_col > t_vec] = s_vec[w_col > t_vec]
        w_q[w_col < -t_vec] = -s_vec[w_col < -t_vec]

        # Calculate the error that occurred due to quantization
        error = w_col - w_q

        # COMPENSATION
        # Push the error to future columns using the Hessian inverse
        if i < n_in - 1:
            update = error.unsqueeze(1) @ (H_inv[i, i + 1:] / H_inv[i, i]).unsqueeze(0)
            W_quant[:, i + 1:] -= update

        # Store the quantized value
        W_quant[:, i] = w_q

    return W_quant


def find_optimal_per_row_threshold(W, H_inv, X_flat, Y_ref):
    best_mse = float("inf")
    best_threshold = 0.0
    best_W_q = None

    # Search only threshold multipliers
    test_thresholds = np.linspace(0.05, 1.1, 35)
    print(f"Threshold Grid Search ({len(test_thresholds)} combinations)")

    for t_mult in test_thresholds:
        W_q_temp = quantize_with_hessian_per_row(
            W,
            H_inv,
            threshold_multiplier=t_mult,
        )

        with torch.no_grad():
            Y_temp = X_flat @ W_q_temp.t()
            mse = torch.nn.functional.mse_loss(Y_temp, Y_ref).item()

        print(f"T-Mult: {t_mult:.3f} | MSE: {mse:.6f}")

        if mse < best_mse:
            best_mse = mse
            best_threshold = t_mult
            best_W_q = W_q_temp

    print(f"Best Found -> T: {best_threshold:.3f} (MSE: {best_mse:.6f})")
    return best_W_q, best_mse


if __name__ == "__main__":
    # Model
    DEVICE = "cuda"  # Or 'cpu' if you do not have a GPU
    CKPT_PATH = "nanogpt/out-shakespeare-char/ckpt.pt"
    LAYER_INDEX = 0

    # Tokenization
    DATA_DIR = r"nanogpt/data/shakespeare_char"
    BLOCK_SIZE = 256
    BATCH_SIZE = 128
    SEED = 555

    # Load the checkpoint
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    config = GPTConfig(**checkpoint["model_args"])

    # REVIVE THE MODEL (The Engine)
    model = GPT(config)

    # Remove the '_orig_mod.' prefix
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()  # Put the model in evaluation mode

    x_tokens = get_calibration_batch(
        data_dir=DATA_DIR,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
    ).to(DEVICE)

    print(f"Batch shape: {tuple(x_tokens.shape)}")

    # Capture activations
    X_big = get_activations(model, x_tokens, layer_index=LAYER_INDEX)

    # Check shape
    print(f"X big shape: {X_big.shape}")

    # Obtain original weights and bias
    W_orig, b_orig = get_layer_weights(model, layer_index=LAYER_INDEX)

    print(f"W_orig Shape: {W_orig.shape}")
    if b_orig is not None:
        print(f"Original Bias Shape: {b_orig.shape}")

    # Flatten X and build reference output
    X_flat = X_big.view(-1, 384)
    print("X_flat size: ", X_flat.shape)
    with torch.no_grad():
        Y_ref = X_flat @ W_orig.t()
        print("Shape of Y_ref: ", Y_ref.shape)

        # Compute Hessian
        H = X_flat.t() @ X_flat

        # Prevent division by zero / bad conditioning
        eps = 0.01 * torch.mean(torch.diag(H))
        H += eps * torch.eye(H.shape[0], device=H.device)

        # Invert Hessian
        H_inv = torch.inverse(H.float())

        best_W_q, best_mse = find_optimal_per_row_threshold(W_orig, H_inv, X_flat, Y_ref)

        print(f"Best mse: {best_mse}")
        print(f"Weights shape: {best_W_q.shape}")
        print(f"Weights unique sum: {best_W_q.unique().sum().item()}")

        # -----------------------------
        # Save a new checkpoint file:
        # Replacing only the modified layer
        target_layer = model.transformer.h[LAYER_INDEX].attn.c_attn
        target_layer.weight.data.copy_(best_W_q.to(target_layer.weight.dtype))

        # Build a new checkpoint object and store the updated model weights
        ckpt_gptq = dict(checkpoint)
        ckpt_gptq["model"] = model.state_dict()

        # Save next to original checkpoint as ckpt_GPTQ.pt
        ckpt_dir = os.path.dirname(CKPT_PATH)
        ckpt_gptq_path = os.path.join(ckpt_dir, "ckpt_GPTQ.pt")
        torch.save(ckpt_gptq, ckpt_gptq_path)
        print(f"Saved quantized checkpoint to: {ckpt_gptq_path}")

    # import matplotlib.pyplot as plt
    # plt.imshow(best_W_q.cpu().numpy())
    # plt.show()
