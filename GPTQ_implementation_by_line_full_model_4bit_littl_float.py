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
        os.path.join(data_dir, "GPTQ_data.bin"),
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


def get_activations(model, x_tokens, target_layer):
    activations = []

    # catching the data
    def hook(module, input, output):
        activations.append(input[0].detach())

    # attaching the hook to the current target layer
    handle = target_layer.register_forward_hook(hook)

    # Run the model
    with torch.no_grad():
        model(x_tokens)

    # Removing the hook
    handle.remove()

    return torch.cat(activations, dim=0)


def get_layer_weights(target_layer):
    # .detach() creates a copy
    # .clone() ensures we do not modify the original weights
    W = target_layer.weight.detach().clone()

    # Extract bias
    b = None
    if target_layer.bias is not None:
        b = target_layer.bias.detach().clone()

    return W, b


def get_target_layer(model, block_index, layer_name):
    """
    Returns a specific hidden linear layer from one transformer block.
    layer_name must be one of:
    - "attn.c_attn"
    - "attn.c_proj"
    - "mlp.c_fc"
    - "mlp.c_proj"
    """
    block = model.transformer.h[block_index]
    if layer_name == "attn.c_attn":
        return block.attn.c_attn
    if layer_name == "attn.c_proj":
        return block.attn.c_proj
    if layer_name == "mlp.c_fc":
        return block.mlp.c_fc
    if layer_name == "mlp.c_proj":
        return block.mlp.c_proj
    raise ValueError(f"Unknown layer name: {layer_name}")


def quantize_with_hessian_per_row(W, H_inv, scale_multiplier=1.0, which_list=3):
    """
    Symmetric int4 quantization with GPTQ-style compensation.
    W: [n_out, n_in]
    H_inv: [n_in, n_in]
    """
    W_quant = W.clone().float()
    n_out, n_in = W.shape

    if which_list == 1:
        quant_list = W_quant.new_tensor([
            -16.0, -8.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0.0,
            0.25,  0.5,  1.0,  2.0,  4.0,  8.0, 16.0
        ])
    elif which_list == 2:
        quant_list = W_quant.new_tensor([
        -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
        0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0
        ])
    elif which_list == 3:
        quant_list = W_quant.new_tensor([
        -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0,
        0.5,  1.0,  1.5,  2.0,  2.5,  3.0,  3.5
        ])
    else:
        print("ERROR ERORR, select quant_list")


    quant_list_max = quant_list.abs().max()  # 16.0, 6 or 3.5

    print("Weight compensation running (int4)")

    for i in range(n_in):
        # Current column
        w_col = W_quant[:, i].clone()

        # Using current working matrix (
        row_absmax = W_quant.abs().amax(dim=1).clamp_min(1e-8) #clamp to avoid div by 0
        s_vec = (row_absmax / quant_list_max) * scale_multiplier  # one scale per row


        #nearest quantization
        u = w_col / s_vec
        #we obtain indices 
        idx = torch.argmin((u.unsqueeze(1) - quant_list.unsqueeze(0)).abs(), dim=1)
        # replacing each u to nearest quantization level
        q_levels = quant_list[idx]

        
        w_q = q_levels * s_vec

        # Error and GPTQ compensation
        error = w_col - w_q
        if i < n_in - 1:
            update = error.unsqueeze(1) @ (H_inv[i, i + 1:] / H_inv[i, i]).unsqueeze(0)
            W_quant[:, i + 1:] -= update

        W_quant[:, i] = w_q

    return W_quant

def find_optimal_scale(W, H_inv, X_flat, Y_ref):
    best_mse = float("inf")
    best_scale = 1.0
    best_W_q = None

    test_scales = np.linspace(0.7, 1.3, 25)
    print(f"Scale Grid Search ({len(test_scales)} combinations)")

    for s_mult in test_scales:
        W_q_temp = quantize_with_hessian_per_row(
            W,
            H_inv,
            scale_multiplier=s_mult,
        )

        with torch.no_grad():
            Y_temp = X_flat @ W_q_temp.t()
            mse = torch.nn.functional.mse_loss(Y_temp, Y_ref).item()

        print(f"S-Mult: {s_mult:.3f} | MSE: {mse:.6f}")

        if mse < best_mse:
            best_mse = mse
            best_scale = s_mult
            best_W_q = W_q_temp

    print(f"Best Found -> S: {best_scale:.3f} (MSE: {best_mse:.6f})")
    return best_W_q, best_mse

if __name__ == "__main__":
    # Model
    DEVICE = "cuda"  # Or 'cpu' if you do not have a GPU
    CKPT_PATH = "nanogpt/out-shakespeare-char/ckpt.pt"
    LAYER_TYPES = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")

    # Tokenization
    DATA_DIR = r"nanogpt/data/shakespeare_char"
    BLOCK_SIZE = 256
    BATCH_SIZE = 128
    SEED = 555

    

    # Load the checkpoint
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    config = GPTConfig(**checkpoint["model_args"])

    # REVIVE THE MODEL
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

    num_blocks = len(model.transformer.h)
    target_layers = [
        (block_index, layer_name)
        for block_index in range(num_blocks)
        for layer_name in LAYER_TYPES
    ]

    x_tokens = get_calibration_batch(
        data_dir=DATA_DIR,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
    ).to(DEVICE)

    print(f"Batch shape: {tuple(x_tokens.shape)}")

    for block_index, layer_name in target_layers:
        print(f"\n=== Quantizing block {block_index} / {layer_name} ===")
        target_layer = get_target_layer(model, block_index, layer_name)

        # Capture activations for this specific layer
        X_big = get_activations(model, x_tokens, target_layer=target_layer)

        # Check shape
        print(f"X big shape: {X_big.shape}")

        # Obtain original weights and bias
        W_orig, b_orig = get_layer_weights(target_layer)

        print(f"W_orig Shape: {W_orig.shape}")
        if b_orig is not None:
            print(f"Original Bias Shape: {b_orig.shape}")

        # Flatten X and build reference output
        x_dim = W_orig.shape[1]
        X_flat = X_big.view(-1, x_dim)
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

            best_W_q, best_mse = find_optimal_scale(W_orig, H_inv, X_flat, Y_ref)

            print(f"Best mse: {best_mse}")
            print(f"Weights shape: {best_W_q.shape}")
            print(f"Weights unique sum: {best_W_q.unique().sum().item()}")

            # Replace only the currently targeted layer
            target_layer.weight.data.copy_(best_W_q.to(target_layer.weight.dtype))

    # Save one final checkpoint after all target hidden layers are quantized
    ckpt_gptq = dict(checkpoint)
    ckpt_gptq["model"] = model.state_dict()
    ckpt_dir = os.path.dirname(CKPT_PATH)
    ckpt_gptq_path = os.path.join(ckpt_dir, "ckpt_GPTQ_all_fp4_e1m2.pt")
    torch.save(ckpt_gptq, ckpt_gptq_path)
    print(f"\nSaved quantized checkpoint to: {ckpt_gptq_path}")

    # import matplotlib.pyplot as plt
    # plt.imshow(best_W_q.cpu().numpy())
    # plt.show()
