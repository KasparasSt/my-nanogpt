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



def get_activations(model, x_tokens, layer_index = 0):
    activations = []

    #catching the data
    def hook(module, input, output):
        activations.append(input[layer_index].detach())

    # attaching the hook to the trget layers
    handle = model.transformer.h[layer_index].attn.c_attn.register_forward_hook(hook)

    # Run the model
    with torch.no_grad():
        model(x_tokens)

    # Removing the hood
    handle.remove()

    return torch.cat(activations, dim=0)



def get_layer_weights(model, layer_index=0):
    # We target the layer - transformer.h[0].attn.c_attn
    target_layer = model.transformer.h[layer_index].attn.c_attn

    # .detach() creates a copy 
    # .clone() ensures we nodify the original weights
    W = target_layer.weight.detach().clone()
    
    # Extract bias
    b = None
    if target_layer.bias is not None:
        b = target_layer.bias.detach().clone()
        
    return W, b


def quantize_with_hessian(W, H_inv, threshold):
    """
    W: [1152, 384] - Original Weights
    H_inv: [384, 384] - Inverse Hessian
    threshold: The value for Ternary clipping
    """
    W_quant = W.clone().float()
    n_out, n_in = W.shape # 1152, 384
    
    # We find the relevent weighs
    relevant_weights = W[W.abs() > threshold]
    if len(relevant_weights) > 0:
        s = relevant_weights.abs().mean()
    else:
        s = W.abs().mean() # Fallback

    print(f"Using Optimal Scale s: {s:.4f} (Max was {W.max():.4f})")

    print("Weigh compensation running")
    
    for i in range(n_in):
        # Extracting column by column
        w_col = W_quant[:, i].clone()
        
        # Ternary quantization, same as before
        # If weight > threshold -> v_max
        # If weight < -threshold -> v_min
        # Else -> 0
        w_q = torch.zeros_like(w_col)
        w_q[w_col > threshold] = s
        w_q[w_col < -threshold] = -s
        
        # Calculate the error, that occured due to quantization
        error = w_col - w_q
        
        # COMPENSATION 
        # We push the error to the 'future' columns using the Hessian inverse
        if i < n_in - 1:
            # This line calculates how much to adjust the remaining weights
            update = error.unsqueeze(1) @ (H_inv[i, i+1:] / H_inv[i, i]).unsqueeze(0)
            W_quant[:, i+1:] -= update
            
        # Store the quantized value
        W_quant[:, i] = w_q
        
    return W_quant


def find_optimal_threshold(W, H_inv, X_flat, Y_ref):
    best_mse = float('inf')
    best_threshold = 0
    best_W_q = None
    
    # We test thresholds from 0.1x to 2.0x the mean absolute weight
    mean_abs = W.abs().mean().item()
    test_thresholds = np.linspace(mean_abs * 0.1, mean_abs * 2.0, 10)
    
    print(f"Starting Grid Search across {len(test_thresholds)} thresholds...")
    
    for t in test_thresholds:
        # Running quantization
        W_q_temp = quantize_with_hessian(W, H_inv, t)
        
        # Computing mse each time
        with torch.no_grad():
            Y_temp = X_flat @ W_q_temp.t()
            mse = torch.nn.functional.mse_loss(Y_temp, Y_ref).item()
        
        print(f"Threshold: {t:.4f} | MSE: {mse:.6f}")
        
        if mse < best_mse:
            best_mse = mse
            best_threshold = t
            best_W_q = W_q_temp
            
    return best_threshold, best_W_q, best_mse


if __name__ == "__main__":

    # Model
    DEVICE = 'cuda' # Or 'cuda' if you have a GPU
    CKPT_PATH = 'nanogpt/out-shakespeare-char/ckpt.pt'
    LAYER_INDEX = 0

    # Tokenization
    DATA_DIR = r"nanogpt/data/shakespeare_char"
    BLOCK_SIZE = 256
    BATCH_SIZE = 128
    SEED = 555

    # Load the checkpoint
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    config = GPTConfig(**checkpoint['model_args'])
    
    # REVIVE THE MODEL (The Engine)
    model = GPT(config)
    
    # Remove the '_orig_mod.' prefix
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval() # Put the model in evaluation mode



    x_tokens = get_calibration_batch(
        data_dir=DATA_DIR,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
    ).to(DEVICE)

    print(f"Batch shape: {tuple(x_tokens.shape)}")

    # Capture activations
    X_big = get_activations(model, x_tokens, layer_index=LAYER_INDEX)

    #Check shape
    print(f"X big shape: {X_big.shape}")


    # Obtaining original weighs and biases
    W_orig, b_orig = get_layer_weights(model, layer_index=LAYER_INDEX)

    print(f"W_orig Shape: {W_orig.shape}")
    if b_orig is not None:
        print(f"Original Bias Shape: {b_orig.shape}")
    
    # Up to here we have original weights (W_orign) and X (X_big)

    # Now we can calculate matrix multiplications, X @ W to get the reference part for our quantization
    #Flatten the X
    X_flat = X_big.view(-1, 384)
    print("X_flat size: ", X_flat.shape)
    with torch.no_grad():
        Y_ref = X_flat @ W_orig.t()
        print("Shape of Y_ref: ",Y_ref.shape)

        #Here we compute the Hessian
        H = X_flat.t() @ X_flat

        #Here we prevent division by zero
        eps = 0.01 * torch.mean(torch.diag(H))
        H += eps * torch.eye(H.shape[0], device=H.device)

        #Invert hessian
        H_inv = torch.inverse(H.float())

        best_threshold, best_W_q, best_mse = find_optimal_threshold(W_orig, H_inv, X_flat, Y_ref)

        print(f"Best threshold: {best_threshold}")
        print(f"Best mse: {best_mse}")
        print(f"Weighs shape: {best_W_q.shape}")


    import matplotlib.pyplot as plt
    plt.imshow(best_W_q.cpu().numpy())
    plt.show()