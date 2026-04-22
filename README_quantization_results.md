## Latest Update: GPTQ By-Line (Threshold Search + Dynamic Scale) (`GPTQ_implementation_by_line.py`)

This is the current best result in this repo. Also placed new weights into a new checkpoint `ckpt_GPTQ.pt` and evaluated perplexity with one modified layer.

### Method

- Quantization is done column-by-column with GPTQ-style Hessian compensation.
- `threshold_multiplier` is found by grid search.
- Scale is **not** grid-searched. It is computed from remaining relevant weights:
  - relevant weights are those where `abs(weight) > threshold` (per row)
  - scale = mean absolute value of relevant weights
  - fallback = row mean absolute value if no relevant weights remain

### Best Result

- `Best Found -> T: 0.915 (MSE: 0.064732)`
- `Best mse: 0.06473217904567719`
- `Quantized Test PPL: 6.0187 (Non-quantized = 6.0011)` -- almost no increase in perplexity

---

## GPTQ By-Line + 2D Grid Search (`GPTQ_implementation_by_line_gridsearch.py`)

This script implements a GPTQ-style quantization experiment for one attention projection layer, using:
- column-by-column error compensation with inverse Hessian
- per-row ternary thresholds/scales
- 2D grid search over `(threshold_multiplier, scale_multiplier)`

### What it does

1. Loads nanoGPT checkpoint (`ckpt.pt`) and restores model weights.
2. Creates a calibration token batch from `test.bin`.
3. Captures activations from `transformer.h[layer_index].attn.c_attn` using a forward hook.
4. Builds reference outputs:
   - `X_flat = X_big.view(-1, 384)`
   - `Y_ref = X_flat @ W_orig.T`
5. Computes Hessian approximation:
   - `H = X_flat.T @ X_flat`
   - adds damping `eps = 0.01 * mean(diag(H))`
   - computes `H_inv = inverse(H)`
6. Runs quantization + 2D search:
   - `threshold_multiplier` in `np.linspace(0.05, 1.1, 35)`
   - `scale_multiplier` in `np.linspace(0.05, 1.1, 35)`
   - total combinations: `35 x 35 = 1225`
7. Selects the best quantized weight matrix by minimum output MSE.



### Quantization method (per-row)

For each output row:
- `row_mean = mean(abs(W[row]))`
- `threshold[row] = row_mean * threshold_multiplier`
- `scale[row] = row_mean * scale_multiplier`
- ternary quantization values are `{ -scale[row], 0, +scale[row] }`

Quantization is applied column-by-column, and each column’s quantization error is propagated to future columns using GPTQ-style Hessian inverse compensation.

### Main config in `__main__`

- `DEVICE`
- `CKPT_PATH`
- `LAYER_INDEX`
- `DATA_DIR`
- `BLOCK_SIZE`
- `BATCH_SIZE`
- `SEED`

### Run


```python GPTQ_implementation_by_line_gridsearch.py```


### Result

By running the gridsearch, mse of 0.19 was obtained. The threshold of 0.3 and scaling of 1.1 was found to perform the best. However, the results are worse, compared to previously done layer-wise quantizatiom (mse 0.11) where the thresholds were searched and scaling coefficients were just based on the average on remaining weights.



# GPTQ Prototype Script (`GPTQ_implementation.py`)

This script is a **layer-wise GPTQ-style ternary quantization prototype** for nanoGPT.

## What it does

1. Loads a trained checkpoint (`ckpt.pt`).
2. Builds a calibration token batch from `test.bin`.
3. Captures activations from `transformer.h[layer_index].attn.c_attn` using a forward hook.
4. Computes:
   - reference output `Y_ref = X @ W^T`
   - Hessian approximation `H = X^T @ X`
   - inverse Hessian `H_inv`
5. Runs threshold search and Hessian-aware ternary quantization.
6. Reports best threshold + MSE and visualizes quantized weights.

### Main settings (edit in `__main__`)

- `DEVICE` (`"cpu"` or `"cuda"`)
- `CKPT_PATH` (path to checkpoint)
- `LAYER_INDEX` (transformer block index)
- `DATA_DIR` (folder containing `test.bin`)
- `BLOCK_SIZE`
- `BATCH_SIZE`
- `SEED`

### Run

```bash
python GPTQ_implementation.py
```



# Ternary Quantization of Embedding Layers

'KS code / quantizing_the_model_input_output.py'
This report summarizes the initial results of getting familiar with LLM quantization. In this case two layers **Weight Token Embedding (WTE)** and **Language Modeling Head (LM Head)** were quantized into three discrete weights (-max, 0, +max) by applying different thresholds.

(lr = 2.5e-3, min_lr = 2.5e-4, iterations = 1000, other parameters are the same as described in README.md)


## Methods
* **Layer Tying:** Quantization was applied to the weights of the `wte` and `lm_head` layers.
(tried to quantize only wte first, but apparently you can't change the input without changing the output, since Wight Tying is implemented, using the same shared dictionary for both inout and output)

* **Weights:** $Min \approx -0.2447$, $Zero = 0.0000$, $Max \approx 0.2477$. Since the distribution is symetrical, decided to use +- max.

* **Logic:** Threshold $\Delta$ was used which is obtained by $Max / N$, where $N$ is an integer. Values within $[-\Delta, \Delta]$ were set to 0. Values $>\Delta$ were set to $Max$. Values $<-\Delta$ were set to $-Max$.


## Results

| Configuration | Threshold ($\Delta$) | Zero-Zone | Test Loss | **Test Perplexity** |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Float32)** | None | 0% | 1.7919 | **6.0011** |
| **Ternary (Max / 3)** | 0.0826 | **84.6%** | 2.4476 | **11.5607** |
| **Ternary (Max / 5)** | 0.0495 | 60.8% | 3.7533 | **42.6625** |
| **Ternary (Max / 2)** | 0.1238 | ~97% | 4.0094 | **55.1143** |

----


# Ternary Quantization of hidden layers

As I understand, no one actually quantizes input and output layers, that's why I moved on to test hidden attention layer quantization.

The same approach was used as with the Embedding layers. Here the weighs are $Min \approx -0.2888, Max \approx 0.2538$.

## Results for first attention layer


| Configuration | Zero-Zone | Test Loss | **Test Perplexity** |
| :--- | :--- | :--- | :--- |
| **Max / 2** | 99.2% | 2.7618 | **15.8278** |
| **Max / 3** | 95.1% | 2.6812 | **14.6027** |
| **Max / 4** | 89.0% | 2.2920 | **9.8950** |
| **Max / 5** | **82.3%** | 2.1374 | **8.4773** (Best) |
| **Max / 7** | 70.0% | 2.1684 | **8.7446** |
| **Max / 8** | 64.6% | 2.2114 | **9.1281** |


The optimal quantization threshold was found to be around Max / 5. Since these values were computed by guessing, later the GPTQ or similar approach will be used to find more a more optimal result.
