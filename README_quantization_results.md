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