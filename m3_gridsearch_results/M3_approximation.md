# M3 Approximation Experiments


Scripts:

- [M3_single_layer_eval.py](./M3_single_layer_eval.py)
- [M3_multi_layer_eval.py](./M3_multi_layer_eval.py)

## Table

`transformer.h.0.attn.c_attn` layer under test

| k | Baseline PPL | Patched PPL | Ratio |
|---:|---:|---:| ---: |
| 64 | 5.889 | 60.576 | 10.286 |
| 128 | 5.889 | 32.272 | 5.480 |
| 256 | 5.889 | 12.921 | 3.477 |
| 512 | 5.889 | 6.221 | **1.0563** |
| 1024 | 5.889 | 5.8441 | 0.992 |
| 2048 | 5.889 | 5.912 | 1.0038 |

Only MLP layers in first block under test

- `transformer.h.0.mlp.c_fc`
- `transformer.h.0.mlp.c_proj`


| k | Baseline PPL | Patched PPL | Ratio |
|---:|---:|---:| ---: |
| 256 | 5.889 | 36.837 | 6.255 |
| 512 | 5.889 | 18.118 | 3.076 |
| 1024 | 5.889 | 8.833 | 1.500 |
| 2048 | 5.889 | 6.288 | **1.068** |
| 4096  | 5.889 | 5.791 | 0.983 |


Only attention layers in first block under test

- `transformer.h.0.attn.c_attn`
- `transformer.h.0.attn.c_proj`


| k | Baseline PPL | Patched PPL | Ratio |
|---:|---:|---:| ---: |
| 256 | 5.889 | 25.303 | 4.296 |
| 512 | 5.889 | 7.243 | 1.230 |
| 1024 | 5.889 | 5.940 | **1.008** |
| 2048 | 5.889 | 5.940 | 1.008 |




Entire first block linear layers under test

- `transformer.h.0.attn.c_attn`
- `transformer.h.0.attn.c_proj`
- `transformer.h.0.mlp.c_fc`
- `transformer.h.0.mlp.c_proj`

| k | Baseline PPL | Patched PPL | Ratio |
|---:|---:|---:| ---: |
| 512 | 5.889 | 47.4771 | 8.061 |
| 1024 | 5.889 | 11.379 | 1.932 |
| 2048 | 5.889 | 6.495 | 1.627 |
| 4096  | 5.889 | 6.040 | **1.026** |



All transformer layers approximated


| k | Baseline PPL | Patched PPL | Ratio |
|---:|---:|---:| ---: |
| 512 | 5.889 | 221.805 | 37.662 |
| 1024 | 5.889 | 70.203 | 11.920 |
| 2048 | 5.889 | 16.086 | 2.731 |
| 4096  | 5.889 | 7.838 | 1.331 |
| 8192  | 5.889 | 6.147 | **1.0438** |
| 12288 (32*n*)  | 5.889 | 5.990 | 1.017 |



All MLP layers approximated


| k | Baseline PPL | Patched PPL | Ratio |
|---:|---:|---:| ---: |
| 512 | 5.889 | 146.719 | 24.913 |
| 1024 | 5.889 | 45.083 | 7.655 |
| 2048 | 5.889 | 13.490 | 2.291 |
| 4096  | 5.889 | 6.894 | 1.171 |
| 8192  | 5.889 | 6.117 | 1.039 |
| 12288 (32*n*)  | 5.889 | 5.896 | 1.001 |



All attention layers approximated


| k | Baseline PPL | Patched PPL | Ratio |
|---:|---:|---:| ---: |
| 512 | 5.889 | 16.561 | 2.812 |
| 1024 | 5.889 | 7.383 | 1.254 |
| 2048 | 5.889 | 6.178 | 1.049 |
| 4096  | 5.889 | 6.022 | 1.022 |
| 8192  | 5.889 | 5.926 | 1.006 |
| 12288 (32*n*)  | 5.889 | 5.887 | 1.000 |

**Conclusion:** MLP layers are a lot more sensitive to M3 simplification.


## Custom combinations

Attn layers: `k = 4096`, MLP layers: `k = 8192`. **PPL: 6.416**. Ratio: 1.090

All layers approximated except mlp.c_proj

According to [M3_layerwise_gridsearch](nanogpt\m3_gridsearch_results\M3_approximation.md), the least stable layers are  mlp.c_proj. So I will try to evaluate PPL while these layers are left unquantized.


**Here we quantize all linear layers except for: `h.5.mlp_fc`**

| k | Baseline PPL | Patched PPL | Ratio |
|---:|---:|---:| ---: |
| 512 | 5.889 | 65.775 | 11.168 |
| 1024 | 5.889 | 15.679 | 2.662 |
| 2048 | 5.889 | 7.087 | 1.203 |
| 4096  | 5.889 | 6.118 | **1.039** |
| 8192  | 5.889 | 5.933 | 1.007 |
| 12288 (32*n*)  | 5.889 | 5.883 | 0.999 |

We obtain PPL of 6.118 at `k = 4096`, for comparison for same k, if we leave all MLP layers intact we obtain PPL = 6.022. Here we simplify 6 more layers with decrease in PPL increasing around 0.1.


We can further optimize the M3 approximation by leaving the h.5.mlp.c_fc layer intact as well, since it is one of the least stable ones for M3 approximation ([M3_layerwise_gridsearch](nanogpt\m3_gridsearch_results\M3_approximation.md)). 

THe ppl changes noticeably: PPL 6.118 $\rightarrow$ 6.044 

Recudcing most stable layer (attn.c_proj) k to 2048, makes the model even lighter without a large sacrifice on PPL.

PPL 6.044 $\rightarrow$ 6.119