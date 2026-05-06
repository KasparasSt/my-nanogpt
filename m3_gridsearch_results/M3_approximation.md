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

