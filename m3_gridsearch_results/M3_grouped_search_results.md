# M3 Grouped Search Results

Baseline PPL: **5.8894**
Acceptable threshold: **PPL <= 6.5000**

Group meanings:
- `attn.c_attn` = all 6 attention input projections
- `attn.c_proj` = all 6 attention output projections
- `mlp.c_fc early` = blocks 0-3
- `mlp.c_fc b4` = block 4
- `mlp.c_fc b5` = block 5
- `mlp.c_proj b0` = block 0
- `mlp.c_proj b1` = block 1
- `mlp.c_proj b2` = block 2
- `mlp.c_proj b3` = block 3
- `mlp.c_proj b4` = block 4
- `mlp.c_proj b5` = block 5

| ID | attn.c_attn | attn.c_proj | mlp.c_fc early | mlp.c_fc b4 | mlp.c_fc b5 | mlp.c_proj b0 | mlp.c_proj b1 | mlp.c_proj b2 | mlp.c_proj b3 | mlp.c_proj b4 | mlp.c_proj b5 | PPL | Acceptable | Proxy Cost | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| B0 | 2048 | 2048 | 4096 | 4096 | none | none | none | none | none | none | none | 6.3264 | yes | 45056 | Current best backbone: no mlp.c_proj approximated. |
| B1 | 4096 | 2048 | 4096 | 4096 | none | none | none | none | none | none | none | 6.1515 | yes | 57344 | Raise attn.c_attn to 4096. |
| B2 | 2048 | 4096 | 4096 | 4096 | none | none | none | none | none | none | none | 6.2144 | yes | 57344 | Raise attn.c_proj to 4096. |
| B3 | 4096 | 4096 | 4096 | 4096 | none | none | none | none | none | none | none | 6.0117 | yes | 69632 | Raise both attention groups to 4096. |
| B4 | 2048 | 2048 | 2048 | 4096 | none | none | none | none | none | none | none | 6.5664 | no | 36864 | Lower mlp.c_fc early to 2048. |
| B5 | 2048 | 2048 | 4096 | 2048 | none | none | none | none | none | none | none | 6.4281 | yes | 43008 | Lower mlp.c_fc b4 to 2048. |
| B6 | 2048 | 2048 | 4096 | 4096 | 4096 | none | none | none | none | none | none | 6.4344 | yes | 49152 | Turn on mlp.c_fc b5 at 4096. |
| S1_2048 | 2048 | 2048 | 4096 | 4096 | none | none | 2048 | none | none | none | none | 6.5156 | no | 47104 | Single block trial: h1.mlp.c_proj = 2048. |
| S1_4096 | 2048 | 2048 | 4096 | 4096 | none | none | 4096 | none | none | none | none | 6.4490 | yes | 49152 | Single block trial: h1.mlp.c_proj = 4096. |
| S1_8192 | 2048 | 2048 | 4096 | 4096 | none | none | 8192 | none | none | none | none | 6.3592 | yes | 53248 | Single block trial: h1.mlp.c_proj = 8192. |
| S3_2048 | 2048 | 2048 | 4096 | 4096 | none | none | none | none | 2048 | none | none | 6.7213 | no | 47104 | Single block trial: h3.mlp.c_proj = 2048. |
| S3_4096 | 2048 | 2048 | 4096 | 4096 | none | none | none | none | 4096 | none | none | 6.5686 | no | 49152 | Single block trial: h3.mlp.c_proj = 4096. |
| S3_8192 | 2048 | 2048 | 4096 | 4096 | none | none | none | none | 8192 | none | none | 6.3847 | yes | 53248 | Single block trial: h3.mlp.c_proj = 8192. |
| S0_2048 | 2048 | 2048 | 4096 | 4096 | none | 2048 | none | none | none | none | none | 7.3045 | no | 47104 | Single block trial: h0.mlp.c_proj = 2048. |
| S0_4096 | 2048 | 2048 | 4096 | 4096 | none | 4096 | none | none | none | none | none | 6.8189 | no | 49152 | Single block trial: h0.mlp.c_proj = 4096. |
| S0_8192 | 2048 | 2048 | 4096 | 4096 | none | 8192 | none | none | none | none | none | 6.4844 | yes | 53248 | Single block trial: h0.mlp.c_proj = 8192. |
| S2_2048 | 2048 | 2048 | 4096 | 4096 | none | none | none | 2048 | none | none | none | 6.5951 | no | 47104 | Single block trial: h2.mlp.c_proj = 2048. |
| S2_4096 | 2048 | 2048 | 4096 | 4096 | none | none | none | 4096 | none | none | none | 6.4847 | yes | 49152 | Single block trial: h2.mlp.c_proj = 4096. |
| S2_8192 | 2048 | 2048 | 4096 | 4096 | none | none | none | 8192 | none | none | none | 6.4026 | yes | 53248 | Single block trial: h2.mlp.c_proj = 8192. |
| P13_2048 | 2048 | 2048 | 4096 | 4096 | none | none | 2048 | none | 2048 | none | none | 6.9896 | no | 49152 | Pair trial: h1+h3.mlp.c_proj at 2048. |
| P13_4096 | 2048 | 2048 | 4096 | 4096 | none | none | 4096 | none | 4096 | none | none | 6.6091 | no | 53248 | Pair trial: h1+h3.mlp.c_proj at 4096. |
| P13_8192 | 2048 | 2048 | 4096 | 4096 | none | none | 8192 | none | 8192 | none | none | 6.4149 | yes | 61440 | Pair trial: h1+h3.mlp.c_proj at 8192. |
| P13_mixA | 2048 | 2048 | 4096 | 4096 | none | none | 4096 | none | 8192 | none | none | 6.4924 | yes | 57344 | Pair trial: h1=4096, h3=8192. |
| P13_mixB | 2048 | 2048 | 4096 | 4096 | none | none | 8192 | none | 4096 | none | none | 6.5114 | no | 57344 | Pair trial: h1=8192, h3=4096. |
| Q130_2048 | 2048 | 2048 | 4096 | 4096 | none | 2048 | 4096 | none | 4096 | none | none | 8.0322 | no | 55296 | Add h0 on top of h1+h3 at 4096. |
| Q130_4096 | 2048 | 2048 | 4096 | 4096 | none | 4096 | 4096 | none | 4096 | none | none | 7.3303 | no | 57344 | Add h0 on top of h1+h3 at 4096. |
| Q132_2048 | 2048 | 2048 | 4096 | 4096 | none | none | 4096 | 2048 | 4096 | none | none | 7.0010 | no | 55296 | Add h2 on top of h1+h3 at 4096. |
| Q132_4096 | 2048 | 2048 | 4096 | 4096 | none | none | 4096 | 4096 | 4096 | none | none | 6.7270 | no | 57344 | Add h2 on top of h1+h3 at 4096. |
| Q1302_2048 | 2048 | 2048 | 4096 | 4096 | none | 2048 | 4096 | 2048 | 4096 | none | none | 8.6611 | no | 57344 | Add h0+h2 at 2048 on top of h1+h3 at 4096. |
| Q1302_4096 | 2048 | 2048 | 4096 | 4096 | none | 4096 | 4096 | 4096 | 4096 | none | none | 7.4985 | no | 61440 | Add h0+h2 at 4096 on top of h1+h3 at 4096. |
| Q13hi_2048 | 2048 | 2048 | 4096 | 4096 | none | 2048 | 8192 | 2048 | 8192 | none | none | 8.2495 | no | 65536 | Stronger pair h1+h3 at 8192, add h0+h2 at 2048. |
| Q13hi_4096 | 2048 | 2048 | 4096 | 4096 | none | 4096 | 8192 | 4096 | 8192 | none | none | 7.3396 | no | 69632 | Stronger pair h1+h3 at 8192, add h0+h2 at 4096. |
| T4_4096 | 2048 | 2048 | 4096 | 4096 | none | none | none | none | none | 4096 | none | 6.5783 | no | 49152 | Turn on h4.mlp.c_proj only at 4096 on top of base. |
| T4_8192 | 2048 | 2048 | 4096 | 4096 | none | none | none | none | none | 8192 | none | 6.4444 | yes | 53248 | Turn on h4.mlp.c_proj only at 8192 on top of base. |
| T5_8192 | 2048 | 2048 | 4096 | 4096 | none | none | none | none | none | none | 8192 | 6.4497 | yes | 53248 | Turn on h5.mlp.c_proj only at 8192 on top of base. |
| T13_4 | 2048 | 2048 | 4096 | 4096 | none | none | 4096 | none | 4096 | 4096 | none | 6.9617 | no | 57344 | Best early pair h1+h3 at 4096 plus h4 at 4096. |
| T13_5 | 2048 | 2048 | 4096 | 4096 | none | none | 4096 | none | 4096 | none | 8192 | 6.6916 | no | 61440 | Best early pair h1+h3 at 4096 plus h5 at 8192. |
| T13_45 | 2048 | 2048 | 4096 | 4096 | none | none | 4096 | none | 4096 | 4096 | 8192 | 7.0495 | no | 65536 | Best early pair h1+h3 at 4096 plus h4=4096 and h5=8192. |
| Ufull_lo | 2048 | 2048 | 4096 | 4096 | none | 2048 | 4096 | 2048 | 4096 | none | none | 8.6611 | no | 57344 | All early mlp.c_proj on: h1,h3=4096 and h0,h2=2048. |
| Ufull_mid | 2048 | 2048 | 4096 | 4096 | none | 4096 | 4096 | 4096 | 4096 | none | none | 7.4985 | no | 61440 | All early mlp.c_proj on: h0-h3 all at 4096. |
| Ufull_hi | 2048 | 2048 | 4096 | 4096 | none | 4096 | 8192 | 4096 | 8192 | none | none | 7.3396 | no | 69632 | All early mlp.c_proj on: h1,h3=8192 and h0,h2=4096. |
| V1 | 2048 | 2048 | 4096 | 4096 | none | 2048 | 4096 | 2048 | 4096 | 4096 | none | 9.4031 | no | 61440 | All early mlp.c_proj on plus h4 at 4096. |
| V2 | 2048 | 2048 | 4096 | 4096 | none | 2048 | 4096 | 2048 | 4096 | none | 8192 | 8.8791 | no | 65536 | All early mlp.c_proj on plus h5 at 8192. |
| V3 | 2048 | 2048 | 4096 | 4096 | none | 2048 | 4096 | 2048 | 4096 | 4096 | 8192 | 9.6567 | no | 69632 | All early mlp.c_proj on plus h4=4096 and h5=8192. |
