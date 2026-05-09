# M3 Grouped Search Plan

This plan reflects the current search direction:

- keep the best known non-`mlp.c_proj` backbone as the starting point
- stop using `k = 1024`
- focus mainly on `k = 2048, 4096, 8192`
- spend most of the search budget on the earlier `mlp.c_proj` layers

Acceptable threshold:

- `PPL <= 6.5`

## Current Starting Point

The search now starts from this base configuration:

- `attn.c_attn = 2048`
- `attn.c_proj = 2048`
- `mlp.c_fc early = 4096`
- `mlp.c_fc b4 = 4096`
- `mlp.c_fc b5 = none`
- all `mlp.c_proj = none`

Reason:

- last time `mlp.c_proj` was not optimized at all
- `mlp.c_fc` in block 4 already looked usable at `4096`
- the next most informative step is to turn on the earlier `mlp.c_proj` blocks gradually

## Groups Used

Shared groups:

1. `attn.c_attn` = all 6 blocks
2. `attn.c_proj` = all 6 blocks
3. `mlp.c_fc early` = blocks `0-3`
4. `mlp.c_fc b4` = block `4`
5. `mlp.c_fc b5` = block `5`

Per-block `mlp.c_proj` groups:

6. `mlp.c_proj b0`
7. `mlp.c_proj b1`
8. `mlp.c_proj b2`
9. `mlp.c_proj b3`
10. `mlp.c_proj b4`
11. `mlp.c_proj b5`

Why this split:

- `mlp.c_proj` is the most sensitive part
- earlier `mlp.c_proj` layers are the main optimization target now
- block 4 and block 5 should be kept separate because the tail behaves differently

## Search Logic

The search is organized into stages.

### Stage 1: Backbone checks

Purpose:
- confirm or slightly improve the current non-`mlp.c_proj` backbone

Examples:
- raise/lower attention groups
- lower `mlp.c_fc early`
- lower `mlp.c_fc b4`
- optionally turn on `mlp.c_fc b5`

### Stage 2: Single-block `mlp.c_proj`

Purpose:
- test earlier `mlp.c_proj` blocks individually with:
  - `2048`
  - `4096`
  - `8192`

Priority order:

1. `b1`
2. `b3`
3. `b0`
4. `b2`
5. `b4`
6. `b5`

This order comes from the layerwise sensitivity table.

### Stage 3: Pair / small-combination search

Purpose:
- combine the most promising earlier `mlp.c_proj` blocks

Main pair:

- `b1 + b3`

Then add:

- `b0`
- `b2`

### Stage 4: Tail additions

Purpose:
- only after the earlier `mlp.c_proj` layers look promising, test:
  - `b4`
  - `b5`

These should remain late-stage additions.

## Implemented Shortlist

The Python script implements a concrete shortlist with roughly 40 runs:

### Backbone

- `B0-B6`

### Single-block `mlp.c_proj`

- `S1_*`
- `S3_*`
- `S0_*`
- `S2_*`

### Pair / combination search

- `P13_*`
- `Q*`

### Tail tests

- `T*`
- `U*`
- `V*`

## Interpretation Rule

If a single-block `mlp.c_proj` setting is already bad on its own:

- do not trust larger combinations involving it

If `b1` and `b3` work well:

- use them as the main early pair
- only then add `b0` and `b2`

If `b4` or `b5` hurt too much:

- keep them exact

## Objective

The goal is not just the lowest PPL.

Better objective:

1. keep `PPL <= 6.5`
2. among acceptable configs, prefer lower approximation cost

Current proxy cost:

```text
sum(k over all approximated groups)
```

This is only a rough ranking heuristic, but good enough for the current search.
