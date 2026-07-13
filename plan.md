# Plan — ConnectionMiner on FlyWire

_Last updated: 2026-07-12_

## Status

`scripts/build_all_matrices.py` completed successfully. All pre-built solver
matrices are in `output/`.

| Metric | Value |
|---|---|
| Cells | 109,743 |
| Types | 741 |
| Connectome density | 34,551 / 549,081 pairs (6.3%) |
| HVGs | 3,000 |
| Metacells | 7,979 (4,059 named / 1,986 numeric / 1,934 orphan) |
| Output dir | `output/` |

The solver has **not been run yet**.

## Confirmed facts

- `cm_visual` solver (`cm_visual.solver.cm_solve(prep, cfg)`) is ready in
  `connectionMiner/cm_visual/` (fork `aravindan2/connectionMiner`).
- Pre-built matrices in `output/` match the solver's expected input format
  (non-negative G, transposed P constraints).
- Dependencies installed: `numpy`, `scipy`, `pandas`, `scikit-learn`,
  `matplotlib`, `openpyxl`, `plotly=5.24.1`, `anndata=0.12.16`.
  `kaleido` is NOT needed (viz uses `write_html` only).

## Pipeline

```
1. scripts/build_all_matrices.py   → builds matrices to output/
2. scripts/run_flywire.py          → loads matrices from output/, runs solver,
                                     generates visualizations
```

## Next steps

1. Run smoke test: `python3 scripts/run_flywire.py --num-iter 2 --smoke`
2. Run full: `python3 scripts/run_flywire.py --num-iter 100 --lambda-sparsity 0.001`
3. Optionally run ablation: `python3 scripts/run_ablation.py --num-iter 20`

## Notes

- HVG count: 3,000 (repo default 4,000) — deliberate FlyWire choice
- Metacells: tiered named/numeric/orphan, 8k target (not repo's ~10/cell)
