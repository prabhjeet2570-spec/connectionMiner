# Metacell Clustering Ablation — Implementation Plan

## Experiments: 2 gene sets × 6 clustering methods = 12

### Gene set (G)
- `hvg_3000` — top 3,000 highly-variable genes
- `tfs_adhesion` — 799 TFs + adhesion molecules (union)

### Clustering method (P)
- `kmeans` (baseline)
- `minibatch_kmeans`
- `ward` (agglomerative)
- `gmm` (Gaussian mixture)
- `hdbscan` (density-based)
- `spectral` (sklearn SpectralClustering, affinity="nearest_neighbors")

## Changes

### 1. `cm_visual/preprocess.py`
- Add `_get_clusterer(n_clusters, cfg, seed)` factory returning a `.fit_predict()`-compatible object
- Replace hardcoded `KMeans(...)` in `_build_metacells_from_features` with `_get_clusterer(...)`

### 2. `cm_visual/config.py`
- Add `"cluster_method": "kmeans"` to the `metacell` config dict

### 3. `scripts/run_metacell_ablation.py`
- Loop over 2 gene sets × 5 methods
- Same structure as `run_ablation.py` (resolve genes, build RawData, call cm_preprocess_binary, call cm_solve, save outputs, compute stats)
- Each experiment overrides `cfg["metacell"]["cluster_method"]`

### 4. `scripts/build_clustering_explorer.py`
- Parallel to `build_ablation_explorer.py`, reads from `clustering_ablation/` directory

## Output

```
output/connectionMiner_ablation/
└── clustering_ablation/
    ├── exp_01_hvg_3000_kmeans/
    ├── exp_02_hvg_3000_minibatch_kmeans/
    ├── exp_03_hvg_3000_ward/
    ├── exp_04_hvg_3000_gmm/
    ├── exp_05_hvg_3000_hdbscan/
    ├── exp_06_tfs_adhesion_kmeans/
    ├── exp_07_tfs_adhesion_minibatch_kmeans/
    ├── exp_08_tfs_adhesion_ward/
    ├── exp_09_tfs_adhesion_gmm/
    ├── exp_10_tfs_adhesion_hdbscan/
    ├── exp_11_hvg_3000_spectral/
    ├── exp_12_tfs_adhesion_spectral/
    ├── all_stats.csv
    ├── viz_ablation_comparison.html
    └── viz_ablation_explorer.html
```

Each experiment subdir: `beta_learned.npy`, `P_refined.npz`, `C_reconstructed.npy`, `cell_to_metacell_solver.npy`, `solver_loss.csv`, `run_stats.json`, `run_config.json`, visualizations.
