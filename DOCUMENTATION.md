# ConnectionMiner — FlyWire Visual System: Complete Documentation

**Last updated: 2026-07-20**

---

## 1. Project Overview

ConnectionMiner is a computational method that jointly infers **neuronal cell-type identities** and **synaptic gene interaction programs** from single-cell RNA-seq data, using the measured synaptic connectome as a structural constraint.

This implementation adapts the original method (Gupta et al. 2025, motor system) to the **FlyWire visual system** — the *Drosophila* optic lobe — with **99,656 cells** (after glia removal) spanning **741 visual neuron types** and a **741×741 binary connectome**.

---

## 2. Problem Statement

### Given
- **scRNA-seq**: expression matrix for 99,656 cells × 12,028 genes (10,087 glial cells removed)
- **Connectome**: binary adjacency matrix for 741 visual neuron types (6.3% density: 34,551 edges)

### Infer
1. **P** — Soft cell-type assignment (741 types × metacells)
2. **β** — Gene interaction matrix (genes × genes)

### Core Modeling Assumption

```
C_recon = P @ G @ β @ Gᵀ @ Pᵀ
```

Where:
- **C** (741×741): observed binary connectome
- **P** (741 × n_meta): soft assignment of each metacell to each type (row-stochastic)
- **G** (n_meta × n_genes): metacell-level gene detection probability in [0,1]
- **β** (n_genes × n_genes): non-negative, L1-sparse gene interaction matrix

### Optimization Objective

```
min_{P, β ≥ 0}  ‖W ⊙ (P G β Gᵀ Pᵀ − C)‖²_F  +  λ‖β‖₁  +  ε·H(P)
```

---

## 3. Data Sources

| File | Description | Location |
|------|-------------|----------|
| `Adult.h5ad` | FlyWire scRNA-seq (109,743 cells × 12,028 genes) | `data/` |
| `connections_princeton.csv.gz` | ~84M synapse rows (2GB compressed) | `data/` |
| `visual_neuron_types.csv.gz` | 95,079 neurons mapped to 741 types | `data/` |
| `consolidated_anchor_types.csv` | Root-ID to type mapping | `data/` |
| `TFs_groups.xlsx` | 628 transcription factors | `data/gene_list/` |
| `cell adhesion molecules_new.xlsx` | 322 adhesion molecules | `data/gene_list/` |
| `Interactome_v3.xlsx` | 119 interactome pairs (87 unique) | `data/gene_list/` |

---

## 4. Pipeline Architecture

### Two-Stage Pipeline

```
Stage 1: scripts/build_all_matrices.py
         → Produces all pre-built matrices (C, G, P, metacells, P_constraints, C_mask)
         → Runs independently, outputs to output/

Stage 2a: scripts/run_flywire.py
          → Loads pre-built matrices + h5ad
          → Runs solver (alternating β/P optimization)
          → Generates 7 Plotly visualizations
          → Output: output/connectionMiner_solve/

Stage 2b: scripts/run_ablation.py
          → Runs 10 gene-set ablation experiments
          → Each experiment: preprocess → solve → postprocess → viz
          → Output: output/connectionMiner_ablation/
```

---

## 5. Matrix Construction (`scripts/build_all_matrices.py`)

10 sequential steps that produce every matrix the solver needs:

### Step 0: Glia Removal (added 2026-07-20)
- Before any processing, glial cells are identified by the `G_` prefix in the `MultiomeAnnotated` column
- **10,087 glial cells removed** (9.2% of total): G_Astro (4,749), G_InChi (2,898), G_Prn (589), G_dSat (335), G_Cortex (312), G_PsCrtrg (271), G_Chalise (250), G_SubPrn (243), G_Fenst (157), G_OutChi (107), G_pSat (87), G_Epith (48), G_Marg (41)
- Remaining: **99,656 cells** for all downstream processing
- Applied consistently in `build_all_matrices.py`, `run_visual.py`, `run_ablation.py`, and `build_ablation_explorer.py`

### Step 1: Load Adult.h5ad
- Loads the full h5ad into memory
- Extracts PCA coordinates from `adata.obsm["X_pca"]`

### Step 2: Build C — Binary Connectome (741×741)
- **Input**: `connections_princeton.csv.gz` (streamed in 2M-row chunks)
- Filters to optic lobe neuropils only: `{ME_R, ME_L, LO_R, LO_L, LOP_R, LOP_L, LA_R, LA_L, AME_R, AME_L, MCE_R, MCE_L, ICL_R, ICL_L, OCG}`
- Maps root_ids → type names → type indices using `visual_neuron_types.csv.gz`
- Accumulates edges as sparse COO matrix, binarizes to bool
- **2,245,062 / 5,342,446 synapse rows kept** (42% after neuropil filter)
- **Output**: `C_matrix.npz` (741×741, 34,551 edges = 6.3% dense), `type_index.csv`

### Step 3: Build G_cells — HVG Expression (99,656×3,000, z-scored)
- Computes per-gene variance sparsely:
  - Detects raw counts (max > 20 in sample of 500 rows)
  - Processes genes in 500-column chunks: log1p-normalizes, computes variance
- Selects top 3,000 highly variable genes (HVGs)
- Densifies only the 3,000 HVG columns (avoids ~6.6 GB full densification)
- Z-score normalizes (mean=0, std=1) — for clustering only, NOT the solver
- **Output**: `G_matrix.npy` (99,656×3,000, 1.20 GB), `gene_index.csv`

### Step 4: Build P_cells — Type Constraints (99,656×741, row-stochastic)
- Three-tier system:
  - **Named** (55,749 cells): hard P=1.0 for matching type
  - **Numeric** (27,264 cells): cosine similarity between cluster centroids in HVG space, threshold ≥ 0.70
  - **Orphan** (16,643 cells): uniform over empty types (667 types with no named cells)
- **Output**: `P_matrix.npz` (99,656×741, 11.2M nnz), `cell_index.csv`

### Step 5: Build B_cells — Gene Covariance (3,000×3,000, reference only)
- `GᵀG / n_cells` via chunked accumulation
- **Not used by the real solver** — saved for diagnostics
- **Output**: `B_matrix.npy`

### Step 6: Build Metacells (99,656 → 7,981)
- Proportionally allocated to tiers:
  - Named: 55,749 cells → 4,464 metacells
  - Numeric: 27,264 cells → 2,189 metacells
  - Orphan: 16,643 cells → 1,328 metacells
- Each tier: group by constraint signature → PCA → K-means → merge tiny clusters
- **Output**: `metacell_index.csv`, `cell_to_metacell.csv`

### Step 7: Build P_meta, G_meta — Diagnostic (z-scored)
- Row-stochastic P_meta, z-scored G_meta for diagnostics/visualization only
- NOT fed to solver
- **Output**: `G_meta.npy` (7,981×3,000, 96 MB), `P_meta.npz`

### Step 8: Build G_metacell_p — REAL Solver G ([0,1] detection probability)
- **This is the actual G fed to the solver**
- Per metacell: fraction of cells expressing each gene (binarized > 0)
- Bounded in [0,1] — required for multiplicative NMF β updates
- **Output**: `G_metacell_p.npy` (7,981×3,000)

### Step 9: Build P_constraints — REAL Solver P (TRANSPOSED)
- Solver expects `P_constraints` as `(N_types × n_X)`, NOT `(n_X × N_types)`
- Transpose: `P_constraints_cells = (P_cells > 0).T`
- **Output**: `P_constraints_cells.npz` (741×99,656), `P_constraints_metacell.npz` (741×7,981)

### Step 10: Build C_mask — Observed-Entries Mask (741×741, all-ones)
- For FlyWire: all type pairs are potentially measurable → full mask
- **Output**: `C_mask.npy`

---

## 6. Glia Removal Strategy

### Identification
Glial cells are identified by the `G_` prefix in the `MultiomeAnnotated` column of `Adult.h5ad`:
```python
glia_mask = adata.obs["MultiomeAnnotated"].astype(str).str.startswith("G_")
```

### 13 Glial Types Removed

| Annotation | Count |
|---|---|
| G_Astro | 4,749 |
| G_InChi | 2,898 |
| G_Prn | 589 |
| G_dSat | 335 |
| G_Cortex | 312 |
| G_PsCrtrg | 271 |
| G_Chalise | 250 |
| G_SubPrn | 243 |
| G_Fenst | 157 |
| G_OutChi | 107 |
| G_pSat | 87 |
| G_Epith | 48 |
| G_Marg | 41 |
| **Total** | **10,087** |

### Where Filtering Is Applied

| File | Stage | How |
|------|-------|-----|
| `scripts/build_all_matrices.py` | After step 1 | Subsets AnnData: `adata = adata[~glia_mask]` |
| `cm_visual/run_visual.py` | Step B (h5ad loading) | Masks `G_cells_raw`, `umap_xy`, `raw_cluster_id` |
| `scripts/run_ablation.py` | h5ad loading + per-experiment | Masks `umap_xy`, `raw_cluster_id`, all G_cells chunks |
| `scripts/build_ablation_explorer.py` | Shared data loading | Masks `umap_xy` to match filtered `cell_index` |

### Impact
- Cells: 109,743 → 99,656 (−9.2%)
- Orphan metacells: previously ~1,934 (including glia in orphan tier) → now 1,328
- Named/Numeric tiers: unaffected
- C matrix (741×741): unaffected (type-level, no glia)
- HVG selection: slightly changed (glia expression variance removed)

---

## 7. Cell-Type Constraint Tiers

### Tier 1: Named Cells (55,749 cells, hard constraints)
- `MultiomeAnnotated` matches a known type in `type_index`
- Gets `P[c, t] = 1.0` — hard assignment
- "Gold standard" annotations from FlyWire's Multiome pipeline

### Tier 2: Numeric Cells (27,264 cells, soft cosine-similarity constraints)
- `MultiomeAnnotated` is a numeric string (e.g., "47", "128")
- Centroid in HVG space compared to all named cluster centroids via cosine similarity
- Threshold ≥ 0.70 → uniform assignment to matching named types
- Fallback: top-3 most similar named types

### Tier 3: Orphan Cells (16,643 cells, uniform over empty types)
- Label doesn't match any known type AND isn't numeric
- Uniform probability over all 667 types with zero named cells
- Allows discovery of new type assignments from connectome constraints

---

## 8. The Alternating Solver (`cm_visual/solver.py`)

### β Update (`cm_beta_update`)
- **Objective**: `min_{β ≥ 0} ‖W ⊙ (A β B − C)‖²_F + λ‖β‖₁`
- **Algorithm**: Multiplicative NMF-style updates (coordinate descent in log-space)
- Naturally enforces non-negativity; no step size tuning needed

### P Update (`cm_P_update`)
- **Objective**: `min_P ‖W ⊙ (P Z Gᵀ − C)‖²_F + ε·H(P)` subject to `P ∈ support(D)`
- **Algorithm**: Entropic Sinkhorn with backtracking line search
- Row pass + column pass per iteration
- Exponentiated gradients with row/column normalization

### Backend
- Auto-detection: `torch` if GPU available, else `numpy`
- FlyWire run: NumPy backend (no CUDA available)

---

## 9. Visualization Suite (`cm_visual/viz_plotly.py`)

7 interactive Plotly HTML dashboards per experiment:

| # | Name | Description |
|---|------|-------------|
| 1 | Raw MultiomeNN Clusters | UMAP colored by cluster ID (Viridis) |
| 2 | Cell Type Constraints | Named (blue) / Numeric (orange) / Orphan (grey) |
| 3 | Metacells | Two panels: metacell ID + metacell size |
| 4 | Inferred Type Assignments | argmax type + normalized entropy (RdYlGn_r) |
| 5 | Connectome Fit | True C vs reconstructed C_hat (scatter + heatmap) |
| 6 | Loss Trajectory | β loss, P fit, P entropy, total loss over iterations |
| 7 | Combined Three-Panel | Original annotations + Metacells + Inferred Type (synchronized) |

Key features:
- All use `Scattergl` (WebGL) for 100k-point performance
- Cross-panel highlighting on hover (JS-injected post-hoc)
- Synchronized zoom/pan across subplots
- Consolidated info panel (15 fields per cell)
- 30-color discrete palette (D30)

---

## 10. Ablation Experiments (`scripts/run_ablation.py`)

### Gene Sets Compared

| # | Name | Genes | Source |
|---|------|-------|--------|
| 01 | hvg_3000 | 3,000 | Top HVGs by variance (**baseline**) |
| 02 | hvg_5000 | 5,000 | Top HVGs by variance |
| 03 | tfs_only | 521 | Transcription factors |
| 04 | adhesion_only | 278 | Cell adhesion molecules |
| 05 | interactome_only | 87 | Known PPI pairs |
| 06 | tfs_adhesion | 799 | Union of TFs + adhesion |
| 07 | tfs_interactome | 608 | Union of TFs + interactome |
| 08 | adhesion_interactome | 287 | Union of adhesion + interactome |
| 09 | all_three_union | 808 | Union of all three curated sets |
| 10 | all_three_hvg3000 | 212 | Intersection: curated ∩ 3000 HVGs |

### Key Results

| Experiment | Pearson r | Final Loss | Genes |
|---|---|---|---|
| **tfs_adhesion** | **0.587** | **21,355** | **799** |
| all_three_union | 0.574 | 21,840 | 808 |
| tfs_interactome | 0.568 | 22,072 | 608 |
| tfs_only | 0.546 | 22,852 | 521 |
| hvg_5000 | 0.493 | 24,659 | 5,000 |
| hvg_3000 (baseline) | 0.474 | 25,372 | 3,000 |

**Finding**: Curated gene sets (TF + adhesion molecules) dramatically outperform HVGs — **24% relative improvement** in Pearson r with **3.8× fewer genes**.

---

## 11. Repository Layout

```
connectionMiner/
├── cm_visual/                      # Core pipeline Python package
│   ├── __init__.py                 # Exports cm_run_visual
│   ├── run_visual.py               # Pipeline entrypoint orchestrating steps A–I
│   ├── config.py                   # Default config (HVG count, solver params)
│   ├── models.py                   # RawData, PrepData, CmResult dataclasses
│   ├── preprocess.py               # Binary preprocessing + metacell construction
│   ├── solver.py                   # Alternating β/P optimization
│   ├── postprocess.py              # Type×gene probabilities + identifiability scoring
│   ├── exports.py                  # Excel export of type×gene probabilities
│   ├── viz_plotly.py               # 7 interactive Plotly HTML dashboards
│   ├── validate.py                 # Shape/value invariant checks
│   └── utils.py                    # JSON/MAT export, z-score, helpers
├── scripts/                        # CLI entry points
│   ├── build_all_matrices.py       # Matrix construction (10 steps + glia removal)
│   ├── run_flywire.py              # Main pipeline CLI
│   ├── run_ablation.py             # Ablation experiment orchestrator (10 experiments)
│   └── build_ablation_explorer.py  # Interactive ablation explorer HTML
├── data/                           # Input data (large files gitignored)
│   ├── Adult.h5ad                  # scRNA-seq (109,743 cells × 12,028 genes)
│   ├── connections_princeton.csv.gz# Synapse-level connectome
│   ├── visual_neuron_types.csv.gz  # Neuron type metadata (95,079 neurons)
│   ├── consolidated_anchor_types.csv # Root-ID to type mapping
│   └── gene_list/                  # Curated gene lists for ablation
├── output/                         # Pre-built solver matrices
│   ├── C_matrix.npz                # 741×741 binary connectome
│   ├── G_matrix.npy                # 99,656×3,000 z-scored HVG expression
│   ├── G_metacell_p.npy            # 7,981×3,000 [0,1] detection probability
│   ├── P_matrix.npz                # 99,656×741 row-stochastic constraints
│   ├── P_constraints_cells.npz     # 741×99,656 binary support masks
│   ├── P_constraints_metacell.npz  # 741×7,981 binary support masks
│   ├── C_mask.npy                  # 741×741 (all-ones)
│   ├── cell_index.csv              # Per-cell metadata
│   ├── type_index.csv              # 741 type metadata
│   ├── gene_index.csv              # 3,000 HVG metadata
│   ├── metacell_index.csv          # 7,981 metacell metadata
│   ├── cell_to_metacell.csv        # Cell → metacell mapping
│   ├── connectionMiner_solve/      # Solver outputs + visualizations
│   └── connectionMiner_ablation/   # Ablation experiment outputs
├── DOCUMENTATION.md                # This file
├── README.md                       # Quick-start guide
├── plan.md                         # Implementation status
├── experiments.md                  # Ablation experiment details
├── plotly_interactive_features.md  # Plotly JS injection docs
├── PPT_EXPLANATION.md              # Detailed project explanation
└── requirements.txt                # Python dependencies
```

---

## 12. How to Run

### Step 0: Setup
```bash
pip install -r requirements.txt
pip install anndata plotly
```

### Step 1: Build Matrices (with glia removed)
```bash
cd scripts
python3 build_all_matrices.py
```
Runtime: ~30-45 minutes. Removes 10,087 glial cells automatically.

### Step 2: Run Main Pipeline
```bash
# Smoke test (2 iterations)
python3 run_flywire.py --num-iter 2 --smoke

# Full run (100 iterations)
python3 run_flywire.py --num-iter 100 --lambda-sparsity 0.001
```

### Step 3: Run Ablation Experiments
```bash
# All 10 experiments, 20 iterations each
python3 run_ablation.py --num-iter 20

# Quick test
python3 run_ablation.py --smoke
```

### CLI Flags (run_flywire.py)
| Flag | Default | Description |
|------|---------|-------------|
| `--num-iter` | 100 | Solver iterations |
| `--lambda-sparsity` | 0.001 | L1 penalty on β |
| `--beta-rank` | 0 | Low-rank β dimension |
| `--smoke` | off | Fast test (2 iters) |
| `--seed` | 750 | Random seed |

---

## 13. Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Glia filtering | `G_` prefix in MultiomeAnnotated | Removes 10,087 non-neuronal cells |
| HVG count | 3,000 | Computational efficiency; curated sets beat both |
| Expression binarization | Binary (count > 0) | Required for NMF-compatible [0,1] G |
| Connectome representation | Binary (not weighted) | Simpler reconstruction target |
| Metacell target | ~8,000 | Balance between granularity and tractability |
| Metacell construction | Tiered PCA+K-means | Named/numeric/orphan tiers reflect annotation confidence |
| P constraints orientation | Transposed (types × cells) | Matches solver expectation |
| Solver iterations | 100 (main), 20 (ablation) | Convergence by ~50 |
| Best gene set | TF + adhesion molecules (799 genes) | r=0.587, 24% improvement over HVGs |
