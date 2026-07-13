from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .config import default_config, ensure_run_dir
from .exports import cm_export_type_gene_probabilities
from .models import CmResult, PrepData, RawData
from .postprocess import cm_build_type_gene_probabilities
from .preprocess import cm_preprocess_binary
from .solver import cm_solve
from .validate import cm_validate
from .viz_plotly import run_all_visualizations


def cm_run_visual(
    output_dir: str | Path = "output",
    h5ad_path: str | Path = "output/Adult.h5ad",
    cfg_overrides: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    h5ad_path = Path(h5ad_path)

    cfg = default_config(input_mode="binary")
    if cfg_overrides:
        from .config import merge_config
        cfg = merge_config(cfg, cfg_overrides)

    # STEP A — Load pre-built matrices
    print("=== STEP A: Loading pre-built matrices ===")
    C_counts = sparse.load_npz(str(output_dir / "C_matrix.npz")).toarray()
    C_mask = np.load(output_dir / "C_mask.npy")
    P_constraints_cells = sparse.load_npz(str(output_dir / "P_constraints_cells.npz"))

    type_index = pd.read_csv(output_dir / "type_index.csv")
    cell_index = pd.read_csv(output_dir / "cell_index.csv")
    gene_index = pd.read_csv(output_dir / "gene_index.csv")
    cell_to_metacell_csv = pd.read_csv(output_dir / "cell_to_metacell.csv")
    metacell_index = pd.read_csv(output_dir / "metacell_index.csv")

    n_types = C_counts.shape[0]
    n_cells = P_constraints_cells.shape[1]

    # STEP B — Load Adult.h5ad
    print("=== STEP B: Loading h5ad ===")
    adata = ad.read_h5ad(str(h5ad_path), backed="r")

    hvg_names = gene_index["gene_name"].values.astype(str)
    hvg_idx = np.where(np.isin(adata.var_names.values, hvg_names))[0]

    if hvg_idx.size == 0:
        hvg_idx = gene_index["col_idx"].values.astype(int)
        hvg_idx = hvg_idx[hvg_idx < adata.shape[1]]

    print(f"  Found {len(hvg_idx)} of {len(hvg_names)} HVGs in h5ad")

    G_cells_raw = adata[:, hvg_idx].X
    if sparse.issparse(G_cells_raw):
        G_cells_raw = G_cells_raw.toarray()
    else:
        G_cells_raw = np.asarray(G_cells_raw, dtype=float)

    umap_key = "X_umap"
    if umap_key not in adata.obsm_keys():
        umap_key = "X_tsne"
    umap_xy = np.asarray(adata.obsm[umap_key], dtype=float)

    raw_cluster_id = adata.obs["MultiomeNN"].values.astype(float)

    genes_shared = adata.var_names.values[hvg_idx].astype(str)

    # STEP C — Build RawData
    print("=== STEP C: Building RawData ===")
    raw = RawData(
        G_cells=G_cells_raw,
        genes_shared=genes_shared,
        P_constraints_cells=P_constraints_cells,
        C_counts=C_counts,
        C_mask=C_mask,
        umap_xy=umap_xy,
        raw_cluster_id=raw_cluster_id,
        meta={
            "Ncells": n_cells,
            "Ntypes": n_types,
            "Ng_shared": len(genes_shared),
            "all_names": type_index["type"].values,
        },
    )

    cm_validate(raw, cfg, "raw")

    run_dir = ensure_run_dir(cfg, run_tag_prefix="flywire")
    print(f"  Run directory: {run_dir}")

    # STEP D — Preprocess (binary mode)
    print("=== STEP D: cm_preprocess_binary ===")
    prep = cm_preprocess_binary(raw, cfg)
    cm_validate(prep, cfg, "prep")

    # STEP E — Solve
    print("=== STEP E: cm_solve ===")
    cm = cm_solve(prep, cfg)
    cm_validate(cm, cfg, "cm")

    # compute G_type_prob for export
    cm.meta["G_type_prob"] = cm.P @ prep.G_metacell_p

    # STEP F — Type gene probabilities
    if cfg.get("compute_type_gene_probabilities", True):
        print("=== STEP F: cm_build_type_gene_probabilities ===")
        cm_build_type_gene_probabilities(raw, prep, cm, cfg)

    # STEP G — Export
    if cfg.get("export_type_gene_probabilities", True):
        print("=== STEP G: Exporting type_gene_probabilities ===")
        try:
            cm_export_type_gene_probabilities(raw, cm, cfg)
        except Exception as exc:
            print(f"Warning: export failed: {exc}")

    # STEP H — Save solver outputs
    print("=== STEP H: Saving solver outputs ===")
    run_dir = Path(cfg["run_dir"])
    save_dir = output_dir / "connectionMiner_solve"
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / "beta_learned.npy", cm.beta)
    sparse.save_npz(save_dir / "P_refined.npz", sparse.csr_matrix(cm.P))
    np.save(save_dir / "C_reconstructed.npy", cm.C_recon)
    np.save(save_dir / "cell_to_metacell_solver.npy", prep.cell_to_metacell)

    loss_df = pd.DataFrame({
        "iteration": np.arange(1, len(cm.loss) + 1),
        "obj_beta": cm.obj_beta,
        "obj_P_fit": cm.obj_P_fit,
        "obj_P_ent": cm.obj_P_ent,
        "total_loss": cm.loss,
    })
    loss_df.to_csv(save_dir / "solver_loss.csv", index=False)

    # Also copy the Excel export and solver objectives here for convenience
    for fn in ["type_gene_probabilities.xlsx", "solver_objectives.txt"]:
        src = run_dir / fn
        if src.exists():
            shutil.copy2(src, save_dir / fn)

    print(f"  Saved solver outputs to {save_dir}")

    # STEP I — Visualizations
    print("=== STEP I: Visualizations ===")
    meta_dfs = {
        "cell_index": cell_index,
        "type_index": type_index,
        "gene_index": gene_index,
        "cell_to_metacell": cell_to_metacell_csv,
        "metacell_index": metacell_index,
    }
    run_all_visualizations(raw, prep, cm, cfg, meta_dfs, save_dir)

    print(f"=== Pipeline complete: {save_dir} ===")
    return save_dir
