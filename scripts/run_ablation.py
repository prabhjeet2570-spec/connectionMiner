#!/usr/bin/env python3
"""Ablation experiments: compare xlsx gene sets vs HVG baselines."""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cm_visual.config import default_config, merge_config
from cm_visual.exports import cm_export_type_gene_probabilities
from cm_visual.models import CmResult, PrepData, RawData
from cm_visual.postprocess import cm_build_type_gene_probabilities
from cm_visual.preprocess import cm_preprocess_binary
from cm_visual.solver import cm_solve
from cm_visual.validate import cm_validate
from cm_visual.viz_plotly import run_all_visualizations

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"
H5AD_PATH = DATA_DIR / "Adult.h5ad"
GENE_LIST_DIR = DATA_DIR / "gene_list"
ABLATION_ROOT = OUTPUT_DIR / "connectionMiner_ablation"
NUM_ITER = 20
SEED = 750

EXP_DEFS = [
    {"name": "hvg_3000",     "mode": "hvg",        "n_genes": 3000},
    {"name": "hvg_5000",     "mode": "hvg",        "n_genes": 5000},
    {"name": "tfs_only",     "mode": "xlsx",       "file": "TFs_groups.xlsx",                      "cols": ["gene"]},
    {"name": "adhesion_only","mode": "xlsx",       "file": "cell adhesion molecules_new.xlsx",      "cols": ["Gene"]},
    {"name": "interactome_only","mode": "xlsx",    "file": "Interactome_v3.xlsx",                  "cols": ["Partner 1", "Partner 2"]},
    {"name": "tfs_adhesion", "mode": "union",      "parents": ["tfs_only", "adhesion_only"]},
    {"name": "tfs_interactome","mode": "union",    "parents": ["tfs_only", "interactome_only"]},
    {"name": "adhesion_interactome","mode": "union","parents": ["adhesion_only", "interactome_only"]},
    {"name": "all_three_union","mode": "union",    "parents": ["tfs_only", "adhesion_only", "interactome_only"]},
    {"name": "all_three_hvg3000","mode": "inter_hvg","parents": ["tfs_only", "adhesion_only", "interactome_only"], "n_hvg": 3000},
]


def load_xlsx_gene_sets() -> dict[str, set[str]]:
    """Return {name: set_of_gene_symbols} for each xlsx source."""
    sets = {}

    tf = pd.read_excel(GENE_LIST_DIR / "TFs_groups.xlsx")
    sets["tfs_only"] = set(tf["gene"].dropna().astype(str).str.strip())

    adh = pd.read_excel(GENE_LIST_DIR / "cell adhesion molecules_new.xlsx")
    sets["adhesion_only"] = set(adh["Gene"].dropna().astype(str).str.strip())

    inter = pd.read_excel(GENE_LIST_DIR / "Interactome_v3.xlsx")
    p1 = set(inter["Partner 1"].dropna().astype(str).str.strip())
    p2 = set(inter["Partner 2"].dropna().astype(str).str.strip())
    sets["interactome_only"] = p1 | p2

    return sets


def compute_hvg_indices(adata: ad.AnnData, n_hvg: int) -> np.ndarray:
    """Compute top-N HVG indices via log1p-normalized variance on sparse data."""
    t0 = time.time()
    print(f"  Loading full sparse matrix into memory ...")
    X_sp = adata.X.to_memory()
    n_all = adata.shape[1]
    GCOL_CHUNK = 500

    row_sums = np.asarray(X_sp.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1

    gene_var = np.zeros(n_all, dtype=np.float64)
    print(f"  Computing variance for {n_all:,} genes (sparse, chunks of {GCOL_CHUNK}) ...")
    for gc_start in range(0, n_all, GCOL_CHUNK):
        gc_end = min(gc_start + GCOL_CHUNK, n_all)
        chunk = np.asarray(X_sp[:, gc_start:gc_end].todense(), dtype=np.float32)
        chunk = np.log1p(chunk / row_sums[:, None] * 10_000)
        gene_var[gc_start:gc_end] = chunk.var(axis=0)
        print(f"    {gc_end:>6,} / {n_all:,} genes  ({time.time()-t0:.1f}s)", end="\r")
    print()

    hvg_idx = np.sort(np.argsort(gene_var)[::-1][:n_hvg])
    print(f"  Selected top {n_hvg} HVGs in {time.time()-t0:.1f}s")
    return hvg_idx


def resolve_gene_set(
    exp_def: dict[str, Any],
    xlsx_sets: dict[str, set[str]],
    hvg_5000_idx: np.ndarray | None,
    all_gene_names: np.ndarray,
) -> tuple[list[int], list[str]]:
    """Return (indices into all_gene_names, gene_names) for the experiment."""
    mode = exp_def["mode"]

    if mode == "hvg":
        n = exp_def["n_genes"]
        if n == 3000:
            gene_idx_df = pd.read_csv(OUTPUT_DIR / "gene_index.csv")
            names = gene_idx_df["gene_name"].values.astype(str)
            idx = [int(np.where(all_gene_names == g)[0][0]) for g in names if g in set(all_gene_names)]
            return idx, names.tolist()
        elif n == 5000 and hvg_5000_idx is not None:
            genes = all_gene_names[hvg_5000_idx]
            return hvg_5000_idx.tolist(), genes.tolist()
        else:
            raise ValueError(f"Unknown HVG config: n={n}")

    if mode == "xlsx":
        genes = xlsx_sets[exp_def["name"]]
        idx = [i for i, g in enumerate(all_gene_names) if g in genes]
        names = [all_gene_names[i] for i in idx]
        if not idx:
            raise RuntimeError(f"No genes from {exp_def['name']} found in h5ad")
        print(f"  {exp_def['name']}: {len(genes)} in xlsx, {len(idx)} found in h5ad")
        return idx, names

    if mode == "union":
        all_genes: set[str] = set()
        for p in exp_def["parents"]:
            p_def = next(e for e in EXP_DEFS if e["name"] == p)
            _, p_names = resolve_gene_set(p_def, xlsx_sets, hvg_5000_idx, all_gene_names)
            all_genes |= set(p_names)
        idx = [i for i, g in enumerate(all_gene_names) if g in all_genes]
        names = [all_gene_names[i] for i in idx]
        print(f"  {exp_def['name']}: union of {exp_def['parents']} = {len(idx)} genes")
        return idx, names

    if mode == "inter_hvg":
        all_genes = set()
        for p in exp_def["parents"]:
            p_def = next(e for e in EXP_DEFS if e["name"] == p)
            _, p_names = resolve_gene_set(p_def, xlsx_sets, hvg_5000_idx, all_gene_names)
            all_genes |= set(p_names)
        n_hvg = exp_def.get("n_hvg", 3000)
        if hvg_5000_idx is None:
            raise RuntimeError("hvg_5000_idx required for inter_hvg mode")
        hvg_idx = hvg_5000_idx[:n_hvg]
        hvg_names = set(all_gene_names[hvg_idx])
        inter = all_genes & hvg_names
        idx = [i for i, g in enumerate(all_gene_names) if g in inter]
        names = [all_gene_names[i] for i in idx]
        print(f"  {exp_def['name']}: {len(all_genes)} xlsx ∩ {n_hvg} HVGs = {len(idx)} genes")
        return idx, names

    raise ValueError(f"Unknown mode: {mode}")


def compute_stats(prep: PrepData, cm: CmResult) -> dict[str, Any]:
    """Compute experiment statistics."""
    W = cm.C_mask
    C = cm.C
    C_hat = cm.C_recon
    idx = np.where(W > 0)
    c_obs = C[idx]
    c_pred = C_hat[idx]
    corr = float(np.corrcoef(c_obs, c_pred)[0, 1]) if len(c_obs) > 1 else 0.0
    if np.isnan(corr):
        corr = 0.0
    resid = c_obs - c_pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((c_obs - np.mean(c_obs)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, np.finfo(float).eps)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    if np.isnan(r2):
        r2 = 0.0

    beta_sparsity = float(np.mean(np.abs(cm.beta) < 1e-10))

    epsilon = 1e-30
    P_norm = cm.P / np.maximum(np.sum(cm.P, axis=0, keepdims=True), epsilon)
    P_ent = -np.sum(P_norm * np.log2(np.maximum(P_norm, epsilon)), axis=0)
    mean_P_entropy = float(np.mean(P_ent))
    max_ent = np.log2(cm.P.shape[0]) if cm.P.shape[0] > 1 else 1.0

    return {
        "n_metacells": int(prep.meta["N_metacells"]),
        "n_genes_solver": int(prep.meta["Ng_solver"]),
        "final_total_loss": float(cm.loss[-1]) if len(cm.loss) > 0 else 0.0,
        "final_obj_beta": float(cm.obj_beta[-1]) if len(cm.obj_beta) > 0 else 0.0,
        "final_obj_P_fit": float(cm.obj_P_fit[-1]) if len(cm.obj_P_fit) > 0 else 0.0,
        "final_obj_P_ent": float(cm.obj_P_ent[-1]) if len(cm.obj_P_ent) > 0 else 0.0,
        "pearson_r": corr,
        "r_squared": r2,
        "rmse": rmse,
        "beta_sparsity": beta_sparsity,
        "mean_P_entropy": mean_P_entropy,
        "max_P_entropy": max_ent,
        "elapsed_sec": cm.elapsed_sec,
        "Ng_eff": cm.Ng_eff,
        "is_low_rank": cm.is_low_rank,
    }


def run_single_experiment(
    exp_def: dict[str, Any],
    exp_idx: int,
    adata: ad.AnnData,
    shared: dict[str, Any],
    xlsx_sets: dict[str, set[str]],
    hvg_5000_idx: np.ndarray | None,
    num_iter: int = NUM_ITER,
    backend: str = "auto",
) -> dict[str, Any]:
    """Run a single ablation experiment end-to-end."""
    exp_name = exp_def["name"]
    save_dir = ABLATION_ROOT / f"exp_{exp_idx:02d}_{exp_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"Experiment {exp_idx:02d}/{len(EXP_DEFS)}: {exp_name}")
    print(f"Output: {save_dir}")
    print(f"{'='*70}")

    CHUNK = 500
    all_gene_names = adata.var_names.values.astype(str)
    gene_idx, gene_names = resolve_gene_set(exp_def, xlsx_sets, hvg_5000_idx, all_gene_names)
    gene_idx_sorted = sorted(gene_idx)

    print(f"  Loading {len(gene_idx)} genes from h5ad ...")
    t0 = time.time()

    cols_list = []
    for start in range(0, len(gene_idx_sorted), CHUNK):
        chunk_idx = gene_idx_sorted[start:start + CHUNK]
        X_chunk = adata[:, chunk_idx].X
        if sparse.issparse(X_chunk):
            cols_list.append(X_chunk.toarray())
        else:
            cols_list.append(np.asarray(X_chunk, dtype=float))
    G_cells = np.column_stack(cols_list)

    print(f"  Loaded {G_cells.shape} in {time.time()-t0:.1f}s")

    n_genes_loaded = len(gene_idx)
    custom_gene_idx = list(range(n_genes_loaded))

    cfg = default_config(input_mode="binary")
    cfg_overrides = {
        "seed": SEED,
        "binary": {
            "n_genes_use": n_genes_loaded,
            "custom_gene_idx": custom_gene_idx,
            "min_cells": 1,
        },
        "solver": {
            "num_iter": num_iter,
            "time_limit_per_step": 120,
            "backend": backend,
        },
        "compute_type_gene_probabilities": True,
        "export_type_gene_probabilities": True,
        "smoke_test": {"enabled": False},
        "run_dir": str(save_dir),
    }
    cfg = merge_config(cfg, cfg_overrides)

    raw = RawData(
        G_cells=G_cells.astype(np.float64),
        genes_shared=np.array(gene_names, dtype=str),
        P_constraints_cells=shared["P_constraints_cells"],
        C_counts=shared["C_counts"],
        C_mask=shared["C_mask"],
        umap_xy=shared["umap_xy"],
        raw_cluster_id=shared["raw_cluster_id"],
        meta={
            "Ncells": shared["n_cells"],
            "Ntypes": shared["n_types"],
            "Ng_shared": n_genes_loaded,
            "all_names": shared["type_names"],
        },
    )

    print("  Running cm_preprocess_binary ...")
    prep = cm_preprocess_binary(raw, cfg)

    print("  Running cm_solve ...")
    cm = cm_solve(prep, cfg)

    cm.meta["G_type_prob"] = cm.P @ prep.G_metacell_p

    if cfg.get("compute_type_gene_probabilities", True):
        print("  Building type-gene probabilities ...")
        cm_build_type_gene_probabilities(raw, prep, cm, cfg)

    if cfg.get("export_type_gene_probabilities", True):
        print("  Exporting type-gene probabilities ...")
        try:
            cm_export_type_gene_probabilities(raw, cm, cfg)
        except Exception as exc:
            print(f"  Warning: export failed: {exc}")

    print("  Saving solver outputs ...")
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

    stats = compute_stats(prep, cm)
    stats["n_genes_loaded"] = n_genes_loaded

    with open(save_dir / "run_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    config_saved = {
        "experiment": exp_name,
        "gene_idx": gene_idx,
        "n_genes_loaded": n_genes_loaded,
        "gene_mode": exp_def["mode"],
        "solver_num_iter": NUM_ITER,
        "seed": SEED,
    }
    with open(save_dir / "run_config.json", "w") as f:
        json.dump(config_saved, f, indent=2, default=str)

    print("  Generating visualizations ...")
    meta_dfs = {
        "cell_index": shared["cell_index"],
        "type_index": shared["type_index"],
        "gene_index": pd.DataFrame({"gene_name": gene_names, "col_idx": list(range(len(gene_names)))}),
        "cell_to_metacell": shared["cell_to_metacell_csv"],
        "metacell_index": shared["metacell_index"],
    }
    try:
        run_all_visualizations(raw, prep, cm, cfg, meta_dfs, save_dir)
    except Exception as exc:
        print(f"  Warning: viz failed: {exc}")

    print(f"  Done: {save_dir}")
    return stats


def build_consolidated_viz(all_stats: list[dict[str, Any]], exp_names: list[str], num_iter: int = NUM_ITER) -> None:
    """Build a single HTML dashboard comparing all experiments."""
    save_path = ABLATION_ROOT / "viz_ablation_comparison.html"
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b",
              "#e377c2","#7f7f7f","#bcbd22","#17becf"]

    df = pd.DataFrame(all_stats)
    df["exp_name"] = exp_names

    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            "Final Total Loss", "Connectome Pearson r", "RMSE",
            "Metacell Count", "Beta Sparsity", "Mean P Entropy",
            "Solver Runtime (s)", "Number of Genes", "Ng_eff"
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.1,
    )

    metrics = [
        ("final_total_loss", 0, 0),
        ("pearson_r", 0, 1),
        ("rmse", 0, 2),
        ("n_metacells", 1, 0),
        ("beta_sparsity", 1, 1),
        ("mean_P_entropy", 1, 2),
        ("elapsed_sec", 2, 0),
        ("n_genes_loaded", 2, 1),
        ("Ng_eff", 2, 2),
    ]

    for metric, row, col in metrics:
        values = df[metric].values
        fig.add_trace(
            go.Bar(
                x=exp_names,
                y=values,
                marker_color=colors[:len(exp_names)],
                name=metric,
                text=[f"{v:.4g}" for v in values],
                textposition="outside",
                hovertemplate="%{x}<br>%{y:.6e}<extra></extra>",
                showlegend=False,
            ),
            row=row + 1, col=col + 1,
        )
        fig.update_xaxes(tickangle=45, row=row + 1, col=col + 1)

    fig.update_layout(
        title_text=f"Ablation Experiment Comparison ({len(exp_names)} experiments, {NUM_ITER} iterations each)",
        height=900,
        width=1600,
        barmode="group",
    )

    fig.write_html(save_path, include_plotlyjs="cdn")
    print(f"\nConsolidated viz: {save_path}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Ablation experiments for ConnectionMiner")
    parser.add_argument("--smoke", action="store_true", help="Quick test: 2 experiments, 2 iterations")
    parser.add_argument("--num-iter", type=int, default=None, help="Override iteration count")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "numpy", "torch"],
                        help="Solver backend: auto (GPU if available), numpy, torch")
    args = parser.parse_args()

    num_iter = NUM_ITER if args.num_iter is None else args.num_iter
    if args.smoke:
        print("SMOKE MODE: 2 experiments, 2 iterations")
        num_iter = 2

    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ConnectionMiner — Ablation Experiments")
    print(f"Output root: {ABLATION_ROOT}")
    print(f"Iterations: {num_iter}")
    if args.smoke:
        print("*** SMOKE MODE ***")
    print("=" * 70)

    print("\n--- Loading shared data ---")
    t0 = time.time()
    C_counts = sparse.load_npz(str(OUTPUT_DIR / "C_matrix.npz")).toarray()
    C_mask = np.load(OUTPUT_DIR / "C_mask.npy")
    P_constraints_cells = sparse.load_npz(str(OUTPUT_DIR / "P_constraints_cells.npz"))
    type_index = pd.read_csv(OUTPUT_DIR / "type_index.csv")
    cell_index = pd.read_csv(OUTPUT_DIR / "cell_index.csv")
    cell_to_metacell_csv = pd.read_csv(OUTPUT_DIR / "cell_to_metacell.csv")
    metacell_index = pd.read_csv(OUTPUT_DIR / "metacell_index.csv")
    n_types = C_counts.shape[0]
    n_cells = P_constraints_cells.shape[1]
    type_names = type_index["type"].values.astype(str)
    print(f"  Loaded shared data in {time.time()-t0:.1f}s")

    print("\n--- Loading h5ad ---")
    adata = ad.read_h5ad(str(H5AD_PATH), backed="r")
    all_gene_names = adata.var_names.values.astype(str)

    umap_key = "X_umap"
    if umap_key not in adata.obsm:
        umap_key = "X_tsne"
    umap_xy = np.asarray(adata.obsm[umap_key], dtype=float)
    raw_cluster_id = adata.obs["MultiomeNN"].values.astype(float)
    print(f"  h5ad shape: {adata.shape}, umap: {umap_xy.shape}")

    shared = {
        "C_counts": C_counts,
        "C_mask": C_mask,
        "P_constraints_cells": P_constraints_cells,
        "type_index": type_index,
        "cell_index": cell_index,
        "cell_to_metacell_csv": cell_to_metacell_csv,
        "metacell_index": metacell_index,
        "n_types": n_types,
        "n_cells": n_cells,
        "type_names": type_names,
        "umap_xy": umap_xy,
        "raw_cluster_id": raw_cluster_id,
    }

    print("\n--- Loading xlsx gene sets ---")
    xlsx_sets = load_xlsx_gene_sets()
    for name, genes in xlsx_sets.items():
        in_h5ad = len([g for g in genes if g in set(all_gene_names)])
        print(f"  {name}: {len(genes)} in xlsx, {in_h5ad} in h5ad")

    print("\n--- Computing 5000 HVGs (shared baseline) ---")
    hvg_5000_idx = compute_hvg_indices(adata, 5000)

    exps_to_run = EXP_DEFS[:2] if args.smoke else EXP_DEFS
    all_stats = []
    for idx, exp_def in enumerate(exps_to_run, start=1):
        stats = run_single_experiment(exp_def, idx, adata, shared, xlsx_sets, hvg_5000_idx, num_iter, args.backend)
        stats["exp_idx"] = idx
        all_stats.append(stats)

        del adata
        gc.collect()
        adata = ad.read_h5ad(str(H5AD_PATH), backed="r")

    exp_names = [e["name"] for e in exps_to_run]
    stats_df = pd.DataFrame(all_stats)
    stats_df["exp_name"] = exp_names
    stats_df.to_csv(ABLATION_ROOT / "all_stats.csv", index=False)
    print(f"\nAll stats saved to {ABLATION_ROOT / 'all_stats.csv'}")

    print("\n--- Building consolidated comparison viz ---")
    build_consolidated_viz(all_stats, exp_names, num_iter)

    print(f"\n{'='*70}")
    print(f"All experiments complete. Results in: {ABLATION_ROOT}")
    print(f"Consolidated viz: {ABLATION_ROOT / 'viz_ablation_comparison.html'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
