#!/usr/bin/env python3
"""Metacell clustering ablation: 2 gene sets x 5 clustering methods = 10 experiments."""

from __future__ import annotations

import gc
import json
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
from cm_visual.viz_plotly import run_all_visualizations

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"
H5AD_PATH = DATA_DIR / "Adult.h5ad"
GENE_LIST_DIR = DATA_DIR / "gene_list"
ABLATION_ROOT = OUTPUT_DIR / "connectionMiner_ablation" / "clustering_ablation"
NUM_ITER = 20
SEED = 750

GENE_SETS = [
    ("hvg_3000", "hvg"),
    ("tfs_adhesion", "union"),
]

CLUSTER_METHODS = [
    "kmeans",
    "minibatch_kmeans",
    "ward",
    "gmm",
    "hdbscan",
    "spectral",
]


def load_xlsx_gene_sets() -> dict[str, set[str]]:
    sets = {}
    tf = pd.read_excel(GENE_LIST_DIR / "TFs_groups.xlsx")
    sets["tfs_only"] = set(tf["gene"].dropna().astype(str).str.strip())
    adh = pd.read_excel(GENE_LIST_DIR / "cell adhesion molecules_new.xlsx")
    sets["adhesion_only"] = set(adh["Gene"].dropna().astype(str).str.strip())
    return sets


def resolve_gene_set(
    gene_set_name: str,
    gene_set_mode: str,
    all_gene_names: np.ndarray,
    xlsx_sets: dict[str, set[str]],
) -> tuple[list[int], list[str]]:
    if gene_set_mode == "hvg":
        gene_idx_df = pd.read_csv(OUTPUT_DIR / "gene_index.csv")
        names = gene_idx_df["gene_name"].values.astype(str)
        idx = [int(np.where(all_gene_names == g)[0][0]) for g in names if g in set(all_gene_names)]
        return idx, names.tolist()

    if gene_set_mode == "union":
        tf_genes = xlsx_sets["tfs_only"]
        adh_genes = xlsx_sets["adhesion_only"]
        all_genes = tf_genes | adh_genes
        idx = [i for i, g in enumerate(all_gene_names) if g in all_genes]
        names = [all_gene_names[i] for i in idx]
        print(f"  tfs_adhesion: {len(tf_genes)} TFs + {len(adh_genes)} adhesion = {len(all_genes)} total, {len(idx)} found in h5ad")
        return idx, names

    raise ValueError(f"Unknown gene_set_mode: {gene_set_mode}")


def compute_stats(prep: PrepData, cm: CmResult) -> dict[str, Any]:
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
    gene_set_name: str,
    gene_set_mode: str,
    cluster_method: str,
    exp_idx: int,
    adata: ad.AnnData,
    shared: dict[str, Any],
    xlsx_sets: dict[str, set[str]],
    num_iter: int,
) -> dict[str, Any]:
    save_dir = ABLATION_ROOT / f"exp_{exp_idx:02d}_{gene_set_name}_{cluster_method}"
    stats_path = save_dir / "run_stats.json"
    n_total = 2 * len(CLUSTER_METHODS)
    if stats_path.exists():
        print(f"\n{'='*70}")
        print(f"Experiment {exp_idx:02d}/{n_total}: {gene_set_name} + {cluster_method}")
        print(f"Output: {save_dir}")
        print(f"{'='*70}")
        with open(stats_path) as f:
            stats = json.load(f)
        print(f"  Already complete, skipping.")
        return stats
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"Experiment {exp_idx:02d}/{n_total}: {gene_set_name} + {cluster_method}")
    print(f"Output: {save_dir}")
    print(f"{'='*70}")

    CHUNK = 500
    all_gene_names = adata.var_names.values.astype(str)
    gene_idx, gene_names = resolve_gene_set(gene_set_name, gene_set_mode, all_gene_names, xlsx_sets)
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
        "metacell": {"cluster_method": cluster_method},
        "binary": {
            "n_genes_use": n_genes_loaded,
            "custom_gene_idx": custom_gene_idx,
            "min_cells": 1,
        },
        "solver": {
            "num_iter": num_iter,
            "time_limit_per_step": 120,
            "backend": "auto",
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
        "gene_set": gene_set_name,
        "cluster_method": cluster_method,
        "n_genes_loaded": n_genes_loaded,
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


def build_comparison_viz(all_stats: list[dict[str, Any]], labels: list[str], num_iter: int) -> None:
    save_path = ABLATION_ROOT / "viz_ablation_comparison.html"
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    df = pd.DataFrame(all_stats)
    df["label"] = labels

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
                x=labels,
                y=values,
                marker_color=colors[:len(labels)],
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
        title_text=f"Metacell Clustering Ablation ({len(labels)} experiments, {num_iter} iterations each)",
        height=900,
        width=1600,
        barmode="group",
    )
    fig.write_html(save_path, include_plotlyjs="cdn")
    print(f"\nConsolidated viz: {save_path}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Metacell clustering ablation for ConnectionMiner")
    parser.add_argument("--smoke", action="store_true", help="Quick test: 2 experiments, 2 iterations")
    parser.add_argument("--num-iter", type=int, default=None, help="Override iteration count")
    args = parser.parse_args()

    num_iter = NUM_ITER if args.num_iter is None else args.num_iter
    if args.smoke:
        print("SMOKE MODE: 2 experiments, 2 iterations")
        num_iter = 2

    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ConnectionMiner — Metacell Clustering Ablation")
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

    exps_to_run = []
    for gene_set_name, gene_set_mode in GENE_SETS:
        for cmethod in CLUSTER_METHODS:
            exps_to_run.append((gene_set_name, gene_set_mode, cmethod))

    if args.smoke:
        exps_to_run = exps_to_run[:2]

    all_stats = []
    for idx, (gene_set_name, gene_set_mode, cmethod) in enumerate(exps_to_run, start=1):
        stats = run_single_experiment(gene_set_name, gene_set_mode, cmethod, idx, adata, shared, xlsx_sets, num_iter)
        stats["exp_idx"] = idx
        stats["gene_set"] = gene_set_name
        stats["cluster_method"] = cmethod
        all_stats.append(stats)

        del adata
        gc.collect()
        adata = ad.read_h5ad(str(H5AD_PATH), backed="r")

    labels = [f"{gs}_{cm}" for gs, _, cm in exps_to_run]
    stats_df = pd.DataFrame(all_stats)
    stats_df["label"] = labels
    stats_df.to_csv(ABLATION_ROOT / "all_stats.csv", index=False)
    print(f"\nAll stats saved to {ABLATION_ROOT / 'all_stats.csv'}")

    print("\n--- Building comparison viz ---")
    build_comparison_viz(all_stats, labels, num_iter)

    print("\n--- Building interactive clustering explorer ---")
    try:
        from build_clustering_explorer import build_html, load_shared, load_experiments
        s = load_shared()
        experiments = load_experiments()
        html = build_html(s, experiments)
        explorer_path = ABLATION_ROOT / "viz_clustering_explorer.html"
        with open(explorer_path, "w") as f:
            f.write(html)
        print(f"  Saved: {explorer_path}")
    except Exception as exc:
        print(f"  Warning: clustering explorer failed: {exc}")

    print(f"\n{'='*70}")
    print(f"All experiments complete. Results in: {ABLATION_ROOT}")
    print(f"Comparison viz: {ABLATION_ROOT / 'viz_ablation_comparison.html'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
