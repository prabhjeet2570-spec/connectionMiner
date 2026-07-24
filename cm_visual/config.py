from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .paths import cm_get_paths


def default_config(input_mode: str = "binary") -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "seed": 750,
        "input_mode": input_mode,
        "paths": cm_get_paths(),
        "load": {
            "read_size": 1000,
            "numeric_class": "float32",
            "allow_reorder": True,
        },
        "binary": {
            "n_genes_use": 3000,
            "min_cells": 5,
        },
        "metacell": {
            "target_size": 15,
            "min_size": 5,
            "min_samples_prior": 10,
            "n_pcs": 50,
            "cluster_method": "kmeans",
            "kmeans_reps": 5,
            "kmeans_maxiter": 200,
        },
        "compute_type_gene_probabilities": True,
        "export_type_gene_probabilities": True,
        "solver": {
            "num_iter": 100,
            "lambda_sparsity": 0.001,
            "optimal_transport_epsilon": 1e-12,
            "optimal_transport_step": 0.04,
            "optimal_transport_iterations": 50,
            "regression_iterations": 50,
            "use_binary_connectome": True,
            "beta_rank": 0,
            "use_complement": False,
            "P_init": "random_proportional",
            "beta_init": "random",
            "time_limit_per_step": 60,
            "backend": "auto",
        },
        "compute_type_gene_probabilities": True,
        "smoke_test": {
            "enabled": False,
            "max_cells": 999999,
            "max_genes": 999999,
        },
    }
    return cfg


def merge_config(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Deep-merge override into base and return a new config."""
    if override is None:
        return base

    out = copy.deepcopy(base)

    def _merge(a: dict[str, Any], b: dict[str, Any]) -> None:
        for key, value in b.items():
            if isinstance(value, dict) and isinstance(a.get(key), dict):
                _merge(a[key], value)
            else:
                a[key] = value

    _merge(out, override)
    return out


def ensure_run_dir(cfg: dict[str, Any], run_tag_prefix: str = "run") -> Path:
    repo_root = Path(cfg["paths"]["repo_root"])
    run_root = repo_root / "cm_visual" / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / f"{run_tag_prefix}_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["run_dir"] = str(run_dir)
    return run_dir
