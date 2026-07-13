#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))

from cm_visual.run_visual import cm_run_visual


def main() -> None:
    parser = argparse.ArgumentParser(description="FlyWire Visual System × ConnectionMiner")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "output"), help="Path to pre-built matrices directory")
    parser.add_argument("--h5ad", default=str(REPO_ROOT / "data" / "Adult.h5ad"), help="Path to Adult.h5ad")
    parser.add_argument("--num-iter", type=int, default=None, help="Number of solver iterations")
    parser.add_argument("--lambda-sparsity", type=float, default=None, help="Sparsity regularization")
    parser.add_argument("--beta-rank", type=int, default=None, help="Low-rank beta dimension (0=full)")
    parser.add_argument("--smoke", action="store_true", help="Enable smoke test mode")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--time-limit", type=float, default=None, help="Time limit per solver step (seconds)")
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.num_iter is not None:
        overrides.setdefault("solver", {})["num_iter"] = args.num_iter
    if args.lambda_sparsity is not None:
        overrides.setdefault("solver", {})["lambda_sparsity"] = args.lambda_sparsity
    if args.beta_rank is not None:
        overrides.setdefault("solver", {})["beta_rank"] = args.beta_rank
    if args.time_limit is not None:
        overrides.setdefault("solver", {})["time_limit_per_step"] = args.time_limit
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.smoke:
        overrides.setdefault("smoke_test", {})["enabled"] = True

    output_dir = cm_run_visual(
        output_dir=args.output_dir,
        h5ad_path=args.h5ad,
        cfg_overrides=overrides if overrides else None,
    )
    print(f"Done: {output_dir}")


if __name__ == "__main__":
    main()
