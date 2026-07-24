from __future__ import annotations

import time
from typing import Any

import numpy as np

from .models import CmResult, PrepData


def _get_torch():
    """Lazy-import torch — call only inside functions when CUDA is confirmed."""
    import torch
    _has_cuda = torch.cuda.is_available()
    _DEVICE = torch.device("cuda" if _has_cuda else "cpu")
    _FLOAT = torch.float64
    _EPS = torch.finfo(_FLOAT).eps

    def _to(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=_FLOAT, device=_DEVICE)

    def _from(t: torch.Tensor) -> np.ndarray:
        return t.cpu().numpy()

    return torch, _DEVICE, _FLOAT, _EPS, _to, _from


def cm_solve_torch(prep: PrepData, cfg: dict[str, Any]) -> CmResult:
    torch, DEV, FLOAT, EPS, _to, _from = _get_torch()
    torch.manual_seed(cfg["seed"])
    t0 = time.time()

    backend = f"torch ({DEV})"
    print(f"  Using Torch backend: {backend}")

    G_full = prep.G_metacell_p
    if prep.G_metacell_p_solve is not None and prep.G_metacell_p_solve.size > 0:
        G = _to(prep.G_metacell_p_solve)
        ng_solver = G.shape[1]
        print(f"  Solver base gene count (Ng_solver) = {ng_solver} (of {G_full.shape[1]} total)")
    else:
        G = _to(G_full)
        ng_solver = G.shape[1]
        print(f"  Solver using all {ng_solver} genes as base solver genes")

    use_complement = bool(cfg["solver"].get("use_complement", False))
    if use_complement:
        G_base = G
        G_comp = torch.clamp(1.0 - G_base, 0.0, 1.0)
        G = torch.cat([G_base, G_comp], dim=1)
        ng_solve = G.shape[1]
        print(f"  Using complement features: Ng_solver={ng_solver}, Ng_eff_raw={ng_solve} (=2*Ng_solver)")
    else:
        ng_solve = G.shape[1]
        print(f"  Complements disabled: Ng_solver={ng_solver}")

    beta_rank = int(cfg["solver"].get("beta_rank", 0) or 0)
    is_low_rank = False
    U_r_np = None
    S_r_np = None

    if 0 < beta_rank < ng_solve:
        print(f"  Low-rank beta: projecting G to rank {beta_rank}")
        G_np = _from(G)
        G_centered = G_np - np.mean(G_np, axis=0, keepdims=True)
        U_np, s_np, _ = np.linalg.svd(G_centered, full_matrices=False)
        ord_idx = np.argsort(-s_np)[:beta_rank]
        U_r_np = U_np[:, ord_idx]
        S_r_np = np.diag(s_np[ord_idx])
        G_proj = _to(U_r_np @ S_r_np)
        ng_eff = beta_rank
        is_low_rank = True
        print(f"  Effective gene dimension: {ng_eff} (was {ng_solve})")
    else:
        G_proj = G
        ng_eff = ng_solve

    C_np = prep.C_counts.astype(float).copy()
    C_mask_np = prep.C_mask.astype(float)
    if cfg["solver"].get("use_binary_connectome", True):
        C_np = (C_np > 0).astype(float)
    W_np = ((C_mask_np > 0) & (~np.isnan(C_np))).astype(float)
    C_np[W_np == 0] = 0.0

    C = _to(C_np)
    W_t = _to(W_np)

    P_constraints = _to(prep.P_constraints_metacell.astype(float))
    D = P_constraints

    with torch.no_grad():
        P = cm_init_P_torch(P_constraints, cfg["solver"].get("P_init", "blend"), torch, DEV, FLOAT, _to, _from)

    print(f"  P init: {cfg['solver'].get('P_init', 'blend')}")
    print(f"  Beta init: {cfg['solver'].get('beta_init', 'random')}")

    beta_init_type = str(cfg["solver"].get("beta_init", "random")).lower()
    if beta_init_type == "identity":
        beta = torch.eye(ng_eff, dtype=FLOAT, device=DEV)
    elif beta_init_type == "ones":
        beta = torch.ones(ng_eff, ng_eff, dtype=FLOAT, device=DEV)
    else:
        beta = torch.rand(ng_eff, ng_eff, dtype=FLOAT, device=DEV)
    beta_max = torch.full((ng_eff, ng_eff), float("inf"), dtype=FLOAT, device=DEV)

    beta_mask = None
    interactome_mode = str(cfg["solver"].get("interactome_constraint", "none"))
    if interactome_mode == "hard" and hasattr(prep, "beta_mask"):
        if is_low_rank:
            print("  Warning: interactome hard mask not applied in low-rank mode")
        else:
            beta_mask_np = getattr(prep, "beta_mask")
            beta_mask = _to(beta_mask_np.astype(float))
            beta = torch.where(beta_mask > 0, beta, torch.zeros_like(beta))
            beta_max = torch.where(beta_mask > 0, beta_max, torch.zeros_like(beta_max))

    num_iter = int(cfg["solver"]["num_iter"])
    lamb = float(cfg["solver"]["lambda_sparsity"])
    epsilon = float(cfg["solver"]["optimal_transport_epsilon"])
    step_size = float(cfg["solver"]["optimal_transport_step"])
    ot_max_iter = int(cfg["solver"]["optimal_transport_iterations"])
    reg_max_iter = int(cfg["solver"]["regression_iterations"])
    time_limit = float(cfg["solver"].get("time_limit_per_step", 30.0))

    loss = np.zeros(num_iter, dtype=float)
    obj_beta = np.zeros(num_iter, dtype=float)
    obj_P_fit = np.zeros(num_iter, dtype=float)
    obj_P_ent = np.zeros(num_iter, dtype=float)

    print(f"  Solver: {num_iter} outer iterations, Ng_eff={ng_eff}, beta size={ng_eff}x{ng_eff}")

    for it in range(num_iter):
        t_iter = time.time()

        with torch.no_grad():
            PG = P @ G_proj
            beta, train_loss_val, _ = cm_beta_update_torch(
                A=PG, B=PG.T, C=C, W=W_t, beta=beta, beta_max=beta_max,
                lamb=lamb, max_iter=reg_max_iter, time_limit=time_limit,
                beta_mask=beta_mask, torch=torch, DEV=DEV, FLOAT=FLOAT, EPS=EPS, _to=_to,
            )
            obj_beta[it] = train_loss_val

            Z = G_proj @ beta @ G_proj.T
            P = cm_P_update_torch(
                P=P, Z=Z, C=C, W=W_t, D=D, epsilon=epsilon, step_size=step_size,
                max_iter=ot_max_iter, time_limit=time_limit, torch=torch, DEV=DEV, FLOAT=FLOAT, EPS=EPS, _to=_to, _from=_from,
            )

            recon = P @ G_proj @ beta @ G_proj.T @ P.T
            diff = W_t * (recon - C)
            obj_P_fit[it] = float(torch.sum(diff ** 2).item())

            mask_P = P > 0
            ent = torch.sum(P[mask_P] * torch.log(P[mask_P])) if mask_P.any() else torch.zeros(1, device=DEV)
            obj_P_ent[it] = float((epsilon * ent).item())
            loss[it] = obj_P_fit[it] + obj_P_ent[it]

        print(
            "  Solver iter {}/{}: obj_beta={:.6e}  obj_P_fit={:.6e}  obj_P_ent={:.6e}  total={:.6e} ({:.1f}s)".format(
                it + 1, num_iter, obj_beta[it], obj_P_fit[it], obj_P_ent[it], loss[it], time.time() - t_iter,
            )
        )

    with torch.no_grad():
        final_recon = P @ G_proj @ beta @ G_proj.T @ P.T

    meta: dict[str, Any] = {}
    if is_low_rank:
        meta["U_r"] = U_r_np
        meta["S_r"] = S_r_np

    cm = CmResult(
        P=_from(P), beta=_from(beta), G_proj=_from(G_proj),
        loss=loss, obj_beta=obj_beta, obj_P_fit=obj_P_fit, obj_P_ent=obj_P_ent,
        P_constraints=_from(P_constraints), C=_from(C), C_mask=_from(W_t), C_recon=_from(final_recon),
        elapsed_sec=time.time() - t0, Ng_solve=ng_solver, Ng_eff=ng_eff, is_low_rank=is_low_rank, meta=meta,
    )

    if cfg.get("run_dir"):
        obj_path = f"{cfg['run_dir']}/solver_objectives.txt"
        with open(obj_path, "w", encoding="utf-8") as f:
            f.write("iter\tobj_beta\tobj_P_fit\tobj_P_ent\ttotal_loss\n")
            for i in range(num_iter):
                f.write(f"{i + 1}\t{obj_beta[i]:.6e}\t{obj_P_fit[i]:.6e}\t{obj_P_ent[i]:.6e}\t{loss[i]:.6e}\n")
        print(f"  Wrote {obj_path}")

    print(f"  Solver done in {cm.elapsed_sec:.1f} s. Final loss: {loss[-1]:.6e}")
    return cm


def cm_init_P_torch(D, init_type, torch, DEV, FLOAT, _to, _from):
    init_type = (init_type or "blend").lower()

    if init_type == "uniform":
        P = D / torch.clamp(D.sum(dim=1, keepdim=True), min=1e-16)
        return torch.nan_to_num(P, 0.0)

    if init_type == "binary":
        return _normalize_rows_torch(_random_binary_init_torch(D, torch, DEV, FLOAT, _to, _from), torch)

    if init_type == "random_proportional":
        P = torch.zeros_like(D)
        idx = D > 0
        P[idx] = torch.rand(P[idx].shape, dtype=FLOAT, device=DEV)
        P = _normalize_rows_torch(P, torch)
        P[~idx] = 0.0
        return P

    P1 = _normalize_rows_torch(_random_binary_init_torch(D, torch, DEV, FLOAT, _to, _from), torch)
    P2 = D / torch.clamp(D.sum(dim=1, keepdim=True), min=1e-16)
    P2 = torch.nan_to_num(P2, 0.0)
    return 0.5 * P1 + 0.5 * P2


def _random_binary_init_torch(D, torch, DEV, FLOAT, _to, _from):
    D_np = _from(D)
    N, M = D_np.shape
    P_np = np.zeros_like(D_np, dtype=float)

    groups = {}
    for c in range(M):
        rows = tuple(np.where(D_np[:, c] > 0)[0].tolist())
        groups.setdefault(rows, []).append(c)

    rng = np.random.default_rng()
    for rows, cols in groups.items():
        if not rows or not cols:
            continue
        rows_arr = np.array(rows, dtype=int)
        cols_arr = np.array(cols, dtype=int)
        rng.shuffle(rows_arr)
        rng.shuffle(cols_arr)
        for j, col in enumerate(cols_arr):
            P_np[rows_arr[j % rows_arr.size], col] = 1.0

    return _to(P_np)


def _normalize_rows_torch(P, torch):
    row_sums = P.sum(dim=1, keepdim=True)
    inv = torch.where(row_sums > 0, 1.0 / row_sums, torch.zeros_like(row_sums))
    return P * inv


def cm_beta_update_torch(A, B, C, W, beta, beta_max, lamb, max_iter, time_limit, beta_mask, torch, DEV, FLOAT, EPS, _to):
    tol = 1e-6
    t0 = time.time()

    Wsq = W ** 2
    Numer = A.T @ (Wsq * C) @ B.T

    beta = torch.clamp(beta, torch.tensor(0.0, device=DEV, dtype=FLOAT), beta_max)
    if beta_mask is not None:
        beta = torch.where(beta_mask > 0, beta, torch.zeros_like(beta))

    prev_obj = _beta_obj_torch(A, B, C, Wsq, beta, torch)

    for _ in range(max_iter):
        M_recon = A @ (beta @ B)
        Denom = A.T @ (Wsq * M_recon) @ B.T
        beta_raw = beta * (Numer / (Denom + lamb))

        beta = torch.clamp(beta_raw, torch.tensor(0.0, device=DEV, dtype=FLOAT), beta_max)
        if beta_mask is not None:
            beta = torch.where(beta_mask > 0, beta, torch.zeros_like(beta))

        curr_obj = _beta_obj_torch(A, B, C, Wsq, beta, torch)
        rel_change = abs(curr_obj - prev_obj) / (prev_obj + EPS)

        if rel_change < tol or (time.time() - t0) > time_limit:
            break
        prev_obj = curr_obj

    beta = beta + 100 * EPS * torch.rand(*beta.shape, dtype=FLOAT, device=DEV)
    if beta_mask is not None:
        beta = torch.where(beta_mask > 0, beta, torch.zeros_like(beta))

    train_loss = _beta_obj_torch(A, B, C, Wsq, beta, torch)
    val_loss = _beta_obj_torch(A, B, C, 1.0 - Wsq, beta, torch)
    return beta, train_loss, val_loss


def _beta_obj_torch(A, B, C, Wsq, X, torch):
    R = A @ (X @ B) - C
    return float(torch.sum(Wsq * (R ** 2)).item())


def cm_P_update_torch(P, Z, C, W, D, epsilon, step_size, max_iter, time_limit, torch, DEV, FLOAT, EPS, _to, _from):
    D_norm = D / torch.clamp(D.sum(dim=1, keepdim=True), min=1e-16)
    D_norm = torch.nan_to_num(D_norm, 0.0)
    row_c = D_norm.sum(dim=1)
    col_c = D_norm.sum(dim=0)

    A_mat = Z @ P.T
    B_mat = C
    unfixed = ~torch.all(B_mat == 0, dim=1)
    P_tmp = _entropic_sinkhorn_torch(A_mat, B_mat, W, row_c, col_c, D, epsilon, step_size, max_iter, time_limit, P, torch, DEV, FLOAT, EPS)
    P[unfixed, :] = P_tmp[unfixed, :]

    A_mat = Z.T @ P.T
    B_mat = C.T
    unfixed = ~torch.all(B_mat == 0, dim=1)
    P_tmp = _entropic_sinkhorn_torch(A_mat, B_mat, W.T, row_c, col_c, D, epsilon, step_size, max_iter, time_limit, P, torch, DEV, FLOAT, EPS)
    P[unfixed, :] = P_tmp[unfixed, :]

    return P


def _entropic_sinkhorn_torch(A, B, W, a, b, D, epsilon, step_size, max_iter, time_limit, P0, torch, DEV, FLOAT, EPS):
    row_col_iters = 10
    max_backtrack = 20
    tol = 1e-6

    if P0 is not None:
        P = P0.clone()
    else:
        P = 0.5 * D
    P = _normalize_clip_torch(P, a, b, D, row_col_iters, torch)

    W2 = W ** 2
    prev_obj = _sinkhorn_obj_torch(P, A, B, W2, epsilon, torch)

    t0 = time.time()
    for _ in range(max_iter):
        PA = P @ A
        residual = PA - B
        WR = W2 * residual
        grad_ls = 2.0 * (WR @ A.T)
        grad_ent = epsilon * (1.0 + _safe_log_torch(P, torch))
        G = grad_ls + grad_ent

        trial_step = step_size
        new_obj = prev_obj
        P_trial = P
        for bt in range(max_backtrack + 1):
            P_trial = P * torch.exp(-trial_step * G)
            P_trial = torch.clamp(P_trial, torch.zeros_like(D), D)
            P_trial = _normalize_clip_torch(P_trial, a, b, D, row_col_iters, torch)
            new_obj = _sinkhorn_obj_torch(P_trial, A, B, W2, epsilon, torch)
            if new_obj <= prev_obj or bt >= max_backtrack:
                break
            trial_step /= 2.0

        P = P_trial
        rel_change = abs(new_obj - prev_obj) / max(1.0, abs(prev_obj))

        if rel_change < tol:
            P = _normalize_clip_torch(P, a, b, D, 1000, torch)
            break
        if (time.time() - t0) > time_limit:
            break

        prev_obj = new_obj

    return P


def _sinkhorn_obj_torch(P, A, B, W2, epsilon, torch):
    R = P @ A - B
    val_ls = float(torch.sum(W2 * (R ** 2)).item())
    mask = P > 0
    val_ent = float(torch.sum(P[mask] * torch.log(P[mask])).item()) if mask.any() else 0.0
    return val_ls + epsilon * val_ent


def _safe_log_torch(X, torch):
    return torch.log(torch.clamp(X, min=1e-30))


def _normalize_clip_torch(P, a, b, D, passes, torch):
    for _ in range(passes):
        rs = P.sum(dim=1, keepdim=True)
        scale_r = a[:, None] / torch.clamp(rs, min=1e-16)
        P = torch.clamp(P * scale_r, torch.zeros_like(D), D)

        cs = P.sum(dim=0, keepdim=True)
        scale_c = b[None, :] / torch.clamp(cs, min=1e-16)
        P = torch.clamp(P * scale_c, torch.zeros_like(D), D)
    return P
