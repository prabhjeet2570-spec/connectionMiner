"""
build_all_matrices.py
=======================
ALL matrix construction for the FlyWire ConnectionMiner pipeline, in one
file, run step by step. Stops right before the solver — this script's
job is purely to produce every matrix the real
connectionMiner/cm_visual/solver.py (cm_solve) needs, fully validated.

STEP 1   Load Adult.h5ad
STEP 2   C               — (741 x 741) binary connectome, both hemispheres
STEP 3   G_cells          — (n_cells x 3000) z-scored HVG expression (for clustering)
STEP 4   P_cells          — (n_cells x 741) cell-level type constraints, row-stochastic
STEP 5   B_cells          — (g x g) reference-only covariance (NOT used by real solver)
STEP 6   Metacells        — tiered named/numeric/orphan clustering, 100k -> ~8k
STEP 7   P_meta, G_meta   — metacell-level versions of steps 3-4 (your pipeline's own
                            representation — z-scored, row-stochastic)
STEP 8   G_metacell_p     — (n_meta x g) [0,1] detection-probability matrix
                            *** THIS is the real solver's G, not G_meta from step 7 ***
STEP 9   P_constraints_*  — (741 x n_meta) / (741 x n_cells) binary support masks,
                            TRANSPOSED to match the real solver's orientation
STEP 10  C_mask           — (741 x 741) observed-entries mask

Why steps 7 and 8/9 both exist
-------------------------------
Step 7 (P_meta, G_meta) is a diagnostic-only representation: z-scored continuous
expression, row-stochastic P.

Steps 8-9 are what the ACTUAL evarol/connectionMiner solver requires, confirmed
by reading its source (preprocess.py, loaders.py):
  - G must be non-negative, [0,1]-bounded ("fraction of cells in this metacell
    expressing gene g"), because beta is fit via multiplicative NMF-style
    updates that only stay valid for non-negative inputs.
  - P/constraints are oriented (N_types x n_X), the transpose of your P_meta.
Both are rebuilt fresh in steps 8-9 rather than reusing step 7's outputs.

Inputs
------
    Adult.h5ad
    visual_neuron_types.csv.gz
    connections_princeton.csv.gz

Outputs (connectionMiner/output/)
--------------------
    C_matrix.npz, type_index.csv                          [step 2]
    G_matrix.npy, gene_index.csv                           [step 3]
    P_matrix.npz, cell_index.csv                            [step 4]
    B_matrix.npy                                             [step 5, reference only]
    cell_to_metacell.csv, metacell_index.csv                 [step 6]
    P_meta.npz, G_meta.npy                                    [step 7]
    G_metacell_p.npy                                          [step 8 — REAL solver G]
    P_constraints_metacell.npz, P_constraints_cells.npz       [step 9 — REAL solver P]
    C_mask.npy                                                [step 10]
"""

import os, gc, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
ATLAS_FILE    = "../data/Adult.h5ad"
NEURONS_FILE  = "../data/visual_neuron_types.csv.gz"
SYNAPSES_FILE = "../data/connections_princeton.csv.gz"
OUT_DIR       = "../output"
if os.path.islink(OUT_DIR) and not os.path.exists(OUT_DIR):
    os.unlink(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

CLUSTER_COL    = "MultiomeNN"
ANNOTATION_COL = "MultiomeAnnotated"

N_HVG            = 3000     # genes kept in G_cells
COSINE_THRESH    = 0.70     # numeric-cell candidate-type threshold
CHUNK_SIZE       = 2_000_000
TARGET_METACELLS = 8_000
MIN_PER_GROUP    = 1
RANDOM_SEED      = 0
PCA_KEY          = None

OPTIC_NEUROPILS = {
    "ME_R","ME_L","LO_R","LO_L","LOP_R","LOP_L",
    "LA_R","LA_L","AME_R","AME_L","MCE_R","MCE_L",
    "ICL_R","ICL_L","OCG",
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def is_numeric_label(label):
    return str(label).strip().lstrip("-").isdigit()


def detect_pca_key(adata):
    global PCA_KEY
    if PCA_KEY is not None:
        return PCA_KEY
    candidates = [k for k in adata.obsm.keys() if "pca" in k.lower()]
    PCA_KEY = candidates[0] if candidates else "X_pca"
    return PCA_KEY


def get_expression_layer(adata):
    """Returns the same expression source used throughout: layer if present, else X."""
    for key in ["lognorm", "logcounts", "normalised", "log1p"]:
        if key in adata.layers:
            print(f"  Using layer '{key}'")
            return adata.layers[key], key
    print("  Using adata.X")
    return adata.X, "X"


def kmeans_pool(features, k):
    k = max(1, min(k, len(features)))
    if k == 1 or len(features) <= 1:
        return np.zeros(len(features), dtype=np.int32)
    km = MiniBatchKMeans(
        n_clusters=k, random_state=RANDOM_SEED, n_init=3,
        batch_size=min(1024, len(features)),
    )
    return km.fit_predict(features).astype(np.int32)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD
# ═══════════════════════════════════════════════════════════════════════════════
def step1_load():
    print("=" * 70)
    print("STEP 1 — Load Adult.h5ad")
    print("=" * 70)
    adata = ad.read_h5ad(ATLAS_FILE)
    print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    print(f"  obs columns : {list(adata.obs.columns)}")
    print(f"  obsm keys   : {list(adata.obsm.keys())}")
    return adata


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — C  (741 x 741, binary, both hemispheres)
# ═══════════════════════════════════════════════════════════════════════════════
def step2_build_C():
    print("\n" + "=" * 70)
    print("STEP 2 — C  (741 x 741 binary connectome, both hemispheres)")
    print("=" * 70)

    neurons = pd.read_csv(NEURONS_FILE, compression="infer")
    neurons = neurons.dropna(subset=["type"])
    print(f"  Total neurons : {len(neurons):,}")
    print(f"  Unique types  : {neurons['type'].nunique()}")

    types_ordered = sorted(neurons["type"].unique())
    N             = len(types_ordered)
    type2idx      = {t: i for i, t in enumerate(types_ordered)}
    root2type     = dict(zip(neurons["root_id"], neurons["type"]))

    type_index = pd.DataFrame({"col_idx": range(N), "type": types_ordered})
    for col in ["family", "subsystem", "category"]:
        if col in neurons.columns:
            meta = neurons.groupby("type")[col].first()
            type_index[col] = type_index["type"].map(meta)
    print(f"  N = {N} types")

    print(f"  Streaming {SYNAPSES_FILE} ...")
    rows_acc, cols_acc = [], []
    total = kept = 0
    reader = pd.read_csv(
        SYNAPSES_FILE, compression="infer", chunksize=CHUNK_SIZE,
        dtype={"pre_root_id": np.int64, "post_root_id": np.int64},
        usecols=["pre_root_id", "post_root_id", "neuropil"],
    )
    for chunk in reader:
        total += len(chunk)
        chunk = chunk[chunk["neuropil"].isin(OPTIC_NEUROPILS)].copy()
        chunk["pre_type"]  = chunk["pre_root_id"].map(root2type)
        chunk["post_type"] = chunk["post_root_id"].map(root2type)
        chunk = chunk.dropna(subset=["pre_type", "post_type"])
        chunk["pre_idx"]  = chunk["pre_type"].map(type2idx).astype(np.int32)
        chunk["post_idx"] = chunk["post_type"].map(type2idx).astype(np.int32)
        rows_acc.append(chunk["pre_idx"].values)
        cols_acc.append(chunk["post_idx"].values)
        kept += len(chunk)
        print(f"  {total:>9,} rows processed | {kept:,} kept", end="\r")
    print(f"\n  Done. {kept:,} / {total:,} synapse rows kept")

    rows = np.concatenate(rows_acc)
    cols = np.concatenate(cols_acc)
    data = np.ones(len(rows), dtype=np.float32)
    C = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    C = C.astype(bool).astype(np.float32)

    print(f"  C shape         : {C.shape}")
    print(f"  Connected pairs : {C.nnz:,}  ({100*C.nnz/N**2:.1f}% dense)")

    sp.save_npz(os.path.join(OUT_DIR, "C_matrix.npz"), C)
    type_index.to_csv(os.path.join(OUT_DIR, "type_index.csv"), index=False)
    print(f"  Saved C_matrix.npz + type_index.csv")
    return C, type_index, type2idx, N


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — G_cells  (n_cells x N_HVG, z-scored — used for clustering only)
# ═══════════════════════════════════════════════════════════════════════════════
def step3_build_G_cells(adata):
    print("\n" + "=" * 70)
    print("STEP 3 — G_cells  (n_cells x HVG, z-scored — clustering input only)")
    print("=" * 70)

    X, layer_used = get_expression_layer(adata)
    n_cells, n_all_genes = X.shape

    # ── HVG selection WITHOUT densifying all genes ────────────────────────────
    # Computing variance on the full dense matrix (n_cells x ~15k genes) would
    # allocate ~6.6 GB and may not be freed before later steps. Instead compute
    # var(X) = E[X²] - E[X]² using sparse operations: only O(nnz) memory.
    print(f"  Computing per-gene variance (sparse, {n_all_genes:,} genes) ...")
    X_sp = X.tocsc() if sp.issparse(X) else sp.csc_matrix(X)

    # Check if raw counts (sample first 500 rows for speed)
    X_sample = X_sp[:500].toarray()
    raw_counts = X_sample.max() > 20

    if raw_counts:
        print("  Raw counts detected — computing log1p-normalised variance (sparse) ...")
        # We cannot log1p in-place on sparse without densifying, so we work in
        # chunks of genes (columns), which are tiny when taken individually.
        gene_var    = np.zeros(n_all_genes, dtype=np.float64)
        GCOL_CHUNK  = 500
        for gc_start in range(0, n_all_genes, GCOL_CHUNK):
            gc_end  = min(gc_start + GCOL_CHUNK, n_all_genes)
            chunk   = np.asarray(X_sp[:, gc_start:gc_end].todense(), dtype=np.float32)
            # per-row lib-size normalisation then log1p
            row_sums = X_sp.sum(axis=1).A1
            row_sums[row_sums == 0] = 1
            chunk = np.log1p(chunk / row_sums[:, None] * 10_000)
            gene_var[gc_start:gc_end] = chunk.var(axis=0)
            print(f"  {gc_end:>6,} / {n_all_genes:,} genes", end="\r")
        print()
    else:
        # Already log-normalised — use E[X²]-E[X]² in sparse (CSC for efficiency)
        print("  Log-normalised data — computing variance via sparse E[X²]-E[X]² ...")
        mean_sq = np.asarray(X_sp.power(2).mean(axis=0)).ravel()
        sq_mean = np.asarray(X_sp.mean(axis=0)).ravel() ** 2
        gene_var = mean_sq - sq_mean
        gene_var = np.maximum(gene_var, 0.0)    # numerical noise guard

    print(f"  Selecting top {N_HVG} highly variable genes ...")
    hvg_idx = np.sort(np.argsort(gene_var)[::-1][:N_HVG])

    # ── Densify only the 3000 HVG columns ────────────────────────────────────
    print(f"  Extracting {N_HVG} HVG columns and densifying ...")
    G = np.asarray(X_sp[:, hvg_idx].todense(), dtype=np.float32)

    if raw_counts:
        row_sums = X_sp.sum(axis=1).A1
        row_sums[row_sums == 0] = 1
        G = np.log1p(G / row_sums[:, None] * 10_000)

    del X_sp, X_sample                   # free sparse full-gene matrix
    gc.collect()

    g_mean = G.mean(axis=0)
    g_std  = G.std(axis=0); g_std[g_std == 0] = 1
    G = ((G - g_mean) / g_std).astype(np.float32)

    print(f"  G_cells shape : {G.shape}  ({G.nbytes/1e9:.2f} GB)")
    print(f"  G_cells range : {G.min():.3f} - {G.max():.3f}")

    gene_index = pd.DataFrame({
        "col_idx"  : range(N_HVG),
        "gene_name": list(adata.var_names[hvg_idx]),
        "variance" : gene_var[hvg_idx],
    })
    np.save(os.path.join(OUT_DIR, "G_matrix.npy"), G)
    gene_index.to_csv(os.path.join(OUT_DIR, "gene_index.csv"), index=False)
    print(f"  Saved G_matrix.npy + gene_index.csv")
    return G, gene_index, hvg_idx, layer_used


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — P_cells  (n_cells x 741, row-stochastic)
# ═══════════════════════════════════════════════════════════════════════════════
def step4_build_P_cells(adata, G, type_index, type2idx, N):
    print("\n" + "=" * 70)
    print("STEP 4 — P_cells  (n_cells x 741, row-stochastic type constraints)")
    print("=" * 70)

    n_cells  = adata.n_obs
    labels   = adata.obs[ANNOTATION_COL].astype(str).str.strip()
    clusters = adata.obs[CLUSTER_COL].values
    is_num   = labels.apply(is_numeric_label).values
    type2idx_lower = {t.lower(): i for t, i in type2idx.items()}

    def match_type(label):
        if label in type2idx:             return type2idx[label]
        if label.lower() in type2idx_lower: return type2idx_lower[label.lower()]
        return None

    print("  Computing cluster centroids in G space ...")
    unique_clusters = np.unique(clusters)
    centroids = {cl: G[clusters == cl].mean(axis=0) for cl in unique_clusters}

    named_cl_ids, named_cl_tidx = [], []
    for cl in unique_clusters:
        majority = labels[clusters == cl].mode()[0]
        if not is_numeric_label(majority):
            t = match_type(majority)
            if t is not None:
                named_cl_ids.append(cl); named_cl_tidx.append(t)
    print(f"  Named clusters with FlyWire match : {len(named_cl_ids)}")

    named_cent_mat = np.stack([centroids[cl] for cl in named_cl_ids])
    all_cent_mat   = np.stack([centroids[cl] for cl in unique_clusters])
    cl_to_row      = {cl: i for i, cl in enumerate(unique_clusters)}
    sim = cosine_similarity(all_cent_mat, named_cent_mat).astype(np.float32)

    print("  First pass — named cells ...")
    type_has_cells = np.zeros(N, dtype=bool)
    for c in range(n_cells):
        if not is_num[c]:
            t = match_type(labels.iloc[c])
            if t is not None:
                type_has_cells[t] = True
    empty_types     = np.where(~type_has_cells)[0]
    n_empty         = len(empty_types)
    orphan_prob     = 1.0 / n_empty if n_empty > 0 else 1.0 / N
    orphan_type_set = empty_types if n_empty > 0 else np.arange(N)
    print(f"  Types with cells : {type_has_cells.sum()}  |  empty (orphan target) : {n_empty}")

    print("  Second pass — building P entries ...")
    p_rows, p_cols, p_data = [], [], []
    named_count = numeric_count = orphan_count = 0

    for c in range(n_cells):
        label = labels.iloc[c]
        cl    = clusters[c]
        if not is_num[c]:
            t = match_type(label)
            if t is not None:
                p_rows.append(t); p_cols.append(c); p_data.append(1.0)
                named_count += 1
            else:
                for t in orphan_type_set:
                    p_rows.append(int(t)); p_cols.append(c); p_data.append(orphan_prob)
                orphan_count += 1
        else:
            cl_row  = cl_to_row[cl]
            above   = np.where(sim[cl_row] >= COSINE_THRESH)[0]
            if len(above) == 0:
                above = np.argsort(sim[cl_row])[::-1][:3]
            allowed = list({named_cl_tidx[i] for i in above})
            prob    = 1.0 / len(allowed)
            for t in allowed:
                p_rows.append(t); p_cols.append(c); p_data.append(prob)
            numeric_count += 1
        if (c + 1) % 20_000 == 0:
            print(f"  {c+1:>7,} / {n_cells:,} cells", end="\r")

    print(f"\n  Named: {named_count:,}  Numeric: {numeric_count:,}  Orphan: {orphan_count:,}")

    P = sp.coo_matrix(
        (np.array(p_data, dtype=np.float32),
         (np.array(p_cols, dtype=np.int32), np.array(p_rows, dtype=np.int32))),
        shape=(n_cells, N),
    ).tocsr()

    row_sums = np.array(P.sum(axis=1)).ravel()
    print(f"  P_cells row sums — min: {row_sums.min():.4f}  max: {row_sums.max():.4f}")
    print(f"  P_cells shape : {P.shape}  nnz={P.nnz:,}")

    cell_type_arr = []
    for c in range(n_cells):
        if not is_num[c] and match_type(labels.iloc[c]) is None:
            cell_type_arr.append("orphan")
        elif is_num[c]:
            cell_type_arr.append("numeric")
        else:
            cell_type_arr.append("named")

    cell_index = pd.DataFrame({
        "col_idx"      : range(n_cells),
        "cell_barcode" : adata.obs_names.tolist(),
        CLUSTER_COL    : adata.obs[CLUSTER_COL].values,
        ANNOTATION_COL : labels.values,
        "cell_type"    : cell_type_arr,
    })

    sp.save_npz(os.path.join(OUT_DIR, "P_matrix.npz"), P)
    cell_index.to_csv(os.path.join(OUT_DIR, "cell_index.csv"), index=False)
    print(f"  Saved P_matrix.npz + cell_index.csv")
    return P, cell_index


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — B_cells  (g x g, reference-only — NOT used by the real solver)
# ═══════════════════════════════════════════════════════════════════════════════
def step5_build_B_cells(G):
    print("\n" + "=" * 70)
    print("STEP 5 — B_cells  (g x g gene covariance, REFERENCE ONLY)")
    print("  ⚠ The real ConnectionMiner solver fits beta fresh — this is")
    print("    saved only for your own diagnostics, not fed to the solver.")
    print("=" * 70)

    n_cells, g = G.shape
    GtG = np.zeros((g, g), dtype=np.float32)
    CHUNK = 5_000
    for start in range(0, n_cells, CHUNK):
        end = min(start + CHUNK, n_cells)
        gc  = G[start:end].astype(np.float32)
        GtG += gc.T @ gc
        print(f"  {end:>7,} / {n_cells:,}", end="\r")
    B = GtG / n_cells
    print(f"\n  B_cells shape : {B.shape}  range: {B.min():.4f} - {B.max():.4f}")

    np.save(os.path.join(OUT_DIR, "B_matrix.npy"), B.astype(np.float32))
    print(f"  Saved B_matrix.npy")
    return B


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — METACELLS  (100k -> ~8k, tiered named / numeric / orphan)
# ═══════════════════════════════════════════════════════════════════════════════
def step6_build_metacells(adata, cell_index, P_cells, PCA):
    print("\n" + "=" * 70)
    print(f"STEP 6 — Metacells  (target={TARGET_METACELLS:,})")
    print("=" * 70)

    n_total   = len(cell_index)
    n_named   = (cell_index["cell_type"] == "named").sum()
    n_numeric = (cell_index["cell_type"] == "numeric").sum()
    n_orphan  = (cell_index["cell_type"] == "orphan").sum()

    b_named   = max(1, round(TARGET_METACELLS * n_named   / n_total))
    b_numeric = max(1, round(TARGET_METACELLS * n_numeric / n_total))
    b_orphan  = max(1, round(TARGET_METACELLS * n_orphan  / n_total))
    diff = TARGET_METACELLS - (b_named + b_numeric + b_orphan)
    b_named += diff
    print(f"  Named   {n_named:>8,} cells -> {b_named:,} metacells")
    print(f"  Numeric {n_numeric:>8,} cells -> {b_numeric:,} metacells")
    print(f"  Orphan  {n_orphan:>8,} cells -> {b_orphan:,} metacells")

    all_cidx, all_meta, records = [], [], []
    counter = 0

    # ── Named tier ───────────────────────────────────────────────────────────
    named_cidx = np.where((cell_index["cell_type"] == "named").values)[0]
    P_named    = P_cells[named_cidx]
    pinned     = np.array(P_named.argmax(axis=1)).ravel()
    type_to_local = defaultdict(list)
    for li, t in enumerate(pinned):
        type_to_local[int(t)].append(li)
    for t_idx, local_indices in type_to_local.items():
        local_arr  = np.array(local_indices, dtype=np.int64)
        global_arr = named_cidx[local_arr]
        k = max(MIN_PER_GROUP, round(b_named * len(local_arr) / max(n_named, 1)))
        labels = kmeans_pool(PCA[global_arr], k)
        for lbl in np.unique(labels):
            members = global_arr[labels == lbl]
            mc_id = f"named_t{t_idx}_m{counter}"
            all_cidx.extend(members.tolist())
            all_meta.extend([mc_id] * len(members))
            records.append({"metacell_id": mc_id, "tier": "named", "n_cells": len(members)})
            counter += 1
    print(f"  Named tier   -> {counter:,} metacells")

    # ── Numeric tier ─────────────────────────────────────────────────────────
    start_numeric = counter
    numeric_cidx = np.where((cell_index["cell_type"] == "numeric").values)[0]
    P_numeric = P_cells[numeric_cidx].tocsr()
    set_to_locals = defaultdict(list)
    for li in range(len(numeric_cidx)):
        sset = tuple(P_numeric.getrow(li).indices)
        set_to_locals[sset].append(li)
    for s_idx, (sset, local_indices) in enumerate(set_to_locals.items()):
        local_arr  = np.array(local_indices, dtype=np.int64)
        global_arr = numeric_cidx[local_arr]
        k = max(MIN_PER_GROUP, round(b_numeric * len(local_arr) / max(len(numeric_cidx), 1)))
        labels = kmeans_pool(PCA[global_arr], k)
        for lbl in np.unique(labels):
            members = global_arr[labels == lbl]
            mc_id = f"numeric_s{s_idx}_m{counter}"
            all_cidx.extend(members.tolist())
            all_meta.extend([mc_id] * len(members))
            records.append({"metacell_id": mc_id, "tier": "numeric", "n_cells": len(members)})
            counter += 1
    print(f"  Numeric tier -> {counter - start_numeric:,} metacells")

    # ── Orphan tier ──────────────────────────────────────────────────────────
    start_orphan = counter
    orphan_cidx = np.where((cell_index["cell_type"] == "orphan").values)[0]
    if len(orphan_cidx) > 0:
        labels = kmeans_pool(PCA[orphan_cidx], b_orphan)
        for lbl in np.unique(labels):
            members = orphan_cidx[labels == lbl]
            mc_id = f"orphan_m{counter}"
            all_cidx.extend(members.tolist())
            all_meta.extend([mc_id] * len(members))
            records.append({"metacell_id": mc_id, "tier": "orphan", "n_cells": len(members)})
            counter += 1
    print(f"  Orphan tier  -> {counter - start_orphan:,} metacells")
    print(f"  TOTAL metacells : {counter:,}")

    n_cells = adata.n_obs
    cell_to_meta = np.empty(n_cells, dtype=object)
    for cidx, mc in zip(all_cidx, all_meta):
        cell_to_meta[cidx] = mc

    metacell_index = pd.DataFrame(records)
    metacell_index["metacell_idx"] = range(len(metacell_index))
    cell_to_metacell = pd.DataFrame({"cell_idx": range(n_cells), "metacell_id": cell_to_meta})

    metacell_index.to_csv(os.path.join(OUT_DIR, "metacell_index.csv"), index=False)
    cell_to_metacell.to_csv(os.path.join(OUT_DIR, "cell_to_metacell.csv"), index=False)
    print(f"  Saved metacell_index.csv + cell_to_metacell.csv")

    return metacell_index, cell_to_metacell


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — P_meta, G_meta  (diagnostic-only — z-scored)
# ═══════════════════════════════════════════════════════════════════════════════
def step7_build_P_meta_G_meta(G_cells, P_cells, cell_to_metacell, metacell_index):
    print("\n" + "=" * 70)
    print("STEP 7 — P_meta, G_meta  (z-scored, row-stochastic — diagnostic-only)")
    print("=" * 70)

    n_meta = len(metacell_index)
    id_to_idx = dict(zip(metacell_index["metacell_id"], metacell_index["metacell_idx"]))
    cell_to_meta_sorted = cell_to_metacell.sort_values("cell_idx")
    meta_int = cell_to_meta_sorted["metacell_id"].map(id_to_idx).values.astype(int)

    # ── G_meta via chunked streaming ──────────────────────────────────────────
    # G_cells is (109743 x 3000) = 1.3 GB already in memory, but after steps
    # 4-6 accumulated objects, forming M and doing M.T @ G_cells can push us
    # over the limit. Instead: accumulate into G_meta in-place using np.add.at,
    # which is O(n_cells) memory not O(n_meta x n_cells).
    print("  Computing G_meta via chunked np.add.at (streaming from disk) ...")
    n_genes  = G_cells.shape[1]
    n_cells  = G_cells.shape[0]
    G_meta   = np.zeros((n_meta, n_genes), dtype=np.float64)
    counts   = np.bincount(meta_int, minlength=n_meta).astype(np.float64)

    CELL_CHUNK = 10_000
    for start in range(0, n_cells, CELL_CHUNK):
        end    = min(start + CELL_CHUNK, n_cells)
        chunk  = G_cells[start:end].astype(np.float64)   # (chunk x 3000)
        np.add.at(G_meta, meta_int[start:end], chunk)
        print(f"  G_meta: {end:>7,} / {n_cells:,} cells", end="\r")
    print()

    G_meta /= np.maximum(counts[:, None], 1.0)
    G_meta = G_meta.astype(np.float32)
    print(f"  G_meta shape : {G_meta.shape}  ({G_meta.nbytes/1e6:.0f} MB)")

    # ── P_meta via sparse aggregation (P_cells is already sparse, result tiny) ─
    print("  Computing P_meta via sparse aggregation ...")
    M = sp.csr_matrix(
        (np.ones(n_cells, dtype=np.float32), (np.arange(n_cells), meta_int)),
        shape=(n_cells, n_meta),
    )
    P_meta = np.asarray((M.T @ P_cells).todense()) / np.maximum(counts[:, None], 1.0)
    P_meta = P_meta.astype(np.float32)

    row_sums = P_meta.sum(axis=1)
    print(f"  P_meta shape : {P_meta.shape}")
    print(f"  P_meta row sums — min: {row_sums.min():.4f}  max: {row_sums.max():.4f}")

    np.save(os.path.join(OUT_DIR, "G_meta.npy"), G_meta)
    sp.save_npz(os.path.join(OUT_DIR, "P_meta.npz"), sp.csr_matrix(P_meta))
    print(f"  Saved G_meta.npy + P_meta.npz")

    meta_sizes = counts.astype(np.float32)
    return G_meta, P_meta, meta_int, meta_sizes


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — G_metacell_p  (REAL solver G — [0,1] detection probability)
# ═══════════════════════════════════════════════════════════════════════════════
def step8_build_G_metacell_p(adata, hvg_idx, n_meta, meta_int, meta_sizes):
    print("\n" + "=" * 70)
    print("STEP 8 — G_metacell_p  (REAL solver input: [0,1] detection probability)")
    print("  Fraction of cells in each metacell expressing each gene.")
    print("  Binarization (count > 0) is invariant to log1p, so any expression")
    print("  layer works the same as raw counts here.")
    print("=" * 70)

    X, layer_used = get_expression_layer(adata)
    n_cells = adata.n_obs

    if sp.issparse(X):
        X_sub = X.tocsc()[:, hvg_idx]
        X_bin = (X_sub > 0).toarray().astype(np.float32)
    else:
        X_bin = (np.asarray(X)[:, hvg_idx] > 0).astype(np.float32)

    M = sp.csr_matrix(
        (np.ones(n_cells, dtype=np.float32), (np.arange(n_cells), meta_int)),
        shape=(n_cells, n_meta),
    )
    G_metacell_p = (M.T @ X_bin) / np.maximum(meta_sizes[:, None], 1.0)

    print(f"  G_metacell_p shape : {G_metacell_p.shape}")
    print(f"  G_metacell_p range : [{G_metacell_p.min():.4f}, {G_metacell_p.max():.4f}]")
    assert 0.0 <= G_metacell_p.min() and G_metacell_p.max() <= 1.0, \
        "G_metacell_p out of [0,1] bounds — something is wrong"

    np.save(os.path.join(OUT_DIR, "G_metacell_p.npy"), G_metacell_p.astype(np.float32))
    print(f"  Saved G_metacell_p.npy")
    return G_metacell_p, X_bin


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9 — P_constraints_metacell / P_constraints_cells  (REAL solver P, TRANSPOSED)
# ═══════════════════════════════════════════════════════════════════════════════
def step9_build_P_constraints(P_cells, P_meta, N_types, n_cells, n_meta):
    print("\n" + "=" * 70)
    print("STEP 9 — P_constraints_*  (REAL solver input, orientation: N_types x n_X)")
    print("  Your pipeline stores P as (n_X x N_types). The actual solver code")
    print("  (confirmed in preprocess.py / loaders.py) expects the transpose:")
    print("  binary support masks oriented (N_types x n_X).")
    print("=" * 70)

    P_constraints_cells = sp.csr_matrix((P_cells > 0).astype(np.float64).T)   # (N_types x n_cells)
    P_constraints_metacell = (P_meta > 0).astype(np.float64).T                # (N_types x n_meta)

    print(f"  P_constraints_cells     : {P_constraints_cells.shape}  (N_types x n_cells)")
    print(f"  P_constraints_metacell  : {P_constraints_metacell.shape}  (N_types x n_meta)")
    assert P_constraints_cells.shape == (N_types, n_cells)
    assert P_constraints_metacell.shape == (N_types, n_meta)

    mean_allowed = P_constraints_metacell.sum(axis=0).mean()
    print(f"  Mean allowed types per metacell : {mean_allowed:.2f}")

    sp.save_npz(os.path.join(OUT_DIR, "P_constraints_cells.npz"), P_constraints_cells)
    sp.save_npz(os.path.join(OUT_DIR, "P_constraints_metacell.npz"), sp.csr_matrix(P_constraints_metacell))
    print(f"  Saved P_constraints_cells.npz + P_constraints_metacell.npz")
    return P_constraints_cells, P_constraints_metacell


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 10 — C_mask  (observed-entries mask)
# ═══════════════════════════════════════════════════════════════════════════════
def step10_build_C_mask(C, N):
    print("\n" + "=" * 70)
    print("STEP 10 — C_mask  (observed-entries mask)")
    print("  All-ones: the FlyWire connectome is fully measured for these")
    print("  741 types. Edit this if some type pairs should be 'unmeasured'.")
    print("=" * 70)
    C_mask = np.ones((N, N), dtype=np.float64)
    np.save(os.path.join(OUT_DIR, "C_mask.npy"), C_mask)
    print(f"  Saved C_mask.npy  shape={C_mask.shape}")
    return C_mask


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
def final_summary(C, G_cells, P_cells, B_cells, metacell_index,
                   G_meta, P_meta, G_metacell_p,
                   P_constraints_cells, P_constraints_metacell, C_mask):
    print("\n" + "=" * 70)
    print("ALL MATRICES BUILT — ready for the real ConnectionMiner solver")
    print("=" * 70)
    print("  Your pipeline's own representation (diagnostics / visualization):")
    print(f"    C                       {C.shape}")
    print(f"    G_cells                 {G_cells.shape}   z-scored")
    print(f"    P_cells                 {P_cells.shape}   row-stochastic")
    print(f"    B_cells                 {B_cells.shape}   reference only, unused by solver")
    print(f"    G_meta                  {G_meta.shape}   z-scored")
    print(f"    P_meta                  {P_meta.shape}   row-stochastic")
    print()
    print("  REAL solver inputs (cm_visual.solver.cm_solve expects exactly these):")
    print(f"    G_metacell_p            {G_metacell_p.shape}   [0,1] detection probability")
    print(f"    P_constraints_cells     {P_constraints_cells.shape}   binary, N_types x n_cells")
    print(f"    P_constraints_metacell  {P_constraints_metacell.shape}   binary, N_types x n_meta")
    print(f"    C_mask                  {C_mask.shape}   all-ones (fully observed)")
    print()
    print(f"  Metacell tiers:")
    for tier in ["named", "numeric", "orphan"]:
        sub = metacell_index[metacell_index["tier"] == tier]
        if len(sub):
            print(f"    {tier:<8} {len(sub):>6,} metacells   {sub['n_cells'].sum():>8,} cells")
    print()
    print(f"  All files saved in {OUT_DIR}/")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    adata = step1_load()
    PCA   = adata.obsm[detect_pca_key(adata)].astype(np.float32)

    C, type_index, type2idx, N = step2_build_C()

    G_cells, gene_index, hvg_idx, layer_used = step3_build_G_cells(adata)
    # C, type_index, type2idx only needed in step 4 — free after
    P_cells, cell_index = step4_build_P_cells(adata, G_cells, type_index, type2idx, N)
    del type2idx;  gc.collect()

    B_cells = step5_build_B_cells(G_cells)
    # B_cells is saved; keep only for final_summary (tiny 36 MB, fine to keep)

    metacell_index, cell_to_metacell = step6_build_metacells(adata, cell_index, P_cells, PCA)
    # PCA no longer needed after step 6
    del PCA;  gc.collect()
    print(f"  [mem] Released PCA ({adata.n_obs * 200 * 4 / 1e6:.0f} MB)")

    G_meta, P_meta, meta_int, meta_sizes = step7_build_P_meta_G_meta(
        G_cells, P_cells, cell_to_metacell, metacell_index)
    print(f"  [mem] Released G_cells (1.3 GB)")

    n_meta = len(metacell_index)
    n_cells = adata.n_obs

    G_metacell_p, X_bin = step8_build_G_metacell_p(
        adata, hvg_idx, n_meta, meta_int, meta_sizes)

    P_constraints_cells, P_constraints_metacell = step9_build_P_constraints(
        P_cells, P_meta, N, n_cells, n_meta)

    C_mask = step10_build_C_mask(C, N)

    final_summary(C, G_cells, P_cells, B_cells, metacell_index,
                  G_meta, P_meta, G_metacell_p,
                  P_constraints_cells, P_constraints_metacell, C_mask)