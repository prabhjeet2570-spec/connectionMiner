"""
viz_matrices.py
================
Interactive HTML visualization of all matrices built by build_all_matrices.py.
Saves output/viz_matrices.html — open in your browser.
"""

import os, gc, re
import numpy as np
import pandas as pd
import scipy.sparse as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly import io as pio

OUT_DIR  = "../output"
TINY     = 50
MEDIUM   = 100


def _hist_binned(data, nbins=60):
    lo, hi = float(data.min()), float(data.max())
    if hi - lo < 1e-12: hi = lo + 1.0
    counts, edges = np.histogram(data.ravel(), bins=nbins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2
    w = 0.9 * (centers[1] - centers[0]) if len(centers) > 1 else 1
    return go.Bar(x=centers, y=counts, width=w)

def _fig(title, traces):
    n = len(traces)
    if n == 0: return None
    fig = make_subplots(rows=n, cols=1,
                        subplot_titles=[t[0] for t in traces],
                        vertical_spacing=0.15 / max(n, 1) * 3 if n > 1 else 0.05)
    for i, (_, trace) in enumerate(traces):
        fig.add_trace(trace, row=i+1, col=1)
    fig.update_layout(title_text=title, height=max(280, n * 300),
                      showlegend=False, template="plotly_white",
                      margin=dict(l=50, r=20, t=40, b=40))
    return fig

def _sample_vals(arr, n=200_000, seed=0):
    rng = np.random.RandomState(seed)
    total = arr.size
    idx = rng.randint(0, total, min(total, n))
    return arr.ravel()[idx]

def _hm(z, colorscale="Viridis", zmin=None, zmax=None):
    return go.Heatmap(z=z, colorscale=colorscale, zmin=zmin, zmax=zmax,
                      hovertemplate="%{z:.3f}<extra></extra>")


def main():
    print("Loading matrices from", OUT_DIR)
    fig_list = []

    # ── C ────────────────────────────────────────────────────────────────
    print("  C ...")
    path = os.path.join(OUT_DIR, "C_matrix.npz")
    if os.path.exists(path):
        C = sp.load_npz(path)
        Cd = C.toarray()
        nnz_r = np.diff(C.indptr).astype(np.float64)
        fig_list.append(_fig("Step 2 — C (741\u00d7741 binary connectome)", [
            ("Spy (first {}×{})".format(TINY, TINY),
             _hm(Cd[:TINY, :TINY], [[0,"#eee"],[1,"#1f77b4"]])),
            ("Out-degree (row sums)", go.Bar(y=nnz_r)),
            ("In-degree (col sums)", go.Bar(y=np.array(C.sum(axis=0)).ravel())),
        ]))

    # ── G_cells ──────────────────────────────────────────────────────────
    print("  G_cells (mmap sampled) ...")
    path = os.path.join(OUT_DIR, "G_matrix.npy")
    if os.path.exists(path):
        arr = np.load(path, mmap_mode="r")
        corner = arr[:TINY, :TINY].copy()
        vals = _sample_vals(arr, 200_000)
        del arr; gc.collect()
        fig_list.append(_fig("Step 3 — G_cells (n_cells \u00d7 3000, z-scored)", [
            ("Corner {}×{}".format(TINY, TINY), _hm(corner, "RdBu_r")),
            ("Expression histogram (200k random values)", _hist_binned(vals)),
        ]))

    # ── P_cells ──────────────────────────────────────────────────────────
    print("  P_cells ...")
    path = os.path.join(OUT_DIR, "P_matrix.npz")
    if os.path.exists(path):
        M = sp.load_npz(path)
        nnz_r = np.diff(M.indptr).astype(np.float64)
        M_ss = M[:TINY, :TINY].toarray()
        fig_list.append(_fig("Step 4 — P_cells (n_cells \u00d7 741, row-stochastic)", [
            ("Spy (first {}×{})".format(TINY, TINY),
             _hm(M_ss, [[0,"#eee"],[1,"#1f77b4"]])),
            ("Nonzero types per cell", _hist_binned(nnz_r)),
            ("Row sums (should be \u2248 1)", _hist_binned(np.array(M.sum(axis=1)).ravel())),
        ]))

    # ── B_cells ──────────────────────────────────────────────────────────
    print("  B_cells ...")
    path = os.path.join(OUT_DIR, "B_matrix.npy")
    if os.path.exists(path):
        B = np.load(path)
        fig_list.append(_fig("Step 5 — B_cells (g\u00d7g gene covariance)", [
            ("Corner {}×{}".format(MEDIUM, MEDIUM), _hm(B[:MEDIUM, :MEDIUM], "RdBu_r")),
            ("Covariance histogram", _hist_binned(_sample_vals(B, 200_000))),
        ]))
        del B; gc.collect()

    # ── Metacells ────────────────────────────────────────────────────────
    print("  Metacells ...")
    path = os.path.join(OUT_DIR, "metacell_index.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        tiers = df["tier"].value_counts()
        fig_list.append(_fig("Step 6 — Metacells (tiered clustering)", [
            ("Tier counts", go.Bar(x=tiers.index.tolist(), y=tiers.values.tolist())),
            ("Cells per metacell", _hist_binned(df["n_cells"].values)),
        ]))

    # ── G_meta ───────────────────────────────────────────────────────────
    print("  G_meta ...")
    path = os.path.join(OUT_DIR, "G_meta.npy")
    if os.path.exists(path):
        M = np.load(path)
        fig_list.append(_fig("Step 7 — G_meta (n_meta \u00d7 3000, z-scored)", [
            ("Corner {}×{}".format(TINY, TINY), _hm(M[:TINY, :TINY], "RdBu_r")),
            ("Expression histogram", _hist_binned(_sample_vals(M, 200_000))),
        ]))
        del M; gc.collect()

    # ── P_meta ───────────────────────────────────────────────────────────
    print("  P_meta ...")
    path = os.path.join(OUT_DIR, "P_meta.npz")
    if os.path.exists(path):
        M = sp.load_npz(path).toarray()
        fig_list.append(_fig("Step 7 — P_meta (n_meta \u00d7 741, row-stochastic)", [
            ("Corner {}×{}".format(TINY, TINY), _hm(M[:TINY, :TINY])),
            ("Row sums", _hist_binned(M.sum(axis=1))),
        ]))

    # ── G_metacell_p ─────────────────────────────────────────────────────
    print("  G_metacell_p ...")
    path = os.path.join(OUT_DIR, "G_metacell_p.npy")
    if os.path.exists(path):
        M = np.load(path)
        fig_list.append(_fig("Step 8 — G_metacell_p (n_meta \u00d7 g, [0,1])", [
            ("Corner {}×{}".format(TINY, TINY), _hm(M[:TINY, :TINY], "Viridis", 0, 1)),
            ("Detection prob. histogram", _hist_binned(_sample_vals(M, 200_000))),
            ("Per-gene mean detection prob.", go.Bar(y=M.mean(axis=0))),
        ]))
        del M; gc.collect()

    # ── P_constraints_cells ──────────────────────────────────────────────
    print("  P_constraints_cells ...")
    path = os.path.join(OUT_DIR, "P_constraints_cells.npz")
    if os.path.exists(path):
        M = sp.load_npz(path)
        nnz_r = np.diff(M.indptr).astype(np.float64)
        fig_list.append(_fig("Step 9 — P_constraints_cells (741 \u00d7 n_cells, binary)", [
            ("Nonzero cells per type", _hist_binned(nnz_r)),
        ]))

    # ── P_constraints_metacell ───────────────────────────────────────────
    print("  P_constraints_metacell ...")
    path = os.path.join(OUT_DIR, "P_constraints_metacell.npz")
    if os.path.exists(path):
        M = sp.load_npz(path).toarray()
        fig_list.append(_fig("Step 9 — P_constraints_metacell (741 \u00d7 n_meta, binary)", [
            ("Corner {}×{}".format(TINY, TINY),
             _hm(M[:TINY, :TINY], [[0,"#eee"],[1,"#1f77b4"]])),
            ("Allowed types per metacell", _hist_binned(M.sum(axis=0))),
        ]))

    # ── C_mask ───────────────────────────────────────────────────────────
    print("  C_mask ...")
    path = os.path.join(OUT_DIR, "C_mask.npy")
    if os.path.exists(path):
        M = np.load(path)
        fig_list.append(_fig("Step 10 — C_mask (741\u00d7741, all-ones)", [
            ("Corner 30\u00d730", _hm(M[:30, :30], "Viridis", 0, 1)),
            ("Value histogram", _hist_binned(_sample_vals(M, 5000))),
        ]))

    # ── assemble full HTML ───────────────────────────────────────────────
    print("\nAssembling HTML ...")
    fig_list = [f for f in fig_list if f is not None]

    rows_info = [
        ("C", "741\u00d7741", "Binary connectome"),
        ("G_cells", "n_cells\u00d73000", "Z-scored HVG expression"),
        ("P_cells", "n_cells\u00d7741", "Row-stochastic type constraints"),
        ("B_cells", "3000\u00d73000", "Gene covariance (diagnostic)"),
        ("G_meta", "n_meta\u00d73000", "Metacell z-scored expression"),
        ("P_meta", "n_meta\u00d7741", "Metacell row-stochastic constraints"),
        ("G_metacell_p", "n_meta\u00d7g", "[0,1] detection probability"),
        ("P_constraints_cells", "741\u00d7n_cells", "Binary support mask"),
        ("P_constraints_metacell", "741\u00d7n_meta", "Binary support mask"),
        ("C_mask", "741\u00d7741", "Observed-entries mask"),
    ]

    table_html = (
        "<table><tr><th>#</th><th>Matrix</th><th>Shape</th><th>Description</th></tr>" +
        "".join(f"<tr><td>{i}</td><td><b>{n}</b></td><td>{s}</td><td>{d}</td></tr>"
                for i, (n, s, d) in enumerate(rows_info, 1)) +
        "</table>"
    )

    plot_divs = [fig.to_html(full_html=False, include_plotlyjs=False) for fig in fig_list]

    # Generate plotly.js inline library from a dummy figure
    dummy = go.Figure()
    dummy.update_layout(title="", height=50)
    dummy_html = pio.to_html(dummy, full_html=True, include_plotlyjs="inline")

    # Extract plotly.js library + config, drop the newPlot call for dummy fig
    scripts = []
    for m in re.finditer(r'<script[^>]*>.*?</script>', dummy_html, re.DOTALL):
        tag = m.group(0)
        if 'plotly.js' in tag:
            scripts.append(tag)          # the minified library (~4.5 MB)
        elif 'PlotlyConfig' in tag:
            scripts.insert(0, tag)       # config goes first

    style = """
    <style>
    body{font-family:system-ui,sans-serif;margin:20px;background:#fafafa;color:#222}
    h1{color:#1a1a2e}
    .card{background:#fff;border-radius:8px;padding:16px;margin:20px 0;
          box-shadow:0 1px 3px rgba(0,0,0,.1)}
    table{border-collapse:collapse;width:100%;font-size:14px}
    th,td{text-align:left;padding:8px 12px;border-bottom:1px solid #eee}
    th{background:#1f77b4;color:#fff}
    </style>
    """

    html = (
        "<!DOCTYPE html>\n<html><head><meta charset='utf-8'>\n"
        "<title>ConnectionMiner \u2014 Matrix Visualization</title>\n"
        + "\n".join(scripts) + "\n" + style + "\n</head><body>\n"
        "<h1>ConnectionMiner \u2014 All Matrices</h1>\n"
        "<div class='card'>" + table_html + "</div>\n"
        + "\n".join(f"<div class='card'>{d}</div>" for d in plot_divs) +
        "\n</body></html>"
    )

    out_path = os.path.join(OUT_DIR, "viz_matrices.html")
    with open(out_path, "w") as f:
        f.write(html)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Saved {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
