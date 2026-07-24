#!/usr/bin/env python3
"""Build an interactive clustering ablation explorer HTML with experiment dropdown + 3-panel UMAP view."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cm_visual.viz_plotly import D30, _discrete_colorscale

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"
ABLATION_ROOT = OUTPUT_DIR / "connectionMiner_ablation" / "clustering_ablation"


def load_shared() -> dict:
    cell_index = pd.read_csv(OUTPUT_DIR / "cell_index.csv")
    type_index = pd.read_csv(OUTPUT_DIR / "type_index.csv")

    import anndata as ad
    adata = ad.read_h5ad(str(REPO_ROOT / "data" / "Adult.h5ad"), backed="r")
    uk = "X_umap" if "X_umap" in adata.obsm else "X_tsne"
    umap_xy = np.asarray(adata.obsm[uk], dtype=float)

    annotations = cell_index["MultiomeAnnotated"].values.astype(str)
    type_names = type_index["type"].values.astype(str)
    type_name_to_idx = {name: i for i, name in enumerate(type_names)}
    unique_ann = np.unique(annotations)
    ann_to_idx = {a: i for i, a in enumerate(unique_ann)}
    ann_colors = []
    for a in annotations:
        if a in ("nan", ""):
            ann_colors.append("#cccccc")
        else:
            tidx = type_name_to_idx.get(a)
            if tidx is not None:
                ann_colors.append(D30[tidx % len(D30)])
            else:
                ann_colors.append(D30[ann_to_idx[a] % len(D30)])

    P_cons = sparse.load_npz(str(OUTPUT_DIR / "P_constraints_cells.npz"))
    n_allowed = np.array(P_cons.sum(axis=0)).flatten().astype(int).tolist()

    rng = np.random.RandomState(42)
    n_cells = len(annotations)
    u0_arr = np.asarray(umap_xy[:, 0])
    u1_arr = np.asarray(umap_xy[:, 1])
    nbins = 40
    u0_min, u0_max = float(u0_arr.min()), float(u0_arr.max())
    u1_min, u1_max = float(u1_arr.min()), float(u1_arr.max())
    u0_bins = np.linspace(u0_min, u0_max + 1e-12, nbins + 1)
    u1_bins = np.linspace(u1_min, u1_max + 1e-12, nbins + 1)
    u0_idx = np.clip(np.searchsorted(u0_bins, u0_arr) - 1, 0, nbins - 1)
    u1_idx = np.clip(np.searchsorted(u1_bins, u1_arr) - 1, 0, nbins - 1)
    cell_bin = u0_idx * nbins + u1_idx
    bin_keys, bin_counts = np.unique(cell_bin, return_counts=True)
    bin_to_cells = {k: [] for k in bin_keys}
    for i, b in enumerate(cell_bin):
        bin_to_cells[b].append(i)
    lod_indices = {0: np.arange(n_cells).tolist(), 1: [], 2: []}
    lod_frac = [0, 0.20, 0.05]
    for level in [1, 2]:
        subset = []
        frac = lod_frac[level]
        for bk in bin_keys:
            cells_in_bin = bin_to_cells[bk]
            keep = max(1, int(len(cells_in_bin) * frac))
            chosen = rng.choice(cells_in_bin, size=keep, replace=False).tolist()
            subset.extend(chosen)
        lod_indices[level] = sorted(subset)
    lod = [lod_indices[0], lod_indices[1], lod_indices[2]]
    ac_lod = [[ann_colors[i] for i in lod[0]],
              [ann_colors[i] for i in lod[1]],
              [ann_colors[i] for i in lod[2]]]

    return {
        "u0": umap_xy[:, 0].tolist(),
        "u1": umap_xy[:, 1].tolist(),
        "bc": cell_index["cell_barcode"].values.astype(str).tolist(),
        "ann": annotations.tolist(),
        "tier": cell_index["cell_type"].values.tolist(),
        "cl": cell_index["MultiomeNN"].values.astype(str).tolist(),
        "nal": n_allowed,
        "tn": type_index["type"].values.astype(str).tolist(),
        "fam": type_index["family"].values.astype(str).tolist(),
        "sub": type_index["subsystem"].values.astype(str).tolist(),
        "cat": type_index["category"].values.astype(str).tolist(),
        "ac": ann_colors,
        "lod": lod,
        "ac_lod": ac_lod,
    }


def load_experiments() -> list[dict]:
    import glob as gb
    exp_dirs = sorted(gb.glob(str(ABLATION_ROOT / "exp_*")))
    experiments = []

    for d in exp_dirs:
        exp_path = Path(d)
        exp_name = exp_path.name
        parts = exp_name.split("_")
        idx = int(parts[1])

        stats_path = exp_path / "run_stats.json"
        stats = {}
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)

        cm_path = exp_path / "cell_to_metacell_solver.npy"
        if not cm_path.exists():
            continue
        cell_to_meta = np.load(cm_path).astype(np.int32)

        P_path = exp_path / "P_refined.npz"
        if not P_path.exists():
            continue
        P = sparse.load_npz(str(P_path)).toarray()
        n_meta = P.shape[1]

        inferred_type = np.argmax(P, axis=0).astype(np.int16).tolist()
        P_max = np.max(P, axis=0).tolist()

        eps = 1e-30
        P_norm = P / np.maximum(np.sum(P, axis=0, keepdims=True), eps)
        P_ent = -np.sum(P_norm * np.log2(np.maximum(P_norm, eps)), axis=0)
        max_ent = np.log2(P.shape[0]) if P.shape[0] > 1 else 1.0
        entropy = (P_ent / max_ent).tolist()

        n_genes = stats.get("n_genes_solver", 0)
        pearson_r = stats.get("pearson_r", 0)
        final_loss = stats.get("final_total_loss", 0)

        meta_colors = _discrete_colorscale(n_meta)
        meta_sizes = np.bincount(cell_to_meta, minlength=n_meta).astype(int).tolist()

        experiments.append({
            "i": idx,
            "n": exp_name.replace(f"exp_{idx:02d}_", ""),
            "g": n_genes,
            "nm": n_meta,
            "cm": cell_to_meta.tolist(),
            "it": inferred_type,
            "pm": P_max,
            "en": entropy,
            "mc": meta_colors,
            "ms": meta_sizes,
            "r": round(pearson_r, 4),
            "loss": round(final_loss, 1),
        })

    experiments.sort(key=lambda e: e["i"])
    return experiments


def build_html(shared: dict, experiments: list[dict]) -> str:
    shared_json = json.dumps(shared, separators=(",", ":"))
    exps_json = json.dumps(experiments, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8" /></head>
<body>
<script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>

<div style="font-family:'Segoe UI',Arial,sans-serif;margin:10px 20px;display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
  <h2 style="margin:0;font-size:18px;color:#9467bd;">Clustering Ablation Explorer</h2>
  <label style="font-weight:600;">Experiment:
    <select id="exp-select" style="font-size:14px;padding:4px 8px;min-width:200px;"></select>
  </label>
  <span id="exp-stats" style="color:#555;font-size:13px;"></span>
</div>

<div id="plot-container" style="position:relative;">
  <div id="three-panel" class="plotly-graph-div" style="height:500px;width:1500px;"></div>
</div>

<div id="cell-info-panel">
  <div id="cell-info-header">Cell Information</div>
  <div id="cell-info-body">
    <div id="cell-info-content"><div class="no-selection">Hover over a cell to see details</div></div>
  </div>
</div>

<style>
#cell-info-panel {{
  position:fixed;top:60px;right:10px;width:280px;
  background:white;border:1px solid #ddd;border-radius:8px;padding:0;
  font-family:'Segoe UI',Arial,sans-serif;font-size:13px;
  box-shadow:0 4px 12px rgba(0,0,0,0.15);z-index:1000;max-height:80vh;overflow-y:auto;
}}
#cell-info-header {{padding:10px 15px;border-radius:8px 8px 0 0;color:white;font-weight:700;font-size:13px;letter-spacing:0.3px;}}
#cell-info-body {{padding:10px 15px;}}
#cell-info-body h3 {{margin:0 0 6px 0;font-size:13px;border-bottom:1px solid #eee;padding-bottom:3px;color:#333;}}
.info-row {{margin:2px 0;display:flex;}}
.info-label {{font-weight:600;width:95px;color:#555;flex-shrink:0;font-size:12px;}}
.info-value {{flex:1;color:#222;word-break:break-all;font-size:12px;}}
.no-selection {{color:#999;font-style:italic;text-align:center;padding:20px 0;}}
.key-info {{margin:4px 0 8px 0;padding:6px 8px;background:#f7f7f7;border-radius:4px;font-size:12px;}}
.key-info .key-label {{font-weight:600;color:#333;}}
.key-info .key-value {{color:#9467bd;font-weight:700;}}
</style>

<script id="clustering-data" type="application/json">{{"shared":{shared_json},"exps":{exps_json}}}</script>

<script>
var DATA = JSON.parse(document.getElementById('clustering-data').textContent);
var S = DATA.shared;
var EXPS = DATA.exps;
var D30 = {json.dumps(D30, separators=(",",":"))};

var COLORS = {{}};
(function() {{
  for (var i = 0; i < S.tn.length; i++) {{
    COLORS[i] = D30[i % D30.length];
  }}
}})();

var mainIdx = [0, 1, 2];
var _syncing = false;

var panelMeta = [
  {{ name: 'Original Annotations', color: '#2ca02c', keyLabel: 'Annotation', keyIdx: 1 }},
  {{ name: 'Metacells', color: '#9467bd', keyLabel: 'Metacell', keyIdx: 3 }},
  {{ name: 'Inferred Type', color: '#d62728', keyLabel: 'Inferred Type', keyIdx: 4 }},
];

var CD_KEYS = [
  'Barcode','Annotation','Tier','Metacell','Inferred Type','Confidence','Norm. Entropy','Metacell Size',
  'MultiomeNN Cluster','# Allowed Types','Family','Subsystem','Category','UMAP1','UMAP2'
];

function buildCustomdata(exp, cellIdx) {{
  var m = exp.cm[cellIdx];
  var t = exp.it[m];
  return [
    S.bc[cellIdx], S.ann[cellIdx], S.tier[cellIdx], String(m),
    S.tn[t], String(exp.pm[m]), String(exp.en[m]), String(exp.ms[m]),
    S.cl[cellIdx], String(S.nal[cellIdx]),
    S.fam[t], S.sub[t], S.cat[t],
    String(S.u0[cellIdx].toFixed(4)), String(S.u1[cellIdx].toFixed(4))
  ];
}}

function renderInfo(exp, cellIdx, cn) {{
  var pm = panelMeta[cn];
  var m = exp.cm[cellIdx];
  var cd = buildCustomdata(exp, cellIdx);

  document.getElementById('cell-info-header').style.background = pm.color;
  document.getElementById('cell-info-header').textContent = pm.name;

  var keyVal = cd[pm.keyIdx] || 'N/A';
  var keyExtra = '';
  if (cn === 1) keyExtra = ' (Size: ' + exp.ms[m] + ' cells)';
  if (cn === 2) keyExtra = ' (confidence: ' + cd[5] + ', entropy: ' + cd[6] + ')';

  var sections = [
    {{ title: 'Identity', rows: [['Barcode',0],['MultiomeNN Cluster',8],['Constraint Tier',2],['Allowed Types',9]] }},
    {{ title: 'Type Annotation', rows: [['Annotated Type',1],['Family',10],['Subsystem',11],['Category',12]] }},
    {{ title: 'Metacell', rows: [['Metacell ID',3],['Size (cells)',7]] }},
    {{ title: 'ConnectionMiner Result', rows: [['Inferred Type',4],['Confidence',5],['Norm. Entropy',6]] }},
    {{ title: 'UMAP Position', rows: [['UMAP1',13],['UMAP2',14]] }},
  ];

  var html = '<div class="key-info"><span class="key-label">' + pm.keyLabel + ': </span><span class="key-value">' + keyVal + '</span>' + keyExtra + '</div>';
  for (var s = 0; s < sections.length; s++) {{
    html += '<h3>' + sections[s].title + '</h3>';
    for (var r = 0; r < sections[s].rows.length; r++) {{
      var row = sections[s].rows[r];
      html += '<div class="info-row"><span class="info-label">' + row[0] + ':</span><span class="info-value">' + (cd[row[1]] || '') + '</span></div>';
    }}
  }}
  document.getElementById('cell-info-content').innerHTML = html;
}}

function cellInferredType(exp, i) {{ return exp.it[exp.cm[i]]; }}
function cellMetaColor(exp, i) {{ return exp.mc[exp.cm[i]]; }}
function cellInfColor(exp, i) {{ return COLORS[cellInferredType(exp, i)]; }}

var currentExp = null;
var currentLod = 1;
var n_full = S.u0.length;
var _lodTimer = null;
var _hoverActive = false;
var u0min = Infinity, u0max = -Infinity;
for (var _i = 0; _i < n_full; _i++) {{
  if (S.u0[_i] < u0min) u0min = S.u0[_i];
  if (S.u0[_i] > u0max) u0max = S.u0[_i];
}}
var u0span = u0max - u0min;

function realCellIdx(lodIdx) {{
  return currentLod === 0 ? lodIdx : S.lod[currentLod][lodIdx];
}}

function buildLodData(level, exp) {{
  var lod = S.lod[level];
  var m = lod.length;
  var x = new Array(m);
  var y = new Array(m);
  var mcArr = new Array(m);
  var icArr = new Array(m);
  var acArr = S.ac_lod[level];
  for (var i = 0; i < m; i++) {{
    var j = lod[i];
    x[i] = S.u0[j];
    y[i] = S.u1[j];
    if (exp) {{
      var mi = exp.cm[j];
      mcArr[i] = exp.mc[mi];
      icArr[i] = COLORS[exp.it[mi]];
    }}
  }}
  return {{ x: x, y: y, ac: acArr, mc: mcArr, ic: icArr }};
}}

function switchLod(level) {{
  if (level === currentLod || !currentExp || _hoverActive) return;
  currentLod = level;
  var lod = S.lod[level];
  var m = lod.length;
  var x = new Array(m);
  var y = new Array(m);
  var mcArr = new Array(m);
  var icArr = new Array(m);
  var acArr = S.ac_lod[level];
  for (var i = 0; i < m; i++) {{
    var j = lod[i];
    x[i] = S.u0[j];
    y[i] = S.u1[j];
    var mi = currentExp.cm[j];
    mcArr[i] = currentExp.mc[mi];
    icArr[i] = COLORS[currentExp.it[mi]];
  }}
  Plotly.restyle('three-panel', {{
    'x': [x, x, x],
    'y': [y, y, y],
    'marker.color': [acArr, mcArr, icArr]
  }}, [0, 1, 2]);
  Plotly.restyle('three-panel', {{
    'x': [[null],[null],[null]],
    'y': [[null],[null],[null]]
  }}, [3, 4, 5]);
}}

function renderExperiment(expIdx) {{
  var exp = EXPS[expIdx];
  currentExp = exp;
  document.getElementById('exp-stats').textContent =
    exp.n + ' | ' + exp.g + ' genes | ' + exp.nm + ' metacells | r=' + exp.r + ' | loss=' + exp.loss;

  var lod = S.lod[currentLod];
  var m = lod.length;
  var mcArr = new Array(m);
  var icArr = new Array(m);
  for (var i = 0; i < m; i++) {{
    var j = lod[i];
    var mi = exp.cm[j];
    mcArr[i] = exp.mc[mi];
    icArr[i] = COLORS[exp.it[mi]];
  }}

  Plotly.restyle('three-panel', {{
    'marker.color': [S.ac_lod[currentLod], mcArr, icArr]
  }}, [0, 1, 2]);
}}

function initPlot() {{
  var lodData = buildLodData(1, EXPS[0]);
  currentLod = 1;

  var trace1 = {{
    x: lodData.x, y: lodData.y, mode: 'markers', type: 'scattergl',
    marker: {{ size: 2, color: lodData.ac, opacity: 0.7 }},
    customdata: new Array(lodData.x.length),
    hovertemplate: '<b>%{{customdata[1]}}</b><br>Barcode: %{{customdata[0]}}<br>Tier: %{{customdata[2]}}<br>Metacell: %{{customdata[3]}}<extra></extra>',
    showlegend: false, name: 'annotation',
    xaxis: 'x', yaxis: 'y',
  }};

  var trace2 = {{
    x: lodData.x, y: lodData.y, mode: 'markers', type: 'scattergl',
    marker: {{ size: 2, color: lodData.mc, opacity: 0.7 }},
    customdata: new Array(lodData.x.length),
    hovertemplate: '<b>Metacell %{{customdata[3]}}</b><br>Annotation: %{{customdata[1]}}<br>Tier: %{{customdata[2]}}<br>Barcode: %{{customdata[0]}}<extra></extra>',
    showlegend: false, name: 'metacell',
    xaxis: 'x2', yaxis: 'y2',
  }};

  var trace3 = {{
    x: lodData.x, y: lodData.y, mode: 'markers', type: 'scattergl',
    marker: {{ size: 2, color: lodData.ic, opacity: 0.7 }},
    customdata: new Array(lodData.x.length),
    hovertemplate: '<b>%{{customdata[4]}}</b><br>Confidence: %{{customdata[5]}}<br>Annotation: %{{customdata[1]}}<br>Barcode: %{{customdata[0]}}<extra></extra>',
    showlegend: false, name: 'inferred',
    xaxis: 'x3', yaxis: 'y3',
  }};

  var hlTrace = function(axis) {{ return {{
    x: [null], y: [null], mode: 'markers', type: 'scattergl',
    marker: {{ size: 14, color: 'rgba(255,50,50,0.9)', line: {{width:2,color:'white'}}, symbol: 'circle-open' }},
    showlegend: false, hoverinfo: 'none', name: 'highlight', xaxis: axis.x, yaxis: axis.y,
  }}; }};

  var layout = {{
    title: {{ text: 'Clustering Ablation Explorer &mdash; 3-Stage Pipeline Overview' }},
    width: 1500, height: 500, hovermode: 'closest',
    margin: {{ r: 320, t: 40 }},
    grid: {{ rows: 1, columns: 3, pattern: 'independent' }},
    xaxis: {{ title: 'UMAP1', domain: [0, 0.30] }},
    yaxis: {{ title: 'UMAP2', domain: [0, 1] }},
    xaxis2: {{ title: 'UMAP1', domain: [0.33, 0.63] }},
    yaxis2: {{ domain: [0, 1] }},
    xaxis3: {{ title: 'UMAP1', domain: [0.66, 0.96] }},
    yaxis3: {{ domain: [0, 1] }},
    annotations: [
      {{ text: 'Original Cell Type Annotations', x: 0.15, y: 1.02, showarrow: false, xref: 'paper', yref: 'paper', font: {{size:13}} }},
      {{ text: 'Cells Colored by Metacell', x: 0.48, y: 1.02, showarrow: false, xref: 'paper', yref: 'paper', font: {{size:13}} }},
      {{ text: 'After ConnectionMiner (Inferred Type)', x: 0.81, y: 1.02, showarrow: false, xref: 'paper', yref: 'paper', font: {{size:13}} }},
    ]
  }};

  var traces = [trace1, trace2, trace3,
    hlTrace({{x:'x',y:'y'}}), hlTrace({{x:'x2',y:'y2'}}), hlTrace({{x:'x3',y:'y3'}})];

  Plotly.newPlot('three-panel', traces, layout);

  var plotDiv = document.getElementById('three-panel');
  var _lastHoveredCell = -1;

  plotDiv.on('plotly_hover', function(data) {{
    try {{
      if (!data.points || data.points.length === 0) return;
      var pt = data.points[0];
      var idx = pt.pointIndex;
      var realIdx = realCellIdx(idx);

      if (realIdx === _lastHoveredCell || realIdx < 0 || realIdx >= n_full) return;
      _lastHoveredCell = realIdx;
      _hoverActive = true;

      var x = S.u0[realIdx];
      var y = S.u1[realIdx];
      Plotly.restyle('three-panel', {{ 'x': [[x],[x],[x]], 'y': [[y],[y],[y]] }}, [3, 4, 5]);

      var cn = pt.curveNumber;
      if (cn < 0 || cn > 2) cn = 0;
      if (currentExp) renderInfo(currentExp, realIdx, cn);
    }} catch(e) {{ /* ignore hover errors */ }}
  }});

  plotDiv.on('plotly_unhover', function() {{
    _lastHoveredCell = -1;
    _hoverActive = false;
    Plotly.restyle('three-panel', {{ 'x': [[null],[null],[null]], 'y': [[null],[null],[null]] }}, [3, 4, 5]);
    document.getElementById('cell-info-header').style.background = '#9467bd';
    document.getElementById('cell-info-header').textContent = 'Cell Information';
    document.getElementById('cell-info-content').innerHTML =
      '<div class="no-selection">Hover over a cell to see details</div>';
  }});

  function getRange(ed, key) {{
    if (ed[key] && Array.isArray(ed[key].range) && ed[key].range.length === 2) return ed[key].range;
    var r0 = ed[key + '.range[0]'], r1 = ed[key + '.range[1]'];
    if (r0 !== undefined && r1 !== undefined) return [r0, r1];
    return null;
  }}

  var allAxes = ['xaxis','xaxis2','xaxis3','yaxis','yaxis2','yaxis3'];

  plotDiv.on('plotly_relayout', function(ed) {{
    if (_syncing) return;
    if (!ed || Object.keys(ed).length === 0) return;
    var update = {{}};
    for (var a = 0; a < allAxes.length; a++) {{
      var key = allAxes[a];
      var range = getRange(ed, key);
      if (!range) continue;
      var type = key.replace(/[0-9]/g, '');
      for (var b = 0; b < allAxes.length; b++) {{
        var target = allAxes[b];
        if (target.startsWith(type) && target !== key) {{
          update[target + '.range[0]'] = range[0];
          update[target + '.range[1]'] = range[1];
        }}
      }}
    }}
    if (Object.keys(update).length > 0) {{
      _syncing = true;
      Plotly.relayout('three-panel', update);
      setTimeout(function() {{ _syncing = false; }}, 200);
    }}
    if (_lodTimer) clearTimeout(_lodTimer);
    _lodTimer = setTimeout(function() {{
      if (!plotDiv._fullLayout || !plotDiv._fullLayout.xaxis) return;
      var xr = plotDiv._fullLayout.xaxis.range;
      if (!xr) return;
      var span = xr[1] - xr[0];
      var fraction = span / u0span;
      var targetLod;
      if (fraction < 0.3) targetLod = 0;
      else if (fraction < 0.7) targetLod = 1;
      else targetLod = 2;
      switchLod(targetLod);
    }}, 300);
  }});

  var sel = document.getElementById('exp-select');
  for (var e = 0; e < EXPS.length; e++) {{
    var opt = document.createElement('option');
    opt.value = e;
    opt.textContent = EXPS[e].n + ' (' + EXPS[e].g + ' genes)';
    sel.appendChild(opt);
  }}
  sel.addEventListener('change', function() {{
    renderExperiment(parseInt(this.value));
  }});
  renderExperiment(0);
}}

document.addEventListener('DOMContentLoaded', initPlot);
</script>
</body>
</html>"""


def main() -> None:
    print("Loading shared data ...")
    shared = load_shared()
    print(f"  {len(shared['bc']):,} cells, {len(shared['tn'])} types")

    print("Loading experiment data ...")
    experiments = load_experiments()
    print(f"  {len(experiments)} experiments loaded")

    print("Generating HTML ...")
    html = build_html(shared, experiments)

    out_path = ABLATION_ROOT / "viz_clustering_explorer.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
