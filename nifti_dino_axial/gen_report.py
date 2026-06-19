#!/usr/bin/env python3
"""Génère un rapport HTML standalone (Chart.js via CDN) des runs MI300A.
Gère precision fp32/bf16 et lignes OOM. Re-exécutable après chaque ajout au CSV."""
import csv
import json

def _try_int(v):
    try: return int(v)
    except (ValueError, TypeError): return None

def _try_float(v):
    try: return float(v)
    except (ValueError, TypeError): return None

rows = []
with open("results.csv") as f:
    for r in csv.DictReader(f, delimiter=";"):
        for k in ("#Nodes", "#GPUs", "BS_local", "BS_global"):
            r[k] = int(r[k])
        r["DDP_eff_pct"] = _try_int(r["DDP_eff_pct"])
        for k in ("it/s", "image/s", "ETA_10M_hours", "ETA_10M_days"):
            r[k] = _try_float(r[k])
        r["is_oom"] = r["image/s"] is None
        rows.append(r)

ok = [r for r in rows if not r["is_oom"]]
oom = [r for r in rows if r["is_oom"]]

def pts(rs, x_key, y_key):
    return [{"x": r[x_key], "y": r[y_key]} for r in sorted(rs, key=lambda r: r[x_key])]

b40_fp32 = [r for r in ok if r["BS_local"] == 40 and r["precision"] == "fp32"]
b40_bf16 = [r for r in ok if r["BS_local"] == 40 and r["precision"] == "bf16"]
b10_fp32 = [r for r in ok if r["BS_local"] == 10 and r["precision"] == "fp32"]

xs_b40_fp32 = sorted({r["#GPUs"] for r in b40_fp32})
xs_b40_bf16 = sorted({r["#GPUs"] for r in b40_bf16})
xs_b10_fp32 = sorted({r["#GPUs"] for r in b10_fp32})
max_gpus = max([r["#GPUs"] for r in ok])
x_ticks = sorted({1, 2, 4, 8, 16, 32} | {r["#GPUs"] for r in ok})
x_ticks = [x for x in x_ticks if x <= max_gpus]

throughput_data = {
    "datasets": [
        {"label": "b40 fp32", "data": pts(b40_fp32, "#GPUs", "image/s"),
         "borderColor": "#1f77b4", "backgroundColor": "#1f77b4",
         "tension": 0, "borderWidth": 2.5, "pointRadius": 7, "pointStyle": "circle"},
        {"label": "b40 bf16", "data": pts(b40_bf16, "#GPUs", "image/s"),
         "borderColor": "#d62728", "backgroundColor": "#d62728",
         "tension": 0, "borderWidth": 2.5, "pointRadius": 8, "pointStyle": "rectRot"},
        {"label": "b10 fp32", "data": pts(b10_fp32, "#GPUs", "image/s"),
         "borderColor": "#2ca02c", "backgroundColor": "#2ca02c",
         "tension": 0, "borderWidth": 1.5, "pointRadius": 5, "pointStyle": "triangle"},
    ]
}

bf16_4gpu = sorted([r for r in ok if r["#GPUs"] == 4 and r["precision"] == "bf16"],
                    key=lambda r: r["BS_local"])
oom_4gpu_bf16 = [r for r in oom if r["#GPUs"] == 4 and r["precision"] == "bf16"]

batch_data = {
    "labels": [str(r["BS_local"]) for r in bf16_4gpu] + [str(r["BS_local"]) + " (OOM)" for r in oom_4gpu_bf16],
    "datasets": [{
        "label": "img/s (bf16, 1n × 4 APUs)",
        "data": [r["image/s"] for r in bf16_4gpu] + [0] * len(oom_4gpu_bf16),
        "backgroundColor": ["#d62728" if r["BS_local"] == 40 else "#ff9896" for r in bf16_4gpu]
                           + ["#666"] * len(oom_4gpu_bf16),
        "borderColor": "#000", "borderWidth": 1,
    }]
}

eta_sorted = sorted(ok, key=lambda r: r["ETA_10M_hours"])
eta_labels = [f'{r["#Nodes"]}n×{r["#GPUs"]//r["#Nodes"]}G b{r["BS_local"]} {r["precision"]}'
              for r in eta_sorted]
eta_data = {
    "labels": eta_labels,
    "datasets": [{
        "label": "ETA 10M (heures)",
        "data": [r["ETA_10M_hours"] for r in eta_sorted],
        "backgroundColor": ["#d62728" if r["precision"] == "bf16" else "#1f77b4" for r in eta_sorted],
        "borderColor": "#000", "borderWidth": 1,
    }]
}
eta_days_str = [f'{r["ETA_10M_days"]:.2f}' for r in eta_sorted]
eta_prec = [r["precision"] for r in eta_sorted]

def _fmt_float(v, fmt="{:.2f}", missing="OOM"):
    return fmt.format(v) if v is not None else missing

def _bg_color(r):
    if r["is_oom"]: return "#ffe6e6"
    return "#fff3f3" if r["precision"] == "bf16" else "#f3f7ff"

def _row(r):
    bg = _bg_color(r)
    setup = "{}n×{}G".format(r["#Nodes"], r["#GPUs"]//r["#Nodes"])
    its = _fmt_float(r["it/s"], "{:.2f}")
    imgs = _fmt_float(r["image/s"], "{:.1f}")
    etah = _fmt_float(r["ETA_10M_hours"], "{:.1f}", "—")
    etaj = _fmt_float(r["ETA_10M_days"], "{:.2f}", "—")
    return (
        f'<tr style="background:{bg}">'
        f'<td>{setup}</td><td>{r["BS_local"]}</td><td>{r["BS_global"]}</td>'
        f'<td><strong>{r["precision"]}</strong></td>'
        f'<td>{its}</td><td>{imgs}</td><td>{etah}</td><td>{etaj}</td>'
        f'<td><code>{r["JOB_ID"]}</code></td>'
        f'<td style="font-size:0.85em">{r["commentaire"]}</td></tr>'
    )

table_rows = "\n".join(_row(r) for r in rows)

best_fp32 = max(b40_fp32, key=lambda r: r["image/s"])
best_bf16 = max(b40_bf16, key=lambda r: r["image/s"])
gain_pct = 100 * (best_bf16["image/s"] - best_fp32["image/s"]) / best_fp32["image/s"]

html = f"""<!DOCTYPE html>
<html lang="fr"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MI300A — scaling DDP fp32 vs bf16 (HPE_adastra vs dev_hpe)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         max-width: 1150px; margin: 1.5em auto; padding: 0 1.5em; color: #222; line-height: 1.5; }}
  h1 {{ border-bottom: 2px solid #d62728; padding-bottom: 0.3em; margin-bottom: 0.5em; }}
  h2 {{ margin-top: 2em; color: #444; }}
  .takeaways {{ background: #fff8e1; border-left: 4px solid #ffa000;
                padding: 0.8em 1.2em; margin: 1em 0; }}
  .takeaways ul {{ margin: 0.3em 0; padding-left: 1.3em; }}
  .chart-container {{ position: relative; height: 380px; margin: 1.5em 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.88em; }}
  th, td {{ border: 1px solid #ddd; padding: 5px 9px; text-align: right; }}
  th {{ background: #f5f5f5; }}
  td:first-child, td:nth-child(4), td:last-child {{ text-align: left; }}
  code {{ background: #f5f5f5; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }}
  footer {{ font-size: 0.85em; color: #666; margin-top: 2em; padding-top: 1em;
            border-top: 1px solid #ddd; }}
  .legend-fp32 {{ color: #1f77b4; font-weight: bold; }}
  .legend-bf16 {{ color: #d62728; font-weight: bold; }}
</style>
</head><body>

<h1>MI300A — scaling DDP <span class="legend-fp32">fp32</span> vs <span class="legend-bf16">bf16</span></h1>
<p>Bench reproductibles sur Adastra, partition MI300A (4 APUs/node).
fp32 = HPE_adastra branch (pre-bf16 switch), bf16 = dev_hpe branch (dev's choice da20e9f).
Source CSV: <code>results.csv</code>, séparée par colonne <code>precision</code>.</p>

<div class="takeaways">
<strong>3 takeaways :</strong>
<ul>
<li><strong>bf16 gagne à toutes les échelles testées</strong> : +25% mono (1n×4), <strong>+33% sur 2n×4</strong>, +26% sur 4n×4.</li>
<li><strong>Hero number : 4n×4 b40 bf16 = {best_bf16["image/s"]:.1f} img/s</strong> ({gain_pct:+.0f}% vs fp32 {best_fp32["image/s"]:.1f}). ETA Phase 1 (10M images) = {best_bf16["ETA_10M_hours"]:.1f} h.</li>
<li><strong>Sweet spot batch mono-node = b40</strong> en bf16. b48-b64 ouvrent (vs fp32 stuck/OOM) mais plateau ~74 img/s. b80 reste OOM même en bf16.</li>
</ul>
</div>

<h2>1. Throughput scaling — fp32 vs bf16 (par batch)</h2>
<div class="chart-container"><canvas id="throughput"></canvas></div>

<h2>2. Scan batch mono-node 4-GPU bf16 (plateau ~74, peak b40)</h2>
<div class="chart-container"><canvas id="batch"></canvas></div>

<h2>3. ETA 10M images — ranking</h2>
<div class="chart-container"><canvas id="eta"></canvas></div>

<h2>Données brutes</h2>
<table>
<thead><tr>
<th>Setup</th><th>BS_local</th><th>BS_global</th><th>precision</th><th>it/s</th>
<th>image/s</th><th>ETA 10M (h)</th><th>ETA 10M (j)</th><th>JOB_ID</th><th>commentaire</th>
</tr></thead><tbody>
{table_rows}
</tbody></table>

<footer>
  Source : <code>results.csv</code> · Détails par run bf16 : <code>report_bf16.md</code> ·
  Régénérer : <code>python3 gen_report.py</code>
</footer>

<script>
const X_TICKS = {json.dumps(x_ticks)};
const COMMON_X = {{
  type: 'linear', title: {{ display: true, text: "Nombre total d'APUs MI300A" }},
  min: 0.5, max: {max_gpus + 0.5},
  ticks: {{ stepSize: 1, callback: v => X_TICKS.includes(v) ? v : '' }}
}};
const ETA_DAYS = {json.dumps(eta_days_str)};
const ETA_PREC = {json.dumps(eta_prec)};

new Chart(document.getElementById('throughput'), {{
  type: 'line',
  data: {json.dumps(throughput_data)},
  options: {{
    responsive: true, maintainAspectRatio: false,
    scales: {{ x: COMMON_X,
              y: {{ title: {{ display: true, text: 'images / seconde' }}, beginAtZero: true }} }},
    plugins: {{ legend: {{ position: 'top' }} }}
  }}
}});

new Chart(document.getElementById('batch'), {{
  type: 'bar',
  data: {json.dumps(batch_data)},
  options: {{
    responsive: true, maintainAspectRatio: false,
    scales: {{ x: {{ title: {{ display: true, text: 'local_batch_size' }} }},
              y: {{ title: {{ display: true, text: 'images / seconde' }}, beginAtZero: true }} }},
    plugins: {{ legend: {{ position: 'top' }} }}
  }}
}});

new Chart(document.getElementById('eta'), {{
  type: 'bar',
  data: {json.dumps(eta_data)},
  options: {{
    indexAxis: 'y', responsive: true, maintainAspectRatio: false,
    scales: {{ x: {{ title: {{ display: true, text: 'heures' }}, beginAtZero: true }},
              y: {{ title: {{ display: true, text: 'Setup (par ETA croissant)' }} }} }},
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{
        label: c => `${{c.parsed.x.toFixed(1)}}h (${{ETA_DAYS[c.dataIndex]}}j) - ${{ETA_PREC[c.dataIndex]}}`
      }} }}
    }}
  }}
}});
</script>
</body></html>
"""

with open("report.html", "w") as f:
    f.write(html)

print(f"OK report.html ({len(html)} chars = {len(html)//1024} KB, {html.count(chr(10))} lignes)")
print(f"  fp32 rows: {sum(1 for r in rows if r['precision']=='fp32')}  bf16 rows: {sum(1 for r in rows if r['precision']=='bf16')}  OOM: {len(oom)}")
