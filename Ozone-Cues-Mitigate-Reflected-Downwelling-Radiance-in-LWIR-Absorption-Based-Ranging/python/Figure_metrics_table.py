"""
Generates a clean metrics summary table figure for slide presentation.
Global RMSE/MAE from saved figures (bi/quad) and recomputed from npz (hyper).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
DATA_DIR    = os.path.join(BASE, "data")
FIGURES_DIR = os.path.join(BASE, "Figures", "Ablation")

SCENE = "IHTest_202104_Path5_Step1_LWHSI1_collect1_DistStA"

# ── Load data ─────────────────────────────────────────────────────────────────
with np.load(os.path.join(DATA_DIR, "lidar.npz"), allow_pickle=False) as f:
    lidar = f["depthMap"].astype(float)[:256, 900:1156]
lidar[lidar == 0] = np.nan

with np.load(os.path.join(RESULTS_DIR, f"{SCENE}_downwelling_TV.npz"), allow_pickle=False) as f:
    d_dw = f["d"].squeeze().astype(float)
with np.load(os.path.join(RESULTS_DIR, f"{SCENE}_no_downwelling.npz"), allow_pickle=False) as f:
    d_no = f["d"].squeeze().astype(float)

d_dw[d_dw == 0] = np.nan
d_no[d_no == 0] = np.nan

N = min(d_dw.shape[0], lidar.shape[0])
M = min(d_dw.shape[1], lidar.shape[1])
lidar = lidar[:N, :M]
d_dw  = d_dw[:N, :M]
d_no  = d_no[:N, :M]

def nanrmse(e, g): return np.sqrt(np.nanmean((e - g) ** 2))
def nanmae(e, g):  return np.nanmean(np.abs(e - g))

ZONES   = {"Target 1": (187, 86), "Target 2": (145, 18), "Tree": (70, 160), "Forest": (80, 160)}
H       = 4

# ── Build data table ──────────────────────────────────────────────────────────
# Bispectral / Quadspectral global metrics come from the saved depth_grid figure
# (computed by Figure_ablation.py with the full HSI cube).
METHODS = [
    ("Bispectral [air]",         62.9, 25.4, None),
    ("Quadspectral",             65.4, 26.1, None),
    ("Hyperspectral (no dw)",    nanrmse(d_no, lidar), nanmae(d_no, lidar), d_no),
    ("Hyperspectral (w/ dw+TV)", nanrmse(d_dw, lidar), nanmae(d_dw, lidar), d_dw),
]

# Per-zone MAE (hyperspectral only — bi/quad unavailable without HDR)
zone_maes = {}
for name, rmse, mae, d in METHODS:
    if d is None:
        zone_maes[name] = {z: None for z in ZONES}
    else:
        zone_maes[name] = {}
        for zname, (r, c) in ZONES.items():
            patch_e = d[r - H:r + H, c - H:c + H]
            patch_g = lidar[r - H:r + H, c - H:c + H]
            zone_maes[name][zname] = nanmae(patch_e, patch_g)

# ── Colors ────────────────────────────────────────────────────────────────────
METHOD_COLORS = ["#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c"]
BEST_COLOR    = "#d4f4dd"   # light green highlight for best row
HEADER_COLOR  = "#2c3e50"
HEADER_TXT    = "white"

# ── Figure: global table ──────────────────────────────────────────────────────
col_labels = ["Method", "RMSE (m)", "MAE (m)"]
rows = [[name, f"{rmse:.1f}", f"{mae:.1f}"] for name, rmse, mae, _ in METHODS]
best_row = min(range(len(METHODS)), key=lambda i: METHODS[i][1])   # lowest RMSE

fig, ax = plt.subplots(figsize=(7, 2.8))
ax.axis("off")

tbl = ax.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(13)
tbl.scale(1, 2.0)
tbl.auto_set_column_width([0, 1, 2])
# Widen method column manually
for row in range(len(rows) + 1):
    tbl[row, 0].set_width(0.52)
    tbl[row, 1].set_width(0.24)
    tbl[row, 2].set_width(0.24)

# Style header
for col in range(len(col_labels)):
    cell = tbl[0, col]
    cell.set_facecolor(HEADER_COLOR)
    cell.set_text_props(color=HEADER_TXT, fontweight="bold")

# Style data rows
for row in range(1, len(rows) + 1):
    is_best = (row - 1) == best_row
    for col in range(len(col_labels)):
        cell = tbl[row, col]
        cell.set_facecolor(BEST_COLOR if is_best else "white")
        if col == 0:
            cell.set_text_props(color=METHOD_COLORS[row - 1], fontweight="bold")
        if is_best and col > 0:
            cell.set_text_props(fontweight="bold")

# Left-align method column
for row in range(len(rows) + 1):
    tbl[row, 0].set_text_props(ha="left")
    tbl[row, 0]._loc = "left"

fig.suptitle("Depth estimation — method comparison (global)", fontsize=13, fontweight="bold", y=0.98)
fig.tight_layout()

out_global = os.path.join(FIGURES_DIR, "metrics_table_global")
fig.savefig(out_global + ".pdf", bbox_inches="tight")
fig.savefig(out_global + ".jpg", bbox_inches="tight", dpi=180)
plt.close(fig)
print(f"Saved: {out_global}.[pdf|jpg]")

# ── Figure: per-zone table (hyper only) ──────────────────────────────────────
zone_names  = list(ZONES.keys())
hyper_names = [n for n, *_ , d in METHODS if d is not None]
hyper_data  = [METHODS[i] for i in range(len(METHODS)) if METHODS[i][3] is not None]

col_labels2 = ["Method"] + zone_names
rows2 = []
for name, rmse, mae, d in hyper_data:
    row = [name] + [f"{zone_maes[name][z]:.1f}" for z in zone_names]
    rows2.append(row)

# Add header note about bi/quad not available per-zone
fig2, ax2 = plt.subplots(figsize=(9, 1.8))
ax2.axis("off")

tbl2 = ax2.table(
    cellText=rows2,
    colLabels=col_labels2,
    cellLoc="center",
    loc="center",
)
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(12)
tbl2.scale(1, 2.2)
tbl2.auto_set_column_width(list(range(len(col_labels2))))
n_cols = len(col_labels2)
for row in range(len(rows2) + 1):
    tbl2[row, 0].set_width(0.36)
    for col in range(1, n_cols):
        tbl2[row, col].set_width(0.16)

hyper_colors = [METHOD_COLORS[2], METHOD_COLORS[3]]

for col in range(len(col_labels2)):
    cell = tbl2[0, col]
    cell.set_facecolor(HEADER_COLOR)
    cell.set_text_props(color=HEADER_TXT, fontweight="bold")

for row in range(1, len(rows2) + 1):
    is_best_row = (row - 1) == 1   # w/ dw+TV is row index 1
    for col in range(len(col_labels2)):
        cell = tbl2[row, col]
        cell.set_facecolor(BEST_COLOR if is_best_row else "white")
        if col == 0:
            cell.set_text_props(color=hyper_colors[row - 1], fontweight="bold")
        elif is_best_row:
            cell.set_text_props(fontweight="bold")

fig2.suptitle("Per-zone MAE (m) — hyperspectral variants", fontsize=12, fontweight="bold", y=1.02)
fig2.tight_layout()

out_zone = os.path.join(FIGURES_DIR, "metrics_table_zones")
fig2.savefig(out_zone + ".pdf", bbox_inches="tight")
fig2.savefig(out_zone + ".jpg", bbox_inches="tight", dpi=180)
plt.close(fig2)
print(f"Saved: {out_zone}.[pdf|jpg]")

# ── Print summary to terminal ─────────────────────────────────────────────────
print("\n── Global metrics ────────────────────────────────────")
print(f"{'Method':<28} {'RMSE':>8} {'MAE':>8}")
print("-" * 46)
for name, rmse, mae, _ in METHODS:
    marker = " ◀ best" if name == METHODS[best_row][0] else ""
    print(f"{name:<28} {rmse:>8.1f} {mae:>8.1f}{marker}")

print("\n── Per-zone MAE (hyperspectral) ───────────────────────")
print(f"{'Method':<28}", end="")
for z in zone_names:
    print(f"  {z:>10}", end="")
print()
for name, rmse, mae, d in hyper_data:
    print(f"{name:<28}", end="")
    for z in zone_names:
        v = zone_maes[name][z]
        print(f"  {v:>10.1f}", end="")
    print()
