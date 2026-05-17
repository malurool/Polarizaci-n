"""
Ablation study: Bispectral vs Quadspectral vs Hyperspectral depth estimation.

Environment variables
---------------------
HDR_PATH        : path to the .hdr file
DATA_DIR        : directory with data .npz files (attenuation, lidar, I_downwelling_res)
RESULTS_DIR     : directory with hyperspectral .npz result files
SCENE_NAME      : basename of the scene (without extension), used to find result .npz files
T_AIR           : air temperature in Kelvin (default 289.7)
SELECTED_PIXELS : semicolon-separated "row,col" pairs, 1-indexed
                  e.g. "143,16;160,16;130,16;40,16;80,160;5,16"
"""

import os
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap
from spectral.io import envi

FUNCTIONS_DIR = os.path.join(os.path.dirname(__file__), "Functions")
sys.path.insert(0, FUNCTIONS_DIR)

from bispectral_estimation import bispectral_estimation
from quadspectral_estimation import quadspectral_estimation
from blackbody import blackbody
from forward_model import forward_model

REPO_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO_DIR, "Figures", "Ablation")

# ── Configuration ────────────────────────────────────────────────────────────

HDR_PATH    = os.getenv("HDR_PATH", "")
DATA_DIR    = os.getenv("DATA_DIR", "")
RESULTS_DIR = os.getenv("RESULTS_DIR", "")
SCENE_NAME  = os.getenv("SCENE_NAME", "")
T_AIR       = float(os.getenv("T_AIR", "289.7"))

# Selected pixels for per-pixel analysis (1-indexed row, col).
# Change these directly or override via SELECTED_PIXELS env var.
DEFAULT_PIXELS = [
    (143, 16),   # Reflective panel 1
    (160, 16),   # Reflective panel 2
    (130, 16),   # Grass
    ( 40, 16),   # Tree
    ( 80, 160),  # Background forest
    (  5, 16),   # Sky
]
PIXEL_LABELS = [
    "Refl. panel 1",
    "Refl. panel 2",
    "Grass",
    "Tree",
    "Background forest",
    "Sky",
]

_env_pixels = os.getenv("SELECTED_PIXELS", "")
if _env_pixels.strip():
    DEFAULT_PIXELS = [tuple(int(x) for x in p.split(",")) for p in _env_pixels.split(";")]
    PIXEL_LABELS   = [f"Pixel ({r},{c})" for r, c in DEFAULT_PIXELS]

# Zones for box-plot analysis (center pixel, patch half-size)
ZONES = {
    "Target 1": (187, 86),
    "Target 2": (145, 18),
    "Tree":     ( 70, 160),
    "Forest":   ( 80, 160),
}
PATCH_HALF = 4

# Band indices for bispectral / quadspectral (1-indexed arrays → 0-indexed below)
WATER_ABS_BANDS = np.array([4,10,14,19,23,28,31,37,42,48,54,189,230]) - 1
CLEAR_BANDS     = np.array([8,13,17,21,25,30,34,39,45,50,59,195,236]) - 1
OZONE_BAND      = 77 - 1
OZONE_CLEAR     = 81 - 1
PAIR_INDEX      = 7   # which water-vapor pair to use for bispectral

K = 247   # number of spectral bands used
Q = 10    # number of downwelling radiance components

DEPTH_CLIM = (0, 150)   # colorbar limits for depth maps (m)

# ── Helpers ──────────────────────────────────────────────────────────────────

def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".jpg", bbox_inches="tight", dpi=150)
    plt.close(fig)


def get_parula():
    try:
        return matplotlib.colormaps["parula"]
    except KeyError:
        return matplotlib.colormaps["viridis"]


def load_npz_key(path, key):
    with np.load(path, allow_pickle=False) as f:
        return f[key]


def load_hdr(hdr_path, n_bands=247):
    img = envi.open(hdr_path)
    cube = np.ascontiguousarray(np.asarray(img.load())[:, :, :n_bands])
    wl = img.metadata.get("wavelength")
    if wl is not None:
        wl = np.array([float(v) for v in wl])[:n_bands]
        if np.nanmax(wl) > 100:
            wl = wl / 1000.0
    return cube, wl


def patch(M, center, half=PATCH_HALF):
    r, c = center
    return M[r - half: r + half, c - half: c + half]


def nanrmse(est, gt):
    diff = est - gt
    return np.sqrt(np.nanmean(diff ** 2))


def nanmae(est, gt):
    return np.nanmean(np.abs(est - gt))


# ── Load data ────────────────────────────────────────────────────────────────

print("Loading data...")
meas, lambda_vals = load_hdr(HDR_PATH, n_bands=247)
meas = meas[:, -256:, :]   # same crop as hyperspectral_estimation

with np.load(os.path.join(DATA_DIR, "attenuation.npz"), allow_pickle=False) as f:
    attenuation = f["attenuation"].ravel()[:247]
with np.load(os.path.join(DATA_DIR, "I_downwelling_res.npz"), allow_pickle=False) as f:
    I_downwelling = f["I_downwelling_res"][:247, :]
with np.load(os.path.join(DATA_DIR, "lidar.npz"), allow_pickle=False) as f:
    lidar_full = f["depthMap"].astype(float)

# Align lidar to same crop as meas
lidar = lidar_full[:meas.shape[0], -meas.shape[1]:]
lidar[lidar == 0] = np.nan

# ── Compute bispectral depth ─────────────────────────────────────────────────

print("Computing bispectral depth...")
d_bispectral = bispectral_estimation(
    lambda_vals, meas, attenuation,
    WATER_ABS_BANDS[PAIR_INDEX], CLEAR_BANDS[PAIR_INDEX], T_AIR
)

# ── Compute quadspectral depth ────────────────────────────────────────────────

print("Computing quadspectral depth...")
cor_coeff = np.zeros(len(WATER_ABS_BANDS))
for i in range(len(WATER_ABS_BANDS)):
    Y = I_downwelling[WATER_ABS_BANDS[i], :] - I_downwelling[CLEAR_BANDS[i], :]
    X = I_downwelling[OZONE_BAND, :] - I_downwelling[OZONE_CLEAR, :]
    cor_coeff[i] = (X @ Y) / (X @ X)

d_quadspectral = quadspectral_estimation(
    lambda_vals, meas, attenuation,
    WATER_ABS_BANDS[PAIR_INDEX], CLEAR_BANDS[PAIR_INDEX],
    OZONE_BAND, OZONE_CLEAR,
    T_AIR, cor_coeff[PAIR_INDEX]
)

# ── Load hyperspectral depth ──────────────────────────────────────────────────

print("Loading hyperspectral results...")
hyper_dw_path = os.path.join(RESULTS_DIR, f"{SCENE_NAME}_downwelling.npz")
hyper_no_path = os.path.join(RESULTS_DIR, f"{SCENE_NAME}_no_downwelling.npz")

with np.load(hyper_dw_path, allow_pickle=False) as f:
    d_hyper_dw   = f["d"].squeeze().astype(float)
    T_hyper_dw   = f["T"].squeeze().astype(float)
    eps_hyper_dw = f["emissivity"].squeeze().astype(float)
    V_hyper_dw   = f["V"].astype(float)[:, :, :, :Q]   # (N,M,1,Q)

with np.load(hyper_no_path, allow_pickle=False) as f:
    d_hyper_no   = f["d"].squeeze().astype(float)
    T_hyper_no   = f["T"].squeeze().astype(float)
    eps_hyper_no = f["emissivity"].squeeze().astype(float)
    V_hyper_no   = f["V"].astype(float)[:, :, :, :Q]   # (N,M,1,Q)

d_hyper_dw[d_hyper_dw == 0] = np.nan
d_hyper_no[d_hyper_no == 0] = np.nan

# Align all depth maps to common shape
N_rows = min(d_bispectral.shape[0], d_hyper_dw.shape[0], lidar.shape[0])
N_cols = min(d_bispectral.shape[1], d_hyper_dw.shape[1], lidar.shape[1])

d_bispectral  = d_bispectral[:N_rows, :N_cols]
d_quadspectral = d_quadspectral[:N_rows, :N_cols]
d_hyper_dw    = d_hyper_dw[:N_rows, :N_cols]
d_hyper_no    = d_hyper_no[:N_rows, :N_cols]
lidar         = lidar[:N_rows, :N_cols]
meas          = meas[:N_rows, :N_cols, :]

# ── Method registry ───────────────────────────────────────────────────────────

METHODS = {
    "Bispectral":              d_bispectral,
    "Quadspectral":            d_quadspectral,
    "Hyperspectral (no dw)":   d_hyper_no,
    "Hyperspectral (w/ dw)":   d_hyper_dw,
}
METHOD_COLORS = ["#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c"]

PIXEL_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

plt.close("all")
cmap_depth = ListedColormap(get_parula()(np.linspace(0, 1, 256))[::-1])
cmap_err   = "hot"

# ── Plot A: 3×N grid — estimated | LiDAR GT | |error| ────────────────────────

print("Generating depth grid figure...")
n_methods = len(METHODS)
fig_grid, axes = plt.subplots(n_methods, 3, figsize=(14, 4 * n_methods))
fig_grid.suptitle("Depth estimation ablation: Estimated | Ground Truth | |Error|", fontsize=13)

vmin, vmax = DEPTH_CLIM
err_max = 50  # cap error colorscale at 50 m

for row_idx, (name, d_est) in enumerate(METHODS.items()):
    err = np.abs(d_est - lidar)

    ax_est, ax_gt, ax_err = axes[row_idx]

    im_est = ax_est.imshow(d_est, cmap=cmap_depth, vmin=vmin, vmax=vmax)
    ax_est.set_title(f"{name}\nEstimated", fontsize=9)
    ax_est.axis("off")
    fig_grid.colorbar(im_est, ax=ax_est, orientation="horizontal", pad=0.02, fraction=0.046)

    im_gt = ax_gt.imshow(lidar, cmap=cmap_depth, vmin=vmin, vmax=vmax)
    ax_gt.set_title("LiDAR GT", fontsize=9)
    ax_gt.axis("off")
    fig_grid.colorbar(im_gt, ax=ax_gt, orientation="horizontal", pad=0.02, fraction=0.046)

    im_err = ax_err.imshow(err, cmap=cmap_err, vmin=0, vmax=err_max)
    rmse = nanrmse(d_est, lidar)
    mae  = nanmae(d_est, lidar)
    ax_err.set_title(f"|Error|  RMSE={rmse:.1f}m  MAE={mae:.1f}m", fontsize=9)
    ax_err.axis("off")
    fig_grid.colorbar(im_err, ax=ax_err, orientation="horizontal", pad=0.02,
                      fraction=0.046, label="m")

    # Mark selected pixels on all three panels
    for px_idx, (pr, pc) in enumerate(DEFAULT_PIXELS):
        col = PIXEL_COLORS[px_idx % len(PIXEL_COLORS)]
        for ax in (ax_est, ax_gt, ax_err):
            ax.plot(pc - 1, pr - 1, "o", color=col, markersize=5,
                    markeredgecolor="white", markeredgewidth=0.5)

fig_grid.tight_layout()
save_fig(fig_grid, os.path.join(FIGURES_DIR, "depth_grid"))

# ── Plot B: CDF of absolute error ─────────────────────────────────────────────

print("Generating CDF figure...")
fig_cdf, ax_cdf = plt.subplots(figsize=(8, 5))

for (name, d_est), col in zip(METHODS.items(), METHOD_COLORS):
    err = np.abs(d_est - lidar).ravel()
    err = err[~np.isnan(err)]
    err_sorted = np.sort(err)
    cdf = np.arange(1, len(err_sorted) + 1) / len(err_sorted)
    ax_cdf.plot(err_sorted, cdf * 100, color=col, linewidth=2, label=name)

ax_cdf.set_xlabel("Absolute error (m)", fontsize=12)
ax_cdf.set_ylabel("% of pixels", fontsize=12)
ax_cdf.set_title("CDF of absolute depth error", fontsize=13)
ax_cdf.set_xlim([0, 80])
ax_cdf.set_ylim([0, 100])
ax_cdf.legend(fontsize=11)
ax_cdf.grid(True, alpha=0.3)
ax_cdf.axhline(50, color="gray", linestyle="--", linewidth=0.8)
ax_cdf.axhline(80, color="gray", linestyle="--", linewidth=0.8)
ax_cdf.text(0.5, 51, "50%", color="gray", fontsize=9)
ax_cdf.text(0.5, 81, "80%", color="gray", fontsize=9)
fig_cdf.tight_layout()
save_fig(fig_cdf, os.path.join(FIGURES_DIR, "cdf_error"))

# ── Plot C: Box plot of |error| by zone ───────────────────────────────────────

print("Generating box-plot by zone...")
fig_box, axes_box = plt.subplots(1, len(ZONES), figsize=(4 * len(ZONES), 5), sharey=True)
fig_box.suptitle("Absolute depth error by zone", fontsize=13)

for ax_z, (zone_name, zone_center) in zip(axes_box, ZONES.items()):
    data_per_method = []
    for name, d_est in METHODS.items():
        p_est   = patch(d_est, zone_center)
        p_lidar = patch(lidar, zone_center)
        err = np.abs(p_est - p_lidar).ravel()
        err = err[~np.isnan(err)]
        data_per_method.append(err)

    bp = ax_z.boxplot(data_per_method, patch_artist=True, widths=0.5,
                      medianprops=dict(color="black", linewidth=2))
    for patch_obj, col in zip(bp["boxes"], METHOD_COLORS):
        patch_obj.set_facecolor(col)
        patch_obj.set_alpha(0.7)

    ax_z.set_title(zone_name, fontsize=11)
    ax_z.set_xticks(range(1, len(METHODS) + 1))
    ax_z.set_xticklabels(list(METHODS.keys()), rotation=25, ha="right", fontsize=8)
    ax_z.set_ylabel("Absolute error (m)" if ax_z is axes_box[0] else "")
    ax_z.grid(True, axis="y", alpha=0.3)

fig_box.tight_layout()
save_fig(fig_box, os.path.join(FIGURES_DIR, "boxplot_by_zone"))

# ── Plot D: Depth profile comparison ──────────────────────────────────────────

print("Generating depth profile figure...")
line_col = N_cols // 4   # vertical line at 1/4 of the image width

fig_prof, ax_prof = plt.subplots(figsize=(10, 4))
for (name, d_est), col in zip(METHODS.items(), METHOD_COLORS):
    ax_prof.plot(d_est[:, line_col], color=col, linewidth=1.5, label=name)
ax_prof.plot(lidar[:, line_col], "--", color="black", linewidth=2, label="LiDAR GT")
ax_prof.set_xlabel("Pixel row", fontsize=11)
ax_prof.set_ylabel("Distance (m)", fontsize=11)
ax_prof.set_title(f"Depth profile at column {line_col}", fontsize=12)
ax_prof.legend(fontsize=10)
ax_prof.grid(True, alpha=0.3)
fig_prof.tight_layout()
save_fig(fig_prof, os.path.join(FIGURES_DIR, "depth_profile"))

# ── Plot E: Per-pixel radiance analysis ───────────────────────────────────────
# For each selected pixel: [radiance fit] + [residual]
# Uses the hyperspectral with-downwelling result as the primary method.

print("Generating per-pixel radiance analysis...")
lambda_vec = lambda_vals.ravel()

with np.load(os.path.join(DATA_DIR, "I_downwelling_res.npz"), allow_pickle=False) as f:
    reflected = f["I_downwelling_res"][:247, :]

lv  = lambda_vals.reshape(1, 1, K)
att = attenuation.reshape(1, 1, K)
ref = reflected.reshape(1, 1, K, Q)

n_px = len(DEFAULT_PIXELS)
fig_px, axes_px = plt.subplots(n_px, 2, figsize=(12, 3.5 * n_px))
if n_px == 1:
    axes_px = axes_px[np.newaxis, :]
fig_px.suptitle("Per-pixel radiance fit & residual (Hyperspectral w/ downwelling)", fontsize=12)

for px_idx, ((pr, pc), label) in enumerate(zip(DEFAULT_PIXELS, PIXEL_LABELS)):
    r, c = pr - 1, pc - 1
    col  = PIXEL_COLORS[px_idx % len(PIXEL_COLORS)]

    V_px   = V_hyper_dw[r, c, 0, :].reshape(1, 1, 1, Q)
    T_px   = float(T_hyper_dw[r, c])
    eps_px = eps_hyper_dw[r, c, :].reshape(1, 1, K)
    d_px   = float(d_hyper_dw[r, c]) if not np.isnan(d_hyper_dw[r, c]) else 0.0
    fit    = forward_model(lv, T_px, eps_px, V_px, ref, att, d_px, T_AIR).squeeze()
    measured_px = meas[r, c, :]
    residual    = measured_px - fit

    ax_fit, ax_res = axes_px[px_idx]

    ax_fit.plot(lambda_vec, measured_px, color=col, linewidth=1.2, label="Measured")
    ax_fit.plot(lambda_vec, fit, "k--", linewidth=1, label="Model fit")
    ax_fit.set_ylabel("Radiance\n(μW cm⁻² sr⁻¹ μm⁻¹)", fontsize=8)
    ax_fit.set_title(f"{label}  ({pr},{pc})", fontsize=9)
    ax_fit.set_xlim([lambda_vec[0], lambda_vec[-1]])
    ax_fit.legend(fontsize=8)
    ax_fit.grid(True, alpha=0.3)

    ax_res.plot(lambda_vec, residual, color=col, linewidth=1)
    ax_res.axhline(0, color="black", linewidth=0.7)
    ax_res.set_ylabel("Residual", fontsize=8)
    ax_res.set_xlabel("Wavelength (μm)", fontsize=8)
    ax_res.set_xlim([lambda_vec[0], lambda_vec[-1]])
    ax_res.grid(True, alpha=0.3)

fig_px.tight_layout()
save_fig(fig_px, os.path.join(FIGURES_DIR, "pixel_analysis", "radiance_and_residual"))

# ── Plot F: Pixel locations overlaid on T and emissivity mean maps ────────────

print("Generating pixel overlay maps...")
eps_mean_dw = eps_hyper_dw[:N_rows, :N_cols, :].mean(axis=2)
eps_mean_no = eps_hyper_no[:N_rows, :N_cols, :].mean(axis=2)
T_dw        = T_hyper_dw[:N_rows, :N_cols]

fig_ov, axes_ov = plt.subplots(1, 3, figsize=(15, 5))
fig_ov.suptitle("Selected pixels overlaid on scene maps", fontsize=12)

overlay_data = [
    (d_hyper_dw, "Depth — Hyper w/ dw (m)", cmap_depth, DEPTH_CLIM),
    (T_dw,       "Temperature (K)",           "hot",       (285, 294)),
    (eps_mean_dw, "Mean emissivity",           "viridis",   (0.7, 1.0)),
]

for ax_ov, (data, title, cmap_ov, clim_ov) in zip(axes_ov, overlay_data):
    im = ax_ov.imshow(data, cmap=cmap_ov, vmin=clim_ov[0], vmax=clim_ov[1])
    ax_ov.set_title(title, fontsize=10)
    ax_ov.axis("off")
    fig_ov.colorbar(im, ax=ax_ov, orientation="horizontal", pad=0.02, fraction=0.046)

    for px_idx, ((pr, pc), label) in enumerate(zip(DEFAULT_PIXELS, PIXEL_LABELS)):
        col = PIXEL_COLORS[px_idx % len(PIXEL_COLORS)]
        ax_ov.plot(pc - 1, pr - 1, "o", color=col, markersize=7,
                   markeredgecolor="white", markeredgewidth=0.8)
        ax_ov.annotate(label, xy=(pc - 1, pr - 1), xytext=(5, -10),
                       textcoords="offset points", fontsize=6,
                       color="white",
                       bbox=dict(boxstyle="round,pad=0.1", fc=col, alpha=0.7))

fig_ov.tight_layout()
save_fig(fig_ov, os.path.join(FIGURES_DIR, "pixel_overlay_maps"))

# ── Summary statistics ────────────────────────────────────────────────────────

print("\n── Depth estimation summary ──────────────────────────────────────")
print(f"{'Method':<30} {'RMSE (m)':>10} {'MAE (m)':>10}")
print("-" * 52)
for name, d_est in METHODS.items():
    print(f"{name:<30} {nanrmse(d_est, lidar):>10.2f} {nanmae(d_est, lidar):>10.2f}")

print(f"\nFigures saved to: {FIGURES_DIR}")
