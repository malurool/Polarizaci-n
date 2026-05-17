import os
import sys
from types import SimpleNamespace

import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from spectral.io import envi

FUNCTIONS_DIR = os.path.join(os.path.dirname(__file__), "Functions")
sys.path.insert(0, FUNCTIONS_DIR)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO_DIR, "Figures")


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".jpg", bbox_inches="tight", dpi=150)


from forward_model import forward_model


def get_parula():
    try:
        return matplotlib.colormaps["parula"]
    except KeyError:
        return matplotlib.colormaps["viridis"]


def load_npz(results_dir, name):
    path = os.path.join(results_dir, f"{name}.npz")
    with np.load(path, allow_pickle=False) as data:
        ns = SimpleNamespace(**{key: data[key] for key in data.files})
    # Squeeze trailing singleton dims: (N,M,1,1)->(N,M), (N,M,K,1)->(N,M,K)
    ns.d          = ns.d.squeeze()
    ns.T          = ns.T.squeeze()
    ns.emissivity = ns.emissivity.squeeze()
    return ns


def load_hdr_cube(hdr_path, n_bands=247):
    img = envi.open(hdr_path)
    cube = np.asarray(img.load())[:, :, :n_bands]
    wavelength = img.metadata.get("wavelength")
    if wavelength is not None:
        wavelength = np.array([float(v) for v in wavelength])[:n_bands]
        if np.nanmax(wavelength) > 100:
            wavelength = wavelength / 1000.0
    return cube, wavelength


plt.close("all")

HDR_PATH   = os.getenv("HDR_PATH", "")
DATA_DIR   = os.getenv("DATA_DIR", "")
RESULTS_DIR = os.getenv("RESULTS_DIR", "")
SCENE_NAME  = os.getenv("SCENE_NAME", "")

K = 247
Q = 10

no_downwelling      = load_npz(RESULTS_DIR, f"{SCENE_NAME}_no_downwelling")
with_downwelling    = load_npz(RESULTS_DIR, f"{SCENE_NAME}_downwelling")

no_downwelling.d = no_downwelling.d.astype(float)
with_downwelling.d = with_downwelling.d.astype(float)

no_downwelling.d[no_downwelling.d == 0] = np.nan
with_downwelling.d[with_downwelling.d == 0] = np.nan

no_downwelling.V  = no_downwelling.V[:, :, :, :Q]
with_downwelling.V = with_downwelling.V[:, :, :, :Q]

# Load measurements and wavelength from .hdr/.bsq
meas, lambda_vals = load_hdr_cube(HDR_PATH, n_bands=K)
meas = meas[:, -256:, :]

with np.load(os.path.join(DATA_DIR, "attenuation.npz"), allow_pickle=False) as f:
    attenuation = f["attenuation"].ravel()[:K]
with np.load(os.path.join(DATA_DIR, "I_downwelling_res.npz"), allow_pickle=False) as f:
    reflected = f["I_downwelling_res"][:K, :]

# Lidar is optional — skip if not present
lidar_path = os.path.join(DATA_DIR, "lidar.npz")
if os.path.exists(lidar_path):
    with np.load(lidar_path, allow_pickle=False) as f:
        lidar = f["depthMap"].astype(float)
    lidar[lidar == 0] = np.nan
    has_lidar = True
else:
    lidar = None
    has_lidar = False

lambda_vals = lambda_vals.reshape(1, 1, K)
attenuation = attenuation.reshape(1, 1, K)
reflected   = reflected.reshape(1, 1, K, Q)

# ── Selected pixels for per-pixel analysis (1-indexed row, col).
# Change these directly or override via SELECTED_PIXELS env var.
_DEFAULT_PIXELS = [
    (143, 16),   # Reflective panel 1
    (160, 16),   # Reflective panel 2
    (130, 16),   # Grass
    ( 40, 16),   # Tree
    ( 80, 160),  # Background forest
    (  5, 16),   # Sky
]
_DEFAULT_LABELS = ["Refl. panel 1", "Refl. panel 2", "Grass", "Tree", "Background forest", "Sky"]
_PIXEL_COLORS   = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

_env_px = os.getenv("SELECTED_PIXELS", "")
if _env_px.strip():
    _DEFAULT_PIXELS = [tuple(int(x) for x in p.split(",")) for p in _env_px.split(";")]
    _DEFAULT_LABELS = [f"Pixel ({r},{c})" for r, c in _DEFAULT_PIXELS]

pixel_reflective_panel_1 = _DEFAULT_PIXELS[0]
pixel_reflective_panel_2 = _DEFAULT_PIXELS[1] if len(_DEFAULT_PIXELS) > 1 else _DEFAULT_PIXELS[0]
pixel_grass_area         = _DEFAULT_PIXELS[2] if len(_DEFAULT_PIXELS) > 2 else _DEFAULT_PIXELS[0]
pixel_tree               = _DEFAULT_PIXELS[3] if len(_DEFAULT_PIXELS) > 3 else _DEFAULT_PIXELS[0]
pixel_background_forest  = _DEFAULT_PIXELS[4] if len(_DEFAULT_PIXELS) > 4 else _DEFAULT_PIXELS[0]
pixel_sky                = _DEFAULT_PIXELS[5] if len(_DEFAULT_PIXELS) > 5 else _DEFAULT_PIXELS[0]

center_target_1 = (187, 86)
center_target_2 = (145, 18)
center_tree = (70, 160)

line_index = 20

cut_off_1 = 0
cut_off_2 = 150

cmap = get_parula()
cmap = ListedColormap(cmap(np.linspace(0, 1, 256))[::-1])

fig1, ax1 = plt.subplots(num=1)
ax1.imshow(no_downwelling.d[:, :], cmap=cmap)
ax1.images[0].set_clim(cut_off_1, cut_off_2)
cb1 = fig1.colorbar(ax1.images[0], ax=ax1, orientation="horizontal", pad=0.05)
cb1.set_label("Distance (m)")
cb1.set_ticks([cut_off_1, cut_off_2])
cb1.set_ticklabels([str(cut_off_1), str(cut_off_2)])
ax1.tick_params(labelsize=20)
ax1.set_aspect("equal")
ax1.axis("off")
ax1.axvline(line_index - 1, color=[1, 0, 0], linewidth=2, linestyle="--")

fig2, ax2 = plt.subplots(num=2)
ax2.imshow(with_downwelling.d[:, :], cmap=cmap)
ax2.images[0].set_clim(cut_off_1, cut_off_2)
cb2 = fig2.colorbar(ax2.images[0], ax=ax2, orientation="horizontal", pad=0.05)
cb2.set_label("Distance (m)")
cb2.set_ticks([cut_off_1, cut_off_2])
cb2.set_ticklabels([str(cut_off_1), str(cut_off_2)])
ax2.tick_params(labelsize=20)
ax2.set_aspect("equal")
ax2.axis("off")
ax2.axvline(line_index - 1, color=[1, 0, 0], linewidth=2, linestyle="--")

fig5 = plt.figure(num=5)
fig5.set_size_inches(3.5 * 5.6, 2.5)
ax5 = fig5.add_subplot(111)
ax5.plot(no_downwelling.d[:, line_index - 1], linewidth=2, label="Without downwelling correction")
ax5.plot(with_downwelling.d[:, line_index - 1], linewidth=2, label="Downwelling correction")
if has_lidar:
    ax5.plot(lidar[:, line_index - 1], "--", linewidth=3, color=[0, 0, 0], label="Lidar (ground truth)")
ax5.set_ylim([15, 200])
ax5.set_xlim([0, 256])
ax5.set_xlabel("Pixel Index")
ax5.set_ylabel("Distance (m)")
ax5.legend(loc="upper right", ncol=2)
ax5.tick_params(labelsize=15)

patch_size = 8
half_patch = patch_size // 2


def extract_patch(M, cp):
    start_row = cp[0] - half_patch + 1
    end_row = cp[0] + half_patch
    start_col = cp[1] - half_patch + 1
    end_col = cp[1] + half_patch
    return M[start_row - 1 : end_row, start_col - 1 : end_col]


patch_t1_d1 = extract_patch(no_downwelling.d, center_target_1)
patch_t1_d2 = extract_patch(with_downwelling.d, center_target_1)

patch_t2_d1 = extract_patch(no_downwelling.d, center_target_2)
patch_t2_d2 = extract_patch(with_downwelling.d, center_target_2)

patch_tree_d1 = extract_patch(no_downwelling.d, center_tree)
patch_tree_d2 = extract_patch(with_downwelling.d, center_tree)

all_values_list = [
    patch_t1_d1.ravel(), patch_t1_d2.ravel(),
    patch_t2_d1.ravel(), patch_t2_d2.ravel(),
    patch_tree_d1.ravel(), patch_tree_d2.ravel(),
]
if has_lidar:
    all_values_list += [
        extract_patch(lidar, center_target_1).ravel(),
        extract_patch(lidar, center_target_2).ravel(),
        extract_patch(lidar, center_tree).ravel(),
    ]
all_values = np.concatenate(all_values_list)

bin_width = 5
bin_min = np.floor(np.nanmin(all_values) / bin_width) * bin_width
bin_max = 200
bin_edges = np.arange(bin_min, bin_max + bin_width, bin_width)

color_d1 = [0, 0, 1]
color_d2 = [1, 0, 0]
color_gt = [0, 0, 0]

fig101, ax101 = plt.subplots(num=101)
fig101.set_size_inches(7, 4.2)
ax101.hist(patch_t1_d1.ravel(), bins=bin_edges, color=color_d1, alpha=0.6, label="w/o down. corr.")
ax101.hist(patch_t1_d2.ravel(), bins=bin_edges, color=color_d2, alpha=0.6, label="w/ down. corr.")
if has_lidar:
    ax101.hist(extract_patch(lidar, center_target_1).ravel(), bins=bin_edges, color=color_gt, alpha=0.6, label="Lidar (GT)")
ax101.legend()
ax101.set_xlabel("Distance (m)")
ax101.set_ylabel("Count")
ax101.tick_params(labelsize=25)

fig102, ax102 = plt.subplots(num=102)
fig102.set_size_inches(7, 4.2)
ax102.hist(patch_t2_d1.ravel(), bins=bin_edges, color=color_d1, alpha=0.6, label="w/o down. corr.")
ax102.hist(patch_t2_d2.ravel(), bins=bin_edges, color=color_d2, alpha=0.6, label="w/ down. corr.")
if has_lidar:
    ax102.hist(extract_patch(lidar, center_target_2).ravel(), bins=bin_edges, color=color_gt, alpha=0.6, label="Lidar (GT)")
ax102.legend()
ax102.set_xlabel("Distance (m)")
ax102.set_ylabel("Count")
ax102.tick_params(labelsize=25)

fig103, ax103 = plt.subplots(num=103)
fig103.set_size_inches(7, 4.2)
ax103.hist(patch_tree_d1.ravel(), bins=bin_edges, color=color_d1, alpha=0.6, label="w/o down. corr.")
ax103.hist(patch_tree_d2.ravel(), bins=bin_edges, color=color_d2, alpha=0.6, label="w/ down. corr.")
if has_lidar:
    ax103.hist(extract_patch(lidar, center_tree).ravel(), bins=bin_edges, color=color_gt, alpha=0.6, label="Lidar (GT)")
ax103.legend()
ax103.set_xlabel("Distance (m)")
ax103.set_ylabel("Count")
ax103.tick_params(labelsize=25)

print("\n--- Patch Statistics (Mean ± Std, ignoring NaNs) ---")

print("\nCalibration Target 1:")
print(f"  Without downwelling correction: {np.nanmean(patch_t1_d1):.2f} ± {np.nanstd(patch_t1_d1, ddof=1):.2f} m")
print(f"  Downwelling correction:         {np.nanmean(patch_t1_d2):.2f} ± {np.nanstd(patch_t1_d2, ddof=1):.2f} m")

print("\nCalibration Target 2:")
print(f"  Without downwelling correction: {np.nanmean(patch_t2_d1):.2f} ± {np.nanstd(patch_t2_d1, ddof=1):.2f} m")
print(f"  Downwelling correction:         {np.nanmean(patch_t2_d2):.2f} ± {np.nanstd(patch_t2_d2, ddof=1):.2f} m")

print("\nTree:")
print(f"  Without downwelling correction: {np.nanmean(patch_tree_d1):.2f} ± {np.nanstd(patch_tree_d1, ddof=1):.2f} m")
print(f"  Downwelling correction:         {np.nanmean(patch_tree_d2):.2f} ± {np.nanstd(patch_tree_d2, ddof=1):.2f} m")

T_air = 289.7

pixel_reflective_panel_1 = (143, 16)
pixel_reflective_panel_2 = (160, 16)
pixel_grass_area = (130, 16)
pixel_tree = (40, 16)
pixel_background_forest = (80, 160)
pixel_sky = (5, 16)

lambda_vec = lambda_vals.reshape(-1)

# Reflective Panel 1
pixel_index = pixel_reflective_panel_1
V = no_downwelling.V[pixel_index[0] - 1, pixel_index[1] - 1, 0, :].reshape(1, 1, 1, -1)
T = no_downwelling.T[pixel_index[0] - 1, pixel_index[1] - 1]
emissivity = no_downwelling.emissivity[pixel_index[0] - 1, pixel_index[1] - 1, :].reshape(1, 1, -1)
d = no_downwelling.d[pixel_index[0] - 1, pixel_index[1] - 1]
model_fit_no_downwelling = forward_model(lambda_vals, T, emissivity, V, reflected, attenuation, d, T_air)

V = with_downwelling.V[pixel_index[0] - 1, pixel_index[1] - 1, 0, :].reshape(1, 1, 1, -1)
T = with_downwelling.T[pixel_index[0] - 1, pixel_index[1] - 1]
emissivity = with_downwelling.emissivity[pixel_index[0] - 1, pixel_index[1] - 1, :].reshape(1, 1, -1)
d = with_downwelling.d[pixel_index[0] - 1, pixel_index[1] - 1]
model_fit_with_downwelling = forward_model(lambda_vals, T, emissivity, V, reflected, attenuation, d, T_air)

fig6 = plt.figure(num=6)
fig6.set_size_inches(6, 5.2)
ax6 = fig6.add_subplot(111)
ax6.plot(lambda_vec, meas[pixel_index[0] - 1, pixel_index[1] - 1, :].squeeze(), linewidth=1)
ax6.plot(lambda_vec, model_fit_with_downwelling.squeeze(), linewidth=1)
ax6.plot(lambda_vec, model_fit_no_downwelling.squeeze(), "--", linewidth=2)
ax6.set_xlabel("Wavelength (μm)")
ax6.set_ylabel("Radiance (μW cm$^{-2}$ sr$^{-1}$ μm$^{-1}$)")
ax6.set_xlim([8, 13])
ax6.set_xticks(np.arange(8, 14))
ax6.tick_params(labelsize=20)

# Reflective Panel 2
pixel_index = pixel_reflective_panel_2
V = no_downwelling.V[pixel_index[0] - 1, pixel_index[1] - 1, 0, :].reshape(1, 1, 1, -1)
T = no_downwelling.T[pixel_index[0] - 1, pixel_index[1] - 1]
emissivity = no_downwelling.emissivity[pixel_index[0] - 1, pixel_index[1] - 1, :].reshape(1, 1, -1)
d = no_downwelling.d[pixel_index[0] - 1, pixel_index[1] - 1]
model_fit_no_downwelling = forward_model(lambda_vals, T, emissivity, V, reflected, attenuation, d, T_air)

V = with_downwelling.V[pixel_index[0] - 1, pixel_index[1] - 1, 0, :].reshape(1, 1, 1, -1)
T = with_downwelling.T[pixel_index[0] - 1, pixel_index[1] - 1]
emissivity = with_downwelling.emissivity[pixel_index[0] - 1, pixel_index[1] - 1, :].reshape(1, 1, -1)
d = with_downwelling.d[pixel_index[0] - 1, pixel_index[1] - 1]
model_fit_with_downwelling = forward_model(lambda_vals, T, emissivity, V, reflected, attenuation, d, T_air)

fig7 = plt.figure(num=7)
fig7.set_size_inches(6, 5.2)
ax7 = fig7.add_subplot(111)
ax7.plot(lambda_vec, meas[pixel_index[0] - 1, pixel_index[1] - 1, :].squeeze(), linewidth=1)
ax7.plot(lambda_vec, model_fit_with_downwelling.squeeze(), linewidth=1)
ax7.plot(lambda_vec, model_fit_no_downwelling.squeeze(), "--", linewidth=2)
ax7.set_xlabel("Wavelength (μm)")
ax7.set_ylabel("Radiance (μW cm$^{-2}$ sr$^{-1}$ μm$^{-1}$)")
ax7.set_xlim([8, 13])
ax7.set_xticks(np.arange(8, 14))
ax7.tick_params(labelsize=20)
ax7_inset = fig7.add_axes([0.35, 0.25, 0.3, 0.3])
ax7_inset.plot(lambda_vec[69:90], meas[pixel_index[0] - 1, pixel_index[1] - 1, 69:90].squeeze(), linewidth=1)
ax7_inset.plot(lambda_vec[69:90], model_fit_with_downwelling.squeeze()[69:90], linewidth=1)
ax7_inset.plot(lambda_vec[69:90], model_fit_no_downwelling.squeeze()[69:90], "--", linewidth=2)
ax7_inset.set_xlim([lambda_vec[69], lambda_vec[89]])

# Grass area
pixel_index = pixel_grass_area
V = no_downwelling.V[pixel_index[0] - 1, pixel_index[1] - 1, 0, :].reshape(1, 1, 1, -1)
T = no_downwelling.T[pixel_index[0] - 1, pixel_index[1] - 1]
emissivity = no_downwelling.emissivity[pixel_index[0] - 1, pixel_index[1] - 1, :].reshape(1, 1, -1)
d = no_downwelling.d[pixel_index[0] - 1, pixel_index[1] - 1]
model_fit_no_downwelling = forward_model(lambda_vals, T, emissivity, V, reflected, attenuation, d, T_air)

V = with_downwelling.V[pixel_index[0] - 1, pixel_index[1] - 1, 0, :].reshape(1, 1, 1, -1)
T = with_downwelling.T[pixel_index[0] - 1, pixel_index[1] - 1]
emissivity = with_downwelling.emissivity[pixel_index[0] - 1, pixel_index[1] - 1, :].reshape(1, 1, -1)
d = with_downwelling.d[pixel_index[0] - 1, pixel_index[1] - 1]
model_fit_with_downwelling = forward_model(lambda_vals, T, emissivity, V, reflected, attenuation, d, T_air)

fig8 = plt.figure(num=8)
fig8.set_size_inches(6, 5.2)
ax8 = fig8.add_subplot(111)
ax8.plot(lambda_vec, meas[pixel_index[0] - 1, pixel_index[1] - 1, :].squeeze(), linewidth=1)
ax8.plot(lambda_vec, model_fit_with_downwelling.squeeze(), linewidth=1)
ax8.plot(lambda_vec, model_fit_no_downwelling.squeeze(), "--", linewidth=2)
ax8.set_xlabel("Wavelength (μm)")
ax8.set_ylabel("Radiance (μW cm$^{-2}$ sr$^{-1}$ μm$^{-1}$)")
ax8.set_xlim([8, 13])
ax8.set_xticks(np.arange(8, 14))
ax8.tick_params(labelsize=20)
ax8_inset = fig8.add_axes([0.35, 0.25, 0.3, 0.3])
ax8_inset.plot(lambda_vec[69:90], meas[pixel_index[0] - 1, pixel_index[1] - 1, 69:90].squeeze(), linewidth=1)
ax8_inset.plot(lambda_vec[69:90], model_fit_with_downwelling.squeeze()[69:90], linewidth=1)
ax8_inset.plot(lambda_vec[69:90], model_fit_no_downwelling.squeeze()[69:90], "--", linewidth=2)
ax8_inset.set_xlim([lambda_vec[69], lambda_vec[89]])

# Tree area
pixel_index = pixel_tree
V = no_downwelling.V[pixel_index[0] - 1, pixel_index[1] - 1, 0, :].reshape(1, 1, 1, -1)
T = no_downwelling.T[pixel_index[0] - 1, pixel_index[1] - 1]
emissivity = no_downwelling.emissivity[pixel_index[0] - 1, pixel_index[1] - 1, :].reshape(1, 1, -1)
d = no_downwelling.d[pixel_index[0] - 1, pixel_index[1] - 1]
model_fit_no_downwelling = forward_model(lambda_vals, T, emissivity, V, reflected, attenuation, d, T_air)

V = with_downwelling.V[pixel_index[0] - 1, pixel_index[1] - 1, 0, :].reshape(1, 1, 1, -1)
T = with_downwelling.T[pixel_index[0] - 1, pixel_index[1] - 1]
emissivity = with_downwelling.emissivity[pixel_index[0] - 1, pixel_index[1] - 1, :].reshape(1, 1, -1)
d = with_downwelling.d[pixel_index[0] - 1, pixel_index[1] - 1]
model_fit_with_downwelling = forward_model(lambda_vals, T, emissivity, V, reflected, attenuation, d, T_air)

fig9 = plt.figure(num=9)
fig9.set_size_inches(6, 6.5)
ax9 = fig9.add_subplot(111)
ax9.plot(lambda_vec, meas[pixel_index[0] - 1, pixel_index[1] - 1, :].squeeze(), linewidth=1)
ax9.plot(lambda_vec, model_fit_with_downwelling.squeeze(), linewidth=1)
ax9.plot(lambda_vec, model_fit_no_downwelling.squeeze(), "--", linewidth=2)
ax9.set_xlim([8, 13])
ax9.set_xticks(np.arange(8, 14))
ax9.set_xlabel("Wavelength (μm)")
ax9.set_ylabel("Radiance (μW cm$^{-2}$ sr$^{-1}$ μm$^{-1}$)")
ax9.legend(
    [
        "Measurement",
        "Model fit (including downwelling)",
        "Model fit (neglecting downwelling)",
    ],
    loc="lower center",
)
ax9.tick_params(labelsize=20)
ax9_inset = fig9.add_axes([0.35, 0.45, 0.3, 0.3])
ax9_inset.plot(lambda_vec[69:90], meas[pixel_index[0] - 1, pixel_index[1] - 1, 69:90].squeeze(), linewidth=1)
ax9_inset.plot(lambda_vec[69:90], model_fit_with_downwelling.squeeze()[69:90], linewidth=1)
ax9_inset.plot(lambda_vec[69:90], model_fit_no_downwelling.squeeze()[69:90], "--", linewidth=2)
ax9_inset.set_xlim([lambda_vec[69], lambda_vec[89]])

# Background forest area
pixel_index = pixel_background_forest
V = no_downwelling.V[pixel_index[0] - 1, pixel_index[1] - 1, 0, :].reshape(1, 1, 1, -1)
T = no_downwelling.T[pixel_index[0] - 1, pixel_index[1] - 1]
emissivity = no_downwelling.emissivity[pixel_index[0] - 1, pixel_index[1] - 1, :].reshape(1, 1, -1)
d = no_downwelling.d[pixel_index[0] - 1, pixel_index[1] - 1]
model_fit_no_downwelling = forward_model(lambda_vals, T, emissivity, V, reflected, attenuation, d, T_air)

V = with_downwelling.V[pixel_index[0] - 1, pixel_index[1] - 1, 0, :].reshape(1, 1, 1, -1)
T = with_downwelling.T[pixel_index[0] - 1, pixel_index[1] - 1]
emissivity = with_downwelling.emissivity[pixel_index[0] - 1, pixel_index[1] - 1, :].reshape(1, 1, -1)
d = with_downwelling.d[pixel_index[0] - 1, pixel_index[1] - 1]
model_fit_with_downwelling = forward_model(lambda_vals, T, emissivity, V, reflected, attenuation, d, T_air)

fig10 = plt.figure(num=10)
fig10.set_size_inches(6, 6.5)
ax10 = fig10.add_subplot(111)
ax10.plot(lambda_vec, meas[pixel_index[0] - 1, pixel_index[1] - 1, :].squeeze(), linewidth=1)
ax10.plot(lambda_vec, model_fit_with_downwelling.squeeze(), linewidth=1)
ax10.plot(lambda_vec, model_fit_no_downwelling.squeeze(), "--", linewidth=2)
ax10.set_xlim([8, 13])
ax10.set_xticks(np.arange(8, 14))
ax10.set_xlabel("Wavelength (μm)")
ax10.set_ylabel("Radiance (μW cm$^{-2}$ sr$^{-1}$ μm$^{-1}$)")
ax10.legend(
    [
        "Measurement",
        "Model fit (including downwelling)",
        "Model fit (neglecting downwelling)",
    ],
    loc="lower center",
)
ax10.tick_params(labelsize=20)
ax10_inset = fig10.add_axes([0.35, 0.45, 0.3, 0.3])
ax10_inset.plot(lambda_vec[69:90], meas[pixel_index[0] - 1, pixel_index[1] - 1, 69:90].squeeze(), linewidth=1)
ax10_inset.plot(lambda_vec[69:90], model_fit_with_downwelling.squeeze()[69:90], linewidth=1)
ax10_inset.plot(lambda_vec[69:90], model_fit_no_downwelling.squeeze()[69:90], "--", linewidth=2)
ax10_inset.set_xlim([lambda_vec[69], lambda_vec[89]])

# Sky pixels
pixel_index = pixel_sky
V = no_downwelling.V[pixel_index[0] - 1, pixel_index[1] - 1, 0, :].reshape(1, 1, 1, -1)
T = no_downwelling.T[pixel_index[0] - 1, pixel_index[1] - 1]
emissivity = no_downwelling.emissivity[pixel_index[0] - 1, pixel_index[1] - 1, :].reshape(1, 1, -1)
d = no_downwelling.d[pixel_index[0] - 1, pixel_index[1] - 1]
model_fit_no_downwelling = forward_model(lambda_vals, T, emissivity, V, reflected, attenuation, d, T_air)

V = with_downwelling.V[pixel_index[0] - 1, pixel_index[1] - 1, 0, :].reshape(1, 1, 1, -1)
T = with_downwelling.T[pixel_index[0] - 1, pixel_index[1] - 1]
emissivity = with_downwelling.emissivity[pixel_index[0] - 1, pixel_index[1] - 1, :].reshape(1, 1, -1)
d = with_downwelling.d[pixel_index[0] - 1, pixel_index[1] - 1]
model_fit_with_downwelling = forward_model(lambda_vals, T, emissivity, V, reflected, attenuation, d, T_air)

fig11 = plt.figure(num=11)
fig11.set_size_inches(6, 5.2)
ax11 = fig11.add_subplot(111)
ax11.plot(lambda_vec, meas[pixel_index[0] - 1, pixel_index[1] - 1, :].squeeze(), linewidth=1)
ax11.plot(lambda_vec, model_fit_with_downwelling.squeeze(), linewidth=1)
ax11.plot(lambda_vec, model_fit_no_downwelling.squeeze(), "--", linewidth=2)
ax11.set_xlim([8, 13])
ax11.set_xticks(np.arange(8, 14))
ax11.set_xlabel("Wavelength (μm)")
ax11.set_ylabel("Radiance (μW cm$^{-2}$ sr$^{-1}$ μm$^{-1}$)")
ax11.tick_params(labelsize=20)

# Other plots

downwelling_name = [
    "0deg",
    "30deg",
    "60deg",
    "70deg",
    "80deg",
    "82deg",
    "84deg",
    "86deg",
    "88deg",
    "89deg",
]

for i in range(10):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(with_downwelling.V[:, :, 0, i])
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05)
    cb.set_ticks([0, 0.2])
    cb.set_ticklabels(["0", "0.2"])
    im.set_clim(0, 0.2)
    ax.tick_params(labelsize=25)
    ax.axis("off")
    ax.set_aspect("equal")
    save_fig(fig, os.path.join(FIGURES_DIR, "Hyperspectral_correction", "Normalized_Solid_Angles", downwelling_name[i]))

_dm = os.path.join(FIGURES_DIR, "Hyperspectral_correction", "Model_Fits")
save_fig(fig6, os.path.join(_dm, "reflective_panel_1"))
save_fig(fig7, os.path.join(_dm, "reflective_panel_2"))
save_fig(fig8, os.path.join(_dm, "grass"))
save_fig(fig9, os.path.join(_dm, "tree"))
save_fig(fig10, os.path.join(_dm, "background_forest"))
save_fig(fig11, os.path.join(_dm, "sky"))

# Close model-fit figures before reusing numbers 6–11 for K-means / temperature
for _fnum in range(6, 12):
    plt.close(_fnum)

# K-means on emissivity
clusters = 5
max_iterations = 100000

emissivity = no_downwelling.emissivity
N, M, K_em = emissivity.shape
emissivity = emissivity.reshape(N * M, K_em)

np.random.seed(0)
model = KMeans(n_clusters=clusters, max_iter=max_iterations, n_init=10, random_state=0)
idx = model.fit_predict(emissivity)
C = model.cluster_centers_

mean_emissivity = C.mean(axis=1)
sort_idx = np.argsort(mean_emissivity)[::-1]
C = C[sort_idx, :]

new_idx = np.zeros_like(idx)
for i, old_idx in enumerate(sort_idx):
    new_idx[idx == old_idx] = i + 1
idx = new_idx.reshape(N, M)
idx_no_dw = idx.copy()
C_no_dw = C.copy()

cluster_colors = np.array(
    [
        [0.19, 0.64, 0.28],
        [0.76, 0.70, 0.50],
        [0.55, 0.57, 0.67],
        [0.53, 0.81, 0.92],
        [0.82, 0.82, 0.88],
    ]
)

fig6, ax6 = plt.subplots(num=6)
fig6.set_size_inches(5.6, 5.6)
im = ax6.imshow(idx, vmin=1, vmax=clusters)
ax6.set_aspect("equal")
ax6.axis("off")
ax6.set_cmap(ListedColormap(cluster_colors))
cb6 = fig6.colorbar(im, ax=ax6, orientation="horizontal", pad=0.05)
cb6.set_ticks(np.linspace(1, clusters, clusters))
cb6.set_ticklabels([f"Cluster {i}" for i in range(1, clusters + 1)])
ax6.tick_params(labelsize=17)

fig7, ax7 = plt.subplots(num=7)
fig7.set_size_inches(6, 5.2)
for i in range(clusters):
    ax7.plot(lambda_vec, C[i, :], color=cluster_colors[i], linewidth=2)
ax7.set_ylim([0.5, 1])
ax7.set_xlim([8, 13])
ax7.set_xticks(np.arange(8, 14))
ax7.legend([f"Cluster {i}" for i in range(1, clusters + 1)], loc="lower left", ncol=2)
ax7.set_xlabel("Wavelength (μm)")
ax7.set_ylabel("Emissivity")
ax7.tick_params(labelsize=25)

# Repeat for with_downwelling emissivity
emissivity = with_downwelling.emissivity
N, M, K_em = emissivity.shape
emissivity = emissivity.reshape(N * M, K_em)

np.random.seed(0)
model = KMeans(n_clusters=clusters, max_iter=max_iterations, n_init=10, random_state=0)
idx = model.fit_predict(emissivity)
C = model.cluster_centers_

mean_emissivity = C.mean(axis=1)
sort_idx = np.argsort(mean_emissivity)[::-1]
C = C[sort_idx, :]

new_idx = np.zeros_like(idx)
for i, old_idx in enumerate(sort_idx):
    new_idx[idx == old_idx] = i + 1
idx = new_idx.reshape(N, M)
idx_with_dw = idx.copy()
C_with_dw = C.copy()

fig8, ax8 = plt.subplots(num=8)
fig8.set_size_inches(5.6, 5.6)
im = ax8.imshow(idx, vmin=1, vmax=clusters)
ax8.set_aspect("equal")
ax8.axis("off")
ax8.set_cmap(ListedColormap(cluster_colors))
cb8 = fig8.colorbar(im, ax=ax8, orientation="horizontal", pad=0.05)
cb8.set_ticks(np.linspace(1, clusters, clusters))
cb8.set_ticklabels([f"Cluster {i}" for i in range(1, clusters + 1)])
ax8.tick_params(labelsize=17)

fig9, ax9 = plt.subplots(num=9)
fig9.set_size_inches(6, 5.2)
for i in range(clusters):
    ax9.plot(lambda_vec, C[i, :], color=cluster_colors[i], linewidth=2)
ax9.set_xlabel("Wavelength (μm)")
ax9.set_ylabel("Emissivity")
ax9.set_ylim([0.4, 1])
ax9.set_xlim([8, 13])
ax9.set_xticks(np.arange(8, 14))
ax9.legend([f"Cluster {i}" for i in range(1, clusters + 1)], loc="lower left", ncol=2)
ax9.tick_params(labelsize=25)

# Fig 13: Side-by-side K-means cluster images
cluster_cmap = ListedColormap(cluster_colors)

fig13, (ax13a, ax13b) = plt.subplots(1, 2, num=13, figsize=(11, 5))
im13a = ax13a.imshow(idx_no_dw, vmin=1, vmax=clusters, cmap=cluster_cmap)
ax13a.set_title("Without downwelling correction", fontsize=12)
ax13a.axis("off")
ax13a.set_aspect("equal")
im13b = ax13b.imshow(idx_with_dw, vmin=1, vmax=clusters, cmap=cluster_cmap)
ax13b.set_title("With downwelling correction", fontsize=12)
ax13b.axis("off")
ax13b.set_aspect("equal")
fig13.subplots_adjust(right=0.88, bottom=0.12)
cbar_ax13 = fig13.add_axes([0.91, 0.15, 0.02, 0.7])
cb13 = fig13.colorbar(im13b, cax=cbar_ax13)
cb13.set_ticks(np.linspace(1, clusters, clusters))
cb13.set_ticklabels([f"Cluster {i}" for i in range(1, clusters + 1)], fontsize=10)

# Fig 14: Emissivity spectra at key pixels (with vs without downwelling)
key_pixels = {
    "Reflective panel 1": pixel_reflective_panel_1,
    "Reflective panel 2": pixel_reflective_panel_2,
    "Grass": pixel_grass_area,
    "Tree": pixel_tree,
    "Background forest": pixel_background_forest,
    "Sky": pixel_sky,
}

fig14, axes14 = plt.subplots(2, 3, num=14, figsize=(14, 8), sharex=True, sharey=True)
fig14.suptitle("Emissivity spectra at key pixels", fontsize=13)
axes14_flat = axes14.flatten()

for ax, (label, (row, col)) in zip(axes14_flat, key_pixels.items()):
    r, c = row - 1, col - 1
    eps_no = no_downwelling.emissivity[r, c, :]
    eps_with = with_downwelling.emissivity[r, c, :]
    ax.plot(lambda_vec, eps_no, "--", linewidth=1.5, label="w/o downwelling corr.")
    ax.plot(lambda_vec, eps_with, linewidth=1.5, label="w/ downwelling corr.")
    ax.set_title(label, fontsize=11)
    ax.set_xlim([8, 13])
    ax.set_xticks(np.arange(8, 14))
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

for ax in axes14[1, :]:
    ax.set_xlabel("Wavelength (μm)", fontsize=10)
for ax in axes14[:, 0]:
    ax.set_ylabel("Emissivity", fontsize=10)

axes14_flat[0].legend(loc="lower left", fontsize=9)
fig14.tight_layout()

# Temperature
cut_off_1 = 285
cut_off_2 = 294

fig10, ax10 = plt.subplots(num=10)
ax10.imshow(with_downwelling.T[:, :], cmap="hot")
ax10.images[0].set_clim(cut_off_1, cut_off_2)
cb10 = fig10.colorbar(ax10.images[0], ax=ax10, orientation="horizontal", pad=0.05)
cb10.set_ticks([cut_off_1, cut_off_2])
cb10.set_ticklabels([f"{cut_off_1} K", f"{cut_off_2} K"])
ax10.tick_params(labelsize=20)
ax10.set_aspect("equal")
ax10.axis("off")

fig11, ax11 = plt.subplots(num=11)
ax11.imshow(no_downwelling.T[:, :], cmap="hot")
ax11.images[0].set_clim(cut_off_1, cut_off_2)
cb11 = fig11.colorbar(ax11.images[0], ax=ax11, orientation="horizontal", pad=0.05)
cb11.set_ticks([cut_off_1, cut_off_2])
cb11.set_ticklabels([f"{cut_off_1} K", f"{cut_off_2} K"])
ax11.tick_params(labelsize=20)
ax11.set_aspect("equal")
ax11.axis("off")

fig12, (ax12a, ax12b) = plt.subplots(1, 2, num=12, figsize=(11, 5))
im12a = ax12a.imshow(no_downwelling.T[:, :], cmap="hot")
im12a.set_clim(cut_off_1, cut_off_2)
ax12a.set_title("Without downwelling correction", fontsize=12)
ax12a.axis("off")
ax12a.set_aspect("equal")
im12b = ax12b.imshow(with_downwelling.T[:, :], cmap="hot")
im12b.set_clim(cut_off_1, cut_off_2)
ax12b.set_title("With downwelling correction", fontsize=12)
ax12b.axis("off")
ax12b.set_aspect("equal")
fig12.subplots_adjust(right=0.88)
cbar_ax12 = fig12.add_axes([0.91, 0.15, 0.02, 0.7])
cb12 = fig12.colorbar(im12b, cax=cbar_ax12)
cb12.set_label("Temperature (K)", fontsize=11)
cb12.set_ticks([cut_off_1, cut_off_2])
cb12.set_ticklabels([f"{cut_off_1} K", f"{cut_off_2} K"])

_dr = os.path.join(FIGURES_DIR, "Hyperspectral_correction", "Range")
save_fig(fig1, os.path.join(_dr, "hyperspectral_no_downwelling_range"))
save_fig(fig2, os.path.join(_dr, "hyperspectral_with_downwelling_range"))
save_fig(fig5, os.path.join(_dr, "hyperspectral_profile_comparison"))

_dh = os.path.join(FIGURES_DIR, "Hyperspectral_correction")
save_fig(fig101, os.path.join(_dh, "hyperspectral_histogram_calibration_target_1"))
save_fig(fig102, os.path.join(_dh, "hyperspectral_histogram_calibration_target_2"))
save_fig(fig103, os.path.join(_dh, "hyperspectral_histogram_tree"))

_de = os.path.join(FIGURES_DIR, "Hyperspectral_correction", "Emissivity_clusters")
save_fig(fig6, os.path.join(_de, "cluster_image_no_downwelling"))
save_fig(fig7, os.path.join(_de, "cluster_spectrum_no_downwelling"))
save_fig(fig8, os.path.join(_de, "cluster_image_with_downwelling"))
save_fig(fig9, os.path.join(_de, "cluster_spectrum_with_downwelling"))
save_fig(fig13, os.path.join(_de, "cluster_image_comparison"))
save_fig(fig14, os.path.join(_de, "emissivity_spectra_key_pixels"))

_dt = os.path.join(FIGURES_DIR, "Hyperspectral_correction", "Temperature")
save_fig(fig10, os.path.join(_dt, "hyperspectral_with_downwelling_temperature"))
save_fig(fig11, os.path.join(_dt, "hyperspectral_no_downwelling_temperature"))
save_fig(fig12, os.path.join(_dt, "hyperspectral_temperature_comparison"))

# ── Pixel overlay on T and emissivity mean maps ──────────────────────────────
eps_mean_no = no_downwelling.emissivity.mean(axis=2)
eps_mean_dw = with_downwelling.emissivity.mean(axis=2)

fig_ov, axes_ov = plt.subplots(1, 4, figsize=(18, 5))
fig_ov.suptitle("Selected pixels on scene maps", fontsize=12)

overlay_cfg = [
    (with_downwelling.T[:, :],  "Temperature w/ dw (K)", "hot",     (285, 294)),
    (no_downwelling.T[:, :],    "Temperature no dw (K)",  "hot",     (285, 294)),
    (eps_mean_dw,               "Mean emissivity w/ dw",  "viridis", (0.7, 1.0)),
    (eps_mean_no,               "Mean emissivity no dw",  "viridis", (0.7, 1.0)),
]

for ax_ov, (data, title, cm, clim) in zip(axes_ov, overlay_cfg):
    im = ax_ov.imshow(data, cmap=cm, vmin=clim[0], vmax=clim[1])
    ax_ov.set_title(title, fontsize=9)
    ax_ov.axis("off")
    fig_ov.colorbar(im, ax=ax_ov, orientation="horizontal", pad=0.02, fraction=0.046)
    for px_idx, ((pr, pc), lbl) in enumerate(zip(_DEFAULT_PIXELS, _DEFAULT_LABELS)):
        col = _PIXEL_COLORS[px_idx % len(_PIXEL_COLORS)]
        ax_ov.plot(pc - 1, pr - 1, "o", color=col, markersize=6,
                   markeredgecolor="white", markeredgewidth=0.6)

# Legend outside
handles = [
    matplotlib.patches.Patch(color=_PIXEL_COLORS[i % len(_PIXEL_COLORS)], label=lbl)
    for i, lbl in enumerate(_DEFAULT_LABELS)
]
fig_ov.legend(handles=handles, loc="lower center", ncol=len(_DEFAULT_LABELS),
              fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig_ov.tight_layout()
save_fig(fig_ov, os.path.join(_dt, "pixel_overlay_maps"))

plt.show()
