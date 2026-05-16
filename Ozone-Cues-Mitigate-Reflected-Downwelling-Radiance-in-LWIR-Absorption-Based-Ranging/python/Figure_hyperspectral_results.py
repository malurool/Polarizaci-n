import os
import sys
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.io import loadmat
from sklearn.cluster import KMeans

FUNCTIONS_DIR = os.path.join(os.path.dirname(__file__), "Functions")
sys.path.insert(0, FUNCTIONS_DIR)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO_DIR, "Figures")


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".jpg", bbox_inches="tight", dpi=150)

from forward_model import forward_model


def _mat_struct_to_namespace(mat_obj):
    ns = SimpleNamespace()
    for field in mat_obj._fieldnames:
        value = getattr(mat_obj, field)
        if hasattr(value, "_fieldnames"):
            value = _mat_struct_to_namespace(value)
        setattr(ns, field, value)
    return ns


def load_mat_as_namespace(path):
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    data = {key: value for key, value in mat.items() if not key.startswith("__")}
    if len(data) == 1 and hasattr(list(data.values())[0], "_fieldnames"):
        return _mat_struct_to_namespace(list(data.values())[0])
    return SimpleNamespace(**data)


def get_parula():
    try:
        return matplotlib.colormaps["parula"]
    except KeyError:
        return matplotlib.colormaps["viridis"]


def load_array(data_dir, name, key):
    npz_path = os.path.join(data_dir, f"{name}.npz")
    mat_path = os.path.join(data_dir, f"{name}.mat")
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=False) as data:
            return data[key]
    return loadmat(mat_path, squeeze_me=True)[key]


def load_results(results_dir, name):
    npz_path = os.path.join(results_dir, f"{name}.npz")
    mat_path = os.path.join(results_dir, f"{name}.mat")
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=True) as data:
            return SimpleNamespace(**{key: data[key] for key in data.files})
    return load_mat_as_namespace(mat_path)


plt.close("all")

DATA_DIR = os.getenv("DATA_DIR", "")  # Enter data directory
RESULTS_DIR = os.getenv("RESULTS_DIR", "")  # Enter results directory

K = 247
Q = 10

no_downwelling = load_results(RESULTS_DIR, "no_downwelling")
with_downwelling = load_results(RESULTS_DIR, "downwelling")
with_downwelling_reg = load_results(RESULTS_DIR, "downwelling_reg")

no_downwelling.d = no_downwelling.d.astype(float)
with_downwelling.d = with_downwelling.d.astype(float)
with_downwelling_reg.d = with_downwelling_reg.d.astype(float)

no_downwelling.d[no_downwelling.d == 0] = np.nan
with_downwelling.d[with_downwelling.d == 0] = np.nan
with_downwelling_reg.d[with_downwelling_reg.d == 0] = np.nan

no_downwelling.V = no_downwelling.V[:, :, :, :Q]
with_downwelling.V = with_downwelling.V[:, :, :, :Q]
with_downwelling_reg.V = with_downwelling_reg.V[:, :, :, :Q]

lambda_vals = load_array(DATA_DIR, "lambda", "lambda")[:K]
meas = loadmat(os.path.join(DATA_DIR, "DC2P5S1.mat"), squeeze_me=True)["measurements"][:256, 900:1156, :K]
attenuation = load_array(DATA_DIR, "attenuation", "attenuation")[:K]
reflected = load_array(DATA_DIR, "I_downwelling_res", "I_downwelling_res")[:K, :]

lidar = load_array(DATA_DIR, "lidar", "depthMap")[:256, 900:1156].astype(float)
lidar[lidar == 0] = np.nan

lambda_vals = lambda_vals.reshape(1, 1, K)
attenuation = attenuation.reshape(1, 1, K)
reflected = reflected.reshape(1, 1, K, Q)

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

fig3, ax3 = plt.subplots(num=3)
ax3.imshow(with_downwelling_reg.d[:, :], cmap=cmap)
ax3.images[0].set_clim(cut_off_1, cut_off_2)
cb3 = fig3.colorbar(ax3.images[0], ax=ax3, orientation="horizontal", pad=0.05)
cb3.set_label("Distance (m)")
cb3.set_ticks([cut_off_1, cut_off_2])
cb3.set_ticklabels([str(cut_off_1), str(cut_off_2)])
ax3.tick_params(labelsize=20)
ax3.set_aspect("equal")
ax3.axis("off")
ax3.axvline(line_index - 1, color=[1, 0, 0], linewidth=2, linestyle="--")

fig4, ax4 = plt.subplots(num=4)
colors = np.vstack(([0, 0, 0, 1], get_parula()(np.linspace(0, 1, 256))[::-1]))
ax4.imshow(lidar, cmap=ListedColormap(colors))
ax4.text(center_target_1[1] - 6, center_target_1[0] - 1, "(f)", color="red", fontsize=10, fontweight="bold")
ax4.text(center_target_2[1] - 6, center_target_2[0] - 1, "(g)", color="red", fontsize=10, fontweight="bold")
ax4.text(center_tree[1] - 6, center_tree[0] - 1, "(h)", color="red", fontsize=10, fontweight="bold")
ax4.images[0].set_clim(cut_off_1, cut_off_2)
cb4 = fig4.colorbar(ax4.images[0], ax=ax4, orientation="horizontal", pad=0.05)
cb4.set_label("Distance (m)")
cb4.set_ticks([cut_off_1, cut_off_2])
cb4.set_ticklabels([str(cut_off_1), str(cut_off_2)])
ax4.tick_params(labelsize=20)
ax4.set_aspect("equal")
ax4.axis("off")
ax4.axvline(line_index - 1, color=[1, 0, 0], linewidth=2, linestyle="--")

fig5 = plt.figure(num=5)
fig5.set_size_inches(3.5 * 5.6, 2.5)
ax5 = fig5.add_subplot(111)
ax5.plot(no_downwelling.d[:, line_index - 1], linewidth=2)
ax5.plot(with_downwelling.d[:, line_index - 1], linewidth=2)
ax5.plot(with_downwelling_reg.d[:, line_index - 1], linewidth=2)
ax5.plot(lidar[:, line_index - 1], "--", linewidth=3, color=[0, 0, 0])
ax5.set_ylim([15, 200])
ax5.set_xlim([0, 256])
ax5.set_xlabel("Pixel Index")
ax5.set_ylabel("Distance (m)")
ax5.legend(
    [
        "Without downwelling correction",
        "Downwelling correction",
        "Downwelling correction (TV)",
        "Lidar (ground truth)",
    ],
    loc="upper right",
    ncol=2,
)
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
patch_t1_tv = extract_patch(with_downwelling_reg.d, center_target_1)
patch_t1_lidar = extract_patch(lidar, center_target_1)

patch_t2_d1 = extract_patch(no_downwelling.d, center_target_2)
patch_t2_d2 = extract_patch(with_downwelling.d, center_target_2)
patch_t2_tv = extract_patch(with_downwelling_reg.d, center_target_2)
patch_t2_lidar = extract_patch(lidar, center_target_2)

patch_tree_d1 = extract_patch(no_downwelling.d, center_tree)
patch_tree_d2 = extract_patch(with_downwelling.d, center_tree)
patch_tree_tv = extract_patch(with_downwelling_reg.d, center_tree)
patch_tree_lidar = extract_patch(lidar, center_tree)

all_values = np.concatenate(
    [
        patch_t1_d1.ravel(),
        patch_t1_d2.ravel(),
        patch_t1_tv.ravel(),
        patch_t1_lidar.ravel(),
        patch_t2_d1.ravel(),
        patch_t2_d2.ravel(),
        patch_t2_tv.ravel(),
        patch_t2_lidar.ravel(),
        patch_tree_d1.ravel(),
        patch_tree_d2.ravel(),
        patch_tree_tv.ravel(),
        patch_tree_lidar.ravel(),
    ]
)

bin_width = 5
bin_min = np.floor(np.nanmin(all_values) / bin_width) * bin_width
bin_max = 200
bin_edges = np.arange(bin_min, bin_max + bin_width, bin_width)

color_d1 = [0, 0, 1]
color_d2 = [1, 0, 0]
color_tv = [1, 1, 0]
color_gt = [0, 0, 0]

fig101, ax101 = plt.subplots(num=101)
fig101.set_size_inches(7, 4.2)
ax101.hist(patch_t1_d1.ravel(), bins=bin_edges, color=color_d1, alpha=0.6, label="w/o down. corr.")
ax101.hist(patch_t1_d2.ravel(), bins=bin_edges, color=color_d2, alpha=0.6, label="w/ down. corr.")
ax101.hist(patch_t1_tv.ravel(), bins=bin_edges, color=color_tv, alpha=0.6, label="w/ down. corr. (TV)")
ax101.hist(patch_t1_lidar.ravel(), bins=bin_edges, color=color_gt, alpha=0.6, label="Lidar (GT)")
ax101.legend()
ax101.set_xlabel("Distance (m)")
ax101.set_ylabel("Count")
ax101.tick_params(labelsize=25)

fig102, ax102 = plt.subplots(num=102)
fig102.set_size_inches(7, 4.2)
ax102.hist(patch_t2_d1.ravel(), bins=bin_edges, color=color_d1, alpha=0.6, label="w/o down. corr.")
ax102.hist(patch_t2_d2.ravel(), bins=bin_edges, color=color_d2, alpha=0.6, label="w/ down. corr.")
ax102.hist(patch_t2_tv.ravel(), bins=bin_edges, color=color_tv, alpha=0.6, label="w/ down. corr. (TV)")
ax102.hist(patch_t2_lidar.ravel(), bins=bin_edges, color=color_gt, alpha=0.6, label="Lidar (GT)")
ax102.legend()
ax102.set_xlabel("Distance (m)")
ax102.set_ylabel("Count")
ax102.tick_params(labelsize=25)

fig103, ax103 = plt.subplots(num=103)
fig103.set_size_inches(7, 4.2)
ax103.hist(patch_tree_d1.ravel(), bins=bin_edges, color=color_d1, alpha=0.6, label="w/o down. corr.")
ax103.hist(patch_tree_d2.ravel(), bins=bin_edges, color=color_d2, alpha=0.6, label="w/ down. corr.")
ax103.hist(patch_tree_tv.ravel(), bins=bin_edges, color=color_tv, alpha=0.6, label="w/ down. corr. (TV)")
ax103.hist(patch_tree_lidar.ravel(), bins=bin_edges, color=color_gt, alpha=0.6, label="Lidar (GT)")
ax103.legend()
ax103.set_xlabel("Distance (m)")
ax103.set_ylabel("Count")
ax103.tick_params(labelsize=25)

print("\n--- Patch Statistics (Mean ± Std, ignoring NaNs) ---")

print("\nCalibration Target 1:")
print(
    f"  Without downwelling correction: {np.nanmean(patch_t1_d1):.2f} ± {np.nanstd(patch_t1_d1, ddof=1):.2f} m"
)
print(
    f"  Downwelling correction:         {np.nanmean(patch_t1_d2):.2f} ± {np.nanstd(patch_t1_d2, ddof=1):.2f} m"
)
print(
    f"  Downwelling correction (TV):    {np.nanmean(patch_t1_tv):.2f} ± {np.nanstd(patch_t1_tv, ddof=1):.2f} m"
)
print(
    f"  Lidar (Ground Truth):           {np.nanmean(patch_t1_lidar):.2f} ± {np.nanstd(patch_t1_lidar, ddof=1):.2f} m"
)

print("\nCalibration Target 2:")
print(
    f"  Without downwelling correction: {np.nanmean(patch_t2_d1):.2f} ± {np.nanstd(patch_t2_d1, ddof=1):.2f} m"
)
print(
    f"  Downwelling correction:         {np.nanmean(patch_t2_d2):.2f} ± {np.nanstd(patch_t2_d2, ddof=1):.2f} m"
)
print(
    f"  Downwelling correction (TV):    {np.nanmean(patch_t2_tv):.2f} ± {np.nanstd(patch_t2_tv, ddof=1):.2f} m"
)
print(
    f"  Lidar (Ground Truth):           {np.nanmean(patch_t2_lidar):.2f} ± {np.nanstd(patch_t2_lidar, ddof=1):.2f} m"
)

print("\nTree:")
print(
    f"  Without downwelling correction: {np.nanmean(patch_tree_d1):.2f} ± {np.nanstd(patch_tree_d1, ddof=1):.2f} m"
)
print(
    f"  Downwelling correction:         {np.nanmean(patch_tree_d2):.2f} ± {np.nanstd(patch_tree_d2, ddof=1):.2f} m"
)
print(
    f"  Downwelling correction (TV):    {np.nanmean(patch_tree_tv):.2f} ± {np.nanstd(patch_tree_tv, ddof=1):.2f} m"
)
print(
    f"  Lidar (Ground Truth):           {np.nanmean(patch_tree_lidar):.2f} ± {np.nanstd(patch_tree_lidar, ddof=1):.2f} m"
)

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
save_fig(fig3, os.path.join(_dr, "hyperspectral_with_downwelling_TV"))
save_fig(fig4, os.path.join(_dr, "hyperspectral_lidar"))
save_fig(fig5, os.path.join(_dr, "hyperspectral_profile_comparison_TV"))

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

plt.show()
