import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.io import loadmat
from spectral.io import envi

FUNCTIONS_DIR = os.path.join(os.path.dirname(__file__), "Functions")
sys.path.insert(0, FUNCTIONS_DIR)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO_DIR, "Figures")


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".jpg", bbox_inches="tight", dpi=150)

from bispectral_estimation import bispectral_estimation
from quadspectral_estimation import quadspectral_estimation


def get_parula():
    try:
        return matplotlib.colormaps["parula"]
    except KeyError:
        return matplotlib.colormaps["viridis"]


def read_hdr_cube(hdr_path):
    img = envi.open(hdr_path)
    cube = np.asarray(img.load())
    wavelength = img.metadata.get("wavelength")
    if wavelength is not None:
        wavelength = np.array([float(value) for value in wavelength])
        if np.nanmax(wavelength) > 100:
            wavelength = wavelength / 1000.0
    return cube, wavelength


def load_array(data_dir, name, key):
    npz_path = os.path.join(data_dir, f"{name}.npz")
    mat_path = os.path.join(data_dir, f"{name}.mat")
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=False) as data:
            return data[key]
    return loadmat(mat_path, squeeze_me=True)[key]


plt.close("all")

DATA_DIR = os.getenv("DATA_DIR", "")  # Enter data directory
HDR_PATH = os.getenv("HDR_PATH", "")
PLOT_BOTH = os.getenv("PLOT_BOTH", "1") == "1"

if HDR_PATH:
    measurements, lambda_vals = read_hdr_cube(HDR_PATH)
    if lambda_vals is None:
        lambda_vals = load_array(DATA_DIR, "lambda", "lambda")
else:
    mat = loadmat(os.path.join(DATA_DIR, "DC2P5S1.mat"), squeeze_me=True)
    measurements = mat["measurements"]
    lambda_vals = load_array(DATA_DIR, "lambda", "lambda")
I_downwelling_res = load_array(DATA_DIR, "I_downwelling_res", "I_downwelling_res")
attenuation = load_array(DATA_DIR, "attenuation", "attenuation")

lidar = load_array(DATA_DIR, "lidar", "depthMap").astype(float)

lidar[lidar == 0] = np.nan

lambda_vals = lambda_vals[:247]
attenuation = attenuation[:247]
I_downwelling = I_downwelling_res[:247, :]

water_absorption_bands = np.array([4, 10, 14, 19, 23, 28, 31, 37, 42, 48, 54, 189, 230])
clear_bands = np.array([8, 13, 17, 21, 25, 30, 34, 39, 45, 50, 59, 195, 236])
ozone_bands = 77
ozone_clear = 81

cor_coeff = np.zeros(water_absorption_bands.shape[0])
for i in range(water_absorption_bands.shape[0]):
    Y = I_downwelling[water_absorption_bands[i] - 1, :] - I_downwelling[clear_bands[i] - 1, :]
    X = I_downwelling[ozone_bands - 1, :] - I_downwelling[ozone_clear - 1, :]
    cor_coeff[i] = (X @ Y) / (X @ X)

pair_index = 7


def _strip_nan(arr):
    return arr[~np.isnan(arr)]


def _stat(arr):
    valid = _strip_nan(arr.ravel())
    if len(valid) == 0:
        return "nan ± nan"
    mean = valid.mean()
    std = valid.std(ddof=1) if len(valid) > 1 else 0.0
    return f"{mean:.2f} ± {std:.2f}"


def run_mode(measurements, lidar, mode_label, fig_offset, crop_mode):
    if crop_mode == "crop":
        measurements = measurements[:256, 900:1156, :247]
        lidar = lidar[:256, 900:1156]
    else:
        measurements = measurements[:, :, :247]
        if lidar is not None:
            rows = min(measurements.shape[0], lidar.shape[0])
            cols = min(measurements.shape[1], lidar.shape[1])
            measurements = measurements[:rows, :cols, :]
            lidar = lidar[:rows, :cols]

    fig1 = plt.figure(num=1 + fig_offset)
    fig1.set_size_inches(7, 9.25)
    ax1 = fig1.add_subplot(111)

    angles = [
        r"$0^{\circ}$",
        r"$30^{\circ}$",
        r"$60^{\circ}$",
        r"$70^{\circ}$",
        r"$80^{\circ}$",
        r"$82^{\circ}$",
        r"$84^{\circ}$",
        r"$86^{\circ}$",
        r"$88^{\circ}$",
        r"$89^{\circ}$",
    ]

    num_curves = I_downwelling.shape[1]
    colors = np.column_stack(
        [
            np.linspace(0, 0.5, num_curves),
            np.linspace(0, 0.8, num_curves),
            np.linspace(0.5, 1, num_curves),
        ]
    )

    p1 = []
    for i in range(num_curves):
        line = ax1.plot(lambda_vals, I_downwelling[:, i], color=colors[i, :], linewidth=2)[0]
        p1.append(line)

    p2 = ax1.axvline(lambda_vals[water_absorption_bands[pair_index - 1] - 1], color=[0, 0, 0], linewidth=4)
    p3 = ax1.axvline(lambda_vals[clear_bands[pair_index - 1] - 1], color=[0, 0, 0], linewidth=4, linestyle="--")
    p4 = ax1.axvline(lambda_vals[ozone_bands - 1], color=[1, 0, 0], linewidth=4)
    p5 = ax1.axvline(lambda_vals[ozone_clear - 1], color=[1, 0, 0], linewidth=4, linestyle="--")

    ax1.set_xlim([8.5, 10])
    ax1.set_ylim([10, 820])
    ax1.set_xlabel("Wavelength (μm)")
    ax1.set_ylabel("Radiance (μW cm$^{-2}$ sr$^{-1}$ μm$^{-1}$)")
    ax1.tick_params(labelsize=25)

    legend_entries = angles + [
        r"$\lambda_1$: Water vapor absorption",
        r"$\lambda_2$: Transparent band",
        r"$\lambda_3$: Ozone absorption",
        r"$\lambda_4$: Transparent band",
    ]

    ax1.legend(p1 + [p2, p3, p4, p5], legend_entries, loc="lower center", ncol=3)

    fig2 = plt.figure(num=2 + fig_offset)
    fig2.set_size_inches(7, 9.25)
    ax2 = fig2.add_subplot(111)

    Y = I_downwelling[water_absorption_bands[pair_index - 1] - 1, :] - I_downwelling[clear_bands[pair_index - 1] - 1, :]
    X = I_downwelling[ozone_bands - 1, :] - I_downwelling[ozone_clear - 1, :]

    num_points = X.shape[0]
    colors = np.column_stack(
        [
            np.linspace(0, 0.5, num_points),
            np.linspace(0, 0.8, num_points),
            np.linspace(0.5, 1, num_points),
        ]
    )

    p1 = []
    for i in range(num_points):
        line = ax2.plot(
            X[i],
            Y[i],
            "o",
            markerfacecolor=colors[i, :],
            markeredgecolor=colors[i, :],
            linewidth=7,
        )[0]
        p1.append(line)

    x_line = np.linspace(X.min(), X.max(), 100)
    p2 = ax2.plot(x_line, cor_coeff[pair_index - 1] * x_line, "--", linewidth=3, color=[0, 0, 0])[0]

    ax2.set_xlabel(r"$L_D(\lambda_4) - L_D(\lambda_3)$")
    ax2.set_ylabel(r"$L_D(\lambda_2) - L_D(\lambda_1)$")
    ax2.set_xlim([X.min(), X.max()])
    ax2.tick_params(labelsize=25)

    legend_entries = angles + ["Estimated Correlation"]
    ax2.legend(p1 + [p2], legend_entries, loc="lower center", ncol=3)

    cmap = get_parula()
    colors = np.vstack(([0, 0, 0, 1], cmap(np.linspace(0, 1, 256))[::-1]))
    cmap = ListedColormap(colors)
    cut_off_1 = 0
    cut_off_2 = 150

    center_target_1 = (187, 86)
    center_target_2 = (145, 18)
    center_tree = (70, 160)

    line_index = 20
    range_pair_index = 5
    T_air = 289.7
    index_1 = water_absorption_bands[range_pair_index - 1] - 1
    index_2 = clear_bands[range_pair_index - 1] - 1
    index_3 = ozone_bands - 1
    index_4 = ozone_clear - 1

    lambda_vec = lambda_vals.reshape(-1)
    attenuation_vec = attenuation.reshape(-1)

    d_hat_1 = bispectral_estimation(lambda_vec, measurements, attenuation_vec, index_1, index_2, T_air)
    d_hat_2 = quadspectral_estimation(
        lambda_vec,
        measurements,
        attenuation_vec,
        index_1,
        index_2,
        index_3,
        index_4,
        T_air,
        cor_coeff[range_pair_index - 1],
    )

    fig101, ax101 = plt.subplots(num=101 + fig_offset)
    ax101.imshow(d_hat_1, cmap=cmap)
    ax101.set_aspect("equal")
    ax101.axis("off")
    ax101.images[0].set_clim(cut_off_1, cut_off_2)
    cb101 = fig101.colorbar(ax101.images[0], ax=ax101, orientation="horizontal", pad=0.05)
    cb101.set_label("Distance (m)")
    cb101.set_ticks([cut_off_1, cut_off_2])
    cb101.set_ticklabels([str(cut_off_1), str(cut_off_2)])
    ax101.tick_params(labelsize=20)
    ax101.axvline(line_index - 1, color=[1, 0, 0], linewidth=2, linestyle="--")

    fig102, ax102 = plt.subplots(num=102 + fig_offset)
    ax102.imshow(d_hat_2, cmap=cmap)
    ax102.set_aspect("equal")
    ax102.axis("off")
    ax102.images[0].set_clim(cut_off_1, cut_off_2)
    cb102 = fig102.colorbar(ax102.images[0], ax=ax102, orientation="horizontal", pad=0.05)
    cb102.set_label("Distance (m)")
    cb102.set_ticks([cut_off_1, cut_off_2])
    cb102.set_ticklabels([str(cut_off_1), str(cut_off_2)])
    ax102.tick_params(labelsize=20)
    ax102.axvline(line_index - 1, color=[1, 0, 0], linewidth=2, linestyle="--")

    fig103, ax103 = plt.subplots(num=103 + fig_offset)
    ax103.imshow(np.abs(measurements[:, :, ozone_bands - 1] - measurements[:, :, ozone_clear - 1]), cmap="gray")
    ax103.set_aspect("equal")
    ax103.axis("off")
    ax103.images[0].set_clim(0, 7)
    cb103 = fig103.colorbar(ax103.images[0], ax=ax103, orientation="horizontal", pad=0.05)
    cb103.set_ticks([0, 7])
    cb103.set_label("Radiance (μW cm$^{-2}$ sr$^{-1}$ μm$^{-1}$)")
    ax103.tick_params(labelsize=20)

    fig104, ax104 = plt.subplots(num=104 + fig_offset)
    ax104.imshow(lidar, cmap=cmap)
    ax104.text(center_target_1[1] - 6, center_target_1[0] - 1, "(f)", color="red", fontsize=10, fontweight="bold")
    ax104.text(center_target_2[1] - 6, center_target_2[0] - 1, "(g)", color="red", fontsize=10, fontweight="bold")
    ax104.text(center_tree[1] - 6, center_tree[0] - 1, "(h)", color="red", fontsize=10, fontweight="bold")
    ax104.set_aspect("equal")
    ax104.axis("off")
    ax104.images[0].set_clim(cut_off_1, cut_off_2)
    cb104 = fig104.colorbar(ax104.images[0], ax=ax104, orientation="horizontal", pad=0.05)
    cb104.set_label("Distance (m)")
    cb104.set_ticks([cut_off_1, cut_off_2])
    cb104.set_ticklabels([str(cut_off_1), str(cut_off_2)])
    ax104.tick_params(labelsize=20)
    ax104.axvline(line_index - 1, color=[1, 0, 0], linewidth=2, linestyle="--")

    fig105 = plt.figure(num=105 + fig_offset)
    fig105.set_size_inches(3.5 * 5.6, 2.5)
    ax105 = fig105.add_subplot(111)
    ax105.plot(d_hat_1[:, line_index - 1], linewidth=2)
    ax105.plot(d_hat_2[:, line_index - 1], linewidth=2)
    ax105.plot(lidar[:, line_index - 1], "--", linewidth=3, color=[0, 0, 0])
    ax105.set_xlim([0, 256])
    ax105.set_ylim([0, 200])
    ax105.legend(["Bispectral (baseline)", "Quadspectral", "Lidar (ground truth)"])
    ax105.set_xlabel("Pixel Index")
    ax105.set_ylabel("Distance (m)")
    ax105.tick_params(labelsize=15)
    for spine in ax105.spines.values():
        spine.set_linewidth(1.5)

    patch_size = 8
    half_patch = patch_size // 2

    def extract_patch(M, cp):
        start_row = cp[0] - half_patch + 1
        end_row = cp[0] + half_patch
        start_col = cp[1] - half_patch + 1
        end_col = cp[1] + half_patch
        return M[start_row - 1 : end_row, start_col - 1 : end_col]

    patch_t1_d1 = extract_patch(d_hat_1, center_target_1)
    patch_t1_d2 = extract_patch(d_hat_2, center_target_1)
    patch_t1_lidar = extract_patch(lidar, center_target_1)

    patch_t2_d1 = extract_patch(d_hat_1, center_target_2)
    patch_t2_d2 = extract_patch(d_hat_2, center_target_2)
    patch_t2_lidar = extract_patch(lidar, center_target_2)

    patch_tree_d1 = extract_patch(d_hat_1, center_tree)
    patch_tree_d2 = extract_patch(d_hat_2, center_tree)
    patch_tree_lidar = extract_patch(lidar, center_tree)

    all_values = np.concatenate(
        [
            patch_t1_d1.ravel(),
            patch_t1_d2.ravel(),
            patch_t1_lidar.ravel(),
            patch_t2_d1.ravel(),
            patch_t2_d2.ravel(),
            patch_t2_lidar.ravel(),
            patch_tree_d1.ravel(),
            patch_tree_d2.ravel(),
            patch_tree_lidar.ravel(),
        ]
    )

    bin_width = 5
    valid_values = all_values[~np.isnan(all_values)]
    bin_min = np.floor(valid_values.min() / bin_width) * bin_width if len(valid_values) > 0 else 0
    bin_max = 200
    bin_edges = np.arange(bin_min, bin_max + bin_width, bin_width)

    color_d1 = [0, 0, 1]
    color_d2 = [1, 0, 0]
    color_gt = [0, 0, 0]

    fig201, ax201 = plt.subplots(num=201 + fig_offset)
    fig201.set_size_inches(7, 4.2)
    ax201.hist(_strip_nan(patch_t1_d1.ravel()), bins=bin_edges, color=color_d1, alpha=0.6, label="Bispectral")
    ax201.hist(_strip_nan(patch_t1_d2.ravel()), bins=bin_edges, color=color_d2, alpha=0.6, label="Quadspectral")
    ax201.hist(_strip_nan(patch_t1_lidar.ravel()), bins=bin_edges, color=color_gt, alpha=0.6, label="Lidar (GT)")
    ax201.legend()
    ax201.set_xlabel("Distance (m)")
    ax201.set_ylabel("Count")
    ax201.tick_params(labelsize=25)

    fig202, ax202 = plt.subplots(num=202 + fig_offset)
    fig202.set_size_inches(7, 4.2)
    ax202.hist(_strip_nan(patch_t2_d1.ravel()), bins=bin_edges, color=color_d1, alpha=0.6, label="Bispectral")
    ax202.hist(_strip_nan(patch_t2_d2.ravel()), bins=bin_edges, color=color_d2, alpha=0.6, label="Quadspectral")
    ax202.hist(_strip_nan(patch_t2_lidar.ravel()), bins=bin_edges, color=color_gt, alpha=0.6, label="Lidar (GT)")
    ax202.legend()
    ax202.set_xlabel("Distance (m)")
    ax202.set_ylabel("Count")
    ax202.tick_params(labelsize=25)

    fig203, ax203 = plt.subplots(num=203 + fig_offset)
    fig203.set_size_inches(7, 4.2)
    ax203.hist(_strip_nan(patch_tree_d1.ravel()), bins=bin_edges, color=color_d1, alpha=0.6, label="Bispectral")
    ax203.hist(_strip_nan(patch_tree_d2.ravel()), bins=bin_edges, color=color_d2, alpha=0.6, label="Quadspectral")
    ax203.hist(_strip_nan(patch_tree_lidar.ravel()), bins=bin_edges, color=color_gt, alpha=0.6, label="Lidar (GT)")
    ax203.legend()
    ax203.set_xlabel("Distance (m)")
    ax203.set_ylabel("Count")
    ax203.tick_params(labelsize=25)

    print(f"\n--- Patch Statistics ({mode_label}) (Mean ± Std, ignoring NaNs) ---")

    print("\nCalibration Target 1:")
    print(f"  Bispectral (Baseline): {_stat(patch_t1_d1)} m")
    print(f"  Quadspectral:           {_stat(patch_t1_d2)} m")
    print(f"  Lidar (Ground Truth):   {_stat(patch_t1_lidar)} m")

    print("\nCalibration Target 2:")
    print(f"  Bispectral (Baseline): {_stat(patch_t2_d1)} m")
    print(f"  Quadspectral:           {_stat(patch_t2_d2)} m")
    print(f"  Lidar (Ground Truth):   {_stat(patch_t2_lidar)} m")

    print("\nTree:")
    print(f"  Bispectral (Baseline): {_stat(patch_tree_d1)} m")
    print(f"  Quadspectral:           {_stat(patch_tree_d2)} m")
    print(f"  Lidar (Ground Truth):   {_stat(patch_tree_lidar)} m")

    if crop_mode == "crop":
        _d = os.path.join(FIGURES_DIR, "Bispectral_correction")
        save_fig(fig1, os.path.join(_d, "bispectral_correlation_1"))
        save_fig(fig2, os.path.join(_d, "bispectral_correlation_2"))
        save_fig(fig101, os.path.join(_d, "bispectral_no_downwelling"))
        save_fig(fig102, os.path.join(_d, "bispectral_with_downwelling"))
        save_fig(fig103, os.path.join(_d, "bispectral_ozone_feature"))
        save_fig(fig104, os.path.join(_d, "bispectral_lidar"))
        save_fig(fig105, os.path.join(_d, "bispectral_profile_comparison"))
        save_fig(fig201, os.path.join(_d, "bispectral_histogram_calibration_target_1"))
        save_fig(fig202, os.path.join(_d, "bispectral_histogram_calibration_target_2"))
        save_fig(fig203, os.path.join(_d, "bispectral_histogram_tree"))
if PLOT_BOTH:
    run_mode(measurements, lidar, "crop", 0, "crop")
    run_mode(measurements, lidar, "full", 1000, "full")
else:
    run_mode(measurements, lidar, "crop", 0, "crop")

plt.show()
