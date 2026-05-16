import os
import sys
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.io import loadmat

FUNCTIONS_DIR = os.path.join(os.path.dirname(__file__), "Functions")
sys.path.insert(0, FUNCTIONS_DIR)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO_DIR, "Figures")


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".jpg", bbox_inches="tight", dpi=150)


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


def plot_circle(ax, center, radius, color):
    theta = np.linspace(0, 2 * np.pi, 100)
    x = center[1] + radius * np.cos(theta)
    y = center[0] + radius * np.sin(theta)
    ax.plot(x, y, linewidth=2, color=color, linestyle="--")


# %%
plt.close("all")

DATA_DIR = os.getenv("DATA_DIR", "")  # Enter data directory
RESULTS_DIR = os.getenv("RESULTS_DIR", "")  # Enter results directory

K = 247
Q = 10

no_downwelling = load_results(RESULTS_DIR, "no_downwelling")
with_downwelling = load_results(RESULTS_DIR, "downwelling")
with_downwelling_reg = load_results(RESULTS_DIR, "downwelling_reg")

no_downwelling.V = no_downwelling.V[:, :, :, :Q]
with_downwelling.V = with_downwelling.V[:, :, :, :Q]
with_downwelling_reg.V = with_downwelling_reg.V[:, :, :, :Q]

lambda_vals = load_array(DATA_DIR, "lambda", "lambda")[:K]
meas = loadmat(os.path.join(DATA_DIR, "DC2P5S1.mat"), squeeze_me=True)["measurements"][:256, 900:1156, :K]
attenuation = load_array(DATA_DIR, "attenuation", "attenuation")[:K]
reflected = load_array(DATA_DIR, "I_downwelling_res", "I_downwelling_res")[:K, :]

lambda_vals = lambda_vals.reshape(1, 1, K) - 0.120
attenuation = attenuation.reshape(1, 1, K)
reflected = reflected.reshape(1, 1, K, Q)

cmap = get_parula()
cmap = ListedColormap(cmap(np.linspace(0, 1, 256))[::-1])
cut_off_1 = 0
cut_off_2 = 150

pixel_reflective_panel_1 = (143, 16)
pixel_grass_area = (68, 16)
pixel_sky = (5, 16)

ozone_bands = 77

square_colors = np.array(
    [
        [0.71, 0.27, 0.61],
        [0.13, 0.54, 0.13],
        [0.40, 0.60, 0.85],
    ]
)

circle_centers = np.array([[100, 15], [150, 15], [191, 82]])
circle_radius = 15

# No downwelling
fig1, ax1 = plt.subplots(num=1)
ax1.set_position([0.05, 0.05, 0.9, 0.75])
im1 = ax1.imshow(no_downwelling.d[:, :], cmap=cmap)
im1.set_clim(cut_off_1, cut_off_2)
cb1 = fig1.colorbar(im1, ax=ax1, orientation="horizontal", pad=0.05)
cb1.set_label("Distance (m)")
cb1.set_ticks([cut_off_1, cut_off_2])
cb1.set_ticklabels([str(cut_off_1), str(cut_off_2)])
ax1.tick_params(labelsize=25)
ax1.set_aspect("equal")
ax1.axis("off")

for center in circle_centers:
    plot_circle(ax1, center, circle_radius, [1, 0, 0])

ax1.plot(
    pixel_reflective_panel_1[1] - 1,
    pixel_reflective_panel_1[0] - 1,
    "x",
    color=square_colors[0],
    markersize=15,
    linewidth=3,
)
ax1.plot(
    pixel_grass_area[1] - 1,
    pixel_grass_area[0] - 1,
    "x",
    color=square_colors[1],
    markersize=15,
    linewidth=3,
)
ax1.plot(
    pixel_sky[1] - 1,
    pixel_sky[0] - 1,
    "x",
    color=square_colors[2],
    markersize=15,
    linewidth=3,
)

# With downwelling
fig2, ax2 = plt.subplots(num=2)
ax2.set_position([0.05, 0.05, 0.9, 0.75])
im2 = ax2.imshow(with_downwelling_reg.d[:, :], cmap=cmap)
im2.set_clim(cut_off_1, cut_off_2)
cb2 = fig2.colorbar(im2, ax=ax2, orientation="horizontal", pad=0.05)
cb2.set_label("Distance (m)")
cb2.set_ticks([cut_off_1, cut_off_2])
cb2.set_ticklabels([str(cut_off_1), str(cut_off_2)])
ax2.tick_params(labelsize=25)
ax2.set_aspect("equal")
ax2.axis("off")

for center in circle_centers:
    plot_circle(ax2, center, circle_radius, [1, 0, 0])

ax2.plot(
    pixel_reflective_panel_1[1] - 1,
    pixel_reflective_panel_1[0] - 1,
    "x",
    color=square_colors[0],
    markersize=15,
    linewidth=3,
)
ax2.plot(
    pixel_grass_area[1] - 1,
    pixel_grass_area[0] - 1,
    "x",
    color=square_colors[1],
    markersize=15,
    linewidth=3,
)
ax2.plot(
    pixel_sky[1] - 1,
    pixel_sky[0] - 1,
    "x",
    color=square_colors[2],
    markersize=15,
    linewidth=3,
)

# Figure 3 - Spectral plots
fig3 = plt.figure(num=3)
fig3.set_size_inches(14.5, 8)
ax3_left = fig3.add_subplot(111)
ax3_left.tick_params(labelsize=35)

spec_reflective_panel = meas[pixel_reflective_panel_1[0] - 1, pixel_reflective_panel_1[1] - 1, :].squeeze()
spec_grass_area = meas[pixel_grass_area[0] - 1, pixel_grass_area[1] - 1, :].squeeze()
spec_sky = meas[pixel_sky[0] - 1, pixel_sky[1] - 1, :].squeeze()

lambda_vec = lambda_vals.reshape(-1)

h_ref = ax3_left.plot(lambda_vec, spec_reflective_panel, "-", color=square_colors[0], linewidth=3)[0]
h_sky = ax3_left.plot(lambda_vec, spec_sky, "-", color=square_colors[2], linewidth=3)[0]
ax3_right = ax3_left.twinx()
h_grass = ax3_right.plot(lambda_vec, spec_grass_area, "-", color=square_colors[1], linewidth=3)[0]
ax3_right.set_ylim([670, 800])
ax3_right.tick_params(axis="y", colors=square_colors[1])
ax3_right.set_ylabel("Radiance (μW cm$^{-2}$ sr$^{-1}$ μm$^{-1}$)")

ax3_left.set_xlim([lambda_vec.min(), lambda_vec.max()])
ax3_left.set_ylim([500, 900])
ax3_left.set_xlabel("Wavelength (μm)")
ax3_left.set_ylabel("Radiance (μW cm$^{-2}$ sr$^{-1}$ μm$^{-1}$)")
ax3_left.set_xlim([8, 13])
ax3_left.set_xticks(np.arange(8, 14))
ax3_left.tick_params(labelsize=35)

ozone_band_start = lambda_vec[ozone_bands - 1] - 0.1
ozone_band_end = lambda_vec[ozone_bands - 1] + 0.5
y_limits = [200, 875]
ozone_patch = ax3_left.fill(
    [ozone_band_start, ozone_band_end, ozone_band_end, ozone_band_start],
    [y_limits[0], y_limits[0], y_limits[1], y_limits[1]],
    [0, 0, 0],
    alpha=0.2,
    edgecolor="none",
)[0]

ax3_left.legend(
    [h_ref, h_sky, h_grass, ozone_patch],
    ["Reflective panel", "Sky", "Grass area", "Ozone absorption band"],
    loc="upper center",
    ncol=2,
)

_d = os.path.join(FIGURES_DIR, "Figure_introduction")
save_fig(fig1, os.path.join(_d, "introduction_hyperspectral_no_downwelling_range"))
save_fig(fig2, os.path.join(_d, "introduction_hyperspectral_with_downwelling_range"))
save_fig(fig3, os.path.join(_d, "introduction_hyperspectral_spectral_signatures"))

plt.show()
