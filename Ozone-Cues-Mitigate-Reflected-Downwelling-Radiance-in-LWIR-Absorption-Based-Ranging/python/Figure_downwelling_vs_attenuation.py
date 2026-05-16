import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

FUNCTIONS_DIR = os.path.join(os.path.dirname(__file__), "Functions")
sys.path.insert(0, FUNCTIONS_DIR)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO_DIR, "Figures")


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".jpg", bbox_inches="tight", dpi=150)

from read_transmittance_file import read_transmittance_file


def load_array(data_dir, name, key):
    npz_path = os.path.join(data_dir, f"{name}.npz")
    mat_path = os.path.join(data_dir, f"{name}.mat")
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=False) as data:
            return data[key]
    return loadmat(mat_path, squeeze_me=True)[key]

plt.close("all")

DATA_DIR = os.getenv("DATA_DIR", "")  # Enter data directory

attenuation = load_array(DATA_DIR, "attenuation", "attenuation")
I_downwelling_res = load_array(DATA_DIR, "I_downwelling_res", "I_downwelling_res")

ozone = {}
water_vapor = {}
ozone["wavelength"], ozone["transmittance"] = read_transmittance_file(
    os.path.join(DATA_DIR, "transmittance_ozone.txt")
)
water_vapor["wavelength"], water_vapor["transmittance"] = read_transmittance_file(
    os.path.join(DATA_DIR, "transmittance_water_vapor.txt")
)

try:
    lambda_vals = load_array(DATA_DIR, "attenuation", "lambda")
except KeyError:
    lambda_vals = load_array(DATA_DIR, "lambda", "lambda")

lambda_vals = np.asarray(lambda_vals).squeeze()

ozone_bands = 77
ozone_attenuation = -10 * np.log10(np.interp(lambda_vals, ozone["wavelength"], ozone["transmittance"]))
water_vapor_attenuation = -10 * np.log10(np.interp(lambda_vals, water_vapor["wavelength"], water_vapor["transmittance"]))

fig1 = plt.figure(num=1)
fig1.set_size_inches(2.5 * 5.6, 8)
ax1 = fig1.add_subplot(111)
ax1.plot(lambda_vals, attenuation, linewidth=2)
ax1.set_ylabel("Attenuation (dB/m)")
ax1_right = ax1.twinx()
ax1_right.plot(lambda_vals, I_downwelling_res[:, 0], "--", linewidth=3)

ozone_band_start = lambda_vals[ozone_bands - 1] - 0.1
ozone_band_end = lambda_vals[ozone_bands - 1] + 0.5
y_limits = [0, 800]
ax1_right.fill(
    [ozone_band_start, ozone_band_end, ozone_band_end, ozone_band_start],
    [y_limits[0], y_limits[0], y_limits[1], y_limits[1]],
    [0, 0, 0],
    alpha=0.2,
    edgecolor="none",
)
ax1_right.set_ylabel("Radiance (μW cm$^{-2}$ sr$^{-1}$ μm$^{-1}$)")
ax1.set_xlim([lambda_vals.min(), lambda_vals.max()])
ax1_right.set_ylim([0, 700])
ax1.set_xlabel("Wavelength (μm)")
ax1.legend(["Ground-level attenuation", "Downwelling radiance", "Ozone absorption band"], loc="upper right")
ax1.tick_params(labelsize=35)

fig2 = plt.figure(num=2)
fig2.set_size_inches(2.5 * 5.6, 6)
ax2 = fig2.add_subplot(111)
ax2.plot(lambda_vals, water_vapor_attenuation, color=[0, 0.4470, 0.7410], linewidth=3)
ax2.plot(lambda_vals, ozone_attenuation, "--", color=[0, 0, 0.5], linewidth=3)

ozone_band_start = lambda_vals[ozone_bands - 1] - 0.1
ozone_band_end = lambda_vals[ozone_bands - 1] + 0.5
y_limits = [0, 1000]
ax2.fill(
    [ozone_band_start, ozone_band_end, ozone_band_end, ozone_band_start],
    [y_limits[0], y_limits[0], y_limits[1], y_limits[1]],
    [0, 0, 0],
    alpha=0.2,
    edgecolor="none",
)

ax2.set_ylabel("Attenuation (dB/m)")
ax2.set_xlim([lambda_vals.min(), lambda_vals.max()])
ax2.set_ylim([0, 0.25])
ax2.set_xlabel("Wavelength (μm)")
ax2.legend(["Water vapor attenuation", "Ozone attenuation", "Ozone absorption band"], loc="upper right")
ax2.tick_params(labelsize=35)

_d = os.path.join(FIGURES_DIR, "Ozone_feature")
save_fig(fig1, os.path.join(_d, "ozone_feature"))
save_fig(fig2, os.path.join(_d, "ozone_feature_2"))

plt.show()
