import numpy as np


def read_transmittance_file(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = [float(item) for item in line.split()]
            if values:
                data.append(values)

    if not data:
        raise ValueError(f"No valid data found in file: {filename}")

    data = np.asarray(data, dtype=float)
    wavelength = data[:, 0]
    transmittance = data[:, 1]
    return wavelength, transmittance
