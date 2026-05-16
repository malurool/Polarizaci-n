import numpy as np


def blackbody(lambda_um, T):
    lambda_m = lambda_um * 1e-6
    h = 6.63e-34
    c = 3e8
    k_B = 1.38e-23
    microflicks = 1e-4 * (2 * h * c**2 / lambda_m**5) / (np.exp(h * c / (lambda_m * k_B * T)) - 1)
    return microflicks
