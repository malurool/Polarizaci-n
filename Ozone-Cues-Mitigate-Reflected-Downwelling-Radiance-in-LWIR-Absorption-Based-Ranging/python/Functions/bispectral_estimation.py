import numpy as np

from blackbody import blackbody


def bispectral_estimation(lambda_vals, measurements, attenuation, index_1, index_2, T_air):
    transmittance_hat = (measurements[:, :, index_1] - blackbody(lambda_vals[index_1], T_air)) / (
        measurements[:, :, index_2] - blackbody(lambda_vals[index_2], T_air)
    )
    with np.errstate(invalid="ignore"):
        d_hat = -10 * np.log10(transmittance_hat) / (attenuation[index_1] - attenuation[index_2])
    if np.iscomplexobj(d_hat):
        imag_mask = np.imag(d_hat) != 0
        d_hat = np.real(d_hat)
        d_hat[imag_mask] = np.nan
    d_hat[d_hat < 0] = np.nan
    return d_hat
