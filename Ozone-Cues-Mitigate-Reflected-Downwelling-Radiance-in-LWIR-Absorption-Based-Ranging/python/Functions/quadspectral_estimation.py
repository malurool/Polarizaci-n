import numpy as np

from blackbody import blackbody


def quadspectral_estimation(
    lambda_vals,
    measurements,
    attenuation,
    index_1,
    index_2,
    index_3,
    index_4,
    T_air,
    cor_coeff,
):
    bb_air = blackbody(lambda_vals, T_air)
    L_1_centralized = measurements[:, :, index_1] - bb_air[index_1]
    L_2_centralized = measurements[:, :, index_2] - bb_air[index_2]
    L_3_centralized = measurements[:, :, index_3]
    L_4_centralized = measurements[:, :, index_4]

    downwelling_correction = cor_coeff * (L_4_centralized - L_3_centralized)
    transmittance_hat = (L_2_centralized - downwelling_correction) / (L_1_centralized)
    with np.errstate(invalid="ignore"):
        d_hat = -10 * np.log10(transmittance_hat) / (attenuation[index_2] - attenuation[index_1])

    if np.iscomplexobj(d_hat):
        imag_mask = np.imag(d_hat) != 0
        d_hat = np.real(d_hat)
        d_hat[imag_mask] = np.nan
    d_hat[d_hat < 0] = np.nan
    return d_hat
