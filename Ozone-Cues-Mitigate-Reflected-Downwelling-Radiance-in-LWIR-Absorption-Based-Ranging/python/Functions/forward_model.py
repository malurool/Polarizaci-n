import numpy as np

from blackbody import blackbody


def forward_model(lambda_vals, T, emissivity, V, reflected, attenuation, d, T_air):
    obj_emission = blackbody(lambda_vals, T) * emissivity
    obj_reflection = (1 - emissivity) * np.sum(V * reflected, axis=3)

    tau = 10 ** (-attenuation * d / 10)

    radiance_sensor = tau * (obj_emission + obj_reflection) + (1 - tau) * blackbody(lambda_vals, T_air)
    return radiance_sensor
