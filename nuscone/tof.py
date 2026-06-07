import numpy as np

C = 299_792_458.0
MN_C2 = 939.56542052  # MeV

def energy_error(
    E,
    distance,
    distance_error,
    time_error,
):
    gamma = 1.0 + np.asarray(E) / MN_C2
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    tof = distance / (beta * C)

    dE_dt = -MN_C2 * gamma**3 * beta**2 / tof
    dE_dd = MN_C2 * gamma**3 * beta**2 / distance

    return np.sqrt(
        (dE_dt * time_error) ** 2
        + (dE_dd * distance_error) ** 2
    )