from typing import Callable, Tuple
import numpy as np
from numpy.core.multiarray import interp as compiled_interp


def current_timely_value(data: list, timestep: int) -> float:
    """
    Given a list with data and a timestep, it finds the correct value for that timestep.
    By taking the length of the list and doing 'timestep % list_length', we get the right index.
    E.g. for a list of monthly values with 12 entries, timestep 13 will return data for February (January = 0, length = 12)

    Parameters:
    ----------
    data : list
        List with timely data
    timestep : int
        Timestep

    Returns:
    ----------
    float
        Time-specific value if list length > 0, otherwise
    """
    data_length = len(data)
    if data_length > 0:
        current_index = timestep % data_length
        return data[current_index]
    else:
        return 0.0


def generate_release_range_function_minmax(
    min_release: float, max_release: float
) -> Callable[[dict], Tuple[float, float]]:
    """
    Generates a standard release range function given some min and max release.
    Can be used for nodes that have a constant min and max release.

    Parameters:
    ----------
    min_release : float
        Minimum release bound
    max_release : float
        Maximum release bound

    Returns:
    ----------
    Callable[[dict], Tuple[float, float]]
        Release range function that takes a dict and returns the specified release range
    """
    return lambda info: (min_release, max_release)


def generate_storage_to_release_range_function(storage_to_minmax_rel: np.ndarray):
    return lambda info: (
        np.interp(info["volume"], storage_to_minmax_rel[0], storage_to_minmax_rel[1]),
        np.interp(info["volume"], storage_to_minmax_rel[0], storage_to_minmax_rel[2]),
    )


def generate_evaporation_function(evap_rates, storage_to_surface_rel):
    return (
        lambda info: -current_timely_value(evap_rates, info["timestep"])
        * storage_to_surface(info["volume"], storage_to_surface_rel)
        / 100
    )


def calculate_power_production(
    outflow,
    volume,
    timestep,
    efficiency,
    max_turbine_flow,
    head_start_level,
    max_capacity,
    storage_to_level_rel,
    operating_hours,
):
    M3_TO_KG_FACTOR = 1000
    W_MW_CONVERSION = 1e-6
    G = 9.81
    turbine_flow = min(outflow, max_turbine_flow)

    water_level = storage_to_level(volume, storage_to_level_rel)
    head = max(0.0, water_level - head_start_level)
    power_in_mw = min(
        max_capacity,
        turbine_flow * head * M3_TO_KG_FACTOR * G * efficiency * W_MW_CONVERSION,
    )
    production = power_in_mw * current_timely_value(operating_hours, timestep)
    return production


def modified_interp(x: float, xp: float, fp: float, left=None, right=None) -> float:
    fp = np.asarray(fp)

    return compiled_interp(x, xp, fp, left, right)


def storage_to_level(storage, storage_to_level_rel):
    return modified_interp(storage, storage_to_level_rel[0], storage_to_level_rel[1])


def storage_to_surface(storage, storage_to_surface_rel):
    return modified_interp(
        storage, storage_to_surface_rel[0], storage_to_surface_rel[1]
    )


def storage_to_release(storage, storage_to_surface_rel):
    min_release = np.interp(
        storage, storage_to_surface_rel[0], storage_to_surface_rel[1]
    )
    max_release = np.interp(
        storage, storage_to_surface_rel[0], storage_to_surface_rel[2]
    )
    return (min_release, max_release)
