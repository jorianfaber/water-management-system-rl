import numpy as np
from numpy.core.multiarray import interp as compiled_interp
from utils import current_timely_value
from typing import Callable, Tuple, List


def generate_storage_to_release_range_function(
    storage_to_minmax_rel: np.ndarray,
) -> Callable[[dict], Tuple[float, float]]:
    """
    Uses the storage_to_minmax relationship data to create a release range function which returns a release range based on storage volume.

    Parameters:
    ----------
    storage_to_minmax_rel : np.ndarray
        Storage to minmax relationship data

    Returns:
    ----------
    Callable[[dict], Tuple[float, float]]
        The release range function: takes an info dict and returns a 2-tuple of min and max release (see release_range_function in Node class)
    """
    return lambda info: (
        np.interp(info["volume"], storage_to_minmax_rel[0], storage_to_minmax_rel[1]),
        np.interp(info["volume"], storage_to_minmax_rel[0], storage_to_minmax_rel[2]),
    )


def generate_evaporation_function(
    evap_rates: np.ndarray, storage_to_surface_rel: np.ndarray
) -> Callable[[dict], float]:
    """
    Generates a water update function for evaporation, based on the given evaporation rates and storage to surface relationship data.

    Parameters:
    ----------
    evap_rates : np.ndarray
        Time-dependent evaporation rate data.
    storage_to_surface_rel : nd.array
        Storage to surface relationship data

    Returns:
    ----------
    Callable[[dict], float]
        Water update function that takes info dict and returns a float (see water_update_function in Node class)
    """
    return (
        lambda info: -current_timely_value(evap_rates, info["timestep"])
        * storage_to_surface(info["volume"], storage_to_surface_rel)
        / 100
    )


def calculate_power_production(
    outflow: float,
    volume: float,
    timestep: int,
    efficiency: float,
    max_turbine_flow: float,
    head_start_level: float,
    max_capacity: float,
    storage_to_level_rel: np.ndarray,
    operating_hours: List[int],
) -> float:
    """
    Calculates power production using formula from Nile River case study. Used in custom reward function.

    Parameters:
    ----------
    outflow : float
        Current outflow
    volume: float
        Current volume
    timestep : int
        Current timestep
    efficiency: float
        Efficiency
    max_turbine_flow : float
        Max turbine flow
    head_start_level : float
        Level that the turbine head starts generating power at
    max_capacity : float
        Max power generation
    storage_to_level_rel : np.ndarray
        Storage to level relationship data
    operating_hours : List[int]
        Time-dependent operating hours

    Returns:
    ----------
    float
        Power production
    """
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
    """
    Modified interpolation method, from Nile River case study.
    """
    fp = np.asarray(fp)

    return compiled_interp(x, xp, fp, left, right)


def storage_to_level(storage: float, storage_to_level_rel: np.ndarray) -> float:
    """
    Converts current storage volume to level.

    Parameters:
    ----------
    storage : float
        Current storage volume
    storage_to_level_rel : np.ndarray
        Storage to level relationship data

    Returns:
    ----------
    float
        Water level
    """
    return modified_interp(storage, storage_to_level_rel[0], storage_to_level_rel[1])


def storage_to_surface(storage: float, storage_to_surface_rel: np.ndarray) -> float:
    """
    Converts current storage volume to surface area.

    Parameters:
    ----------
    storage : float
        Current storage volume
    storage_to_surface_rel : np.ndarray
        Storage to surface relationship data

    Returns:
    ----------
    float
        Surface area
    """
    return modified_interp(
        storage, storage_to_surface_rel[0], storage_to_surface_rel[1]
    )


def storage_to_release(
    storage: float, storage_to_surface_rel: np.ndarray
) -> Tuple[float, float]:
    """
    Converts current storage volume to min and max release. Used in release_range_function.

    Parameters:
    ----------
    storage : float
        Current storage volume
    storage_to_minmax_rel : np.ndarray
        Storage to minmax relationship data

    Returns:
    ----------
    Tuple[float, float]
        Minimum and maximum release
    """
    min_release = np.interp(
        storage, storage_to_surface_rel[0], storage_to_surface_rel[1]
    )
    max_release = np.interp(
        storage, storage_to_surface_rel[0], storage_to_surface_rel[2]
    )
    return (min_release, max_release)
