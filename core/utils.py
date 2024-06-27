from typing import Callable, Tuple


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
