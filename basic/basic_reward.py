from gymnasium import Space, spaces
from typing import Callable


class Reward:
    """
    Reward data type that handles reward tracking and calculation.

    Attributes:
    ----------
    id : str
        Identifier
    reward_function: callable
        Function that calculates reward. This function is expected to handle the info dict as its only argument and return a float or int
    reward_range: Space
        Reward range to inform reward_space in gym.Env implementation for multi-objective gymnasium
    value: float
        Value of the reward, used for summing rewards that stem from multiple different nodes

    Methods:
    ----------
    calculate_reward(info)
        Calls reward_function with info dict and returns the updated reward value
    reset()
        Resets the value to zero
    __str__()
        Overrides string function for more informative print statements
    """

    def __init__(
        self,
        id: str,
        reward_function: Callable[[dict], float],
        reward_range: Space = spaces.Box(float("-inf"), float("inf")),
    ):
        self.id = id
        self.reward_function = reward_function
        self.reward_range = reward_range
        # Initialise value to 0
        self.value: float = 0

    def calculate_reward(self, info: dict) -> float:
        """
        Calls reward_function to calculate reward.

        Parameters:
        ----------
        info : dict
            Info about current state

        Returns:
        ----------
        float
            Calculated reward value
        """
        self.value += self.reward_function(info)
        return self.value

    def reset(self):
        """
        Resets reward value to zero.
        """
        self.value = 0

    def __str__(self) -> str:
        return f"'{self.id}' = {self.value}"
