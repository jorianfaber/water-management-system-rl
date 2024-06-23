import gymnasium as gym
from gymnasium import spaces
from graph import Node
from typing import List
from reward import Reward
import numpy as np


class WaterManagement(gym.Env):
    """
    OpenAI Gymnasium Environment that represents our water management graph.

    Attributes:
    ----------
    nodeList : List[Node]
        List of all the nodes in the graph
    rewards : List[Reward]
        List of all the rewards to optimise
    action_space : Space
        Action space of the environment, defined by nodes
    observation_space : Space
        Observation space of the environment, defined by nodes
    reward_space : Space
        Reward space of the environment, necessary for multi-objective RL, defined by rewards
    reward_dim : int
        Size of the reward vector returned
    seed : int
        Seed of the environment, used for np.random
    """

    def __init__(self, node_list: List[Node], seed: int = 0):
        self.node_list = node_list
        # Initialise dicts to save information when iterating over nodes
        action_space_dict = {}
        observation_space_dict = {}
        reward_space_dict = {}
        self.rewards: List[Reward] = []
        for node in node_list:
            # Only add node's action_space if it is not passive
            if not node.passive:
                action_space_dict[node.id] = node.action_space
            observation_space_dict[node.id] = node.observation_space
            for reward in node.rewards:
                # Only add reward to list to track if we haven't added it yet, to avoid duplicates
                if reward not in self.rewards:
                    self.rewards.append(reward)
                    reward_space_dict[reward.id] = reward.reward_range
        # Initialise action, observation and reward space using information from nodes
        self.action_space = spaces.Dict(action_space_dict)
        self.observation_space = spaces.Dict(observation_space_dict)
        self.reward_space = spaces.Dict(reward_space_dict)
        self.reward_dim = len(reward_space_dict)
        self.seed = seed

    def _get_obs(self) -> dict:
        """
        Calls _get_obs() for all nodes and returns total observation.

        Returns:
        ----------
        dict
            Dictionary with observation from all nodes, with their id as key
        """
        observation = {}
        for node in self.node_list:
            observation[node.id] = node._get_obs()
        return observation

    def _get_info(self):
        """
        Calls _get_info() for all nodes and returns total information.

        Returns:
        ----------
        dict
            Dictionary with information from all nodes, with their id as key
        """
        info = {}
        for node in self.node_list:
            info[node.id] = node._get_info()
        return info

    def reset(self, seed=None):
        super().reset(seed=seed)
        # Iterate over every node and reset (node.reset() also resets connected flows)
        for node in self.node_list:
            node.reset()
        return self._get_obs(), self._get_info()

    def step(self, action: np.array):
        # Initialise boolean to check if any of the nodes return terminated = True
        terminated_general = False
        # Reset rewards to 0
        for reward in self.rewards:
            reward.reset()
        # Iterate over all nodes and call their step function with the corresponding action
        for node in self.node_list:
            # Only pass action value if node is not passive
            if not node.passive:
                observation, rewards, terminated, truncated, info = node.step(
                    action[node.id]
                )
            # If node is passive, pass any action value
            else:
                observation, rewards, terminated, truncated, info = node.step(
                    np.array([1.0])
                )
            if terminated:
                terminated_general = True
        observation = self._get_obs()
        info = self._get_info()
        # Returned flattened list with observation and reward values for multi-objective RL policy
        return (
            np.array(
                [
                    value
                    for subdict in observation.values()
                    for value in subdict.values()
                ]
            ).flatten(),
            np.array([reward.value for reward in self.rewards]),
            terminated_general,
            False,
            info,
        )

    def render(self):
        pass
