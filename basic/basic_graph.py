from __future__ import annotations
from typing import List
from gymnasium import spaces
from basic_reward import Reward


class Node:
    """
    Class to represent a node in our graph structure.

    Attributes:
    ----------
    id : str
        Identifier
    inflow : float
        Amount of water flowing in
    outflow : float
        Amount of water flowing out
    volume : float
        Current storage volume
    capacity : float
        Max volume
    demand : float
        Volume demand of the node
    incoming_flow : Edge
        Incoming edge with water flow
    outgoing_flow : Edge
        Outgoing edge with water flow
    rewards : List[Reward]
        List of rewards to calculate and track
    action_space : Space
        Action space of the node, how much water can be released, capped by capacity
    observation_space : Space
        Observation space of the node (volume and inflow, capped by capacity)

    Methods:
    ----------
    add_incoming_flow(flow)
        Adds incoming flow to class attributes
    add_outgoing_flow(flow)
        Adds outgoing flow to class attributes
    add_reward(reward)
        Adds reward object to list of rewards
    _get_obs()
        Returns observation
    _get_info()
        Returns extra info about state
    step(action)
        Classic Gym Env function - performs step based on action value, updates state and returns reward
    """

    def __init__(
        self,
        id: str,
        inflow: float = 0.0,
        outflow: float = 0.0,
        volume: float = 0.0,
        capacity: float = float("inf"),
        demand: float = 0.0,
    ):
        self.id = id
        self.inflow = inflow
        self.outflow = outflow
        self.volume = volume
        self.capacity = capacity
        self.demand = demand

        self.incoming_flow = None
        self.outgoing_flow = None
        # self.incoming_flows = []
        # self.outgoing_flows = []
        self.rewards: List[Reward] = []

        self.action_space = spaces.Box(0.0, capacity)
        # Only volume in observation space
        self.observation_space = spaces.Dict(
            {
                "volume": spaces.Box(0.0, capacity),
            }
        )

    def add_incoming_flow(self, flow: Edge) -> None:
        """
        Adds incoming flow to class attributes.

        Parameters:
        ----------
        flow : Edge
            Flow to add
        """
        self.incoming_flow = flow

    def add_outgoing_flow(self, flow: Edge) -> None:
        """
        Adds outgoing flow to class attributes.

        Parameters:
        ----------
        flow : Edge
            Flow to add
        """
        self.outgoing_flow = flow

    def add_reward(self, reward: Reward) -> None:
        """
        Adds reward to list of rewards.

        Parameters:
        ----------
        reward : Reward
            Reward to add
        """
        self.rewards.append(reward)

    def _get_obs(self) -> dict:
        """
        Gets observation (volume).

        Returns:
        ----------
        dict
            Observation dictionary
        """
        return {"volume": self.volume}

    def _get_info(self) -> dict:
        """
        Gets extra information about state.
            - id
            - volume
            - inflow
            - outflow
            - capacity
            - demand

        Returns:
        ----------
        dict
            Observation dictionary
        """
        return {
            "id": self.id,
            "volume": self.volume,
            "inflow": self.inflow,
            "outflow": self.outflow,
            "capacity": self.capacity,
            "demand": self.demand,
        }

    def step(self, action: float) -> tuple[dict, List[Reward], bool, bool, dict]:
        """
        Classic Gym Env function that updates state based on action and returns a reward.

        Parameters:
        ----------
        action : float
            How much water to release

        Returns:
        tuple[dict, List[Reward], bool, bool, dict]
            observation, rewards, terminated, truncated, info
        """
        # If there is an incoming flow, inflow is determined by that flow value
        if self.incoming_flow:
            self.inflow = self.incoming_flow.flow
        # Add inflow to volume
        self.volume += self.inflow
        # Clip release to volume
        actual_release = min(action, self.volume)
        # Update outflow values and substract from volume
        self.outflow = actual_release
        self.volume -= actual_release
        if self.outgoing_flow:
            self.outgoing_flow.flow = actual_release
        info = self._get_info()
        # Calculate each reward and update its value
        for reward in self.rewards:
            reward.calculate_reward(info)
        observation = self._get_obs()
        terminated = False
        # When volume exceeds capacity, terminated is set to True
        if self.volume > self.capacity:
            terminated = True
        return observation, self.rewards, terminated, False, info


class Edge:
    """
    Class to represent an edge in our graph structure.

    Attributes:
    ----------
    from_node : Node
        Origin of edge
    to_node : Node
        Destination of edge
    flow : float
        Current water flow on edge
    """

    def __init__(self, from_node: Node, to_node: Node, flow: float = 0.0):
        self.from_node = from_node
        self.to_node = to_node
        self.flow = flow

        # Update the Nodes' incoming and outgoing flow attributes
        from_node.add_outgoing_flow(self)
        to_node.add_incoming_flow(self)
