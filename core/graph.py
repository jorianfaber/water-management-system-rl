from __future__ import annotations
from typing import List, Callable, Union
from gymnasium import spaces
from reward import Reward
import numpy as np
from utils import current_timely_value


class Node:
    """
    Class to represent a node in our graph structure.

    Attributes:
    ----------
    id : str
        Identifier
    inflow : List[float]
        List with timely inflow data (m3/s). Used for streams that do not originate from a node. Use timestep to index
    initial_outflow : float
        Initial amount of water flowing out (m3/s). Used for reset()
    outflow : List[float]
        Water flowing out per timestep (m3/s), last element holds current outflow
    initial_volume : float
        Initial storage volume (m3). Used for reset()
    volume : List[float]
        Storage volume (m3) per timestep, last element holds current volume
    capacity : float
        Max volume (m3)
    demand : List[loat]
        Demand (m3/s) of the node, timely value that depends on timestep
    initial_timestep : int
        Initial timestep value. Used for reset()
    timestep : int
        Current timestep, initialised to value of init_timestep argument
    integration_steps : List[int]
        Number of steps to take at each timestep to perform integration. List for timely values (e.g. number of steps might differ per month)
    sec_per_timestep : List[int]
        Number of seconds in each timestep, used for going from m3/s to m3. List for timely values (e.g. seconds might differ per month)
    passive : bool
        Indicates whether the node is passive. If so, no actions will have to be taken but the water just flows through
    incoming_flows : List[Edge]
        Incoming edges with water flow
    outgoing_flows : List[Edge]
        Outgoing edges with water flow
    rewards : List[Reward]
        List of rewards to calculate and track
    water_update_functions : List[callable]
        List of water update functions, that update water volume (e.g. evaporation). Functions are expected to take info dict as argument and return a float
    release_range_function : Callable[[dict], tuple[float, float]]
        Release range function. Takes info dict as argument and returns a tuple with min and max release
    action_space : Space
        Action space of the node, how much water can be released, capped by capacity or default value of 1 million and dimension equal to number of outflows, if number of outflows > 1
    observation_space : Space
        Observation space of the node (volume and inflow, capped by capacity or default value of 1 million)

    Methods:
    ----------
    add_incoming_flow(Edge)
        Adds incoming flow to class attributes
    add_outgoing_flow(Edge)
        Adds outgoing flow to class attributes
    get_current_inflow()
        Gets the current inflow value based on inflow list and incoming flows
    add_reward(reward)
        Adds reward object to list of rewards
    add_water_update_function(callable)
        Adds water update function to list of water update functions
    calculate_water_update()
        Calls all water update functions and returns the sum
    _check_terminated()
        Check if the new state is terminated
    _get_obs()
        Returns observation
    _get_info()
        Returns extra info about state
    clip_action(float, float, np.array)
        Clips the action values to min and max bounds
    integration(inflow, action)
        Divides timestep up into smaller steps and performs mass-balance equations each time for higher accuracy, calculating outflow and updating volume
    reset()
        Resets the node to initial state, along with connected edges
    step(action)
        Classic Gym Env function - performs step based on action value, updates state and returns reward
    """

    def __init__(
        self,
        id: str,
        inflow: List[float] = [],
        initial_outflow: float = 0.0,
        initial_volume: float = 0.0,
        capacity: Union[float, None] = None,
        max_action: float = 10000,
        demand: List[float] = [0.0],
        initial_timestep: int = 0,
        passive: bool = False,
        integration_steps: List[int] = [1],
        sec_per_timestep: List[int] = [3600 * 24],
        release_range_function: Union[callable, None] = None,
    ):
        self.id = id
        self.inflow = inflow
        self.initial_outflow = initial_outflow
        self.outflow = [initial_outflow]
        self.initial_volume = initial_volume
        self.volume = [initial_volume]
        self.capacity = capacity
        self.demand = demand
        self.max_action = max_action

        self.initial_timestep = initial_timestep
        self.timestep = initial_timestep
        self.integration_steps = integration_steps
        self.sec_per_timestep = sec_per_timestep
        self.passive = passive

        self.incoming_flows: List[Edge] = []
        self.outgoing_flows: List[Edge] = []
        self.rewards: List[Reward] = []

        self.water_update_functions: List[Callable[[dict], float]] = []

        self.action_space = spaces.Box(0.0, max_action)
        # If capacity is not given, assign a default value of 100 million to it
        if not capacity:
            capacity = 100000000000
        self.capacity = capacity
        # Only volume in observation space
        self.observation_space = spaces.Dict(
            {
                "volume": spaces.Box(0.0, capacity),
            }
        )
        # If a release range function is not passed, we assign a default one which simply returns [0, volume/]
        # TODO: raise error if wrong function type is passed
        if not release_range_function:
            self.release_range_function: Callable[[dict], tuple[float, float]] = (
                lambda info: (
                    0.0,
                    self.volume[-1]
                    / (current_timely_value(self.sec_per_timestep, self.timestep)),
                )
            )
        else:
            self.release_range_function = release_range_function

    def add_incoming_flow(self, flow: Edge) -> None:
        """
        Adds incoming flow to list of incoming flows.

        Parameters:
        ----------
        flow : Edge
            Flow to add
        """
        self.incoming_flows.append(flow)

    def add_outgoing_flow(self, flow: Edge) -> None:
        """
        Adds outgoing flow to list of outgoing flows. Updates action_space accordingly.

        Parameters:
        ----------
        flow : Edge
            Flow to add
        """
        self.outgoing_flows.append(flow)
        # Change action_space shape depending on amount of outflows. If there are multiple outflows, we want to take an action for each of them
        self.action_space = spaces.Box(
            0.0, self.max_action, (len(self.outgoing_flows),)
        )

    def get_current_inflow(self) -> float:
        """
        Calculates current inflow value and returns it. Takes current inflow value from list and sums values of all incoming flows.

        Returns:
        ----------
        float
            Current inflow (m3/s)
        """
        inflow = current_timely_value(self.inflow, self.timestep)
        for f in self.incoming_flows:
            inflow += f.get_current_flow()
        return inflow

    def add_reward(self, reward: Reward) -> None:
        """
        Adds reward to list of rewards.

        Parameters:
        ----------
        reward : Reward
            Reward to add
        """
        self.rewards.append(reward)

    def add_water_update_function(
        self, water_update_function: Callable[[dict], float]
    ) -> None:
        """
        Adds water update function to list of water update functions.

        Parameters:
        ----------
        water_update_function : callable
            Function that takes an info dict and returns a float.
            This float is added to the current water volume. Hence, a negative value will be substracted
        """
        self.water_update_functions.append(water_update_function)

    def calculate_water_update(self) -> float:
        """
        Calls all water update functions and returns the sum of all returned float values.

        Returns:
        ----------
        float
            Sum of all returned water update function values
        """
        water = 0
        for water_update_function in self.water_update_functions:
            water += water_update_function(self._get_info())
        return water

    def _check_terminated(self) -> bool:
        """
        Check if current state is terminated.

        Returns:
        ----------
        bool
            Indicates whether state is terminated
        """
        return False

    def _get_obs(self) -> dict:
        """
        Gets observation (volume).

        Returns:
        ----------
        dict
            Observation dictionary
        """
        return {"volume": self.volume[-1]}

    def _get_info(self) -> dict:
        """
        Gets extra information about state.
            - id
            - timestep
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
            "timestep": self.timestep,
            "volume": self.volume[-1],
            "inflow": self.get_current_inflow(),
            "outflow": self.outflow[-1],
            "capacity": self.capacity,
            "demand": current_timely_value(self.demand, self.timestep),
        }

    def clip_action(
        self, min_bound: float, max_bound: float, action: np.array
    ) -> np.array:
        """
        Clips action to min and max bounds.
        If there are multiple actions in action array (multiple outgoing flows), perform basic proportional allocation to clip actions.

        Parameters:
        ----------
        min_bound : float
            Minimum bound
        max_bound : float
            Maximum bound
        action : np.array
            Action values

        Returns:
        ----------
        np.array
            Clipped actions
        """
        action_sum = np.sum(action)
        result = np.copy(action)
        # Check if action_sum is higher than max or lower than min, update result accordingly
        if action_sum > max_bound:
            # Direct proportional allocation
            for index, act in enumerate(action):
                proportion = act / action_sum
                result[index] = proportion * max_bound
        elif action_sum < min_bound:
            # Direct proportional allocation
            for index, act in enumerate(action):
                proportion = act / action_sum
                result[index] = proportion * min_bound
        return result

    def integration(self, inflow: float, action: np.array) -> None:
        """
        Performs integration step of mass-balance equation.
        Per integration step, volume is updated using the formula:
            new_volume = old_volume + (inflow * sec) - (outflow * sec) + water_update
            calculate_water_update() is expected to return the water update for one timestep.
            Therefore, its value is normalised by dividing it by the number of integration steps.

        Parameters:
        ----------
        inflow : float
            Inflow (m3/s) amount
        action : np.array
            Release action per outgoing flow
        """
        # Add volume value, copy from most recent. This is the value that will be updated in each integration step
        self.volume.append(self.volume[-1])
        # Calculate amount of seconds per integration step
        integration_steps = current_timely_value(self.integration_steps, self.timestep)
        seconds = (
            current_timely_value(self.sec_per_timestep, self.timestep)
            / integration_steps
        )
        # Set outgoing flows' flow to 0 for updating in integration loop
        for f in self.outgoing_flows:
            f.update_flow(0, self.timestep)
        outflow_sum = 0
        # We don't have to check for passive if we pass an empty action array when a node is passive
        # All the iterations (such as enumerate) over the action array will not execute if the array is empty
        for _ in range(integration_steps):
            # Release range function is called inside the loop because info (volume) changes during every iteration
            min_release, max_release = self.release_range_function(self._get_info())
            # print( f"{self.id} -- volume={self.volume[-1]} min={min_release} max={max_release}")
            actual_action = self.clip_action(min_release, max_release, action)
            # print(actual_action)
            # If there are outgoing flows, we update their flow values
            if len(self.outgoing_flows) > 0:
                for index, act in enumerate(actual_action):
                    self.outgoing_flows[index].add_to_current_flow(
                        act / integration_steps
                    )
            outflow_sum += np.sum(actual_action) / integration_steps
            # Update the current volume, where outflow is the sum of release actions
            # Water update is divided by number of integration steps to normalise its value
            #   (We expect water update functions to return a value for one timestep)
            self.volume[-1] += (
                inflow * seconds
                - np.sum(actual_action) * seconds
                + self.calculate_water_update() / integration_steps
            )
        # Add outflow value and update it
        self.outflow.append(0)
        self.outflow[-1] = outflow_sum

    def reset(self) -> None:
        """
        Resets relevant attributes to initial value, and resets connected edges.
        """
        self.timestep = self.initial_timestep
        self.volume = [self.initial_volume]
        self.outflow = [self.initial_outflow]
        for reward in self.rewards:
            reward.reset()
        # Since we do not know for certain when the graph ends, we cannot only reset incoming nodes on every node
        for f in self.incoming_flows:
            f.reset()
        for f in self.outgoing_flows:
            f.reset()

    def step(self, action: np.array) -> tuple[dict, List[Reward], bool, bool, dict]:
        """
        Classic Gym Env function that updates state based on action and returns a reward.

        Parameters:
        ----------
        action : np.array
            How much water to release

        Returns:
        tuple[dict, List[Reward], bool, bool, dict]
            observation, rewards, terminated, truncated, info
        """
        # Calculate current inflow (m3/s)
        inflow = self.get_current_inflow()

        # If node is passive, divide releases over all outgoing flows by passing volume (let clip_action() handle it)
        if self.passive:
            action = np.array([self.volume[-1] + 1 for _ in self.outgoing_flows])
        # Call integration step with action array
        self.integration(inflow, action)

        info = self._get_info()
        # Calculate each reward and update its value
        for reward in self.rewards:
            reward.calculate_reward(info)
        observation = self._get_obs()
        terminated = self._check_terminated()
        self.timestep += 1
        return observation, self.rewards, terminated, False, info


class Edge:
    """
    Class to represent an edge in our graph structure.

    Attributes:
    ----------
    source : Node
        Origin of edge
    destination : Node
        Destination of edge
    initial_flow : float
        Current water flow on edge
    delay : int
        Delay by which flow arrives at destination node
    replacement_values : list
        Values to replace delayed flows by, can be zero

    Methods:
    ----------
    reset()
        Resets flow to initial value
    get_current_flow()
        Returns current flow, taking into account delay
    add_to_current_flow(flow)
        Adds flow value to current flow value
    update_flow(flow)
        Changes flow value to new flow value
    """

    def __init__(
        self,
        source: Node,
        destination: Node,
        initial_flow: float = 0.0,
        delay: int = 0,
        replacement_values: list = [0],
    ):
        self.source = source
        self.destination = destination
        self.initial_flow = initial_flow
        self.flow = [initial_flow]
        self.delay = delay
        # Set destination of delay to destination.id
        self.delay_destination_id = destination.id
        self.replacement_values = replacement_values
        # Check if there are enough replacement_values for the delay
        if delay > 0 and not delay == len(replacement_values):
            raise ValueError(
                "Delay must be equal to number of values in replacement_values list"
            )

        # Update the Nodes' incoming and outgoing flow attributes
        source.add_outgoing_flow(self)
        destination.add_incoming_flow(self)

    def reset(self) -> None:
        """
        Reset flow to initial value.
        """
        self.flow = [self.initial_flow]

    def get_current_flow(self) -> float:
        """
        Gets the edge's current flow, taking into account delay.

        Returns:
        float
            Current flow value of edge (m3/s)
        """
        if len(self.flow) - 1 - self.delay < 0:
            return self.replacement_values[len(self.flow) - 1]
        return self.flow[-1 - self.delay]

    def add_to_current_flow(self, flow: float):
        """
        Adds flow value to current flow value.

        Parameters:
        ----------
        flow : float
            Value to add to current flow
        """
        self.flow[-1] += flow

    def update_flow(self, flow: float, timestep: int):
        """
        Updates flow value at given timestep. Raises an error if timestep is too large.

        Parameters:
        ----------
        flow : float
            New flow value
        timestep : int
            Timestep
        """
        if timestep > len(self.flow):
            raise ValueError("Timestep too large for flow list, can't update")
        # If the timestep is just too big for indexing, we add append the list with the new value
        elif timestep == len(self.flow):
            self.flow.append(flow)
        else:
            self.flow[timestep] = flow
