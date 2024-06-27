from watermanagement import WaterManagement
from graph import Node, Edge
from reward import Reward
from pprint import pprint
from gymnasium import spaces

dam1 = Node("1", [200, 100, 150], 0, 3000000, 5000000, 1000, [200], 0, False)
dam2 = Node("2", [], 0, 1000000, 6000000, 1000, [400], 0, True)
dam3 = Node("3", [100], 0, 4000000, 7000000, 1000, [0], 0)
dam4 = Node("4", [], 0, 2000000, 4000000, 1000, [100], 0, True)
edge = Edge(dam1, dam2, 0)
edge2 = Edge(dam3, dam2)
edge3 = Edge(dam3, dam4)


def max_supply_reliability(info: dict):
    demand = info["demand"]
    return info["volume"] / demand


def minimise_deficit(info: dict):
    demand = info["demand"]
    return -demand - info["volume"]


def evap(info: dict):
    return 0.1 * -info["volume"]


reward = Reward(
    "dam1_supply_reliability", max_supply_reliability, spaces.Box(0, float("inf"))
)

reward2 = Reward("dam2_deficit", minimise_deficit, spaces.Box(float("-inf"), 0))

dam1.add_reward(reward)
dam2.add_reward(reward2)
dam1.add_water_update_function(evap)
dam1.add_water_update_function(evap)

env = WaterManagement([dam1, dam3, dam2, dam4])

observation, info = env.reset()

for _ in range(3):
    action = env.action_space.sample()
    print(action)
    observation, rewards, terminated, truncated, info = env.step(action)
    pprint(info)
    print(observation)
    print(rewards)

    if terminated or truncated:
        print("reset!")
        observation, info = env.reset()
        pprint(info)
