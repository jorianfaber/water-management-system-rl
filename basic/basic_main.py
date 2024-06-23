from basic_watermanagement import WaterManagement
from basic_graph import Node, Edge
from basic_reward import Reward
from pprint import pprint
from gymnasium import spaces

dam1 = Node("start", 200, 0, 300, 500, 200)
dam2 = Node("end", 0, 0, 100, 600, 400)
edge = Edge(dam1, dam2, 0)


def max_supply_reliability(info: dict):
    return info["volume"] / info["demand"]


def minimise_deficit(info: dict):
    return -info["demand"] - info["volume"]


reward = Reward(
    "dam1_supply_reliability", max_supply_reliability, spaces.Box(0, float("inf"))
)

reward2 = Reward("dam2_deficit", minimise_deficit, spaces.Box(float("-inf"), 0))

dam1.add_reward(reward)
dam2.add_reward(reward2)

env = WaterManagement([dam1, dam2])

observation, info = env.reset()

for _ in range(50):
    action = env.action_space.sample()
    print(action)
    observation, rewards, terminated, truncated, info = env.step(action)
    pprint(info)
    print(observation)
    print(rewards)

    if terminated or truncated:
        print("reset!")
        observation, info = env.reset()
