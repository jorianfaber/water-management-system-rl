from nile_main import generate_nile
import pprint
import numpy as np
import csv
from pathlib import Path


make_csv = True
csv_path = Path(__file__).parent.parent / "verification" / "group13.csv"


def nile_river_simulation(nu_of_timesteps=240):
    # Create power plant, dam and irrigation system. Initialise with semi-random parameters.
    # Set objective functions to identity for power plant, minimum_water_level for dam and water_deficit_minimised
    # for irrigation system.

    water_management_system = generate_nile()

    if make_csv:
        with open(csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Year",
                    "Input",
                    "Gerd_storage",
                    "Gerd_release",
                    "Roseires_storage",
                    "Roseires_release",
                    "Sennar_storage",
                    "Sennar_release",
                    "Had_storage",
                    "Had_release",
                ]
            )
            np.random.seed(42)
            for i in range(nu_of_timesteps):
                print(i)
                action = generateOutput()
                (
                    final_observation,
                    final_reward,
                    final_terminated,
                    final_truncated,
                    final_info,
                ) = water_management_system.step(action)
                writer.writerow(
                    [
                        i,
                        [value[0] for value in action.values()],
                        ensure_float(final_info.get("GERD")["volume"]),
                        ensure_float(final_info.get("GERD")["outflow"]),
                        ensure_float(final_info.get("Roseires")["volume"]),
                        ensure_float(final_info.get("Roseires")["outflow"]),
                        ensure_float(final_info.get("Sennar")["volume"]),
                        ensure_float(final_info.get("Sennar")["outflow"]),
                        ensure_float(final_info.get("HAD")["volume"]),
                        ensure_float(final_info.get("HAD")["outflow"]),
                    ]
                )
                pprint.pprint(final_info)
                # print(final_reward)
                for reward in water_management_system.rewards:
                    print(reward)
    else:
        for _ in range(nu_of_timesteps):
            action = water_management_system.action_space.sample()
            action = np.array(list(action.values())).flatten()
            print("Action:", action, "\n")
            (
                final_observation,
                final_reward,
                final_terminated,
                final_truncated,
                final_info,
            ) = water_management_system.step(action)
            print("Reward:", final_reward)
            pprint.pprint(final_info)
            print("Is finished:", final_truncated, final_terminated)


def generateOutput():
    random_values = np.random.rand(
        4,
    ) * [10000, 10000, 10000, 4000]

    return {
        "GERD": np.array([random_values[0]]),
        "Roseires": np.array([random_values[1]]),
        "Sennar": np.array([random_values[2]]),
        "HAD": np.array([random_values[3]]),
    }


def ensure_float(value):
    if isinstance(value, np.ndarray):
        return value.item()
    return value


if __name__ == "__main__":
    nile_river_simulation()
