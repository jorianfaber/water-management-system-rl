from watermanagement import WaterManagement
from graph import Node, Edge
from reward import Reward
from gymnasium import spaces
from utils import (
    generate_storage_to_release_range_function,
    calculate_power_production,
    storage_to_level,
    generate_evaporation_function,
)
import numpy as np
from pathlib import Path
from typing import Tuple


def generate_nile():
    DATA_DIRECTORY = Path(__file__).parent.parent / "data"
    DAYS_PER_MONTH = [
        31,  # January
        28,  # February (non-leap year)
        31,  # March
        30,  # April
        31,  # May
        30,  # June
        31,  # July
        31,  # August
        30,  # September
        31,  # October
        30,  # November
        31,  # December
    ]
    # SEC_PER_MONTH = [days * 60 * 60 * 24 for days in DAYS_PER_MONTH]
    INTEGRATION_STEPS_PER_MONTH = [2 * 24 * days for days in DAYS_PER_MONTH]
    SEC_PER_TIMESTEP_PER_MONTH = [
        60 * 30 * steps for steps in INTEGRATION_STEPS_PER_MONTH
    ]
    SEED = 2137

    # GERD
    # ------------------------------------------------------------------------
    name = "GERD"
    GERD_inflow = np.loadtxt(DATA_DIRECTORY / "catchments" / "InflowBlueNile.txt")
    GERD_storage_to_level_rel = np.loadtxt(
        DATA_DIRECTORY / "dams" / f"store_level_rel_{name}.txt"
    )
    GERD_storage_to_minmax_rel = np.loadtxt(
        DATA_DIRECTORY / "dams" / f"store_min_max_release_{name}.txt"
    )
    GERD_storage_to_surface_rel = np.loadtxt(
        DATA_DIRECTORY / "dams" / f"store_sur_rel_{name}.txt"
    )
    GERD_evap_rates = np.loadtxt(DATA_DIRECTORY / "dams" / f"evap_{name}.txt")
    GERD_release_range = generate_storage_to_release_range_function(
        GERD_storage_to_minmax_rel
    )
    GERD = Node(
        name,
        GERD_inflow,
        max_action=10000,
        initial_volume=15000000000,
        integration_steps=INTEGRATION_STEPS_PER_MONTH,
        sec_per_timestep=SEC_PER_TIMESTEP_PER_MONTH,
        release_range_function=GERD_release_range,
    )

    GERD.add_water_update_function(
        generate_evaporation_function(GERD_evap_rates, GERD_storage_to_surface_rel)
    )

    def maximise_power_production_gerd(info: dict) -> float:
        outflow = info["outflow"]
        volume = info["volume"]
        timestep = info["timestep"]
        power_production = calculate_power_production(
            outflow,
            volume,
            timestep,
            0.93,
            4320,
            507,
            6000,
            GERD_storage_to_level_rel,
            [24 * days for days in DAYS_PER_MONTH],
        )
        return power_production

    GERD.add_reward(
        Reward(
            "ethiopia_power",
            maximise_power_production_gerd,
            spaces.Box(0, float("inf")),
        )
    )
    # ------------------------------------------------------------------------

    # Sudan
    def minimise_demand_deficit(info: dict) -> float:
        return -max(0.0, info["demand"] - info["inflow"])

    sudan_deficit = Reward(
        "sudan_deficit_minimised", minimise_demand_deficit, spaces.Box(float("-inf"), 0)
    )

    def irrigation_release_range(info: dict) -> Tuple[float, float]:
        release = info["inflow"] - min(info["demand"], info["inflow"])
        return (release, release)

    # ------------------------------------------------------------------------

    # Roseires
    # ------------------------------------------------------------------------
    name = "Roseires"
    Roseires_inflow = np.loadtxt(
        DATA_DIRECTORY / "catchments" / "InflowGERDToRoseires.txt"
    )
    Roseires_storage_to_minmax_rel = np.loadtxt(
        DATA_DIRECTORY / "dams" / f"store_min_max_release_{name}.txt"
    )
    Roseires_storage_to_surface_rel = np.loadtxt(
        DATA_DIRECTORY / "dams" / f"store_sur_rel_{name}.txt"
    )
    Roseires_evap_rates = np.loadtxt(DATA_DIRECTORY / "dams" / f"evap_{name}.txt")
    Roseires_release_range = generate_storage_to_release_range_function(
        Roseires_storage_to_minmax_rel
    )
    Roseires = Node(
        name,
        Roseires_inflow,
        max_action=10000,
        initial_volume=4571250000,
        integration_steps=INTEGRATION_STEPS_PER_MONTH,
        sec_per_timestep=SEC_PER_TIMESTEP_PER_MONTH,
        release_range_function=Roseires_release_range,
    )
    Roseires.add_water_update_function(
        generate_evaporation_function(
            Roseires_evap_rates, Roseires_storage_to_surface_rel
        )
    )
    # ------------------------------------------------------------------------
    Edge(GERD, Roseires)

    # USSennar_irr_system
    # ------------------------------------------------------------------------
    USSennar_irr_system = Node(
        "USSennar_irr",
        inflow=np.loadtxt(
            DATA_DIRECTORY / "catchments" / "InflowRoseiresToAbuNaama.txt"
        ),
        demand=np.loadtxt(DATA_DIRECTORY / "irrigation" / "irr_demand_USSennar.txt"),
        release_range_function=irrigation_release_range,
        passive=True,
    )
    USSennar_irr_system.add_reward(sudan_deficit)
    # ------------------------------------------------------------------------
    Edge(Roseires, USSennar_irr_system)

    # Sennar
    # ------------------------------------------------------------------------
    name = "Sennar"
    Sennar_inflow = np.loadtxt(DATA_DIRECTORY / "catchments" / "InflowSukiToSennar.txt")
    Sennar_storage_to_minmax_rel = np.loadtxt(
        DATA_DIRECTORY / "dams" / f"store_min_max_release_{name}.txt"
    )
    Sennar_storage_to_surface_rel = np.loadtxt(
        DATA_DIRECTORY / "dams" / f"store_sur_rel_{name}.txt"
    )
    Sennar_evap_rates = np.loadtxt(DATA_DIRECTORY / "dams" / f"evap_{name}.txt")
    Sennar_release_range = generate_storage_to_release_range_function(
        Sennar_storage_to_minmax_rel
    )
    Sennar = Node(
        name,
        Sennar_inflow,
        max_action=10000,
        initial_volume=434925000,
        integration_steps=INTEGRATION_STEPS_PER_MONTH,
        sec_per_timestep=SEC_PER_TIMESTEP_PER_MONTH,
        release_range_function=Sennar_release_range,
    )
    Sennar.add_water_update_function(
        generate_evaporation_function(Sennar_evap_rates, Sennar_storage_to_surface_rel)
    )
    # ------------------------------------------------------------------------
    Edge(USSennar_irr_system, Sennar)

    # Gezira_irr_system
    # ------------------------------------------------------------------------
    Gezira_irr_system = Node(
        "Gezira_irr",
        demand=np.loadtxt(DATA_DIRECTORY / "irrigation" / "irr_demand_Gezira.txt"),
        release_range_function=irrigation_release_range,
        passive=True,
    )
    Gezira_irr_system.add_reward(sudan_deficit)
    # ------------------------------------------------------------------------
    Edge(Sennar, Gezira_irr_system)

    # DSSennar_irr_system
    # ------------------------------------------------------------------------
    Dinder_inflow = np.loadtxt(DATA_DIRECTORY / "catchments" / "InflowDinder.txt")
    Rahad_inflow = np.loadtxt(DATA_DIRECTORY / "catchments" / "InflowRahad.txt")
    DSSennar_irr_system = Node(
        "DSSennar_irr",
        inflow=Dinder_inflow + Rahad_inflow,
        demand=np.loadtxt(DATA_DIRECTORY / "irrigation" / "irr_demand_DSSennar.txt"),
        release_range_function=irrigation_release_range,
        passive=True,
    )
    DSSennar_irr_system.add_reward(sudan_deficit)
    # ------------------------------------------------------------------------
    Edge(Gezira_irr_system, DSSennar_irr_system)

    # Tamaniat_irr_system
    # ------------------------------------------------------------------------
    Tamaniat_irr_system = Node(
        "Tamaniat_irr",
        inflow=np.loadtxt(DATA_DIRECTORY / "catchments" / "InflowWhiteNile.txt"),
        demand=np.loadtxt(DATA_DIRECTORY / "irrigation" / "irr_demand_Tamaniat.txt"),
        release_range_function=irrigation_release_range,
        passive=True,
    )
    Tamaniat_irr_system.add_reward(sudan_deficit)
    # ------------------------------------------------------------------------
    Edge(DSSennar_irr_system, Tamaniat_irr_system)

    # Hassanab_irr_system
    # ------------------------------------------------------------------------
    Hassanab_irr_system = Node(
        "Hassanab_irr",
        inflow=np.concatenate(
            ([0], np.loadtxt(DATA_DIRECTORY / "catchments" / "InflowAtbara.txt"))
        ),
        demand=np.loadtxt(DATA_DIRECTORY / "irrigation" / "irr_demand_Hassanab.txt"),
        release_range_function=irrigation_release_range,
        passive=True,
    )
    Hassanab_irr_system.add_reward(sudan_deficit)
    # ------------------------------------------------------------------------
    Edge(Tamaniat_irr_system, Hassanab_irr_system, delay=1, replacement_values=[934.2])

    # HAD
    # ------------------------------------------------------------------------
    name = "HAD"
    HAD_storage_to_level_rel = np.loadtxt(
        DATA_DIRECTORY / "dams" / f"store_level_rel_{name}.txt"
    )
    HAD_storage_to_minmax_rel = np.loadtxt(
        DATA_DIRECTORY / "dams" / f"store_min_max_release_{name}.txt"
    )
    HAD_storage_to_surface_rel = np.loadtxt(
        DATA_DIRECTORY / "dams" / f"store_sur_rel_{name}.txt"
    )
    HAD_evap_rates = np.loadtxt(DATA_DIRECTORY / "dams" / f"evap_{name}.txt")
    HAD_release_range = generate_storage_to_release_range_function(
        HAD_storage_to_minmax_rel
    )
    HAD = Node(
        name,
        # HAD_inflow,
        max_action=4000,
        initial_volume=137025000000,
        integration_steps=INTEGRATION_STEPS_PER_MONTH,
        sec_per_timestep=SEC_PER_TIMESTEP_PER_MONTH,
        release_range_function=HAD_release_range,
    )
    HAD.add_water_update_function(
        generate_evaporation_function(HAD_evap_rates, HAD_storage_to_surface_rel)
    )

    def minimum_water_level(info: dict) -> float:
        water_level = storage_to_level(info["volume"], HAD_storage_to_level_rel)
        min_water_level = 159
        if water_level < min_water_level:
            return 0.0
        else:
            return 1.0

    HAD.add_reward(
        Reward(
            "HAD_minimum_water_level",
            minimum_water_level,
            reward_range=spaces.Box(0, 1),
        )
    )
    # ------------------------------------------------------------------------
    Edge(Hassanab_irr_system, HAD)

    # Egypt_irr_system
    # ------------------------------------------------------------------------
    Egypt_irr_system = Node(
        "Egypt_irr_system",
        demand=np.loadtxt(DATA_DIRECTORY / "irrigation" / "irr_demand_Egypt.txt"),
        release_range_function=irrigation_release_range,
        passive=True,
    )
    Egypt_irr_system.add_reward(
        Reward(
            "egypt_deficit_minimised",
            minimise_demand_deficit,
            spaces.Box(float("-inf"), 0),
        )
    )
    # ------------------------------------------------------------------------
    Edge(HAD, Egypt_irr_system)

    node_list = [
        GERD,
        Roseires,
        USSennar_irr_system,
        Sennar,
        Gezira_irr_system,
        DSSennar_irr_system,
        Tamaniat_irr_system,
        Hassanab_irr_system,
        HAD,
        Egypt_irr_system,
    ]

    env = WaterManagement(node_list, SEED)

    return env
