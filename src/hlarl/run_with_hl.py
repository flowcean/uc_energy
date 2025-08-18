import logging
from pathlib import Path

import flowcean.cli
import numpy as np
import pandas as pd
from flowcean.core.environment.active import ActiveEnvironment
from flowcean.core.model import Model
from flowcean.core.strategies.active import StopLearning
from flowcean.mosaik.energy_system import EnergySystemActive
from midas_palaestrai import ArlDefenderObjective

from hlarl.learner import HlArlLearner
from hlarl.reward import calculate_reward

logger = logging.getLogger("run_with_hl")


def run_simulation(steps: int = 30 * 24 * 60) -> None:
    """Run the simulation.

    Assuming a step size of 60 seconds.

    """
    end = steps * 60
    flowcean.cli.initialize()

    logger.info("Prepare paths ...")
    # Prepare the required paths
    data_path = (Path(__file__) / ".." / "data").resolve()
    scenario_file = data_path / "midas_scenario.yml"
    mapping_file = data_path / "mapping.csv"
    mapping_sw_file = data_path / "mapping_sw.csv"
    sensor_file = data_path / "sensors.txt"
    actuator_file = data_path / "actuators.txt"
    exchange_dir = Path.cwd() / "folder_to_observe"
    exchange_dir.mkdir(exist_ok=True)
    output_path = Path.cwd() / "_outputs"
    model_path = Path.cwd() / "models"
    result_file = output_path / "test_results_00.csv"
    reward_file = output_path / "test_rewards.csv"

    logger.info("Setup the environment ...")
    # Setup the environment
    environment = EnergySystemActive(
        "agenc_demo",
        str(result_file),
        scenario_file=str(scenario_file),
        reward_func=calculate_reward,
        end=end,
    )

    logger.info("Read actuators and sensors ...")
    with open(actuator_file, "r") as f:
        actuator_ids = f.read().splitlines()
    with open(sensor_file, "r") as f:
        sensor_ids = f.read().splitlines()

    logger.info("Setting up learner ...")
    try:
        learner = HlArlLearner(
            actuator_ids, sensor_ids, ArlDefenderObjective()
        )
        learner.setup(
            environment.action,
            environment.observation,
            mapping_file=str(mapping_file),
            mapping_sw_file=str(mapping_sw_file),
            exchange_dir=str(exchange_dir),
            ask_highleit=True,
            inference_mode=True,
        )
        learner.model.load(str(model_path / "defender_model_v2"))
    except Exception:
        logger.exception("Failed to load learner")
        environment.shutdown()
        return

    logger.info("Starting simulation ...")
    rewards: list[float] = []
    try:
        _, rewards = run_active(environment, learner)
        observation = environment.observe()

        print({r.uid: r.value for r in observation.rewards})
    except Exception:
        logger.exception("Error during environment operation.")
    except KeyboardInterrupt:
        print("Interrupted! Attempting to terminate environment ...")
    rewards_df = pd.DataFrame({"defender": rewards})
    rewards_df.to_csv(reward_file, index=False)
    environment.shutdown()
    print(learner.objective_values)
    logger.info("Finished!")


def run_active(
    environment: ActiveEnvironment, learner: HlArlLearner
) -> tuple[Model, list[float]]:
    """Run with an active environment.

    Run with an active environment by interacting with it. No learning
    is involved. The process stops when the environment ends.

    Args:
        environment: The active environment.
        learner: The active learner.

    Returns:
        The model used in the environment.
    """
    model: Model = learner.model
    rewards: list[float] = [np.nan]
    try:
        while True:
            observations = environment.observe()
            action = learner.propose_action(observations)
            environment.act(action)
            environment.step()
            model = learner.learn_active(
                action, observations, calculate_objective_only=True
            )
            rewards.append(learner.objective_values[-1])
    except StopLearning:
        pass
    return model, rewards
