import logging
from pathlib import Path
from typing import cast

import flowcean.cli
import numpy as np
import pandas as pd
from flowcean.core.environment.active import ActiveEnvironment
from flowcean.core.model import Model
from flowcean.core.strategies.active import StopLearning
from flowcean.mosaik.energy_system import EnergySystemActive
from flowcean.palaestrai.sac_learner import SACLearner
from flowcean.palaestrai.sac_model import SACModel
from midas_palaestrai import ArlDefenderObjective

from hlarl.reward import calculate_reward

logger = logging.getLogger("run_with_hl")

# END = 365 * 24 * 60 * 60
END = 5 * 24 * 60 * 60
# END = 10 * 60 * 60


def run_simulation() -> None:
    flowcean.cli.initialize()

    # Prepare the required paths
    data_path = (Path(__file__) / ".." / "data").resolve()
    scenario_file = data_path / "midas_scenario.yml"
    sensor_file = data_path / "sensors.txt"
    actuator_file = data_path / "actuators.txt"
    exchange_dir = Path.cwd() / "folder_to_observe"
    exchange_dir.mkdir(exist_ok=True)
    output_path = Path.cwd() / "_outputs"
    result_file = output_path / "training_results_00.csv"
    reward_file = output_path / "rewards.csv"

    # Setup the environment
    environment = EnergySystemActive(
        "agenc_demo_training",
        str(result_file),
        scenario_file=str(scenario_file),
        reward_func=calculate_reward,
        end=END,
    )

    with open(actuator_file, "r") as f:
        def_actuator_ids = f.read().splitlines()
    with open(sensor_file, "r") as f:
        def_sensor_ids = f.read().splitlines()

    try:
        def_learner = SACLearner(
            def_actuator_ids,
            def_sensor_ids,
            ArlDefenderObjective(),
            update_after=10000,
            update_every=1000,
            batch_size=100,
            fc_dims=(256, 256, 256),
        )
        def_learner.setup(environment.action, environment.observation)

    except Exception:
        logger.exception("Failed to load learner")
        environment.shutdown()
        return

    logger.info("Starting simulation ...")
    episodes = 10
    target_reward = 0.95
    num_steps = 20
    all_rewards = pd.DataFrame()
    try:
        for e in range(episodes):
            done, _, rewards = learn_active(
                environment, def_learner, target_reward, num_steps
            )
            all_rewards = pd.concat(
                [all_rewards, pd.DataFrame(rewards)], ignore_index=True
            )
            if done:
                print("Training done")
                break

            print("Restarting environment")

            environment.shutdown()
            environment = EnergySystemActive(
                "agenc_demo_training",
                str(output_path / f"training_results_{e + 1:02d}.csv"),
                scenario_file=str(scenario_file),
                reward_func=calculate_reward,
                end=END,
            )

        observation = environment.observe()

        print({r.uid: r.value for r in observation.rewards})
    except Exception:
        logger.exception("Error during environment operation.")
    except KeyboardInterrupt:
        print("Interrupted! Attempting to terminate environment ...")

    environment.shutdown()
    all_rewards.to_csv(reward_file, index=False)
    print(def_learner.objective_values)
    logger.info("Finished!")


def check_agent_reward(reward_value, reward_desired, num_steps):
    count = 0
    if len(reward_value) < num_steps:
        return False
    for i in range(num_steps):
        if reward_value[-i - 1] >= reward_desired:  # reward from the end
            count = count + 1
        else:
            count = 0
    if count == num_steps:
        return True
    else:
        return False


def learn_active(
    environment: ActiveEnvironment,
    def_learner: SACLearner,
    reward_desired,
    num_steps,
) -> tuple[bool, Model, dict[str, list[float]]]:
    """Learn from an active environment.

    Learn from an active environment by interacting with it and
    learning from the observations. The learning process stops when the
    environment ends or when the learner requests to stop.

    Args:
        environment: The active environment.
        learner: The active learner.

    Returns:
        The model learned from the environment.
    """
    def_model: SACModel | None = None
    def_rewards: list[float] = []
    # att_rewards: list[float] = []
    all_rewards: dict[str, list[float]] = {"defender": [np.nan]}
    done: bool = False
    sim_step = 0
    try:
        while not done:
            observations = environment.observe()
            action = def_learner.propose_action(observations)
            environment.act(action)
            environment.step()
            sim_step += 1
            observations = environment.observe()
            def_model = cast(
                "SACModel", def_learner.learn_active(action, observations)
            )
            all_rewards["defender"].append(def_learner.objective_values[-1])

            # Check terminatino condition
            def_rewards = def_learner.objective_values
            if len(def_rewards) >= num_steps:
                done = check_agent_reward(
                    def_rewards, reward_desired, num_steps
                )
            # break if good enough
    except StopLearning:
        pass

    if def_model is None:
        message = "No defender model was learned."
        raise RuntimeError(message)

    # saving defender and attacker models
    def_learner.save(str(Path.cwd() / "_outputs" / "defender"))
    def_learner.model.save(
        str(Path.cwd() / "_outputs" / "defender" / "defender_model")
    )

    print(f"Lasts {num_steps} rewards {def_rewards[-num_steps:]}")
    return done, def_model, all_rewards


if __name__ == "__main__":
    run_simulation()
