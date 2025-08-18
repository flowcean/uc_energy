import numpy as np
from flowcean.core.model import Model
from flowcean.core.strategies.active import Action, Observation
from flowcean.palaestrai.sac_learner import MODEL_ID, SACLearner
from flowcean.palaestrai.util import (
    convert_to_actuator_informations,
    convert_to_reward_informations,
    convert_to_sensor_informations,
    filter_action,
    filter_observation,
)
from harl.sac.brain import SACBrain
from typing_extensions import override

from .model import HlArlModel


class HlArlLearner(SACLearner):
    @override
    def setup(
        self,
        action: Action,
        observation: Observation,
        *,
        mapping_file: str = "",
        mapping_sw_file: str = "",
        exchange_dir: str = "",
        ask_highleit: bool = False,
        timeout: int = 30,
        inference_mode: bool = False,
    ) -> None:
        self.action = filter_action(action, self.actuator_ids)
        self.observation = filter_observation(observation, self.sensor_ids)

        self.brain = SACBrain(**self.brain_params)
        self.brain._seed = 0  # noqa: SLF001
        self.brain._sensors = convert_to_sensor_informations(self.observation)  # noqa: SLF001
        self.brain._actuators = convert_to_actuator_informations(self.action)  # noqa: SLF001
        self.brain.setup()

        self.model = HlArlModel(
            self.action,
            self.observation,
            self.brain.thinking(MODEL_ID, None),
            mapping_file=mapping_file,
            mapping_sw_file=mapping_sw_file,
            exchange_dir=exchange_dir,
            ask_highleit=ask_highleit,
            timeout=timeout,
            inference_mode=inference_mode,
        )

    @override
    def learn_active(
        self,
        action: Action,
        observation: Observation,
        calculate_objective_only: bool = False,
    ) -> Model:
        filtered_action = filter_action(action, self.actuator_ids)
        filtered_observation = filter_observation(observation, self.sensor_ids)
        rewards = convert_to_reward_informations(observation)

        self.brain.memory.append(
            muscle_uid=MODEL_ID,
            sensor_readings=convert_to_sensor_informations(
                filtered_observation
            ),
            actuator_setpoints=convert_to_actuator_informations(
                filtered_action
            ),
            rewards=rewards,
            done=False,
        )
        objective = np.array(
            [self.agent_objective.internal_reward(self.brain.memory)]
        )
        self.brain.memory.append(MODEL_ID, objective=objective)

        if not calculate_objective_only:
            update = self.brain.thinking(MODEL_ID, self.model.data_for_brain)
            self.model.update(update)

        self.rewards.append(observation.rewards)
        self.objective_values.append(objective[0])
        return self.model
