import os
from typing import Any, cast

import pandas as pd
from flowcean.core.strategies.active import Action, Observation
from flowcean.palaestrai.sac_model import SACModel
from flowcean.palaestrai.util import (
    convert_to_actuator_informations,
    convert_to_interface,
    convert_to_sensor_informations,
)
from palaestrai.agent.actuator_information import ActuatorInformation
from palaestrai.agent.sensor_information import SensorInformation
from palaestrai.types.mode import Mode
from typing_extensions import override

from hlarl.hl_export import handle_other_exports, handle_switches, prepare_dirs
from hlarl.hl_import import (
    load_imports,
    translate,
    update_actuators,
    wait_for_import,
)


class HlArlModel(SACModel):
    def __init__(
        self,
        action: Action,
        observation: Observation,
        model: Any,
        start_steps: int = 100,
        mapping_file: str = "",
        mapping_sw_file: str = "",
        exchange_dir: str = "",
        ask_highleit: bool = False,
        timeout: int = 30,
        inference_mode: bool = False,
    ) -> None:
        super().__init__(action, observation, model, start_steps)

        if not mapping_file or not os.path.isfile(mapping_file):
            msg = (
                f"Invalid mapping file: {mapping_file}. Check your "
                "parameters and make sure that the file exists."
            )
            raise ValueError(msg)
        if not mapping_sw_file or not os.path.isfile(mapping_sw_file):
            msg = (
                f"Invalid mapping switches file: {mapping_file}. Check "
                "your parameters and make sure that the file exists."
            )
            raise ValueError(msg)

        if not exchange_dir or not os.path.isdir(exchange_dir):
            msg = (
                f"Invalid exchange directory: {mapping_file}. Check "
                "your parameters and make sure that the directory exists."
            )
            raise ValueError(msg)

        if inference_mode:
            self.muscle._mode = Mode.TEST

        self.mapping = pd.read_csv(mapping_file)
        self.mapping_sw = pd.read_csv(mapping_sw_file)
        self.ask_highleit = ask_highleit
        self.exchange_dir = exchange_dir
        self.recommendations = pd.DataFrame()
        self.corrections = pd.DataFrame()
        self.diffs = pd.DataFrame()
        self.timeout = timeout

    @override
    def predict(self, input_features: Observation) -> Action:
        # Convert from flowcean to palaestrAI domain
        actuators_available = convert_to_actuator_informations(self.action)
        sensors = convert_to_sensor_informations(input_features)

        # Query the original SAC muscle
        actuators, self.data_for_brain = self.muscle.propose_actions(
            sensors, actuators_available
        )

        if not self.ask_highleit:
            return Action(actuators=convert_to_interface(actuators))

        self._prepare_recommendations(sensors, actuators)

        if wait_for_import(self.exchange_dir, timeout=self.timeout):
            corrected = self._receive_corrections(sensors, actuators)

            action = Action(actuators=convert_to_interface(corrected))
        else:
            action = Action(actuators=convert_to_interface(actuators))

        return action

    def _prepare_recommendations(
        self,
        sensors: list[SensorInformation],
        actuators: list[ActuatorInformation],
    ) -> None:

        prepare_dirs(self.exchange_dir)
        recs1 = handle_switches(
            self.mapping_sw.copy(), actuators, self.exchange_dir
        )
        recs2 = handle_other_exports(
            self.mapping.copy(), sensors, actuators, self.exchange_dir
        )

        self.recommendations = cast(
            pd.DataFrame,
            pd.concat(
                [self.recommendations, pd.concat([recs1, recs2], axis=1)],
                ignore_index=True,
            ),
        )

    def _receive_corrections(
        self,
        sensors: list[SensorInformation],
        actuators: list[ActuatorInformation],
    ) -> list[ActuatorInformation]:
        corrections = load_imports(
            self.exchange_dir, self.mapping, self.mapping_sw
        )

        corrections = translate(corrections, sensors, actuators)
        corrected, diff = update_actuators(corrections, sensors, actuators)

        self.corrections = cast(
            pd.DataFrame,
            pd.concat([self.corrections, corrections], ignore_index=True),
        )
        self.diffs = cast(
            pd.DataFrame, pd.concat([self.diffs, diff], ignore_index=True)
        )

        return corrected
