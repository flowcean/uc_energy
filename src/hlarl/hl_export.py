import csv
import logging
import os
from typing import cast

import numpy as np
import pandas as pd
from palaestrai.agent.actuator_information import ActuatorInformation
from palaestrai.agent.sensor_information import SensorInformation

from hlarl.util import (
    AttributeMap,
    actuators_to_attribute_map,
    sensors_to_attribute_map,
)

LOG = logging.getLogger(__name__)

INDEX = "HLSatzNr"
B1T = "b1T"
B2T = "b2T"
B3T = "b3T"
B4T = "b4T"
B5T = "b5T"
PROC = "proc"
VISPROC = "visProc"
TWDIMT = "twDimT"
TWAKST = "twAKST"
PU_ATTRS = ["vm_pu", "vm_from_pu", "vm_to_pu", "vm_hv_pu", "vm_lv_pu"]
A_ATTRS = ["i_from_ka", "i_to_ka", "i_hv_ka", "i_lv_ka"]
HEADER = (
    "#Titel: mw\n#\n#Infopunkt-Typ: MW_SVE\n#\n#Trennzeichen:"
    " \t\n#\n#Felder:\n"
)
SA_HEADER = (
    "#Titel: Schaltzust�nde sa_sve Export\n#\n#Infopunkt-Typ: "
    "SA_SVE\n#\n#Trennzeichen: \t\n#\n#Felder:\n"
)
ML_HEADER = (
    "#Titel: Schaltzust�nde ml_sve Export\n#\n#Infopunkt-Typ: "
    "ML_SVE\n#\n#Trennzeichen: \t\n#\n#Felder:\n"
)
SA_STATE = "saZustand"
SA_PREVIEW = "saPreview"
VALUE = "value"
VIS_VALUE = "visValue"
MW_EXPORT_FILE = "import_hlnt_mw.csv"
SA_EXPORT_FILE = "import_hlnt_sa_sve.csv"
ML_EXPORT_FILE = "import_hlnt_ml_sve.csv"


def handle_switches(
    mapping: pd.DataFrame,
    actuator_setpoints: list[ActuatorInformation],
    filedir: str,
) -> pd.DataFrame:
    mapping[SA_STATE] = mapping["default_sa"]
    mapping[SA_PREVIEW] = mapping["default_sa"]
    mapping[VALUE] = mapping["default_ml"]
    mapping[VIS_VALUE] = mapping["default_ml"]

    hl_sw_sa, hl_sw_ml, recs = map_switch_actuators(
        mapping, actuator_setpoints
    )

    write_at_loc = os.path.join(filedir, SA_EXPORT_FILE)
    LOG.info("Writing switch SA export to %s", write_at_loc)
    export_to_csv(hl_sw_sa, write_at_loc, SA_HEADER)

    write_at_loc = os.path.join(filedir, ML_EXPORT_FILE)

    LOG.info("Writing switch ML export to %s", write_at_loc)
    export_to_csv(hl_sw_ml, write_at_loc, ML_HEADER)

    return recs


def map_switch_actuators(
    mapping: pd.DataFrame, actuators: list[ActuatorInformation]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for actuator in actuators:
        # _, short_uid = actuator.uid.split(".", 1)
        if actuator.uid in mapping["ppkey"].values:
            indices = mapping[mapping.ppkey == actuator.uid].index.values
            for index in indices:
                if actuator.value is None or isinstance(
                    actuator.value, np.ndarray
                ):
                    LOG.info(
                        "Can't read value %s of actuator %",
                        str(actuator.value),
                        actuator.uid,
                    )
                    continue

                mapping.loc[index, SA_STATE] = int(actuator.value)
                mapping.loc[index, SA_PREVIEW] = int(actuator.value)
                mapping.loc[index, VALUE] = int(actuator.value + 1)
                mapping.loc[index, VIS_VALUE] = int(actuator.value + 1)

    mapping.index += 1
    mapping.index.name = INDEX
    mapping[INDEX] = mapping.index
    export_sa = cast(
        pd.DataFrame,
        mapping[
            [INDEX, B1T, B2T, B3T, B4T, B5T, TWAKST, SA_STATE, SA_PREVIEW]
        ],
    )
    export_ml = cast(
        pd.DataFrame,
        mapping[[INDEX, B1T, B2T, B3T, B4T, B5T, TWAKST, VALUE, VIS_VALUE]],
    )
    recs = (
        cast(pd.DataFrame, mapping[["ppkey", SA_STATE]])
        .set_index(pd.Index([0] * mapping.shape[0]))
        .dropna()
        .assign(**{SA_STATE: lambda df: df[SA_STATE].astype(bool)})
        .drop_duplicates(subset="ppkey")
        .pivot(columns="ppkey", values=SA_STATE)
    )

    return export_sa, export_ml, recs


def handle_other_exports(
    mapping: pd.DataFrame,
    sensors: list[SensorInformation],
    actuators: list[ActuatorInformation],
    filedir: str,
) -> pd.DataFrame:
    mapping[PROC] = 0.0
    recs = get_recommendations(sensors, actuators)

    mapping = map_sensors(mapping, sensors, recs)

    write_at_loc = os.path.join(filedir, MW_EXPORT_FILE)

    LOG.info("Writing export to %s", write_at_loc)
    export_to_csv(mapping, write_at_loc, HEADER)

    return recs


def map_sensors(
    mapping: pd.DataFrame, sensors: list[SensorInformation], recs: pd.DataFrame
) -> pd.DataFrame:
    for sensor in sensors:
        assert isinstance(sensor.uid, str)
        _, _, attr = sensor.uid.split(".")
        if sensor.uid in mapping["ppkey"].values:
            indices = mapping[mapping.ppkey == sensor.uid].index.values
            for index in indices:
                if "Sollvorgabe" in mapping.loc[index][B5T]:
                    value = recs[sensor.uid].item()
                else:
                    value = sensor.value
                if attr in PU_ATTRS:
                    multiplier = float(mapping.loc[index, B2T].split(" ")[0])
                elif attr in A_ATTRS:
                    multiplier = 1000
                else:
                    multiplier = 1.0
                assert value is not None
                mapping.loc[index, PROC] = value * multiplier
    mapping[VISPROC] = mapping[PROC]
    mapping.index += 1
    mapping.index.name = INDEX
    mapping[INDEX] = mapping.index
    mapping = cast(
        pd.DataFrame,
        mapping[[INDEX, B1T, B2T, B3T, B4T, B5T, PROC, VISPROC, TWDIMT]],
    )

    return mapping


def get_recommendations(
    sensors: list[SensorInformation], actuators: list[ActuatorInformation]
) -> pd.DataFrame:
    data: AttributeMap = {}
    data = sensors_to_attribute_map(data, sensors)
    data = actuators_to_attribute_map(data, actuators)

    recs = {}
    for attrs in data.values():
        for attr, srcs in attrs.items():
            recs[srcs["uid"]] = [
                (
                    cast(float, srcs["bus"])
                    - cast(float, srcs.get("storage_prev", 0.0))
                    + cast(float, srcs.get("storage", 0.0))
                )
            ]

    return pd.DataFrame(recs)


def export_to_csv(data: pd.DataFrame, filename: str, file_header: str) -> None:
    with open(filename, "w") as f:
        f.write(file_header)

        writer = csv.writer(f, delimiter="\t")
        writer.writerow(list(data.columns) + [""])

        for _, row in data.iterrows():
            row_data = [
                str(value).replace(".", ",")
                if isinstance(value, float)
                else value
                for value in row
            ]

            writer.writerow(row_data + [""])
