import logging
import os
from time import time
from typing import cast

import numpy as np
import pandas as pd
from inotify.adapters import Inotify
from palaestrai.agent.actuator_information import ActuatorInformation
from palaestrai.agent.sensor_information import SensorInformation

from hlarl.hl_export import B1T, B2T, SA_PREVIEW, TWAKST
from hlarl.util import (
    AttributeMap,
    actuators_to_attribute_map,
    sensors_to_attribute_map,
    str_contains,
)

LOG = logging.getLogger(__name__)

NORMAL_P = "normal_P_F0"
NORMAL_Q = "normal_Q_F0"
NORMAL_I = "normal_I_F0"
NORMAL_U = "normal_U"
NORMAL_PHASE = "normal_Phase"

IMPORT_FILES = {
    "load": "export_hlnt_belastungen.csv",
    "sgen": "export_hlnt_netzeinspeisung.csv",
    "bus": "export_hlnt_knoten.csv",
    "switch": "export_hlnt_sa_sve.csv",
    "trafo": "export_hlnt_zweiwickler.csv",
}

FILES_TO_LOAD = ["load", "sgen", "bus", "switch"]


def load_imports(
    filedir: str, mapping: pd.DataFrame, mapping_sw: pd.DataFrame
) -> pd.DataFrame:
    data = {}
    for name, path in IMPORT_FILES.items():
        if name not in FILES_TO_LOAD:
            continue

        fp = os.path.join(filedir, path)
        if name == "load":
            loads = pandas_read(fp, [0, 1, 2, 3, 4])
            for i in range(loads.shape[0]):
                for k1, k2, f in zip(
                    [NORMAL_P, NORMAL_Q, NORMAL_I],
                    ["p_mw", "q_mvar", "i_ka"],
                    [1, 1, 0.001],
                ):
                    key = f"bus-{i + 1}.{k2}"
                    data.setdefault(key, [0.0])
                    data[key][0] += float(loads.iloc[i][k1]) * f

        if name == "sgen":
            sgens = pandas_read(fp, [0, 1, 2, 3])
            for i in range(sgens.shape[0]):
                for k1, k2, f in zip(
                    [NORMAL_P, NORMAL_Q, NORMAL_I],
                    ["p_mw", "q_mvar", "i_ka"],
                    [1, 1, 0.001],
                ):
                    key = f"bus-{i}.{k2}"
                    data.setdefault(key, [0.0])
                    data[key][0] += float(sgens.iloc[i][k1]) * f

        if name == "bus":
            nodes = pandas_read(fp, [0, 1, 2])

            for i in range(nodes.shape[0]):
                bus_vmpu = mapping[
                    mapping["ppkey"].str.contains(f"bus-{i}.vm_pu", na=False)
                ]
                key = bus_vmpu["ppkey"].item()
                ref = float(bus_vmpu[B2T].item().split(" ")[0])
                val = float(nodes.iloc[i][NORMAL_U])
                data[key] = [val / ref]
                data.setdefault(f"bus-{i}.va_degree", []).append(
                    float(nodes.iloc[i][NORMAL_PHASE])
                )
        if name == "switch":
            switches = pandas_read(fp, [0, 1, 2, 3, 4, 5, 6, 7, 8])
            for i in range(mapping_sw.shape[0]):
                ppkey = mapping_sw.iloc[i]["ppkey"]
                if pd.isna(ppkey) or ppkey in data:
                    continue

                subset = switches[switches[B1T] == mapping_sw.iloc[i][B1T]]
                state = subset[subset[TWAKST] == mapping_sw.iloc[i][TWAKST]][
                    SA_PREVIEW
                ].item()
                data.setdefault(ppkey, []).append(state == 1)
        if name == "trafo":
            # TODO:
            trafos = pandas_read(fp, [0, 1, 2, 3, 4, 5, 6, 7, 8])  # noqa: F841

    return pd.DataFrame(data)


def translate(
    corrections: pd.DataFrame,
    sensors: list[SensorInformation],
    actuators: list[ActuatorInformation],
) -> pd.DataFrame:
    sensor_ids: list[str] = [str(s.uid) for s in sensors]
    actuator_ids: list[str] = [str(a.uid) for a in actuators]
    renamings = {}
    relevant_cols = []
    for col in corrections.columns:
        if col in sensor_ids or col in actuator_ids:
            relevant_cols.append(col)
            continue

        eid, attr = col.split(".")
        for sid in sensor_ids:
            _, e, a = sid.split(".")
            _, e_ = e.split("-", 1)
            if e_ == eid and a == attr:
                # if str_contains(sid, [[eid, attr]]):
                renamings[col] = sid
                relevant_cols.append(sid)
                break

        if col in renamings:
            continue

        for sid in actuator_ids:
            _, e, a = sid.split(".")
            _, e_ = e.split("-", 1)
            if e_ == eid and a == attr:
                renamings[col] = sid
                relevant_cols.append(sid)
    corrections = cast(
        pd.DataFrame, corrections.rename(columns=renamings)[relevant_cols]
    )
    return corrections


def update_actuators(
    corrections: pd.DataFrame,
    sensors: list[SensorInformation],
    actuators: list[ActuatorInformation],
) -> tuple[list[ActuatorInformation], pd.DataFrame]:
    data: AttributeMap = {}
    relevant_kws: list[str | list[str]] = [
        ["bus", "p_mw"],
        ["bus", "q_mvar"],
        ["storage", "p_mw"],
        ["storage", "q_mvar"],
    ]
    diff: dict[str, list[int | float]] = {}
    actuator_ids: list[str] = [str(a.uid) for a in actuators]

    data = sensors_to_attribute_map(data, sensors)
    data = actuators_to_attribute_map(data, actuators)

    for c in corrections.columns:
        if not str_contains(c, relevant_kws):
            continue

        _, eid, attr = c.split(".")
        _, bidx = eid.rsplit("-", 1)
        data[bidx][attr]["correction"] = float(corrections[c].item())

    corrected = []
    corrected_ids = []
    for attrs in data.values():
        for attr, srcs in attrs.items():
            uid = srcs["uid"]
            assert isinstance(uid, str)
            _, eid, attr = uid.split(".")
            _, bidx = eid.rsplit("-", 1)

            storage_uid = srcs.get("storage_uid", "")
            if storage_uid not in actuator_ids:
                continue

            act = actuators[actuator_ids.index(storage_uid)]
            assert act.value is not None

            val = np.clip(
                cast(float, srcs["correction"])
                + cast(float, srcs.get("storage_prev", 0.0))
                - cast(float, srcs["bus"]),
                act.space.low,  # type: ignore[reportAttributeAccessIssue]
                act.space.high,  # type: ignore[reportAttributeAccessIssue]
            )
            diff[act.uid] = [float(act.value) - float(val)]
            dtype = act.space.dtype  # type: ignore[reportAttributeAccessIssue]
            corrected.append(
                ActuatorInformation(
                    value=np.array(val, dtype=dtype),
                    uid=act.uid,
                    space=act.space,
                )
            )
            corrected_ids.append(act.uid)

    for act in actuators:
        if act.uid not in corrected_ids:
            assert act.value is not None
            old_val = int(act.value)
            if act.uid in corrections.columns:
                if "switch" in act.uid:
                    new_val = 1 if corrections[act.uid].item() else 0
                else:
                    new_val = int(corrections[act.uid].item())

                diff[act.uid] = [old_val - new_val]
                dtype = act.space.dtype  # type: ignore[reportAttributeAccessIssue]
                corrected.append(
                    ActuatorInformation(
                        value=np.array(new_val, dtype=dtype),
                        space=act.space,
                        uid=act.uid,
                    )
                )
            else:
                corrected.append(
                    ActuatorInformation(
                        value=act.value, uid=act.uid, space=act.space
                    )
                )
    return corrected, pd.DataFrame(diff)


def wait_for_import(filedir: str, timeout: int = 30) -> bool:
    files = [f for k, f in IMPORT_FILES.items() if k in FILES_TO_LOAD]

    t_start = time()
    i = Inotify()
    i.add_watch(filedir)

    LOG.info("Start watching %s ...", filedir)
    while True:
        for event in i.event_gen(timeout_s=1, yield_nones=True):
            if event is not None:
                _, etype, path, filename = event
                if filename in files:
                    if etype[0] in ["IN_CREATE", "IN_MODIFY", "IN_MOVED_TO"]:
                        LOG.debug("Event %s for %s", etype[0], filename)
                        files.remove(filename)

        t_elapsed = time() - t_start
        if t_elapsed >= timeout:
            success = False
            break
        if not files:
            success = True
            break

    if success:
        LOG.info("Finished watch after %.3f seconds", t_elapsed)
    else:
        LOG.info("No file changes detected.")
    return success


def pandas_read(filename, usecols, encoding="utf-8") -> pd.DataFrame:
    return cast(
        pd.DataFrame,
        pd.read_csv(
            filename,
            usecols=usecols,
            skiprows=7,
            decimal=",",
            index_col=0,
            sep="\t",
            encoding=encoding,
        ),
    )
