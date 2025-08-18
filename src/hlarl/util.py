from palaestrai.agent.actuator_information import ActuatorInformation
from palaestrai.agent.sensor_information import SensorInformation
from typing_extensions import TypeAlias

AttributeMap: TypeAlias = dict[str, dict[str, dict[str, str | float]]]
RELEVANT_KWS: list[str | list[str]] = [
    ["bus", "p_mw"],
    ["bus", "q_mvar"],
    ["storage", "p_mw"],
    ["storage", "q_mvar"],
]


def sensors_to_attribute_map(
    data: AttributeMap, sensors: list[SensorInformation]
) -> AttributeMap:
    sensor_ids: list[str] = [str(s.uid) for s in sensors]
    for i, uid in enumerate(sensor_ids):
        assert uid is not None
        if not str_contains(uid, RELEVANT_KWS):
            continue

        sensor_val = sensors[i].value
        assert sensor_val is not None

        _, eid, attr = uid.split(".")
        _, bidx = eid.rsplit("-", 1)
        data.setdefault(bidx, {})
        data[bidx].setdefault(attr, {})

        if "bus" in uid:
            data[bidx][attr]["uid"] = uid
            data[bidx][attr]["bus"] = float(sensor_val)
        if "storage" in uid:
            data[bidx][attr]["storage_uid"] = uid
            data[bidx][attr]["storage_prev"] = float(sensor_val)

    return data


def actuators_to_attribute_map(
    data: AttributeMap, actuators: list[ActuatorInformation]
) -> AttributeMap:
    actuator_ids: list[str] = [str(a.uid) for a in actuators]
    for i, uid in enumerate(actuator_ids):
        if not str_contains(uid, RELEVANT_KWS):
            continue

        act_val = actuators[i].value
        assert act_val is not None

        _, eid, attr = uid.split(".")
        _, bidx = eid.rsplit("-", 1)
        data[bidx][attr]["storage"] = float(act_val)
    return data


def str_contains(target: str, kws: list[str | list[str]]) -> bool:
    contained = False
    for kw in kws:
        if isinstance(kw, str) and kw in target:
            return True
        if isinstance(kw, list):
            for k in kw:
                if k not in target:
                    contained = False
                    break
                contained = True
            if contained:
                return True
    return contained
