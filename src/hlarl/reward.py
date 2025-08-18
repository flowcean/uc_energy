from statistics import mean, median, stdev

import numpy as np
from flowcean.core.strategies.active import Interface


def calculate_reward(sensors: list) -> list:
    voltages = sorted([s.value for s in sensors if "vm_pu" in s.uid])
    voltage_rewards = [
        Interface(
            value=voltages[0],
            uid="vm_pu-min",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=1.5,
        ),
        Interface(
            value=voltages[-1],
            uid="vm_pu-max",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=1.5,
        ),
        Interface(
            value=median(voltages),
            uid="vm_pu-median",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=1.5,
        ),
        Interface(
            value=mean(voltages),
            uid="vm_pu-mean",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=1.5,
        ),
        Interface(
            value=stdev(voltages),
            uid="vm_pu-std",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=1.5,
        ),
    ]

    lineloads = sorted(
        [s.value for s in sensors if ".loading_percent" in s.uid]
    )

    lineload_rewards = [
        Interface(
            value=lineloads[0],
            uid="lineload-min",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=200.0,
        ),
        Interface(
            value=lineloads[-1],
            uid="lineload-max",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=200.0,
        ),
        Interface(
            value=median(lineloads),
            uid="lineload-median",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=200.0,
        ),
        Interface(
            value=mean(lineloads),
            uid="lineload-mean",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=200.0,
        ),
        Interface(
            value=stdev(lineloads),
            uid="lineload-std",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=200.0,
        ),
    ]

    return voltage_rewards + lineload_rewards
