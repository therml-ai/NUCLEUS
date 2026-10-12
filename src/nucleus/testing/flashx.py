import copy
from numbers import Number
from typing import Any, Dict, Optional, Sequence

import torch
from boiling_data.flashx.batch import ConfigKeys, FlashXBatch
from boiling_data.flashx.normalizer import NormalizerWrapper
from boiling_data.tasks.forecast.flashx_batch import FlashXForecastBatch

# face velocities hold one more sample than the cells along their own axis
FACE_PADDING = {"velfacex": (0, 1), "velfacey": (1, 0)}


def flashx_config(width: int = 32, height: int = 32, wall_temp_fraction: float = 1.0) -> Dict[str, Any]:
    """A pool boiling config in the grouped BubbleML layout, whose discretization
    describes a ``height`` x ``width`` cell grid of spacing 1/16."""
    return {
        "physical": {"fluid": "FC-72", "bulk_temp": 50.0, "sat_temp": 58.0, "wall_temp_scale": 70.0},
        "non_dimensional": {
            "gravx": 0.0, "gravy": -1.0, "gravz": 0.0, "prandtl": 7.35, "inv_reynolds": 0.0043,
            "cpgas": 0.7997, "mugas": 0.02816, "rhogas": 0.008687, "thcogas": 0.209, "stefan": 0.156,
        },
        "discretization": {
            "num_blocks_x": 1, "num_blocks_y": 1, "nx_block": width, "ny_block": height,
            "x_min": -width / 32, "x_max": width / 32, "y_min": 0.0, "y_max": height / 16,
            "dt": 0.5, "t_initial": 0.0,
        },
        "heaters": [{
            "type": "constant_wall_temp", "wall_temp_fraction": wall_temp_fraction,
            # in C, as boiling-data's SimulationParameters fills it in from the fraction
            "wall_temp": 50.0 + wall_temp_fraction * (70.0 - 50.0),
            "xMin": -0.5, "xMax": 0.5, "yMin": 0.0, "yMax": 1e-13, "advAngle": 45.0, "rcdAngle": 45.0,
        }],
        "boundary": {"left": {}, "right": {}, "top": {}, "bottom": {}},
    }


def random_fields(
    field_names: Sequence[str], batch_size: int, timesteps: int, height: int, width: int,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    fields = {}
    for name in field_names:
        extra_height, extra_width = FACE_PADDING.get(name, (0, 0))
        shape = (batch_size, timesteps, height + extra_height, width + extra_width)
        fields[name] = torch.randn(shape, generator=generator)
    return fields


def random_batch(
    field_names: Sequence[str], config_keys: ConfigKeys, batch_size: int = 2, timesteps: int = 2,
    height: int = 32, width: int = 32, generator: Optional[torch.Generator] = None,
) -> FlashXBatch:
    fields = random_fields(field_names, batch_size, timesteps, height, width, generator)
    configs = [flashx_config(width, height, wall_temp_fraction=1.0 + index / 10) for index in range(batch_size)]
    return FlashXBatch(fields, configs, config_keys)


def random_forecast_batch(
    field_names: Sequence[str], config_keys: ConfigKeys, batch_size: int = 2, input_timesteps: int = 2,
    target_timesteps: int = 2, height: int = 32, width: int = 32,
) -> FlashXForecastBatch:
    window = random_batch(field_names, config_keys, batch_size, input_timesteps + target_timesteps, height, width)
    return FlashXForecastBatch(window.head_time_window(input_timesteps), window.tail_time_window(target_timesteps))


def flashx_statistics(field_names: Sequence[str], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Normalization statistics in boiling-data's layout: unit-normal fields, and a
    range of two around every numeric config value."""
    config = flashx_config() if config is None else config
    return {
        "fields": {name: {"count": 1, "mean": 0.0, "std": 1.0, "min": -3.0, "max": 3.0} for name in field_names},
        "config": _config_entries(copy.deepcopy(config)),
    }


def flashx_normalizer(field_names: Sequence[str], **kwargs: Any) -> NormalizerWrapper:
    return NormalizerWrapper(flashx_statistics(field_names), **kwargs)


def _config_entries(values: Any) -> Any:
    if isinstance(values, dict):
        entries = {name: _config_entries(value) for name, value in values.items()}
        return {name: entry for name, entry in entries.items() if entry is not None}
    if isinstance(values, list) and values and isinstance(values[0], dict):
        # a list of dicts, such as the heaters, shares one set of statistics
        return _config_entries(values[0])
    if isinstance(values, Number) and not isinstance(values, bool):
        return {"count": 1, "mean": float(values), "std": 1.0, "min": float(values) - 1.0, "max": float(values) + 1.0}
    return None
