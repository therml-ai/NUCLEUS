import dataclasses
from typing import Any, Dict, List, Mapping, Sequence

from boiling_data.flashx.batch import FlashXBatch
from boiling_data.flashx.normalizer import NormalizerWrapper

# The conditioning parameters of the Nucleus2 models, by config group, in the order
# of the config tensor's columns.
NUCLEUS2_CONFIG_KEYS: Dict[str, List[str]] = {
    "non_dimensional": ["inv_reynolds", "cpgas", "mugas", "rhogas", "thcogas", "stefan", "prandtl", "gravy"],
    "physical": ["bulk_temp", "sat_temp"],
    "heaters": ["wall_temp_fraction", "xMin", "xMax"],
}


def restored_config(current: Any, saved: Any, runtime_options: Sequence[str]) -> Any:
    """The config a checkpoint saved, but with the runtime options, how the model is
    run rather than what it learned, kept from the config the model was built with,
    so a resumed run follows its own config rather than the checkpoint's."""
    return dataclasses.replace(saved, **{name: getattr(current, name) for name in runtime_options})


def set_runtime_options(model: Any, params: Mapping[str, Any]) -> None:
    """Sets the model's runtime options from params, e.g. a model config's params;
    the params that are not runtime options, such as its architecture, are ignored."""
    options = {name: params[name] for name in model.runtime_options if name in params}
    model.config = dataclasses.replace(model.config, **options)


def num_config_columns(config_keys: Dict[str, List[str]]) -> int:
    return sum(len(names) for names in config_keys.values())


def require_normalized(batch: FlashXBatch) -> None:
    if not batch.is_normalized:
        raise ValueError("the model takes normalized batches; normalize it with the model's normalizer")


def next_frames(prediction: FlashXBatch, num_timesteps: int, target_timesteps: int) -> FlashXBatch:
    """The first num_timesteps frames after the input. A model outputs as many frames
    as its input, of which training compares the last target_timesteps with the
    target window, the frames that come right after the input."""
    if not 1 <= num_timesteps <= target_timesteps <= prediction.num_timesteps:
        raise ValueError(
            f"need 1 <= num_timesteps <= target_timesteps <= {prediction.num_timesteps}, the frames the model "
            f"predicts, not num_timesteps={num_timesteps} and target_timesteps={target_timesteps}"
        )
    return prediction.tail_time_window(target_timesteps).head_time_window(num_timesteps)


def unnormalized_prediction(normalizer: NormalizerWrapper, window: FlashXBatch, prediction: FlashXBatch) -> FlashXBatch:
    """prediction's fields in raw units, for the same samples as the raw window.
    The window's configs are kept rather than unnormalizing the prediction's, which
    differ from them by rounding and would stop the prediction extending the window."""
    return window.with_fields(
        {name: normalizer.unnormalize_field(name, tensor) for name, tensor in prediction.fields.items()}
    )
