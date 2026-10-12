from typing import Dict, List

from boiling_data.flashx.batch import FlashXBatch
from boiling_data.flashx.normalizer import NormalizerWrapper

# The conditioning parameters of the Nucleus2 models, by config group, in the order
# of the config tensor's columns.
NUCLEUS2_CONFIG_KEYS: Dict[str, List[str]] = {
    "non_dimensional": ["inv_reynolds", "cpgas", "mugas", "rhogas", "thcogas", "stefan", "prandtl", "gravy"],
    "physical": ["bulk_temp", "sat_temp"],
    "heaters": ["wall_temp_fraction", "xMin", "xMax"],
}


def num_config_columns(config_keys: Dict[str, List[str]]) -> int:
    return sum(len(names) for names in config_keys.values())


def require_normalized(batch: FlashXBatch) -> None:
    if not batch.is_normalized:
        raise ValueError("the model takes normalized batches; normalize it with the model's normalizer")


def unnormalized_prediction(normalizer: NormalizerWrapper, window: FlashXBatch, prediction: FlashXBatch) -> FlashXBatch:
    """prediction's fields in raw units, for the same samples as the raw window.
    The window's configs are kept rather than unnormalizing the prediction's, which
    differ from them by rounding and would stop the prediction extending the window."""
    return window.with_fields(
        {name: normalizer.unnormalize_field(name, tensor) for name, tensor in prediction.fields.items()}
    )
