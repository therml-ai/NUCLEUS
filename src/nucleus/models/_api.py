import dataclasses
from typing import Any, Callable, List, Optional, TypeVar, Union
from pathlib import Path

import torch
import torch.nn as nn
from boiling_data.flashx.normalizer import NormalizerWrapper

M = TypeVar("M", bound=nn.Module)
MODELS = {}
# checkpoints name the model they were saved from, so a renamed model keeps loading
RENAMED_MODELS = {"nucleus2_moe_divfree": "nucleus2_moe_divfree_projection"}


def register_model(name: Optional[str] = None) -> Callable[[Callable[..., M]], Callable[..., M]]:
    def wrapper(fn: Callable[..., M]) -> Callable[..., M]:
        key = name or fn.__name__
        if key in MODELS:
            raise ValueError(f"Cannot register duplicate model ({key})")
        MODELS[key] = fn
        return fn
    return wrapper


def list_models() -> List[str]:
    return sorted(MODELS.keys())


def get_model_class(name: str) -> type:
    name = _current_name(name)
    if name not in MODELS:
        raise KeyError(f"Model {name} not found. Available Models: {list_models()}")
    return MODELS[name]


def get_model(name: str, normalizer: NormalizerWrapper, **params: Any) -> nn.Module:
    """The registered model built from params, ignoring any that are not fields of
    its config, e.g. training settings kept next to them in a model config."""
    model_class = get_model_class(name)
    valid_fields = {field.name for field in dataclasses.fields(model_class.config_class)}
    config = model_class.config_class(**{key: value for key, value in params.items() if key in valid_fields})
    return _build(name.lower(), config, normalizer)


def load_model(path: Union[str, Path], map_location=None) -> nn.Module:
    """A model rebuilt from a checkpoint written by nucleus.training.checkpoint. Its
    extra state names the model and holds its config, and the normalizer's extra
    state holds the normalization statistics, so nothing else is needed."""
    model_state = torch.load(path, map_location=map_location, weights_only=True)["model"]
    extra = model_state["_extra_state"]
    model_class = get_model_class(extra["model_name"])
    # load_state_dict fills the placeholder statistics from the normalizer's extra state
    placeholder = NormalizerWrapper({"fields": {}, "config": {}})
    model = _build(extra["model_name"], model_class.config_from_dict(extra["config"]), placeholder)
    model.load_state_dict(model_state)
    # loading restores the name the checkpoint was saved under, which may be an old one
    model._model_name = _current_name(extra["model_name"])
    return model


def _build(name: str, config, normalizer: NormalizerWrapper) -> nn.Module:
    model = get_model_class(name)(config, normalizer)
    model._model_name = _current_name(name)
    return model


def _current_name(name: str) -> str:
    return RENAMED_MODELS.get(name.lower(), name.lower())
