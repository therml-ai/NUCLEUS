import dataclasses
from typing import Optional, Callable, TypeVar, List, Any
import torch
import torch.nn as nn

M = TypeVar("M", bound=nn.Module)
MODELS = {}

def register_model(name: Optional[str] = None) -> Callable[[Callable[..., M]], Callable[..., M]]:
    def wrapper(fn: Callable[..., M]) -> Callable[..., M]:
        key = name or fn.__name__
        if key in MODELS:
            raise ValueError(f"Cannot register duplicate model ({key})")
        MODELS[key] = fn
        return fn
    return wrapper

def list_models() -> List[str]:
    return sorted(list(MODELS.keys()), key=lambda x: x[0])

def get_model(name: str, **kwargs: Any) -> nn.Module:
    name = name.lower()
    try:
        fn = MODELS[name]
    except KeyError as exc:
        raise KeyError(f"Model {name} not found. Available Models: {MODELS.keys()}") from exc

    if hasattr(fn, "config_class"):
        valid_fields = {f.name for f in dataclasses.fields(fn.config_class)}
        config = fn.config_class(**{k: v for k, v in kwargs.items() if k in valid_fields})
        model = fn(config)
        model._model_name = name
        return model
    return fn(**kwargs)

def get_model_class(name: str) -> nn.Module:
    name = name.lower()
    try:
        fn = MODELS[name]
    except KeyError as exc:
        raise KeyError(f"Model {name} not found. Available Models: {MODELS.keys()}") from exc
    return fn

def load_model_state_dict(path, map_location=None) -> dict:
    """
    Return the raw model state dict from a checkpoint, using weights_only=True.

    Supports both Lightning checkpoints (state dict nested under "state_dict" with a
    "model." prefix on every key) and raw torch.save(model.state_dict(), path) saves.
    """
    try:
        ckpt = torch.load(path, weights_only=True, map_location=map_location)
    except:
        ckpt = torch.load(path, weights_only=False, map_location=map_location)

    if "state_dict" in ckpt:
        # Lightning checkpoint — strip the "model." prefix
        return {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    return ckpt


def load_model_from_checkpoint(path, map_location=None, model_cfg=None) -> nn.Module:
    """
    Reconstruct a model and load its weights from a checkpoint, using weights_only=True.

    Supports both Lightning checkpoints (state dict nested under "state_dict" with a
    "model." prefix on every key) and raw torch.save(model.state_dict(), path) saves.

    Newer checkpoints embed the architecture in a "_extra_state" blob (model name +
    config), and the model is rebuilt from that. Older checkpoints predate that and
    carry no config; pass ``model_cfg`` (the ``{"name", "params"}`` dict the Lightning
    module keeps in ``self.model_cfg``) to rebuild them from an explicit config instead.

    Usage:
        model = load_model_from_checkpoint("run/checkpoints/last.ckpt")
        model = load_model_from_checkpoint("last.ckpt", map_location="cuda")
        model = load_model_from_checkpoint("old.ckpt", model_cfg=cfg_dict)  # no _extra_state
    """
    model_state = load_model_state_dict(path, map_location=map_location)

    extra = model_state.get("_extra_state")
    if extra is None:
        return _load_from_model_cfg(model_state, model_cfg)

    model_name = extra.get("model_name")
    config_dict = extra.get("config")
    if model_name is None or config_dict is None:
        raise ValueError("'_extra_state' is missing 'model_name' or 'config'.")

    fn = MODELS[model_name]
    if hasattr(fn, "config_from_dict"):
        config = fn.config_from_dict(config_dict)
        model = fn(config)
        model._model_name = model_name
    else:
        model = get_model(model_name, **config_dict)
    model.load_state_dict(model_state)
    return model


def _load_from_model_cfg(model_state: dict, model_cfg) -> nn.Module:
    """Rebuild a model from an explicit ``model_cfg`` for checkpoints that do not embed
    their config, then load ``model_state``. ``model_cfg`` is a ``{"name", "params"}``
    dict (``get_model`` ignores params that are not config fields, so the extra
    training-only keys are harmless). Only the absent ``_extra_state`` key may be
    missing; any other mismatch means the config does not match the saved weights."""
    if model_cfg is None:
        raise ValueError(
            "No '_extra_state' in checkpoint. Was this model saved with a config "
            "dataclass? Pass model_cfg={'name': ..., 'params': {...}} to load a "
            "checkpoint that does not embed its config."
        )
    model = get_model(model_cfg["name"], **model_cfg["params"])
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    missing = [key for key in missing if key != "_extra_state"]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint weights do not match the model built from model_cfg "
            f"(name={model_cfg['name']}): missing={missing}, unexpected={unexpected}"
        )
    return model
