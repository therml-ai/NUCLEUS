from typing import Optional, Callable, TypeVar, List, Any
import torch.nn as nn

M = TypeVar("M", bound=nn.Module)
MODELS = {}

def register_model(name: Optional[str] = None) -> Callable[[Callable[..., M]], Callable[..., M]]:
    """
    Decorator to register a predefined model class
    """
    def wrapper(fn: Callable[..., M]) -> Callable[..., M]:
        key = name or fn.__name__
        if key in MODELS:
            raise ValueError(f"Cannot register duplicate model ({key})")
        MODELS[key] = fn
        return fn
    return wrapper

def list_models() -> List[str]:
    """
    Returns:
        models (list[str]) : List of all predefined models
    """
    print("Available models:")
    return sorted(list(MODELS.keys()))

def get_model(name: str, **config: Any) -> nn.Module:
    """
    Args:
        name (str) : Name of the model to be fetched
    Returns:
        model (nn.Module) : Model class
    """
    name = name.lower()
    try:
        fn = MODELS[name]
    except KeyError as exc:
        raise KeyError(f"Model {name} not found. Available Models: {MODELS.keys()}") from exc

    return fn(**config)
