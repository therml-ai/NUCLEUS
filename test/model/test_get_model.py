import torch.nn as nn
from nucleus.models import get_model_class, list_models

def test_list_models():
    assert any(model_name.startswith("nucleus1") for model_name in list_models())

def test_get_model():
    for model_name in list_models():
        model = get_model_class(model_name)
        assert model is not None
        assert issubclass(model, nn.Module)