import torch.nn as nn
from nucleus.models import get_model_class, list_models


def test_list_models():
    assert list_models() == ["nucleus2_moe", "nucleus2_moe_divfree_projection"]


def test_get_model():
    for model_name in list_models():
        model = get_model_class(model_name)
        assert issubclass(model, nn.Module)
        assert model.config_keys and model.field_names
