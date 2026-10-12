from typing import Optional

import torch
from torch import nn
from boiling_data.tasks.forecast.flashx_batch import FlashXForecastBatch


class FieldDropout(nn.Module):
    """While training, zeroes whole input fields of some samples, so the model learns
    not to lean on any one field. Each field of each sample is dropped with
    probability drop_probability. On normalized fields, zero is the field's mean."""

    def __init__(self, drop_probability: float = 0.04, generator: Optional[torch.Generator] = None):
        super().__init__()
        if not 0.0 <= drop_probability <= 1.0:
            raise ValueError(f"drop_probability must be in [0, 1], not {drop_probability}")
        self.drop_probability = drop_probability
        self.generator = generator

    def forward(self, batch: FlashXForecastBatch) -> FlashXForecastBatch:
        if not self.training:
            return batch
        fields = {name: self._dropped(tensor) for name, tensor in batch.input.fields.items()}
        return FlashXForecastBatch(batch.input.with_fields(fields), batch.target)

    def _dropped(self, field: torch.Tensor) -> torch.Tensor:
        # a generator can only draw on its own device, which need not be the field's
        device = field.device if self.generator is None else self.generator.device
        draws = torch.rand(field.shape[0], generator=self.generator, device=device).to(field.device)
        keep = (draws >= self.drop_probability).to(field.dtype)
        return field * keep.view(-1, *([1] * (field.ndim - 1)))
