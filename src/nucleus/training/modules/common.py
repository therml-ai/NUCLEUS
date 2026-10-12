from boiling_data.flashx.normalizer import NormalizerWrapper
from boiling_data.tasks.forecast.flashx_batch import FlashXForecastBatch


def normalize_forecast_batch(normalizer: NormalizerWrapper, batch: FlashXForecastBatch) -> FlashXForecastBatch:
    return FlashXForecastBatch(normalizer.normalize(batch.input), normalizer.normalize(batch.target))
