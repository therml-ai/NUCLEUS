from .forecast_dataset import ForecastDatasetBase, ForecastDataset
from .in_mem_forecast_dataset import InMemForecastDataset
from .forecast_webdataset import forecast_web_dataset

# Maps the `pydataset` field of a data_cfg to the PyTorch dataset it selects, so
# the choice is made explicitly in config rather than inferred from the dataset
# name. "forecast_web" is built through a separate code path (webdataset shards).
PYDATASETS = {
    "forecast": ForecastDataset,
    "in_mem_forecast": InMemForecastDataset,
    "forecast_web": forecast_web_dataset,
}


def get_pydataset(name: str):
    try:
        return PYDATASETS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown pydataset '{name}'. Available: {sorted(PYDATASETS)}"
        ) from exc