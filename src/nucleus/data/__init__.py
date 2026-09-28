from .forecast_dataset import ForecastDatasetBase, ForecastDataset
from .in_mem_forecast_dataset import InMemForecastDataset
from .in_mem_divfree_forecast_dataset import InMemDivFreeForecastDataset
from .forecast_webdataset import forecast_web_dataset
from .batching import collate
from .in_mem_divfree_forecast_dataset import divfree_collate

# Maps the `pydataset` field of a data_cfg to the PyTorch dataset it selects, so
# the choice is made explicitly in config rather than inferred from the dataset
# name. "forecast_web" is built through a separate code path (webdataset shards).
PYDATASETS = {
    "forecast": (ForecastDataset, collate),
    "in_mem_forecast": (InMemForecastDataset, collate),
    "in_mem_divfree_forecast": (InMemDivFreeForecastDataset, divfree_collate),
    "forecast_web": (forecast_web_dataset, collate),
}


def get_pydataset(name: str):
    try:
        return PYDATASETS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown pydataset '{name}'. Available: {sorted(PYDATASETS)}"
        ) from exc