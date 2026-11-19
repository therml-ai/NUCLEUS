from pathlib import Path

import hydra
import torch
from diffusers import UNet2DModel
from einops import rearrange
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from nucleus.data import InMemForecastDataset
from nucleus.data.batching import collate
from nucleus.data.layout import convert_layout
from nucleus.data.normalize import get_normalizer
from nucleus.models import load_model_from_checkpoint


def make_dataset(cfg: DictConfig, paths: list, normalizer, surrogate, augment: bool):
    return InMemForecastDataset(
        filenames=paths,
        input_fields=cfg.data_cfg.input_fields,
        output_fields=cfg.data_cfg.output_fields,
        history_time_window=cfg.history_time_window,
        future_time_window=cfg.future_time_window,
        time_step=cfg.time_step,
        start_time=cfg.start_time,
        normalizer=normalizer,
        augment=augment,
        layout=cfg.model_cfg.layout,
        fluid_params=surrogate.expected_fluid_params,
        heater_params=surrogate.expected_heater_params,
        global_params=surrogate.expected_global_params,
    )


def flatten_fields(fields: torch.Tensor, layout: str) -> torch.Tensor:
    fields = convert_layout(fields, "t h w c", layout)
    return rearrange(fields, "batch time height width channel -> batch (time channel) height width")


def unflatten_fields(fields: torch.Tensor, window: int, layout: str) -> torch.Tensor:
    fields = rearrange(
        fields, "batch (time channel) height width -> batch time height width channel",
        time=window,
    )
    return convert_layout(fields, layout, "t h w c")


@torch.no_grad()
def surrogate_prediction(surrogate, batch) -> torch.Tensor:
    prediction = surrogate(batch.get_input())
    return prediction[0] if isinstance(prediction, tuple) else prediction


def training_step(unet, surrogate, batch, optimizer, layout: str) -> float:
    history = flatten_fields(batch.input, layout)
    prediction = flatten_fields(surrogate_prediction(surrogate, batch), layout)
    target = flatten_fields(batch.target, layout)
    noise = 0.5 * torch.randn_like(target)
    sigma = torch.rand(target.shape[0], device=target.device)
    mixing_weight = sigma[:, None, None, None]
    noisy_target = (1 - mixing_weight) * target + mixing_weight * noise
    model_input = torch.cat((history, prediction, noisy_target), dim=1)
    velocity = unet(model_input, sigma * 100).sample
    loss = torch.nn.functional.mse_loss(velocity, noise - target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_unet(unet, surrogate, dataloader, cfg: DictConfig, device: torch.device) -> None:
    if len(dataloader) == 0:
        raise ValueError("The training dataset is empty.")
    optimizer = torch.optim.AdamW(unet.parameters(), lr=cfg.get("lr", 2.5e-4))
    unet.train()
    global_step = 0
    while global_step < cfg.max_steps:
        for batch in dataloader:
            loss = training_step(
                unet, surrogate, batch.to(device), optimizer, cfg.model_cfg.layout
            )
            global_step += 1
            if global_step == 1 or global_step % 100 == 0:
                print(f"Step {global_step}: loss={loss:.6f}", flush=True)
            if global_step >= cfg.max_steps:
                break


@torch.no_grad()
def sample_unet(unet, history, prediction, steps: int = 50) -> torch.Tensor:
    sample = 0.5 * torch.randn_like(prediction)
    # Training flows from data to noise, so generation integrates from sigma=1 to 0.
    sigma_grid = torch.linspace(1, 0, steps + 1, device=sample.device)
    for step_index in range(steps):
        sigma = sigma_grid[step_index].expand(sample.shape[0])
        model_input = torch.cat((history, prediction, sample), dim=1)
        velocity = unet(model_input, sigma * 100).sample
        sample += (sigma_grid[step_index + 1] - sigma_grid[step_index]) * velocity
    return sample


@torch.no_grad()
def rollout(unet, surrogate, dataset, normalizer, cfg: DictConfig, device: torch.device):
    predictions, targets = [], []
    history = None
    layout = cfg.model_cfg.layout
    for start_index in range(0, len(dataset), cfg.future_time_window):
        batch = dataset[start_index].to_collated_batch().to(device)
        if history is not None:
            batch.input = history
        prediction = surrogate_prediction(surrogate, batch)
        corrected = sample_unet(
            unet, flatten_fields(batch.input, layout), flatten_fields(prediction, layout)
        )
        history = unflatten_fields(corrected, cfg.future_time_window, layout)
        bulk_temperature = normalizer.unnormalize_params(batch.sim_params_dict)[0]["bulk_temp"]
        for fields, collection in ((history, predictions), (batch.target, targets)):
            physical_fields = normalizer.unnormalize(fields, bulk_temperature, layout=layout)
            collection.append(convert_layout(physical_fields, "t h w c", layout).squeeze(0).cpu())
    if not predictions:
        raise ValueError("The test dataset is empty.")
    return {"preds": torch.cat(predictions), "targets": torch.cat(targets)}


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    if cfg.history_time_window != cfg.future_time_window or cfg.time_step != 1:
        raise ValueError("Use equal history/future windows and time_step=1.")
    if list(cfg.data_cfg.input_fields) != list(cfg.data_cfg.output_fields):
        raise ValueError("Input and output fields must match for rollout feedback.")
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer = get_normalizer(OmegaConf.to_container(cfg.normalizer_cfg, resolve=True))
    surrogate = load_model_from_checkpoint(
        cfg.checkpoint_path, map_location=device,
        model_cfg=OmegaConf.to_container(cfg.model_cfg, resolve=True),
    ).to(device)
    if getattr(surrogate, "_model_name", None) == "nucleus2_moe_divfree":
        raise ValueError("This minimal script requires a surrogate with a standard forward method.")
    surrogate.eval().requires_grad_(False)
    output_dir = Path(cfg.log_dir) / "diffusion_re"
    output_dir.mkdir(parents=True, exist_ok=True)
    channels = cfg.future_time_window * len(cfg.data_cfg.output_fields)
    unet = UNet2DModel(
        sample_size=None, in_channels=3 * channels, out_channels=channels,
        block_out_channels=(16, 32, 64, 128), layers_per_block=2, norm_num_groups=8,
    ).to(device)
    train_dataset = make_dataset(cfg, cfg.data_cfg.train_paths, normalizer, surrogate, True)
    dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
    )
    train_unet(unet, surrogate, dataloader, cfg, device)
    unet.save_pretrained(output_dir / "diffusion_unet")
    train_dataset.data.clear()
    del dataloader, train_dataset
    unet.eval()
    for test_index, test_path in enumerate(cfg.data_cfg.test_paths):
        test_dataset = make_dataset(cfg, [test_path], normalizer, surrogate, False)
        result = rollout(unet, surrogate, test_dataset, normalizer, cfg, device)
        output_path = output_dir / f"{test_index}_{Path(test_path).stem}.pt"
        torch.save(result, output_path)
        print(f"Saved rollout to {output_path}", flush=True)
        test_dataset.data.clear()


if __name__ == "__main__":
    main()
