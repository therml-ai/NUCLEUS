import math

import torch
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
import hydra
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from nucleus.data import InMemForecastDataset
from nucleus.data.batching import collate
from nucleus.data.layout import convert_layout
from nucleus.data.normalize import get_normalizer
from nucleus.models import load_model_from_checkpoint
from nucleus.models.nucleus2_moe_divfree import (
    Nucleus2MoEDivFreeInput,
    cells_to_x_face,
    cells_to_y_face,
    domain_x_coords,
)
from nucleus.utils.set_fp32_precision import set_fp32_precision
from nucleus.utils.sdf_reinit import sdf_reinit_sussman
from nucleus.utils.physical_metrics import PhysicalMetrics, BubbleMetrics, physical_metrics, bubble_metrics
from nucleus.test import TestResults
from nucleus.plot.plotting import plot_rollout
from einops import rearrange
from pathlib import Path


def predict_surrogate(model, batch, normalizer, layout: str) -> torch.Tensor:
    if getattr(model, "_model_name", None) != "nucleus2_moe_divfree":
        output = model(batch.get_input())
        return output[0] if isinstance(output, tuple) else output

    fields = convert_layout(batch.input, target_layout="t h w c", source_layout=layout)
    sdf, temperature, velx, vely = fields.unbind(-1)
    model_input = Nucleus2MoEDivFreeInput(
        sdf=sdf,
        temperature=temperature,
        velx=cells_to_x_face(velx),
        vely=cells_to_y_face(vely),
    )
    output = model.step(
        model_input,
        sim_params=batch.sim_params_tensor,
        normalizer=normalizer,
        x_coords=domain_x_coords(sdf.shape[-1], device=sdf.device),
        sim_params_dict=normalizer.unnormalize_params(batch.sim_params_dict),
    )
    prediction = torch.stack((
        output.sdf,
        output.temperature,
        (output.velx[..., :-1] + output.velx[..., 1:]) / 2,
        (output.vely[..., :-1, :] + output.vely[..., 1:, :]) / 2,
    ), dim=-1)
    return convert_layout(prediction, target_layout=layout, source_layout="t h w c")


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):
    set_fp32_precision()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)

    normalizer = get_normalizer(OmegaConf.to_container(cfg.normalizer_cfg, resolve=True))
    layout = cfg.model_cfg.layout
    num_fields = len(cfg.data_cfg.output_fields)
    window_length = cfg.history_time_window
    if cfg.history_time_window != cfg.future_time_window or cfg.time_step != 1:
        raise ValueError("Diffusion rollout requires equal history/future windows and time_step=1.")
    if list(cfg.data_cfg.input_fields) != list(cfg.data_cfg.output_fields):
        raise ValueError("Diffusion feedback requires matching input/output fields.")

    model = load_model_from_checkpoint(
        cfg.checkpoint_path,
        map_location=device,
        model_cfg=OmegaConf.to_container(cfg.model_cfg, resolve=True),
    ).to(device)
    loaded_model_name = getattr(model, "_model_name", cfg.model_cfg.name)
    if loaded_model_name != cfg.model_cfg.name:
        raise ValueError(
            f"Checkpoint contains {loaded_model_name}, but model_cfg selects "
            f"{cfg.model_cfg.name}. Set checkpoint_path to a checkpoint trained "
            "with the selected model architecture."
        )
    model.eval()
    model.requires_grad_(False)

    train_dataset = InMemForecastDataset(
        filenames=cfg.data_cfg.train_paths,
        input_fields=cfg.data_cfg.input_fields,
        output_fields=cfg.data_cfg.output_fields,
        future_time_window=cfg.future_time_window,
        history_time_window=cfg.history_time_window,
        time_step=cfg.time_step,
        start_time=cfg.start_time,
        normalizer=normalizer,
        augment=True,
        layout=layout,
        fluid_params=model.expected_fluid_params,
        heater_params=model.expected_heater_params,
        global_params=model.expected_global_params,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate,
    )

    unet = UNet2DModel(
        sample_size=None,
        in_channels=3 * window_length * num_fields,
        out_channels=window_length * num_fields,
        block_out_channels=(64, 128, 256, 512),
        layers_per_block=2,
        norm_num_groups=32,
    ).to(device)

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=100)

    lr = cfg.get("lr", 2.5e-4)
    lr_warmup_steps = cfg.get("lr_warmup_steps", 150)
    min_lr_ratio = cfg.get("min_lr_ratio", 0.001)  # decay down to this fraction of `lr` by max_steps
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)

    def _lr_lambda(step):
        if step < lr_warmup_steps:
            return (step + 1) / max(1, lr_warmup_steps)
        progress = (step - lr_warmup_steps) / max(1, cfg.max_steps - lr_warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    def _flatten(x):
        if layout == "t h w c":
            return rearrange(x, "b t h w c -> b (t c) h w")
        return rearrange(x, "b t c h w -> b (t c) h w")

    def _unflatten(x):
        if layout == "t h w c":
            return rearrange(x, "b (t c) h w -> b t h w c", t=window_length)
        return rearrange(x, "b (t c) h w -> b t c h w", t=window_length)


    #training
    unet.train()
    global_step = 0
    for epoch in range(1000):
        for batch in train_dataloader:
            batch = batch.to(device)

            tgt_flat = _flatten(batch.target)

            with torch.no_grad():
                pred_normalized = predict_surrogate(model, batch, normalizer, layout)
            inp_flat = _flatten(batch.input)
            pred_flat = _flatten(pred_normalized)

            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (inp_flat.shape[0],), device=device,
            )
            noise = torch.randn_like(tgt_flat)
            sigma_1d = scheduler.sigmas.to(device)[timesteps]
            # unet_timesteps uses the scheduler's actual convention (sigma * num_train_timesteps),
            # the same thing `scheduler.timesteps` holds and what the inference loop feeds in --
            # NOT the raw sampling index `timesteps`, which runs in the opposite direction
            # (idx=0 -> sigma=1.0, but the proper conditioning value for sigma=1.0 is
            # num_train_timesteps, not 0).
            unet_timesteps = sigma_1d * scheduler.config.num_train_timesteps
            sigmas = sigma_1d
            while sigmas.dim() < tgt_flat.dim():
                sigmas = sigmas.unsqueeze(-1)
            noisy_tgt = (1.0 - sigmas) * tgt_flat + sigmas * noise

            #predicting
            model_input = torch.cat([inp_flat, pred_flat, noisy_tgt], dim=1)
            pred_noise = unet(model_input, unet_timesteps).sample
            loss = torch.nn.functional.mse_loss(pred_noise, noise - tgt_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            global_step += 1

            if global_step % 100 == 0:
                print(f'Step {global_step}: loss = {loss.item():.6f}, lr = {lr_scheduler.get_last_lr()[0]:.2e}')

            if global_step >= cfg.max_steps:
                break
        if global_step >= cfg.max_steps:
            break

    #inference
    unet.eval()
    num_inference_steps = 50
    scheduler.set_timesteps(num_inference_steps)
    pretrained_rollout = None
    if cfg.get("rollout_path"):
        data = torch.load(cfg.rollout_path, map_location="cpu", weights_only=False)
        if isinstance(data, list):
            pretrained_rollout = data[0].preds.squeeze(0)
        elif isinstance(data, dict):
            pretrained_rollout = data["preds"].squeeze(0)
        else:
            pretrained_rollout = data.preds.squeeze(0)
        print(f"Loaded pretrained rollout with shape {pretrained_rollout.shape}")

    save_dir = Path(cfg.log_dir) / "TEST_diffusion_rollout"
    save_dir.mkdir(parents=True, exist_ok=True)

    for test_file_path in cfg.data_cfg.test_paths:
        test_dataset = InMemForecastDataset(
            filenames=[test_file_path],
            input_fields=cfg.data_cfg.input_fields,
            output_fields=cfg.data_cfg.output_fields,
            future_time_window=cfg.future_time_window,
            history_time_window=cfg.history_time_window,
            time_step=1,
            start_time=cfg.start_time,
            normalizer=normalizer,
            augment=False,
            layout=layout,
            fluid_params=model.expected_fluid_params,
            heater_params=model.expected_heater_params,
            global_params=model.expected_global_params,
        )

        skip_itrs = test_dataset.future_time_window
        preds_list = []
        targets_list = []
        prev_clean_flat = None

        with torch.no_grad():
            for itr in range(0, len(test_dataset), skip_itrs):
                data = test_dataset[itr]
                batch = data.to_collated_batch().to(device)

                bulk_temp = normalizer.unnormalize_params(
                    [batch.sim_params_dict[0]]
                )[0]["bulk_temp"]

                if prev_clean_flat is not None:
                    # Feed the diffusion-corrected prediction back as next window's history,
                    # instead of the raw (uncorrected) surrogate output. Otherwise the
                    # surrogate model drifts exactly as it would with no diffusion correction
                    # at all -- the correction never gets a chance to slow down error
                    # accumulation over a long rollout.
                    batch.input = _unflatten(prev_clean_flat)

                inp_flat = _flatten(batch.input)

                if pretrained_rollout is not None:
                    start = itr // skip_itrs * cfg.history_time_window
                    pred_raw = pretrained_rollout[start:start + cfg.history_time_window].unsqueeze(0).to(device)
                    if layout != "t h w c":
                        pred_raw = convert_layout(pred_raw, target_layout=layout, source_layout="t h w c")
                    pred_raw = normalizer.normalize(pred_raw, bulk_temp, layout=layout)
                else:
                    pred_raw = predict_surrogate(model, batch, normalizer, layout)
                pred_flat = _flatten(pred_raw)

                # Start on the same Gaussian-to-data path used during training.
                noisy = torch.randn_like(pred_flat)
                for idx, t in enumerate(scheduler.timesteps):
                    model_input = torch.cat([inp_flat, pred_flat, noisy], dim=1)
                    pn = unet(model_input, torch.full((1,), t, device=device, dtype=torch.float32)).sample
                    sigma_t = scheduler.sigmas[idx]
                    sigma_next = scheduler.sigmas[idx + 1]
                    noisy = noisy + (sigma_next - sigma_t) * pn

                pred_flat_clean = noisy
                prev_clean_flat = pred_flat_clean

                pred = _unflatten(pred_flat_clean)
                pred = normalizer.unnormalize(pred, bulk_temp, layout=layout)
                tgt = normalizer.unnormalize(batch.target, bulk_temp, layout=layout)

                pred = pred.to(torch.float32).squeeze(0).detach().cpu()
                tgt = tgt.to(torch.float32).squeeze(0).detach().cpu()

                if not pred.isfinite().all() or not tgt.isfinite().all():
                    print(f"Hit NaN at iter {itr}")
                    break

                for t_idx in range(pred.shape[0]):
                    if layout == "t h w c":
                        pred[t_idx, :, :, 0] = sdf_reinit_sussman(pred[t_idx, :, :, 0], dx=1 / 4)
                    else:
                        pred[t_idx, 0, :, :] = sdf_reinit_sussman(pred[t_idx, 0, :, :], dx=1 / 4)

                preds_list.append(pred)
                targets_list.append(tgt)

        preds = torch.cat(preds_list, dim=0)[None, ...]
        targets = torch.cat(targets_list, dim=0)[None, ...]

        preds = convert_layout(preds, target_layout="t h w c", source_layout=layout)
        targets = convert_layout(targets, target_layout="t h w c", source_layout=layout)

        fluid_params = test_dataset.sim_params[0]
        B, T_rollout, H, W, _ = preds.shape

        dx = 1/4
        dy = dx
        bulk_temp = fluid_params["bulk_temp"]
        heater_temp = fluid_params["heater"]["wallTemp"]
        pred_pm = physical_metrics(
            preds[..., 0], preds[..., 1], preds[..., 2], preds[..., 3],
            heater_min=-5.25, heater_max=5.25,
            bulk_temp=bulk_temp, heater_temp=heater_temp,
            xcoords=torch.arange(-8, 8, dx) + dx / 2,
            dx=dx, dy=dy,
        )
        pred_bm = bubble_metrics(preds[..., 0], preds[..., 2], preds[..., 3], dx=dx, dy=dy)
        tgt_pm = physical_metrics(
            targets[..., 0], targets[..., 1], targets[..., 2], targets[..., 3],
            heater_min=-5.25, heater_max=5.25,
            bulk_temp=bulk_temp, heater_temp=heater_temp,
            xcoords=torch.arange(-8, 8, dx) + dx / 2,
            dx=dx, dy=dy,
        )
        tgt_bm = bubble_metrics(targets[..., 0], targets[..., 2], targets[..., 3], dx=dx, dy=dy)
        case_name = f"{fluid_params['setup']}_{fluid_params['liquid']}_{fluid_params['heater']['wallTemp']}"
        test_results = TestResults(
            case_name=case_name,
            preds=preds,
            targets=targets,
            pred_physical_metrics=pred_pm,
            target_physical_metrics=tgt_pm,
            pred_bubble_metrics=pred_bm,
            target_bubble_metrics=tgt_bm,
            moe_outputs=[],
            fluid_params=fluid_params,
        )

        case_dir = save_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"Plotting rollout to {case_dir}")
        plot_rollout(
            save_dir=str(case_dir),
            rollout=preds,
            test_results=test_results,
            step_size=5,
            include_ground_truth=True,
        )

if __name__ == "__main__":
    main()
