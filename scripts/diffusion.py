import math
import gc
import os
import random
from datetime import date

import h5py
import numpy as np
import torch
import wandb
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from nucleus.data import ForecastDataset, InMemForecastDataset
from nucleus.data.batching import collate, get_sim_params
from nucleus.data.layout import convert_layout
from nucleus.data.normalize import Normalizer, get_normalizer
from nucleus.data.sim_params import load_sim_params
from nucleus.models import get_model_class
from nucleus.utils.preemption import (
    PreemptionHandler, restore_training_checkpoint, save_training_checkpoint,
)
from nucleus.utils.sdf_metrics import sdf_frame_metrics
from nucleus.utils.physical_metrics import physical_metrics, bubble_metrics
from nucleus.plot.diffusion_gif import save_diffusion_gif, save_rollout_gif
from einops import rearrange
from pathlib import Path


def log_sdf_comparison(run, metric_prefix, variants, target, dx, dy, start_frame):
    frame_metric = f"{metric_prefix}/frame"
    records = {}
    for variant_name, prediction in variants.items():
        for metric_name, values in sdf_frame_metrics(prediction, target, dx, dy).items():
            key = f"{metric_prefix}/{variant_name}/{metric_name}"
            run.define_metric(key, step_metric=frame_metric)
            records[key] = values.tolist()
            if variant_name == "diffusion_reinitialized" and metric_name == "sdf_mae":
                legacy_key = f"{metric_prefix}/sdf_mae"
                run.define_metric(legacy_key, step_metric=frame_metric)
                records[legacy_key] = values.tolist()
    for frame_offset in range(target.shape[0]):
        run.log({
            frame_metric: start_frame + frame_offset,
            **{key: values[frame_offset] for key, values in records.items()},
        })
    return records

#scaled parameters
#reduce std noisy
#channel 
#check noise implementation

def predict_surrogate(model, batch, normalizer, layout: str) -> torch.Tensor:
    if getattr(model, "_model_name", None) != "nucleus2_moe_divfree":
        output = model(batch.get_input())
        return output[0] if isinstance(output, tuple) else output

    from nucleus.models.nucleus2_moe_divfree import (
        Nucleus2MoEDivFreeInput,
        cells_to_x_face,
        cells_to_y_face,
        domain_x_coords,
    )

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

def check_preflight_file(
    filename: str,
    cfg: DictConfig,
    metadata_keys: tuple[list[str], list[str], list[str]],
    normalizer: Normalizer,
    rollout: bool,
) -> None:
    try:
        params = load_sim_params(filename)
        int(params["bulk_temp"])
        normalized_params = normalizer.normalize_params([params])[0]
        get_sim_params(normalized_params, *metadata_keys)
        if rollout:
            for key in ("setup", "liquid"):
                if key not in params:
                    raise ValueError(f"missing rollout metadata {key!r}")
        required_frames = cfg.start_time + cfg.history_time_window + cfg.future_time_window
        with h5py.File(filename, "r") as dataset:
            for field in set(cfg.data_cfg.input_fields) | set(cfg.data_cfg.output_fields):
                shape = dataset[field].shape
                if len(shape) != 3 or shape[0] < required_frames:
                    raise ValueError(f"{field} has shape {shape}, need {required_frames} frames")
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{filename}: {error}") from error


def check_preflight_batch(
    cfg: DictConfig,
    metadata_keys: tuple[list[str], list[str], list[str]],
    normalizer: Normalizer,
) -> None:
    # The on-disk dataset loads windows without materializing whole trajectories.
    fluid_params, heater_params, global_params = metadata_keys
    layout = cfg.model_cfg.get("layout", "t h w c")
    dataset = ForecastDataset(
        filenames=cfg.data_cfg.train_paths[:1],
        input_fields=cfg.data_cfg.input_fields, output_fields=cfg.data_cfg.output_fields,
        future_time_window=cfg.future_time_window, history_time_window=cfg.history_time_window,
        time_step=cfg.time_step, start_time=cfg.start_time, layout=layout,
        fluid_params=fluid_params, heater_params=heater_params, global_params=global_params,
        normalizer=normalizer, augment=False,
    )
    batch_size = min(cfg.batch_size, len(dataset))
    if batch_size < 1:
        raise ValueError("No complete training window available for preflight.")
    batch = collate([dataset[index] for index in range(batch_size)])
    if batch.input.shape != batch.target.shape:
        raise ValueError(f"Input shape {batch.input.shape} differs from target {batch.target.shape}")
    fields = convert_layout(batch.input, "t h w c", layout)
    _, frames, height, width, channels = fields.shape
    patch_size = cfg.model_cfg.params.patch_size
    if frames != cfg.history_time_window or height % patch_size or width % patch_size:
        raise ValueError(f"Batch shape {tuple(fields.shape)} is incompatible with patch size {patch_size}")
    if channels != len(cfg.data_cfg.output_fields):
        raise ValueError(f"Unexpected field count in batch shape {tuple(fields.shape)}")
    bulk_temps = [params["bulk_temp"] for params in normalizer.unnormalize_params(batch.sim_params_dict)]
    print(f"CPU batch {tuple(batch.input.shape)}, physical bulk_temp={bulk_temps}", flush=True)


def run_preflight(cfg: DictConfig) -> None:
    with Path(cfg.checkpoint_path).open("rb") as checkpoint:
        if not checkpoint.read(1):
            raise ValueError(f"Empty checkpoint: {cfg.checkpoint_path}")
    print(f"Readable checkpoint: {cfg.checkpoint_path}", flush=True)

    model_class = get_model_class(cfg.model_cfg.name)
    if cfg.model_cfg.name == "nucleus2_moe" and not hasattr(model_class, "expected_fluid_params"):
        # This preflight reads the legacy sidecar format used by the diffusion job.
        metadata_keys = (
            ["inv_reynolds", "cpgas", "mugas", "rhogas", "thcogas", "stefan",
             "prandtl", "bulk_temp", "sat_temp"],
            ["wallTemp", "xMin", "xMax"],
            ["gravy"],
        )
    else:
        metadata_keys = (
            model_class.expected_fluid_params, model_class.expected_heater_params,
            model_class.expected_global_params,
        )
    normalizer = get_normalizer(OmegaConf.to_container(cfg.normalizer_cfg, resolve=True))
    check_preflight_batch(cfg, metadata_keys, normalizer)
    for paths, rollout in ((cfg.data_cfg.train_paths, False), (cfg.data_cfg.test_paths, True)):
        for filename in paths:
            check_preflight_file(filename, cfg, metadata_keys, normalizer, rollout)
            print(f"Checked dataset headers and metadata: {filename}", flush=True)
    print("Diffusion preflight passed.", flush=True)


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):
    run_config = OmegaConf.to_container(cfg, resolve=True)
    if cfg.get("diffusion_preflight", False):
        run_preflight(cfg)
        return
    run_config.update({
        "learning_rate": cfg.get("lr", 2.5e-4),
        "lr_warmup_steps": cfg.get("lr_warmup_steps", 150),
        "min_lr_ratio": cfg.get("min_lr_ratio", 0.001),
        "architecture": "UNet2DModel",
        "epochs": 1000,
        "num_train_timesteps": 100,
        "num_inference_steps": 50,
    })
    with PreemptionHandler() as preemption:
        with wandb.init(
            entity="hpcforge",
            project="diffusion",
            config=run_config,
        ) as run:
            run_diffusion(cfg, run, preemption)


def run_diffusion(
    cfg: DictConfig, run: wandb.sdk.wandb_run.Run, preemption: PreemptionHandler
) -> None:
    from nucleus.models import load_model_from_checkpoint
    from nucleus.utils.set_fp32_precision import set_fp32_precision
    from nucleus.utils.sdf_reinit import sdf_reinit_sussman
    from nucleus.test import TestResults
    from nucleus.plot.plotting import plot_rollout

    set_fp32_precision()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Stamp each run's outputs with dataset/date/job id so successive runs don't
    # overwrite each other, matching the convention in scripts/train.py.
    run_id_parts = ["diffusion", str(cfg.data_cfg.dataset).lower(), date.today().strftime("%Y-%m-%d")]
    if os.getenv("SLURM_JOB_ID") is not None:
        run_id_parts.append(os.getenv("SLURM_JOB_ID"))
    run_dir = Path(cfg.log_dir) / "_".join(run_id_parts)
    checkpoint_override = cfg.get("diffusion_resume_path") or os.getenv("DIFFUSION_RESUME_PATH")
    if checkpoint_override:
        resume_path = Path(checkpoint_override)
    elif os.getenv("SLURM_JOB_ID"):
        resume_path = Path(cfg.log_dir) / f"diffusion_{os.environ['SLURM_JOB_ID']}" / "last.pt"
    else:
        resume_path = run_dir / "last.pt"
    run_dir = resume_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing diffusion outputs to {run_dir}", flush=True)
    run.summary["run_dir"] = str(run_dir)

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

    #lr reduce over time
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
    global_step, start_epoch = restore_training_checkpoint(
        resume_path, unet, optimizer, lr_scheduler
    )
    checkpoint_interval = int(cfg.get("diffusion_checkpoint_every", 100))
    if checkpoint_interval < 1:
        raise ValueError("diffusion_checkpoint_every must be positive")
    epoch = start_epoch
    for epoch in range(start_epoch, 1000):
        if global_step >= cfg.max_steps:
            break
        for batch in train_dataloader:
            if preemption.requested:
                save_training_checkpoint(
                    resume_path, unet, optimizer, lr_scheduler, global_step, epoch
                )
                raise SystemExit(75)
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
            #NOISE MOD
            noise = 0.5*torch.randn_like(tgt_flat)
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

            model_input = torch.cat([inp_flat, pred_flat, noisy_tgt], dim=1)
            pred_noise = unet(model_input, unet_timesteps).sample
            loss = torch.nn.functional.mse_loss(pred_noise, noise - tgt_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            global_step += 1
            if preemption.requested or global_step % checkpoint_interval == 0:
                save_training_checkpoint(
                    resume_path, unet, optimizer, lr_scheduler, global_step, epoch
                )
            if preemption.requested:
                print(f"Preemption checkpoint saved to {resume_path}", flush=True)
                raise SystemExit(75)

            run.log(
                {
                    "train/loss": loss.item(),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/epoch": epoch,
                },
                step=global_step,
            )

            if global_step % 100 == 0:
                print(f'Step {global_step}: loss = {loss.item():.6f}, lr = {lr_scheduler.get_last_lr()[0]:.2e}')

            if global_step >= cfg.max_steps:
                break
        if global_step >= cfg.max_steps:
            break

    save_training_checkpoint(
        resume_path, unet, optimizer, lr_scheduler, global_step, epoch
    )
    if preemption.requested:
        raise SystemExit(75)
    #chec
    checkpoint_dir = run_dir / "diffusion_unet"
    unet.save_pretrained(checkpoint_dir)
    run.summary["checkpoint_dir"] = str(checkpoint_dir)
    run.summary["training_steps"] = global_step
    print(f"Saved diffusion U-Net to {checkpoint_dir}", flush=True)
    train_dataset.data.clear()
    del train_dataloader, train_dataset, optimizer, lr_scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("Released training data; starting rollout", flush=True)

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

    save_dir = run_dir / "10k_step_diffusion_rollout"
    save_dir.mkdir(parents=True, exist_ok=True)
    sdf_absolute_error_sum = 0.0
    sdf_value_count = 0

    for test_file_path in cfg.data_cfg.test_paths:
        print(f"Loading test trajectory: {test_file_path}", flush=True)
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

        fluid_params = test_dataset.sim_params[0]
        grid_height, grid_width = test_dataset.data[0][cfg.data_cfg.input_fields[0]].shape[-2:]
        dx = (fluid_params["x_max"] - fluid_params["x_min"]) / grid_width
        dy = (fluid_params["y_max"] - fluid_params["y_min"]) / grid_height
        xcoords = fluid_params["x_min"] + (torch.arange(grid_width) + 0.5) * dx
        heater_min = fluid_params["heater"]["xMin"]
        heater_max = fluid_params["heater"]["xMax"]
        case_name = f"{fluid_params['setup']}_{fluid_params['liquid']}_{fluid_params['heater']['wallTemp']}"
        case_dir = save_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        skip_itrs = test_dataset.future_time_window
        preds_list = []
        targets_list = []
        prev_clean_flat = None
        baseline_history = None
        comparison_records = {}
        rollout_window_paths = []
        metric_prefix = f"rollout/{case_name}/{Path(test_file_path).stem}"
        output_prefix = f"output/{case_name}_{Path(test_file_path).stem}"

        with torch.no_grad():
            for itr in range(0, len(test_dataset), skip_itrs):
                if preemption.requested:
                    raise SystemExit(75)
                data = test_dataset[itr]
                batch = data.to_collated_batch().to(device)

                bulk_temp = normalizer.unnormalize_params(
                    [batch.sim_params_dict[0]]
                )[0]["bulk_temp"]

                if baseline_history is not None:
                    batch.input = baseline_history
                baseline_prediction = predict_surrogate(model, batch, normalizer, layout)
                baseline_history = baseline_prediction.detach()

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

                sdf_field_index = list(cfg.data_cfg.output_fields).index("dfun")
                target_sdf = convert_layout(tgt, "t h w c", layout)[..., sdf_field_index]
                raw_sdf = convert_layout(pred, "t h w c", layout)[..., sdf_field_index].clone()
                baseline_fields = normalizer.unnormalize(
                    baseline_prediction, bulk_temp, layout=layout
                ).squeeze(0).float().cpu()
                baseline_sdf = convert_layout(
                    baseline_fields, "t h w c", layout
                )[..., sdf_field_index]

                for t_idx in range(pred.shape[0]):
                    if layout == "t h w c":
                        pred[t_idx, :, :, 0] = sdf_reinit_sussman(pred[t_idx, :, :, 0], dx=dx, dy=dy)
                    else:
                        pred[t_idx, 0, :, :] = sdf_reinit_sussman(pred[t_idx, 0, :, :], dx=dx, dy=dy)

                corrected_sdf = convert_layout(pred, "t h w c", layout)[..., sdf_field_index]
                baseline_reinitialized = torch.stack([
                    sdf_reinit_sussman(frame, dx=dx, dy=dy)
                    for frame in baseline_sdf
                ])
                records = log_sdf_comparison(
                    run, metric_prefix,
                    {"ground_truth": target_sdf,
                     "diffusion_raw": raw_sdf, "diffusion_reinitialized": corrected_sdf,
                     "surrogate": baseline_sdf,
                     "surrogate_reinitialized": baseline_reinitialized},
                    target_sdf, dx, dy, itr,
                )
                for key, values in records.items():
                    comparison_records.setdefault(key, []).extend(values)
                preds_list.append(pred)
                targets_list.append(tgt)
                torch.save(
                    {"preds": pred, "targets": tgt, "start_index": itr,
                     "diffusion_raw_sdf": raw_sdf, "surrogate_sdf": baseline_sdf,
                     "surrogate_reinitialized_sdf": baseline_reinitialized},
                    case_dir / f"window_{itr:06d}.pt",
                )
                rollout_window_paths.append(case_dir / f"window_{itr:06d}.pt")
                if itr // skip_itrs % 10 == 0:
                    print(f"Rollout {case_name}: {itr + skip_itrs}/{len(test_dataset)}", flush=True)

        if rollout_window_paths:
            gif_path = save_diffusion_gif(
                rollout_window_paths, layout, sdf_field_index,
                case_dir / "sdf_comparison.gif",
                max_frames=cfg.get("rollout_gif_max_frames", 120),
                fps=cfg.get("rollout_gif_fps", 10),
            )
            run.log({f"{output_prefix}_comparison_gif": wandb.Video(str(gif_path), format="gif")})
        for key, values in comparison_records.items():
            metric_values = torch.tensor(values)
            run.summary[f"{key}_mean"] = metric_values.nanmean().item()
            run.summary[f"{key}_final"] = values[-1]
        preds = torch.cat(preds_list, dim=0)[None, ...]
        targets = torch.cat(targets_list, dim=0)[None, ...]
        del preds_list, targets_list
        test_dataset.data.clear()
        gc.collect()
        print(f"Computing metrics for {case_name}", flush=True)

        preds = convert_layout(preds, target_layout="t h w c", source_layout=layout)
        targets = convert_layout(targets, target_layout="t h w c", source_layout=layout)

        sdf_field_index = list(cfg.data_cfg.output_fields).index("dfun")
        sdf_absolute_error = (
            preds[..., sdf_field_index] - targets[..., sdf_field_index]
        ).abs()
        sdf_mae_per_frame = sdf_absolute_error.mean(dim=(0, 2, 3))
        metric_prefix = f"rollout/{case_name}/{Path(test_file_path).stem}"
        case_sdf_mae = sdf_absolute_error.mean().item()
        run.summary[f"{metric_prefix}/sdf_mae_mean"] = case_sdf_mae
        run.summary[f"{metric_prefix}/sdf_mae_final"] = sdf_mae_per_frame[-1].item()
        run.summary[f"{metric_prefix}/evaluated_frames"] = len(sdf_mae_per_frame)
        sdf_absolute_error_sum += sdf_absolute_error.sum().item()
        sdf_value_count += sdf_absolute_error.numel()
        print(f"SDF MAE for {case_name}: {case_sdf_mae:.6g}", flush=True)
        del sdf_absolute_error, sdf_mae_per_frame

        fluid_params = test_dataset.sim_params[0]
        B, T_rollout, H, W, _ = preds.shape

        bulk_temp = fluid_params["bulk_temp"]
        heater_temp = fluid_params["heater"]["wallTemp"]
        pred_pm = physical_metrics(
            preds[..., 0], preds[..., 1], preds[..., 2], preds[..., 3],
            heater_min=heater_min, heater_max=heater_max,
            bulk_temp=bulk_temp, heater_temp=heater_temp,
            xcoords=xcoords,
            dx=dx, dy=dy,
        )
        pred_bm = bubble_metrics(preds[..., 0], preds[..., 2], preds[..., 3], dx=dx, dy=dy)
        tgt_pm = physical_metrics(
            targets[..., 0], targets[..., 1], targets[..., 2], targets[..., 3],
            heater_min=heater_min, heater_max=heater_max,
            bulk_temp=bulk_temp, heater_temp=heater_temp,
            xcoords=xcoords,
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
        for include_ground_truth, frame_dir, gif_key in (
            (True, case_dir, "gif"),
            (False, case_dir / "basic_rollout", "basic_gif"),
        ):
            frame_dir.mkdir(parents=True, exist_ok=True)
            plot_rollout(
                save_dir=str(frame_dir),
                rollout=preds,
                test_results=test_results,
                step_size=5,
                include_ground_truth=include_ground_truth,
                n_jobs=1,
            )
            rollout_gif_path = save_rollout_gif(
                [frame_dir / f"rollout_{frame_index:04d}.png"
                 for frame_index in range(0, preds.shape[1], 5)],
                frame_dir / "rollout.gif",
                max_frames=cfg.get("rollout_gif_max_frames", 120),
                fps=cfg.get("rollout_gif_fps", 10),
            )
            run.log({
                f"{output_prefix}_{gif_key}": wandb.Video(
                    str(rollout_gif_path), format="gif",
                ),
            })
        del test_results, preds, targets, test_dataset
        gc.collect()

    if sdf_value_count:
        run.summary["rollout/sdf_mae"] = sdf_absolute_error_sum / sdf_value_count

if __name__ == "__main__":
    main()
    
