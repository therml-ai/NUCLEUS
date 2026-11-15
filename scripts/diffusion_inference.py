"""Run the diffusion-corrected rollout from an already-trained U-Net.

This is the inference half of scripts/diffusion.py: it skips training entirely and
loads the U-Net from a checkpoint, so any `diffusion_unet` directory produced by a
previous run can be evaluated without retraining.

Required overrides beyond the usual data/model config:

    diffusion_checkpoint_path=<path>   the diffusion U-Net to load
    checkpoint_path=<path>             the surrogate model, same as training

`diffusion_checkpoint_path` accepts either a save_pretrained directory (what
scripts/diffusion.py writes), a run directory containing one, or a raw state dict
saved with torch.save.
"""
import gc
import os
from datetime import date

import torch
import wandb
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
import hydra
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf

from nucleus.data import InMemForecastDataset
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
from nucleus.utils.sdf_metrics import sdf_frame_metrics
from nucleus.utils.sdf_smoothing import smooth_sdf_frames
from nucleus.utils.physical_metrics import physical_metrics, bubble_metrics
from nucleus.test import TestResults
from nucleus.plot.plotting import plot_rollout
from nucleus.plot.diffusion_gif import save_diffusion_gif, save_rollout_gif
from einops import rearrange
from pathlib import Path


GAUSSIAN_SIGMA_CELLS = (0.5, 1.0)


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


def load_diffusion_unet(path, in_channels: int, out_channels: int, cfg, device):
    """Load a trained U-Net from a save_pretrained directory or a raw state dict."""
    path = Path(path)
    if path.is_dir():
        # Allow pointing at either the unet directory or the run directory above it.
        if not (path / "config.json").is_file() and (path / "diffusion_unet" / "config.json").is_file():
            path = path / "diffusion_unet"
        if not (path / "config.json").is_file():
            raise FileNotFoundError(
                f"No config.json under {path}. Point diffusion_checkpoint_path at a "
                "directory written by UNet2DModel.save_pretrained."
            )
        unet = UNet2DModel.from_pretrained(path)
        print(f"Loaded diffusion U-Net from {path}", flush=True)
    else:
        if not path.is_file():
            raise FileNotFoundError(f"No diffusion checkpoint at {path}")
        # A bare state dict carries no architecture, so rebuild it from config. These
        # defaults match scripts/diffusion.py; override them if the checkpoint differs.
        unet = UNet2DModel(
            sample_size=None,
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=tuple(cfg.get("block_out_channels", (16, 32, 64, 128))),
            layers_per_block=int(cfg.get("layers_per_block", 2)),
            norm_num_groups=int(cfg.get("norm_num_groups", 8)),
        )
        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        missing, unexpected = unet.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"State dict does not match the configured architecture: "
                f"{len(missing)} missing, {len(unexpected)} unexpected keys. "
                "Set block_out_channels/norm_num_groups to the values used at training time."
            )
        print(f"Loaded diffusion U-Net state dict from {path}", flush=True)
    return unet.to(device)


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):
    if not cfg.get("diffusion_checkpoint_path"):
        raise ValueError(
            "Set diffusion_checkpoint_path to the diffusion_unet directory to evaluate."
        )
    run_config = OmegaConf.to_container(cfg, resolve=True)
    run_config.update({
        "architecture": "UNet2DModel",
        "num_train_timesteps": 100,
        "num_inference_steps": int(cfg.get("num_inference_steps", 50)),
        "postprocessing_gaussian_sigma_cells": list(GAUSSIAN_SIGMA_CELLS),
        "diffusion_checkpoint_path": str(cfg.diffusion_checkpoint_path),
    })
    with wandb.init(
        entity="hpcforge",
        project="diffusion",
        job_type="inference",
        config=run_config,
    ) as run:
        run_inference(cfg, run)


def run_inference(cfg: DictConfig, run: wandb.sdk.wandb_run.Run) -> None:
    set_fp32_precision()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)

    # Same run-directory convention as scripts/diffusion.py, tagged so inference
    # runs sort separately from training runs under the same log_dir.
    run_id_parts = ["diffusion_inference", str(cfg.data_cfg.dataset).lower(),
                    date.today().strftime("%Y-%m-%d")]
    if os.getenv("SLURM_JOB_ID") is not None:
        run_id_parts.append(os.getenv("SLURM_JOB_ID"))
    run_dir = Path(cfg.log_dir) / "_".join(run_id_parts)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing diffusion inference outputs to {run_dir}", flush=True)
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


    unet = load_diffusion_unet(
        cfg.diffusion_checkpoint_path,
        in_channels=3 * window_length * num_fields,
        out_channels=window_length * num_fields,
        cfg=cfg,
        device=device,
    )
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=100)

    def _flatten(x):
        if layout == "t h w c":
            return rearrange(x, "b t h w c -> b (t c) h w")
        return rearrange(x, "b t c h w -> b (t c) h w")

    def _unflatten(x):
        if layout == "t h w c":
            return rearrange(x, "b (t c) h w -> b t h w c", t=window_length)
        return rearrange(x, "b (t c) h w -> b t c h w", t=window_length)



    #inference
    unet.eval()
    num_inference_steps = int(cfg.get("num_inference_steps", 50))
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

    save_dir = run_dir / cfg.get("rollout_dir_name", "diffusion_rollout")
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

                noisy = 0.5 * torch.randn_like(pred_flat)
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
                # Build each reinitialized variant straight from raw_sdf. Keeping the
                # intermediate blur-only tensors alive alongside them doubled the
                # number of full-trajectory SDF tensors held per rollout window.
                smoothed_variants = {
                    f"diffusion_gaussian_{str(sigma).replace('.', '_')}":
                    smooth_sdf_frames(raw_sdf, sigma, dx, dy)
                    for sigma in GAUSSIAN_SIGMA_CELLS
                }
                records = log_sdf_comparison(
                    run, metric_prefix,
                    {"ground_truth": target_sdf,
                     "diffusion_raw": raw_sdf, "diffusion_reinitialized": corrected_sdf,
                     "surrogate": baseline_sdf,
                     "surrogate_reinitialized": baseline_reinitialized,
                     **smoothed_variants},
                    target_sdf, dx, dy, itr,
                )
                for key, values in records.items():
                    comparison_records.setdefault(key, []).extend(values)
                preds_list.append(pred)
                targets_list.append(tgt)
                torch.save(
                    {"preds": pred, "targets": tgt, "start_index": itr,
                     "diffusion_raw_sdf": raw_sdf, "surrogate_sdf": baseline_sdf,
                     "surrogate_reinitialized_sdf": baseline_reinitialized,
                     **smoothed_variants},
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
            for variant in smoothed_variants:
                gif_path = save_diffusion_gif(
                    rollout_window_paths, layout, sdf_field_index,
                    case_dir / f"{variant}.gif",
                    max_frames=cfg.get("rollout_gif_max_frames", 120),
                    fps=cfg.get("rollout_gif_fps", 10), variant=variant,
                )
                run.log({f"{output_prefix}_{variant}_gif": wandb.Video(str(gif_path), format="gif")})
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
