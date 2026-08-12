import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import PolarPairDataset, ensure_dir
from energy import FusionEnergy
from fusion_metrics import compute_fusion_metrics_dir
from polar_flow import PolarFlowFusion


def parse_args():
    parser = argparse.ArgumentParser(description="Train independent polar-flow fusion")
    parser.add_argument("--train-root", default="dataset/train")
    parser.add_argument("--val-root", default="dataset/val")
    parser.add_argument("--work-dir", default="polar_fusion_independent")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--architecture", default="cpif_lite", choices=["cpif_lite", "flow"])
    parser.add_argument("--quality-channels", type=int, default=16, help="Base channels for the CPIF-lite quality-first network")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--polar-gain", type=float, default=2.0, help="Strength of explicit DoLP high-frequency injection")
    parser.add_argument("--detail-gain", type=float, default=0.14, help="Full-resolution DoLP local-detail boost strength")
    parser.add_argument("--saliency-floor", type=float, default=0.25, help="Minimum DoLP saliency used for detail injection")
    parser.add_argument("--detail-hidden", type=int, default=16, help="Hidden channels in the learned DoLP detail injector")
    parser.add_argument("--detail-boost", type=float, default=1.35, help="Extra multiplier for learned DoLP detail injection")
    parser.add_argument("--dolp-highlight-threshold", type=float, default=0.0045, help="Minimum positive DoLP local detail selected for highlight texture injection")
    parser.add_argument("--dolp-fine-threshold", type=float, default=0.0012, help="Minimum fine-scale DoLP detail selected for texture injection")
    parser.add_argument("--dark-threshold", type=float, default=0.10, help="S0 local brightness below which DoLP detail and sharpening are suppressed")
    parser.add_argument("--direct-dolp-detail", type=float, default=0.42, help="Deterministic positive DoLP fine-detail injection strength")
    parser.add_argument("--s0-residual-detail", type=float, default=0.42, help="S0 fine-detail residual preserved outside selected DoLP texture regions")
    parser.add_argument("--output-sharpen", type=float, default=0.08, help="Final lightweight Laplacian sharpening strength")
    parser.add_argument("--highlight-lift", type=float, default=0.16, help="Extra masked brightness lift for high-confidence DoLP highlights")
    parser.add_argument("--residual-scale", type=float, default=0.6, help="Scale of CPIF-lite learned residual added to S0")
    parser.add_argument("--disable-polar-texture-analysis", action="store_false", dest="use_polar_texture_analysis", help="Replace adaptive polarization texture analysis with a uniform texture mask")
    parser.add_argument("--disable-mask-guided-fusion", action="store_false", dest="use_mask_guided_fusion", help="Disable texture-mask gating in multi-scale feature fusion")
    parser.add_argument("--use-correction-head", action="store_true", help="Enable a lightweight LUT-inspired output correction head")
    parser.add_argument("--correction-hidden", type=int, default=12, help="Hidden channels in the correction head")
    parser.add_argument("--correction-scale", type=float, default=0.05, help="Maximum correction-head residual scale")
    parser.add_argument("--s0-detail-suppression", type=float, default=0.75, help="Local S0 detail reduction inside selected DoLP highlight textures")
    parser.add_argument("--image-width", type=int, default=1125)
    parser.add_argument("--image-height", type=int, default=939)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--multi-gpu", action="store_true", default=True, help="Use torch.nn.DataParallel across all visible CUDA devices")
    parser.add_argument("--single-gpu", action="store_false", dest="multi_gpu", help="Disable DataParallel and use one visible CUDA device")
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--train-papif-teacher-root", default=None)
    parser.add_argument("--val-papif-teacher-root", default=None)
    parser.add_argument("--train-lfdt-teacher-root", default=None)
    parser.add_argument("--val-lfdt-teacher-root", default=None)
    parser.add_argument("--train-cpifuse-teacher-root", default=None)
    parser.add_argument("--val-cpifuse-teacher-root", default=None)
    parser.add_argument("--metric-every", type=int, default=1, help="Run fusion metrics every N epochs; set 0 to disable")
    parser.add_argument("--metric-data-root", default="includ", help="Dolp/S0 root used for periodic inference and metrics")
    parser.add_argument("--metric-backend", default="python", choices=["python", "matlab"], help="Metric implementation backend")
    parser.add_argument("--metric-root", default="评价指标", help="Directory containing evaluate_fusion_batch.m and metric functions")
    parser.add_argument("--metric-command", default="auto", help="MATLAB executable, or auto to search PATH")
    parser.add_argument("--metric-max-images", type=int, default=0, help="Limit metric-data images for inference and metrics; 0 means all")
    parser.add_argument("--metric-keep-images", action="store_true", help="Keep fused images generated by periodic metrics")
    parser.add_argument("--metric-discard-images", action="store_false", dest="metric_keep_images", help="Delete periodic inference images after metrics finish")
    parser.add_argument("--metric-timeout", type=int, default=0, help="MATLAB metric timeout in seconds; 0 means no timeout")
    parser.add_argument("--save-previews", action="store_true", help="Save validation preview images when a new best checkpoint is found")
    parser.add_argument("--log-every", type=int, default=1, help="Update/print training loss every N batches")
    parser.add_argument("--plain-progress", action="store_true", default=True, help="Print one flushed loss line every --log-every batches")
    parser.add_argument("--quiet-progress", action="store_false", dest="plain_progress", help="Disable flushed per-batch loss lines")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--best-save-every", type=int, default=5, help="Save the best validation checkpoint within each N-epoch interval; set 0 to disable")
    parser.add_argument("--base-weight", type=float, default=0.0, help="Deprecated; SSIM now handles S0 structure anchoring")
    parser.add_argument("--ssim-weight", type=float, default=0.0, help="Deprecated; SSIM loss has been removed")
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--detail-weight", type=float, default=0.30)
    parser.add_argument("--dolp-detail-weight", type=float, default=0.45)
    parser.add_argument("--highlight-weight", type=float, default=0.35)
    parser.add_argument("--sf-weight", type=float, default=0.30)
    parser.add_argument("--freq-weight", type=float, default=0.30, help="CPIF-style low/high frequency band loss weight")
    parser.add_argument("--dark-noise-weight", type=float, default=0.35)
    parser.add_argument("--exposure-weight", type=float, default=0.05)
    parser.add_argument("--papif-low-weight", type=float, default=0.0)
    parser.add_argument("--papif-low-ssim-weight", type=float, default=0.0)
    parser.add_argument("--lfdt-low-weight", type=float, default=0.0)
    parser.add_argument("--lfdt-ssim-weight", type=float, default=0.0)
    parser.add_argument("--cpif-ssim-weight", type=float, default=0.0)
    parser.add_argument("--smooth-weight", type=float, default=0.0, help="Deprecated; smoothing loss has been removed")
    parser.add_argument("--polar-weight", type=float, default=0.0, help="Deprecated; kept for old command compatibility")
    parser.add_argument("--cpif-grad-weight", type=float, default=0.0, help="Deprecated; kept for old command compatibility")
    parser.add_argument("--sharp-weight", type=float, default=0.0, help="Deprecated; kept for old command compatibility")
    parser.set_defaults(metric_keep_images=True)
    return parser.parse_args()


def select_device(choice):
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def load_model_state(model, state_dict, strict=True):
    target = unwrap_model(model)
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    missing, unexpected = target.load_state_dict(state_dict, strict=strict)
    if not strict:
        print(f"loaded checkpoint with strict=False | missing={len(missing)} unexpected={len(unexpected)}")


def maybe_wrap_multi_gpu(model, device, enabled):
    if not enabled:
        return model
    if device.type != "cuda":
        print("multi-gpu requested but device is not cuda; using single-device mode")
        return model
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        print(f"multi-gpu requested but only {gpu_count} CUDA device is visible; using single-GPU mode")
        return model
    print(f"using DataParallel on {gpu_count} visible CUDA devices")
    return torch.nn.DataParallel(model)


def tensor_to_image(tensor):
    array = tensor.detach().cpu().squeeze().numpy()
    array = (array * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


def save_preview(model, loader, device, output_dir, epoch):
    model.eval()
    ensure_dir(output_dir)
    with torch.no_grad():
        batch = next(iter(loader))
        dolp = batch["dolp"].to(device)
        s0 = batch["s0"].to(device)
        fused = model(dolp, s0)
        count = min(fused.size(0), 4)
        for index in range(count):
            path = Path(output_dir) / f"epoch_{epoch:03d}_{index:02d}.png"
            tensor_to_image(fused[index : index + 1]).save(path)


def save_metric_images(model, loader, device, output_dir, max_images=0):
    model.eval()
    output_dir = Path(output_dir)
    fused_dir = output_dir / "fused"
    ensure_dir(fused_dir)

    saved = 0
    with torch.no_grad():
        for batch in loader:
            dolp = batch["dolp"].to(device, non_blocking=True)
            s0 = batch["s0"].to(device, non_blocking=True)
            fused = model(dolp, s0)
            names = batch["name"]
            for index, name in enumerate(names):
                if max_images > 0 and saved >= max_images:
                    return fused_dir, saved
                filename = f"{name}.png"
                tensor_to_image(fused[index]).save(fused_dir / filename)
                saved += 1
    return fused_dir, saved


def resolve_metric_command(metric_command):
    if metric_command != "auto":
        return metric_command
    return shutil.which("matlab")


def write_metric_runner(path, metric_root, fused_dir, dolp_dir, s0_dir, output_dir, max_images):
    def matlab_string(value):
        return str(value).replace("'", "''")

    script = f"""try
addpath('{matlab_string(metric_root)}');
evaluate_fusion_batch('{matlab_string(fused_dir)}', '{matlab_string(dolp_dir)}', '{matlab_string(s0_dir)}', '{matlab_string(output_dir)}', {int(max_images)});
catch err
disp(getReport(err, 'extended'));
exit(1);
end
exit(0);
"""
    Path(path).write_text(script, encoding="utf-8")


def load_metric_summary(path):
    metrics = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                metrics[row["metric"]] = float(row["value"])
            except (KeyError, ValueError):
                continue
    return metrics


def run_fusion_metrics(model, loader, device, args, epoch, writer):
    epoch_number = epoch + 1
    metric_root = Path(args.metric_root).resolve()
    output_dir = Path(args.work_dir) / "metric_eval" / f"epoch_{epoch_number:03d}"
    image_dir = output_dir / "images"
    ensure_dir(output_dir)
    if image_dir.exists():
        shutil.rmtree(image_dir)

    fused_dir, saved = save_metric_images(
        model,
        loader,
        device,
        image_dir,
        max_images=args.metric_max_images,
    )
    if saved == 0:
        print(f"fusion metrics skipped at epoch {epoch_number}: no metric-data images were saved")
        return {}
    print(f"periodic inference epoch {epoch_number:03d}: saved {saved} fused images to {fused_dir}")
    dolp_dir = Path(args.metric_data_root) / "Dolp"
    s0_dir = Path(args.metric_data_root) / "S0"

    if args.metric_backend == "python":
        metrics, _ = compute_fusion_metrics_dir(fused_dir, dolp_dir, s0_dir, output_dir)
        for name, value in metrics.items():
            writer.add_scalar(f"fusion_metrics/{name}", value, epoch)
        if not args.metric_keep_images:
            shutil.rmtree(image_dir, ignore_errors=True)
        important = ["EN", "AG", "SF", "SD", "SSIM", "MI", "PSNR", "CC"]
        shown = ", ".join(f"{name}={metrics[name]:.4f}" for name in important if name in metrics and np.isfinite(metrics[name]))
        print(f"python fusion metrics epoch {epoch_number:03d} on {args.metric_data_root} ({saved} images): {shown}")
        return metrics

    metric_command = resolve_metric_command(args.metric_command)
    if not metric_command:
        print("fusion metrics skipped: MATLAB executable was not found in PATH")
        return {}

    runner_path = output_dir / "run_metrics.m"
    write_metric_runner(
        runner_path,
        metric_root,
        Path(fused_dir).resolve(),
        Path(dolp_dir).resolve(),
        Path(s0_dir).resolve(),
        output_dir.resolve(),
        args.metric_max_images,
    )

    command = [metric_command, "-nodisplay", "-nosplash", "-nodesktop", "-r", f"run('{runner_path.resolve()}')"]
    timeout = None if args.metric_timeout <= 0 else args.metric_timeout
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        print(f"fusion metrics timed out at epoch {epoch_number} after {args.metric_timeout}s")
        return {}

    (output_dir / "matlab_stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (output_dir / "matlab_stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        print(f"fusion metrics failed at epoch {epoch_number}; see {output_dir / 'matlab_stdout.log'}")
        return {}

    summary_path = output_dir / "summary.csv"
    if not summary_path.is_file():
        print(f"fusion metrics failed at epoch {epoch_number}: summary.csv was not generated")
        return {}

    metrics = load_metric_summary(summary_path)
    for name, value in metrics.items():
        writer.add_scalar(f"fusion_metrics/{name}", value, epoch)

    if not args.metric_keep_images:
        shutil.rmtree(image_dir, ignore_errors=True)

    important = ["EN", "VIF", "AG", "SSIM", "SF", "PSNR", "SCD", "Qabf", "MS_SSIM", "FSIM", "SD", "CC"]
    shown = ", ".join(f"{name}={metrics[name]:.4f}" for name in important if name in metrics and np.isfinite(metrics[name]))
    print(f"fusion metrics epoch {epoch_number:03d} on {args.metric_data_root} ({saved} images): {shown}")
    return metrics


def format_batch_terms(terms):
    order = (
        "total",
        "edge",
        "detail",
        "dolp_detail",
        "highlight",
        "sf",
        "freq",
        "dark_noise",
        "exposure",
        "papif_low",
        "papif_low_ssim",
        "lfdt_low",
        "lfdt_ssim",
        "cpif_ssim",
        "cpif_grad",
    )
    return ", ".join(f"{name}={terms[name].detach().item():.5f}" for name in order if name in terms)


def run_epoch(model, criterion, loader, optimizer, device, phase, epoch, writer, args):
    training = phase == "train"
    model.train(training)
    totals = {}
    progress = tqdm(
        loader,
        desc=f"{phase} {epoch + 1}/{args.epochs}",
        dynamic_ncols=True,
        file=sys.stdout,
        disable=args.no_tqdm,
        mininterval=0.1,
    )
    total_batches = max(len(loader), 1)

    for batch_index, batch in enumerate(progress, start=1):
        dolp = batch["dolp"].to(device, non_blocking=True)
        s0 = batch["s0"].to(device, non_blocking=True)
        teacher_papif = batch.get("teacher_papif")
        teacher_cpifuse = batch.get("teacher_cpifuse")
        teacher_lfdt = batch.get("teacher_lfdt")
        if teacher_papif is not None:
            teacher_papif = teacher_papif.to(device, non_blocking=True)
        if teacher_cpifuse is not None:
            teacher_cpifuse = teacher_cpifuse.to(device, non_blocking=True)
        if teacher_lfdt is not None:
            teacher_lfdt = teacher_lfdt.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            fused = model(dolp, s0)
            loss, terms = criterion(
                fused,
                dolp,
                s0,
                return_dict=True,
                teacher_papif=teacher_papif,
                teacher_cpifuse=teacher_cpifuse,
                teacher_lfdt=teacher_lfdt,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + value.detach().item()

        should_log = batch_index == 1 or batch_index == total_batches or batch_index % max(args.log_every, 1) == 0
        if should_log:
            progress.set_postfix({name: f"{value.detach().item():.4f}" for name, value in terms.items()})
            if args.plain_progress or args.no_tqdm:
                print(
                    f"[epoch {epoch + 1:03d}/{args.epochs:03d}] "
                    f"{phase} batch {batch_index:04d}/{total_batches:04d} | {format_batch_terms(terms)}",
                    flush=True,
                )

    count = max(len(loader), 1)
    averages = {name: value / count for name, value in totals.items()}
    for name, value in averages.items():
        writer.add_scalar(f"{phase}/{name}", value, epoch)
    return averages


def write_epoch_log(path, epoch, train_metrics, val_metrics, fusion_metrics, lr):
    ensure_dir(Path(path).parent)
    row = {"epoch": epoch + 1, "lr": lr}
    row.update({f"train_{key}": value for key, value in train_metrics.items()})
    row.update({f"val_{key}": value for key, value in val_metrics.items()})
    row.update({f"fusion_{key}": value for key, value in fusion_metrics.items()})
    fieldnames = list(row.keys())
    existing_rows = []
    if Path(path).is_file():
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_rows = list(reader)
            if reader.fieldnames:
                fieldnames = list(dict.fromkeys([*reader.fieldnames, *fieldnames]))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for existing_row in existing_rows:
            writer.writerow(existing_row)
        writer.writerow(row)


def format_metric_group(prefix, metrics):
    order = (
        "total",
        "edge",
        "detail",
        "dolp_detail",
        "highlight",
        "sf",
        "freq",
        "dark_noise",
        "exposure",
        "papif_low",
        "papif_low_ssim",
        "lfdt_low",
        "lfdt_ssim",
        "cpif_ssim",
        "cpif_grad",
    )
    parts = [f"{name}={metrics[name]:.5f}" for name in order if name in metrics]
    return f"{prefix}: " + ", ".join(parts)


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, args):
    ensure_dir(Path(path).parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "config": vars(args),
        },
        path,
    )


def main():
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    device = select_device(args.device)
    work_dir = Path(args.work_dir)
    checkpoint_dir = work_dir / "checkpoints"
    preview_dir = work_dir / "previews"
    writer = SummaryWriter(str(work_dir / "runs"))

    image_size = (args.image_width, args.image_height)
    train_teachers = {
        "papif": args.train_papif_teacher_root,
        "lfdt": args.train_lfdt_teacher_root,
        "cpifuse": args.train_cpifuse_teacher_root,
    }
    val_teachers = {
        "papif": args.val_papif_teacher_root,
        "lfdt": args.val_lfdt_teacher_root,
        "cpifuse": args.val_cpifuse_teacher_root,
    }
    train_data = PolarPairDataset(args.train_root, image_size=image_size, teacher_roots=train_teachers)
    val_data = PolarPairDataset(args.val_root, image_size=image_size, teacher_roots=val_teachers)
    metric_data = PolarPairDataset(args.metric_data_root, image_size=image_size) if args.metric_every > 0 else None
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    metric_loader = (
        DataLoader(
            metric_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if metric_data is not None
        else None
    )

    model = PolarFlowFusion(
        architecture=args.architecture,
        quality_channels=args.quality_channels,
        steps=args.steps,
        hidden=args.hidden,
        polar_gain=args.polar_gain,
        detail_gain=args.detail_gain,
        saliency_floor=args.saliency_floor,
        detail_hidden=args.detail_hidden,
        detail_boost=args.detail_boost,
        dolp_highlight_threshold=args.dolp_highlight_threshold,
        dolp_fine_threshold=args.dolp_fine_threshold,
        dark_threshold=args.dark_threshold,
        direct_dolp_detail=args.direct_dolp_detail,
        s0_residual_detail=args.s0_residual_detail,
        output_sharpen=args.output_sharpen,
        highlight_lift=args.highlight_lift,
        residual_scale=args.residual_scale,
        use_polar_texture_analysis=args.use_polar_texture_analysis,
        use_mask_guided_fusion=args.use_mask_guided_fusion,
        use_correction_head=args.use_correction_head,
        correction_hidden=args.correction_hidden,
        correction_scale=args.correction_scale,
    ).to(device)
    criterion = FusionEnergy(
        ssim_weight=args.ssim_weight,
        edge_weight=args.edge_weight,
        detail_weight=args.detail_weight,
        dolp_detail_weight=args.dolp_detail_weight,
        highlight_weight=args.highlight_weight,
        sf_weight=args.sf_weight,
        freq_weight=args.freq_weight,
        dark_noise_weight=args.dark_noise_weight,
        exposure_weight=args.exposure_weight,
        smooth_weight=args.smooth_weight,
        polar_weight=args.polar_weight,
        cpif_grad_weight=args.cpif_grad_weight,
        sharp_weight=args.sharp_weight,
        base_weight=args.base_weight,
        dolp_highlight_threshold=args.dolp_highlight_threshold,
        dolp_fine_threshold=args.dolp_fine_threshold,
        dark_threshold=args.dark_threshold,
        s0_detail_suppression=args.s0_detail_suppression,
        papif_low_weight=args.papif_low_weight,
        papif_low_ssim_weight=args.papif_low_ssim_weight,
        lfdt_low_weight=args.lfdt_low_weight,
        lfdt_ssim_weight=args.lfdt_ssim_weight,
        cpif_ssim_weight=args.cpif_ssim_weight,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    start_epoch = 0
    best_val = float("inf")
    interval_best_val = float("inf")
    interval_best_epoch = None
    interval_best_tmp = checkpoint_dir / "interval_best_tmp.pt"
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        load_model_state(model, checkpoint["model_state_dict"], strict=not args.use_correction_head)
        if not args.use_correction_head:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val = checkpoint.get("metrics", {}).get("val_total", best_val)

    model = maybe_wrap_multi_gpu(model, device, args.multi_gpu)

    metric_count = len(metric_data) if metric_data is not None else 0
    print(
        f"device={device}, train={len(train_data)}, val={len(val_data)}, "
        f"metric={metric_count} ({args.metric_data_root}), params={sum(p.numel() for p in unwrap_model(model).parameters())}"
    )

    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(model, criterion, train_loader, optimizer, device, "train", epoch, writer, args)
        val_metrics = run_epoch(model, criterion, val_loader, optimizer, device, "val", epoch, writer, args)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        metrics = {
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }

        if args.metric_every > 0 and (epoch + 1) % args.metric_every == 0:
            fusion_metrics = run_fusion_metrics(model, metric_loader, device, args, epoch, writer)
            if fusion_metrics:
                metrics.update({f"fusion_{key}": value for key, value in fusion_metrics.items()})
        else:
            fusion_metrics = {}

        save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, scheduler, epoch, metrics, args)
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, scheduler, epoch, metrics, args)
            if args.save_previews:
                save_preview(model, val_loader, device, preview_dir, epoch)

        if args.best_save_every > 0:
            if val_metrics["total"] < interval_best_val:
                interval_best_val = val_metrics["total"]
                interval_best_epoch = epoch
                save_checkpoint(interval_best_tmp, model, optimizer, scheduler, epoch, metrics, args)
            if (epoch + 1) % args.best_save_every == 0 and interval_best_epoch is not None:
                interval_start = epoch + 2 - args.best_save_every
                interval_end = epoch + 1
                interval_path = checkpoint_dir / (
                    f"best_interval_{interval_start:03d}_{interval_end:03d}"
                    f"_epoch_{interval_best_epoch + 1:03d}.pt"
                )
                shutil.copy2(interval_best_tmp, interval_path)
                print(
                    f"saved interval best checkpoint: {interval_path} "
                    f"(val_total={interval_best_val:.6f})",
                    flush=True,
                )
                interval_best_val = float("inf")
                interval_best_epoch = None

        write_epoch_log(work_dir / "loss_log.csv", epoch, train_metrics, val_metrics, fusion_metrics, current_lr)
        print(
            f"epoch {epoch + 1:03d}/{args.epochs:03d} | lr={current_lr:.6g} | "
            f"{format_metric_group('train', train_metrics)} | {format_metric_group('val', val_metrics)}",
            flush=True,
        )

    writer.close()


if __name__ == "__main__":
    main()
