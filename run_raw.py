"""Train and evaluate the original PyTorch PMRID on DatasetGen single RAW."""
import argparse
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models.net_torch import Network
from raw_dataset import RawDataset, datasetgen_api


PROJECT_ROOT = Path(__file__).resolve().parent
METRICS = ("psnr", "ssim", "mu_psnr", "mu_ssim")


class Tee:
    def __init__(self, first, second):
        self.first, self.second = first, second

    def write(self, value):
        self.first.write(value)
        self.second.write(value)

    def flush(self):
        self.first.flush()
        self.second.flush()


def git_value(*args):
    try:
        return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), *args],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not config.get("name") or not config.get("experiment_note"):
        raise ValueError("Config requires a unique name and Chinese experiment_note")
    return config


def platform_root(section):
    key = "windows_root" if platform.system() == "Windows" else "linux_root"
    value = section.get(key)
    if not value:
        raise ValueError("Missing platform path: " + key)
    return os.path.abspath(os.path.expanduser(value))


def setup_device(args):
    if args.gpu_ids:
        ids = args.gpu_ids.split(",")
        if not ids or any(not value.isdigit() for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("--gpu-ids must be unique comma-separated integer IDs")
        if "LOCAL_RANK" in os.environ and "CUDA_VISIBLE_DEVICES" in os.environ:
            if args.gpu_ids != os.environ["CUDA_VISIBLE_DEVICES"]:
                raise ValueError("--gpu-ids disagrees with CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if not distributed and args.gpu_ids and "," in args.gpu_ids:
        raise ValueError("Multiple GPU IDs require torchrun --nproc_per_node=N")
    if args.device == "cpu":
        if distributed:
            raise ValueError("Distributed CPU training is not supported")
        return torch.device("cpu"), False, 0, 1
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a smoke test")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        if torch.cuda.device_count() < world:
            raise ValueError("torchrun process count exceeds visible GPU count")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="gloo" if os.name == "nt" else "nccl")
        return torch.device("cuda", local_rank), True, rank, world
    if args.mode == "train" and torch.cuda.device_count() > 1 and not args.gpu_ids:
        print("[GPU] Single process uses cuda:0; use torchrun for multiple GPUs", flush=True)
    return torch.device("cuda", 0), False, 0, 1


def seed_everything(seed, rank):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def forward_padded(model, raw):
    h, w = raw.shape[-2:]
    ph, pw = (-h) % 16, (-w) % 16
    if ph or pw:
        mode = "reflect" if ph < h and pw < w else "replicate"
        raw = F.pad(raw, (0, pw, 0, ph), mode=mode)
    output = model(raw)
    return output[..., :h, :w]


def forward_tiled(model, raw, tile_size, overlap):
    if tile_size <= 0:
        return forward_padded(model, raw)
    if tile_size % 16 or tile_size <= overlap or overlap < 0:
        raise ValueError("tile_size must be divisible by 16 and exceed tile_overlap")
    _, _, h, w = raw.shape
    result = torch.zeros_like(raw)
    count = torch.zeros_like(raw[:, :1])
    stride = tile_size - overlap
    for top in range(0, h, stride):
        for left in range(0, w, stride):
            bottom, right = min(top + tile_size, h), min(left + tile_size, w)
            patch = raw[..., top:bottom, left:right]
            result[..., top:bottom, left:right] += forward_padded(model, patch)
            count[..., top:bottom, left:right] += 1
    return result / count


def checkpoint_steps(total, count):
    if count < 1 or total < count:
        raise ValueError("total_iter must be >= save_count >= 1")
    return {(total * i + count - 1) // count for i in range(1, count + 1)}


def make_dataset(config, split, datasetgen_root, patch_size=0):
    cfg = config["dataset"]
    return RawDataset(platform_root(cfg), cfg["manifest"], datasetgen_root,
                      split=split, subsets=cfg.get("subsets", "all"),
                      patch_size=patch_size)


def metric_protocol(config, api):
    cfg = config["metrics"]
    return api[4](hdr_scale=cfg["hdr_scale"], mu=cfg["mu"],
                  crop_border=cfg["crop_border"], demosaic=cfg["demosaic"],
                  metric_max=cfg["metric_max"])


def save_rgb(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), rgb[..., ::-1]):
        raise IOError("Failed to save " + str(path))


@torch.no_grad()
def evaluate(model, dataset, config, datasetgen_root, device, output_dir,
             checkpoint=None, args=None):
    api = datasetgen_api(datasetgen_root)
    protocol = metric_protocol(config, api)
    if protocol.demosaic == "net" and not os.path.isfile(
            os.path.join(datasetgen_root, "isp", "model.bin")):
        raise FileNotFoundError("DatasetGen learned demosaic weights missing")
    accumulator = api[7]()
    test_cfg = config["test"]
    interval = (args.save_rgb_interval if args and args.save_rgb_interval is not None
                else int(test_cfg.get("save_rgb_interval", 0)))
    save_max = (args.save_rgb_max if args and args.save_rgb_max is not None
                else int(test_cfg.get("save_rgb_max", 0)))
    save_enabled = (test_cfg.get("save_rgb", False) if not args or args.save_rgb is None
                    else args.save_rgb)
    if interval < 0 or save_max < 0 or (save_enabled and interval < 1):
        raise ValueError("RGB save interval must be positive; max must be nonnegative")
    tile_size = (args.tile_size if args and args.tile_size is not None
                 else int(test_cfg.get("tile_size", 0)))
    overlap = int(test_cfg.get("tile_overlap", 32))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seen, saved = defaultdict(int), defaultdict(int)
    model.eval()
    started = time.monotonic()
    for index in range(len(dataset)):
        data = dataset[index]
        sample = data["sample"]
        subset = sample["subset"]
        raw = data["input"].unsqueeze(0).to(device)
        gt = data["gt"].unsqueeze(0).to(device)
        metadata = {key: value.to(device) for key, value in data["metadata"].items()}
        output = forward_tiled(model, raw, tile_size, overlap)
        metrics = protocol.calculate(output, gt, metadata)[0]
        seen[subset] += 1
        rgb_path = None
        if save_enabled and (seen[subset] - 1) % interval == 0 and (save_max == 0 or saved[subset] < save_max):
            rgb = protocol.raw_to_linear_rgb(output, metadata)[0]
            rgb16 = api[6](api[5](rgb, protocol.mu))
            safe_id = str(sample["sample_id"]).replace("/", "_").replace("\\", "_")
            rgb_path = output_dir / "rgb" / subset / (safe_id + ".png")
            save_rgb(rgb_path, rgb16)
            saved[subset] += 1
        accumulator.add(metrics, sample_id=sample["sample_id"], subset=subset,
                        split=sample["split"], source_video=sample["source_video"],
                        alignratio=data["alignratio"], rgb_output=str(rgb_path) if rgb_path else "",
                        metric_max=metrics["metric_max"], mu_metric_max=metrics["mu_metric_max"])
        print_freq = int(test_cfg.get("print_freq", 50))
        if index == 0 or (index + 1) % print_freq == 0 or index + 1 == len(dataset):
            print("[test] %d/%d elapsed=%s" % (index + 1, len(dataset),
                  timedelta(seconds=int(time.monotonic() - started))), flush=True)
    report_config = {"experiment": config["name"], "experiment_note": config["experiment_note"],
                     "checkpoint": str(checkpoint) if checkpoint else None,
                     "dataset_root": dataset.root, "manifest_sha256": api[8](dataset.manifest_path),
                     "pmrid_git_commit": git_value("rev-parse", "HEAD"),
                     "pmrid_git_branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
                     "pmrid_git_status": git_value("status", "--short"),
                     "test_split": "test", "tile_size": tile_size, "tile_overlap": overlap,
                     "save_rgb": save_enabled, "save_rgb_interval": interval,
                     "save_rgb_max_per_subset": save_max, "saved_rgb_by_subset": dict(saved),
                     "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES")}
    report = api[9](str(output_dir), accumulator, report_config, protocol,
                    datasetgen_root, manifest_path=dataset.manifest_path)
    for subset, values in report["summary"].items():
        print("[metrics] %s count=%d PSNR=%.4f SSIM=%.4f mu-PSNR=%.4f mu-SSIM=%.4f" %
              (subset, values["count"], *(values[key] for key in METRICS)), flush=True)
    print("[report] " + str(output_dir), flush=True)
    return report


def train(args, config, device, distributed, rank, world, datasetgen_root):
    cfg = config["train"]
    seed_everything(int(cfg["seed"]), rank)
    patch_size = int(config["dataset"]["patch_size"])
    if patch_size % 16:
        raise ValueError("PMRID patch_size must be divisible by 16")
    train_data = make_dataset(config, "train", datasetgen_root, patch_size)
    test_data = make_dataset(config, "test", datasetgen_root) if rank == 0 else None
    batch = int(cfg["batch_size_per_gpu"])
    global_batch = batch * world
    total = int(cfg["total_iter"] or math.ceil(int(cfg["target_patch_exposures"]) / global_batch))
    save_steps = checkpoint_steps(total, int(cfg["save_count"]))
    sampler = DistributedSampler(train_data, num_replicas=world, rank=rank,
                                 shuffle=True, seed=int(cfg["seed"])) if distributed else None
    loader = DataLoader(train_data, batch_size=batch, shuffle=sampler is None,
                        sampler=sampler, num_workers=int(cfg["workers"]),
                        pin_memory=device.type == "cuda", drop_last=True)
    if not len(loader):
        raise ValueError("Training split smaller than per-GPU batch; reduce batch size")
    model = Network().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                                  weight_decay=float(cfg.get("weight_decay", 0)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total, eta_min=float(cfg["min_lr"]))
    train_model = DistributedDataParallel(model, device_ids=[device.index]) if distributed else model
    output = PROJECT_ROOT / "experiments" / config["name"]
    if rank == 0:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Experiment directory is not empty: " + str(output))
        output.mkdir(parents=True, exist_ok=True)
        log_handle = open(output / "train.log", "w", encoding="utf-8", buffering=1)
        sys.stdout = Tee(sys.stdout, log_handle)
        shutil.copyfile(args.config, output / "source_config.yaml")
        api = datasetgen_api(datasetgen_root)
        info = {"command": sys.argv, "name": config["name"], "experiment_note": config["experiment_note"],
                "config": config, "python": sys.version, "torch": torch.__version__,
                "cuda": torch.version.cuda, "gpu_names": [torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())] if device.type == "cuda" else [],
                "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES"), "world_size": world,
                "batch_size_per_gpu": batch, "global_batch": global_batch,
                "patch_size": patch_size, "total_iter": total, "save_steps": sorted(save_steps),
                "train_count": len(train_data), "test_count": len(test_data),
                "train_subsets": dict(Counter(x["subset"] for x in train_data.samples)),
                "test_subsets": dict(Counter(x["subset"] for x in test_data.samples)),
                "manifest_sha256": api[8](train_data.manifest_path),
                "datasetgen_root": datasetgen_root, "dataset_root": train_data.root,
                "demosaic_model": os.path.join(datasetgen_root, "isp", "model.bin"),
                "demosaic_model_sha256": api[8](os.path.join(datasetgen_root, "isp", "model.bin")),
                "pmrid_git_commit": git_value("rev-parse", "HEAD"),
                "pmrid_git_branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
                "pmrid_git_status": git_value("status", "--short")}
        with open(output / "reproducibility.json", "w", encoding="utf-8") as handle:
            json.dump(info, handle, ensure_ascii=False, indent=2)
        print("[experiment] " + json.dumps(info, ensure_ascii=False), flush=True)
    if distributed:
        dist.barrier()
    iterator, epoch = iter(loader), 0
    start = time.monotonic()
    tone = datasetgen_api(datasetgen_root)[5]
    scale, mu = float(config["metrics"]["hdr_scale"]), float(config["metrics"]["mu"])
    for step in range(1, total + 1):
        try:
            data = next(iterator)
        except StopIteration:
            epoch += 1
            if sampler:
                sampler.set_epoch(epoch)
            iterator = iter(loader)
            data = next(iterator)
        inp = data["input"].to(device, non_blocking=True)
        gt = data["gt"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prediction = train_model(inp)
        loss = F.l1_loss(tone(torch.clamp(prediction / scale, min=0), mu),
                         tone(torch.clamp(gt / scale, 0, 1), mu))
        loss.backward()
        optimizer.step()
        scheduler.step()
        if rank == 0 and (step == 1 or step % int(cfg["print_freq"]) == 0 or step == total):
            elapsed = time.monotonic() - start
            eta = (total - step) * elapsed / step
            print("[train] %d/%d loss=%.6f lr=%.3g samples/s=%.2f elapsed=%s ETA=%s" %
                  (step, total, loss.item(), optimizer.param_groups[0]["lr"],
                   step * global_batch / elapsed, timedelta(seconds=int(elapsed)),
                   timedelta(seconds=int(eta))), flush=True)
        del prediction, loss, inp, gt, data
        if step in save_steps:
            if rank == 0:
                optimizer.zero_grad(set_to_none=True)
                path = output / ("pmrid_raw_%07d.pth" % step)
                torch.save({"state_dict": model.state_dict(), "step": step,
                            "config": config, "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict()}, path)
                print("[checkpoint] " + str(path), flush=True)
                evaluate(model, test_data, config, datasetgen_root, device,
                         output / "test" / ("iter_%07d" % step), path, args)
                model.train()
            if distributed:
                dist.barrier()


def test(args, config, device, datasetgen_root):
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for test")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Network().to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if checkpoint.get("config", {}).get("name") != config["name"]:
        raise ValueError("Checkpoint experiment name disagrees with config")
    dataset = make_dataset(config, "test", datasetgen_root)
    output = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "results" / config["name"] / Path(args.checkpoint).stem)
    evaluate(model, dataset, config, datasetgen_root, device, output,
             args.checkpoint, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "test"))
    parser.add_argument("--config", default="options/raw_pmrid_v1.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu-ids", help="Physical CUDA IDs, e.g. 0 or 0,1; torchrun uses one process per ID")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--save-rgb-interval", type=int)
    parser.add_argument("--save-rgb-max", type=int)
    save = parser.add_mutually_exclusive_group()
    save.add_argument("--save-rgb", action="store_true", dest="save_rgb")
    save.add_argument("--no-save-rgb", action="store_false", dest="save_rgb")
    parser.set_defaults(save_rgb=None)
    args = parser.parse_args()
    config = load_config(args.config)
    datasetgen_root = platform_root(config["datasetgen"])
    device, distributed, rank, world = setup_device(args)
    try:
        if args.mode == "train":
            train(args, config, device, distributed, rank, world, datasetgen_root)
        elif distributed:
            raise ValueError("Test uses one process; choose one GPU with --gpu-ids")
        else:
            test(args, config, device, datasetgen_root)
    finally:
        if distributed:
            dist.destroy_process_group()
        if isinstance(sys.stdout, Tee):
            tee = sys.stdout
            sys.stdout = tee.first
            tee.second.close()


if __name__ == "__main__":
    main()
