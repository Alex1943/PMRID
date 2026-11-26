# PMRID single RAW comparison baseline

`run_raw.py` trains the original PyTorch `models/net_torch.py` network on one
DatasetGen packed RAW input per sample. The network remains 4-channel with its
original residual output. No PMRID phone ISO normalization or original BGGR
benchmark ISP is applied. Inputs are uint16 / 1023; GT is floating linear HDR.
`alignratio.npy` is validated only. Training uses Bracketing-style L1 on
`mu(clamp(pred/16, min=0))` and `mu(clamp(gt/16, 0, 1))`, with `mu=5000`.

The config uses OneRestore's DatasetGen 1.3 Linux dataset and ISP paths. Train
selects manifest `split=train`; every saved checkpoint is evaluated on the
manifest `split=test` using DatasetGen's learned-demosaic `RawMetricProtocol`.
This is the official test set, so periodic scores are monitoring results and
should not be used as an independent validation set for checkpoint selection.
The four metrics and per-subset/overall sample counts go to CSV/JSON. RGB images
are 16-bit mu-law PNGs sampled independently within each subset; interval 30
saves positions 1, 31, 61, etc. `save_rgb_max: 0` means no limit.

From this directory in the `seesr` environment:

```bash
conda activate seesr
CUDA_VISIBLE_DEVICES=0 python run_raw.py train --config options/raw_pmrid_v1.yaml
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 run_raw.py train --config options/raw_pmrid_v1.yaml
CUDA_VISIBLE_DEVICES=1 python run_raw.py test --config options/raw_pmrid_v1.yaml --checkpoint experiments/PMRID-v1.0-gt13/pmrid_raw_0400000.pth
```

On Windows PowerShell, use `python run_raw.py train --gpu-ids 0` in the active
environment. For two GPUs, set `$env:CUDA_VISIBLE_DEVICES='0,1'` then run
`torchrun --nproc_per_node=2 run_raw.py train`.
`--gpu-ids` can select visible physical IDs for a single process. `--tile-size`
can reduce test memory use; the default zero evaluates the full image. Tiling
changes the inference protocol, so compare results only with the same setting.
`--save-rgb-interval`, `--save-rgb-max`, and `--no-save-rgb` override test PNG
settings. All samples remain in the metric calculation.

The 3.2 million patch budget gives 400000 steps on one GPU or 200000 on two,
at eight samples per GPU. Eight checkpoints are saved at equal step intervals,
and each triggers the complete test. The cosine scheduler follows the resolved
step count. Training records ETA, throughput, source YAML, environment, manifest
and demosaic hashes, Git state, and per-subset counts under `experiments/<name>`.
Use a new experiment name for a fresh run; an existing nonempty directory fails
instead of overwriting results. The local Windows sample dataset has no test
split, so it supports loader checks but not the formal train/test run.
