"""Single-frame DatasetGen manifest adapter for PMRID."""
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def datasetgen_api(root):
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isfile(os.path.join(root, "degradation_metrics", "protocol.py")):
        raise FileNotFoundError("DatasetGen metrics are missing at " + root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from degradation_metrics.manifest import (load_manifest, load_sample,
                                              prepare_sample, select_rows)
    from degradation_metrics.protocol import RawMetricProtocol, mu_tonemap, quantize_uint16
    from degradation_metrics.reporting import (MetricAccumulator, sha256_file,
                                               write_metric_report)
    return (load_manifest, load_sample, prepare_sample, select_rows,
            RawMetricProtocol, mu_tonemap, quantize_uint16,
            MetricAccumulator, sha256_file, write_metric_report)


class RawDataset(Dataset):
    def __init__(self, root, manifest, datasetgen_root, split, subsets="all", patch_size=0):
        (load_manifest, self.load_sample, prepare_sample, select_rows,
         *_) = datasetgen_api(datasetgen_root)
        self.root = os.path.abspath(os.path.expanduser(root))
        self.manifest_path = (manifest if os.path.isabs(manifest)
                              else os.path.join(self.root, manifest))
        rows = select_rows(load_manifest(self.manifest_path), split=split, subsets=subsets)
        self.samples = [prepare_sample(self.root, row) for row in rows]
        if not self.samples:
            raise ValueError("No samples for split=%s, subsets=%s" % (split, subsets))
        ids = [sample["sample_id"] for sample in self.samples]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate sample_id in selected manifest")
        self.patch_size = int(patch_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        data = self.load_sample(sample, bit_depth=10, require_alignratio=True)
        lq, gt = data["input"], data["gt"]
        if self.patch_size:
            size = self.patch_size
            ph, pw = max(0, size - lq.shape[-2]), max(0, size - lq.shape[-1])
            if ph or pw:
                mode = "reflect" if ph < lq.shape[-2] and pw < lq.shape[-1] else "replicate"
                lq = F.pad(lq, (0, pw, 0, ph), mode=mode)
                gt = F.pad(gt, (0, pw, 0, ph), mode=mode)
            top = torch.randint(lq.shape[-2] - size + 1, (1,)).item()
            left = torch.randint(lq.shape[-1] - size + 1, (1,)).item()
            lq = lq[:, top:top + size, left:left + size]
            gt = gt[:, top:top + size, left:left + size]
        return {"input": lq.contiguous(), "gt": gt.contiguous(),
                "metadata": data["metadata"], "alignratio": data["alignratio"],
                "sample": sample}
