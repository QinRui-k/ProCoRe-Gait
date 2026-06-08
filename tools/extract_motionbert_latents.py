import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.utils.learning import load_backbone
from lib.utils.tools import get_config


class PklPoseDataset(Dataset):
    def __init__(self, data_root, subsets, label_list):
        self.data_root = Path(data_root)
        self.subsets = subsets
        self.label_set = set(str(label) for label in label_list)
        self.paths = []
        for subset in subsets:
            subset_root = self.data_root / subset
            for root, _, files in os.walk(subset_root):
                label_name = Path(root).name
                if label_name not in self.label_set:
                    continue
                for name in files:
                    if name.endswith(".pkl") and not name.startswith("._"):
                        self.paths.append(Path(root) / name)
        self.paths = sorted(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with path.open("rb") as f:
            record = pickle.load(f)
        pose = torch.as_tensor(record["pose"], dtype=torch.float32)
        metadata = metadata_from_record(record, path, self.data_root)
        return pose, str(path), metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Extract MotionBERT latent features and metadata.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subsets", nargs="+", default=["pdgait", "3dgait", "AMASS_GAIT_CMU", "H36M-SH"])
    parser.add_argument("--label-list", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output-dir", default="features/e1_patient_dynamic_anchor")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-window-joint", action="store_true")
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument("--window-stride", type=int, default=9)
    return parser.parse_args()


def normalize_for_model(batch, rootrel=True):
    if rootrel:
        return batch - batch[:, :, 0:1, :]
    batch = batch.clone()
    batch[:, :, :, 2] = batch[:, :, :, 2] - batch[:, 0:1, 0:1, 2]
    return batch


def load_model(args, checkpoint_path, device):
    model = load_backbone(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_pos", checkpoint)
    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def metadata_from_record(record, path, data_root):
    rel = str(Path(path).relative_to(data_root))
    parts = Path(rel).parts
    subset = parts[0] if parts else ""
    subject = ""
    label_dir = ""
    if subset == "pdgait" and len(parts) >= 3:
        subject = parts[1]
        label_dir = parts[2]
    elif len(parts) >= 2:
        label_dir = parts[1]
    return {
        "path": rel,
        "subset": subset,
        "subject": subject,
        "label_dir": label_dir,
        "id": str(record.get("id", "")),
        "video_name": str(record.get("video_name", "")),
        "label": str(record.get("label", "")),
        "score_label": str(record.get("score_label", "")),
        "state": str(record.get("state", "")),
        "state_label": str(record.get("state_label", "")),
        "diag": str(record.get("diag", "")),
        "subtype_label": str(record.get("subtype_label", "")),
        "source": str(record.get("source", "")),
        "dataset": str(record.get("dataset", subset)),
    }


def window_joint_pool(feat, window_size, window_stride):
    # feat: [B, F, J, C]
    chunks = []
    for start in range(0, feat.shape[1] - window_size + 1, window_stride):
        chunks.append(feat[:, start : start + window_size].mean(dim=1))
    return torch.stack(chunks, dim=1)


def main():
    opts = parse_args()
    args = get_config(opts.config)
    label_list = opts.label_list if opts.label_list is not None else args.label_list
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = PklPoseDataset(args.data_root, opts.subsets, label_list)
    if opts.max_samples is not None:
        dataset.paths = dataset.paths[: opts.max_samples]
    print(f"Found {len(dataset)} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, opts.checkpoint, device)
    loader = DataLoader(
        dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    clip_features = []
    window_features = []
    paths = []
    rows = []

    with torch.no_grad():
        for batch, batch_paths, batch_meta in tqdm(loader, desc="Extracting latents"):
            batch = normalize_for_model(batch.to(device), rootrel=args.rootrel)
            feat = model(batch, return_rep=True)
            clip_feat = feat.mean(dim=(1, 2)).detach().cpu().to(torch.float32)
            clip_features.append(clip_feat)
            if opts.save_window_joint:
                wj_feat = window_joint_pool(feat, opts.window_size, opts.window_stride)
                window_features.append(wj_feat.detach().cpu().to(torch.float32))
            paths.extend(batch_paths)
            batch_size = len(batch_paths)
            for idx in range(batch_size):
                rows.append({key: values[idx] for key, values in batch_meta.items()})

    clip_features = torch.cat(clip_features, dim=0)
    torch.save(
        {
            "clip_features": clip_features,
            "paths": paths,
            "checkpoint": opts.checkpoint,
            "subsets": opts.subsets,
        },
        output_dir / "clip_features.pt",
    )

    if opts.save_window_joint:
        torch.save(
            {
                "window_joint_features": torch.cat(window_features, dim=0),
                "paths": paths,
                "checkpoint": opts.checkpoint,
                "subsets": opts.subsets,
                "window_size": opts.window_size,
                "window_stride": opts.window_stride,
            },
            output_dir / "window_joint_features.pt",
        )

    metadata_path = output_dir / "metadata.csv"
    if not rows:
        raise RuntimeError("No samples were extracted. Please check --subsets, --label-list, and data_root.")
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_dir / 'clip_features.pt'}")
    print(f"Wrote {metadata_path}")
    if opts.save_window_joint:
        print(f"Wrote {output_dir / 'window_joint_features.pt'}")


if __name__ == "__main__":
    main()
