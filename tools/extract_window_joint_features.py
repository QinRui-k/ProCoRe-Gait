import argparse
import csv
import json
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


PATIENT_SUBSETS = {"pdgait", "3dgait"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Memory-aware extraction of MotionBERT window-joint features for selected clips."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features-dir", required=True, help="Directory with clip_features.pt and metadata.csv.")
    parser.add_argument("--residual-dir", default=None, help="Directory with residual_features.pt.")
    parser.add_argument(
        "--selection",
        choices=["patients", "topk_normals", "indices_file", "subsets"],
        default="patients",
    )
    parser.add_argument("--indices-file", default=None, help="Text/CSV file containing global metadata indices.")
    parser.add_argument("--subsets", nargs="+", default=None, help="Used when --selection subsets.")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument("--window-stride", type=int, default=9)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-output-gb", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_for_model(batch, rootrel=True):
    if rootrel:
        return batch - batch[:, :, 0:1, :]
    batch = batch.clone()
    batch[:, :, :, 2] = batch[:, :, :, 2] - batch[:, 0:1, 0:1, 2]
    return batch


def window_joint_pool(feat, window_size, window_stride):
    chunks = []
    for start in range(0, feat.shape[1] - window_size + 1, window_stride):
        chunks.append(feat[:, start : start + window_size].mean(dim=1))
    if not chunks:
        raise RuntimeError(
            f"No windows produced for feature length {feat.shape[1]} with size={window_size}, stride={window_stride}."
        )
    return torch.stack(chunks, dim=1)


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


def read_metadata(features_dir):
    path = Path(features_dir) / "metadata.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_indices_file(path):
    path = Path(path)
    indices = []
    with path.open(newline="", encoding="utf-8") as f:
        sample = f.read(2048)
        f.seek(0)
        if "," in sample or "global_index" in sample:
            reader = csv.DictReader(f)
            for row in reader:
                value = row.get("global_index") or row.get("index") or next(iter(row.values()))
                indices.append(int(value))
        else:
            for line in f:
                line = line.strip()
                if line:
                    indices.append(int(line.split()[0]))
    return np.asarray(indices, dtype=np.int64)


def select_indices(rows, opts):
    if opts.selection == "patients":
        if opts.residual_dir:
            residual = torch.load(Path(opts.residual_dir) / "residual_features.pt", map_location="cpu")
            indices = residual["patient_indices"].cpu().numpy().astype(np.int64)
        else:
            indices = np.asarray(
                [i for i, row in enumerate(rows) if row.get("subset") in PATIENT_SUBSETS],
                dtype=np.int64,
            )
    elif opts.selection == "topk_normals":
        if not opts.residual_dir:
            raise ValueError("--residual-dir is required for --selection topk_normals.")
        residual = torch.load(Path(opts.residual_dir) / "residual_features.pt", map_location="cpu")
        indices = torch.unique(residual["topk_normal_indices"].reshape(-1)).cpu().numpy().astype(np.int64)
        indices.sort()
    elif opts.selection == "indices_file":
        if not opts.indices_file:
            raise ValueError("--indices-file is required for --selection indices_file.")
        indices = read_indices_file(opts.indices_file)
    elif opts.selection == "subsets":
        if not opts.subsets:
            raise ValueError("--subsets is required for --selection subsets.")
        subset_set = set(opts.subsets)
        indices = np.asarray([i for i, row in enumerate(rows) if row.get("subset") in subset_set], dtype=np.int64)
    else:
        raise ValueError(f"Unknown selection: {opts.selection}")

    indices = np.asarray(sorted(set(int(i) for i in indices)), dtype=np.int64)
    if opts.max_samples is not None:
        indices = indices[: opts.max_samples]
    if len(indices) == 0:
        raise RuntimeError("No indices selected.")
    if indices.min() < 0 or indices.max() >= len(rows):
        raise RuntimeError(f"Selected index out of metadata range: min={indices.min()}, max={indices.max()}, rows={len(rows)}")
    return indices


class MetadataPoseDataset(Dataset):
    def __init__(self, data_root, rows, global_indices):
        self.data_root = Path(data_root)
        self.rows = rows
        self.global_indices = np.asarray(global_indices, dtype=np.int64)

    def __len__(self):
        return len(self.global_indices)

    def __getitem__(self, index):
        global_index = int(self.global_indices[index])
        row = self.rows[global_index]
        path = self.data_root / row["path"]
        with path.open("rb") as f:
            record = pickle.load(f)
        pose = torch.as_tensor(record["pose"], dtype=torch.float32)
        return pose, global_index


def estimate_output_gb(n_samples, n_windows, n_joints, dim, dtype):
    bytes_per_value = 2 if dtype == "float16" else 4
    return n_samples * n_windows * n_joints * dim * bytes_per_value / (1024**3)


def write_metadata(path, rows, indices):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["global_index"] + list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index in indices:
            row = {"global_index": int(index), **rows[int(index)]}
            writer.writerow(row)


def main():
    opts = parse_args()
    output_file = Path(opts.output_file)
    if output_file.exists() and not opts.overwrite:
        raise FileExistsError(f"{output_file} exists. Use --overwrite to replace it.")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    args = get_config(opts.config)
    rows = read_metadata(opts.features_dir)
    indices = select_indices(rows, opts)
    n_windows = ((args.clip_len if hasattr(args, "clip_len") else 81) - opts.window_size) // opts.window_stride + 1
    n_joints = getattr(args, "num_joints", 17)
    dim = getattr(args, "dim_rep", getattr(args, "dim_feat", 512))
    estimated_gb = estimate_output_gb(len(indices), n_windows, n_joints, dim, opts.dtype)
    print(
        json.dumps(
            {
                "selection": opts.selection,
                "n_samples": int(len(indices)),
                "window_size": opts.window_size,
                "window_stride": opts.window_stride,
                "n_windows_estimate": int(n_windows),
                "n_joints": int(n_joints),
                "dim_estimate": int(dim),
                "dtype": opts.dtype,
                "estimated_output_gb": round(estimated_gb, 3),
                "max_output_gb": opts.max_output_gb,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if estimated_gb > opts.max_output_gb:
        raise RuntimeError(
            f"Estimated output {estimated_gb:.2f} GiB exceeds --max-output-gb={opts.max_output_gb:.2f}. "
            "Narrow the selection or raise the limit intentionally."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, opts.checkpoint, device)
    dataset = MetadataPoseDataset(args.data_root, rows, indices)
    loader = DataLoader(
        dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    out_dtype = torch.float16 if opts.dtype == "float16" else torch.float32
    chunks = []
    seen_indices = []

    with torch.no_grad():
        for batch, batch_indices in tqdm(loader, desc="Extracting window-joint features"):
            batch = normalize_for_model(batch.to(device), rootrel=args.rootrel)
            feat = model(batch, return_rep=True)
            wj_feat = window_joint_pool(feat, opts.window_size, opts.window_stride)
            chunks.append(wj_feat.detach().cpu().to(out_dtype))
            seen_indices.extend(int(i) for i in batch_indices.tolist())

    window_joint_features = torch.cat(chunks, dim=0)
    torch.save(
        {
            "window_joint_features": window_joint_features,
            "global_indices": torch.as_tensor(seen_indices, dtype=torch.long),
            "paths": [rows[int(i)]["path"] for i in seen_indices],
            "checkpoint": opts.checkpoint,
            "features_dir": opts.features_dir,
            "residual_dir": opts.residual_dir,
            "selection": opts.selection,
            "window_size": opts.window_size,
            "window_stride": opts.window_stride,
            "dtype": opts.dtype,
        },
        output_file,
    )
    metadata_output = opts.metadata_output
    if metadata_output is None:
        metadata_output = str(output_file.with_suffix("")) + "_metadata.csv"
    write_metadata(metadata_output, rows, seen_indices)
    print(f"Wrote {output_file}")
    print(f"Wrote {metadata_output}")
    print(f"Actual tensor shape: {tuple(window_joint_features.shape)}")


if __name__ == "__main__":
    main()
