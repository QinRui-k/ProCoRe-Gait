import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build memory-aware window/joint/part residual aggregates from selected window-joint features."
    )
    parser.add_argument("--features-dir", required=True, help="Directory with clip_features.pt and metadata.csv.")
    parser.add_argument("--residual-dir", required=True, help="Directory with residual_features.pt.")
    parser.add_argument("--patient-window-file", required=True)
    parser.add_argument("--normal-window-file", required=True)
    parser.add_argument("--body-parts", default="configs/body_parts_h36m17_gait.json")
    parser.add_argument("--output-dir", default="eval/window_joint_residual_e1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-residual-tensors", action="store_true")
    parser.add_argument("--max-residual-gb", type=float, default=12.0)
    return parser.parse_args()


def read_metadata(features_dir):
    with (Path(features_dir) / "metadata.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_body_parts(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    joint_names = data["joint_names"]
    parts = {name: [int(i) for i in indices] for name, indices in data["parts"].items()}
    for name, indices in parts.items():
        bad = [i for i in indices if i < 0 or i >= len(joint_names)]
        if bad:
            raise RuntimeError(f"Part {name} contains invalid joint indices: {bad}")
    return joint_names, parts


def l2_normalize_torch(x, eps=1e-12):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def load_window_file(path, key="window_joint_features"):
    data = torch.load(path, map_location="cpu")
    features = data[key]
    indices = data["global_indices"].long()
    return data, features, indices


def make_index_map(indices, name):
    out = {}
    for pos, index in enumerate(indices.tolist()):
        if int(index) in out:
            raise RuntimeError(f"Duplicate global index {index} in {name}.")
        out[int(index)] = pos
    return out


def estimate_tensor_gb(shape, dtype=torch.float16):
    bytes_per_value = torch.empty((), dtype=dtype).element_size()
    count = math.prod(int(v) for v in shape)
    return count * bytes_per_value / (1024**3)


def safe_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except ValueError:
        return default


def write_sample_csv(path, rows, patient_indices, r_total, r_joint, r_window, r_part, joint_names, part_names):
    fieldnames = [
        "local_index",
        "global_index",
        "path",
        "subset",
        "subject",
        "id",
        "video_name",
        "label",
        "score_label",
        "state_label",
        "diag",
        "subtype_label",
        "r_total",
    ]
    fieldnames += [f"joint_{i:02d}_{name}" for i, name in enumerate(joint_names)]
    fieldnames += [f"window_{i:02d}" for i in range(r_window.shape[1])]
    fieldnames += [f"part_{name}" for name in part_names]

    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for local_i, global_i in enumerate(patient_indices.tolist()):
            meta = rows[int(global_i)]
            row = {
                "local_index": local_i,
                "global_index": int(global_i),
                "path": meta.get("path", ""),
                "subset": meta.get("subset", ""),
                "subject": meta.get("subject", ""),
                "id": meta.get("id", ""),
                "video_name": meta.get("video_name", ""),
                "label": meta.get("label", ""),
                "score_label": meta.get("score_label", ""),
                "state_label": meta.get("state_label", ""),
                "diag": meta.get("diag", ""),
                "subtype_label": meta.get("subtype_label", ""),
                "r_total": float(r_total[local_i]),
            }
            for j, name in enumerate(joint_names):
                row[f"joint_{j:02d}_{name}"] = float(r_joint[local_i, j])
            for w in range(r_window.shape[1]):
                row[f"window_{w:02d}"] = float(r_window[local_i, w])
            for p, name in enumerate(part_names):
                row[f"part_{name}"] = float(r_part[local_i, p])
            writer.writerow(row)


def write_group_stats(path, sample_rows, joint_names, part_names):
    numeric_prefixes = ["r_total"] + [f"joint_{i:02d}_{name}" for i, name in enumerate(joint_names)]
    numeric_prefixes += [key for key in sample_rows[0] if key.startswith("window_")]
    numeric_prefixes += [f"part_{name}" for name in part_names]

    groups = defaultdict(list)
    for row in sample_rows:
        subset = row["subset"]
        score = safe_int(row.get("score_label", row.get("label")), default=-1)
        binary = 0 if score == 0 else 1
        groups[(subset, str(score), str(binary))].append(row)

    fieldnames = ["subset", "score_label", "binary_label", "n"]
    for key in numeric_prefixes:
        fieldnames.extend([f"{key}_mean", f"{key}_median"])

    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (subset, score, binary), rows in sorted(groups.items()):
            out = {"subset": subset, "score_label": score, "binary_label": binary, "n": len(rows)}
            for key in numeric_prefixes:
                vals = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
                out[f"{key}_mean"] = float(vals.mean())
                out[f"{key}_median"] = float(np.median(vals))
            writer.writerow(out)


def read_csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    opts = parse_args()
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_metadata(opts.features_dir)
    joint_names, parts = load_body_parts(opts.body_parts)
    part_names = list(parts.keys())

    clip_data = torch.load(Path(opts.features_dir) / "clip_features.pt", map_location="cpu")
    clip_features = l2_normalize_torch(clip_data["clip_features"].float())
    residual_data = torch.load(Path(opts.residual_dir) / "residual_features.pt", map_location="cpu")
    patient_indices = residual_data["patient_indices"].long()
    topk_normal_indices = residual_data["topk_normal_indices"].long()
    temperature = float(residual_data.get("temperature", 0.05))

    _, patient_window, patient_window_indices = load_window_file(opts.patient_window_file)
    _, normal_window, normal_window_indices = load_window_file(opts.normal_window_file)
    patient_pos = make_index_map(patient_window_indices, "patient-window-file")
    normal_pos = make_index_map(normal_window_indices, "normal-window-file")

    missing_patients = [int(i) for i in patient_indices.tolist() if int(i) not in patient_pos]
    if missing_patients:
        raise RuntimeError(f"Missing {len(missing_patients)} patient window features. First missing: {missing_patients[:5]}")
    missing_normals = [
        int(i) for i in torch.unique(topk_normal_indices.reshape(-1)).tolist() if int(i) not in normal_pos
    ]
    if missing_normals:
        raise RuntimeError(f"Missing {len(missing_normals)} normal window features. First missing: {missing_normals[:5]}")

    patient_order = torch.as_tensor([patient_pos[int(i)] for i in patient_indices.tolist()], dtype=torch.long)
    patient_window = patient_window[patient_order]
    normal_lookup = torch.as_tensor([normal_pos[int(i)] for i in normal_window_indices.tolist()], dtype=torch.long)
    del normal_lookup

    n, w, j, c = patient_window.shape
    residual_gb = estimate_tensor_gb((n, w, j, c), torch.float16)
    print(
        json.dumps(
            {
                "n_patients": int(n),
                "n_topk_unique_normals": int(len(normal_window_indices)),
                "patient_window_shape": list(patient_window.shape),
                "normal_window_shape": list(normal_window.shape),
                "temperature": temperature,
                "save_residual_tensors": opts.save_residual_tensors,
                "estimated_residual_tensor_gb_float16": round(residual_gb, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if opts.save_residual_tensors and residual_gb > opts.max_residual_gb:
        raise RuntimeError(
            f"Residual tensor would be {residual_gb:.2f} GiB, exceeding --max-residual-gb={opts.max_residual_gb:.2f}."
        )

    normal_global_to_pos = make_index_map(normal_window_indices, "normal-window-file")
    r_total_chunks = []
    r_joint_chunks = []
    r_window_chunks = []
    r_part_chunks = []
    weights_chunks = []
    residual_chunks = []

    for start in tqdm(range(0, n, opts.batch_size), desc="Building window-joint residuals"):
        end = min(start + opts.batch_size, n)
        batch_patient_indices = patient_indices[start:end]
        batch_topk = topk_normal_indices[start:end]
        batch_raw_clip = clip_features[batch_patient_indices]
        batch_normal_clip = clip_features[batch_topk.reshape(-1)].reshape(batch_topk.shape[0], batch_topk.shape[1], -1)
        sim = (batch_raw_clip.unsqueeze(1) * batch_normal_clip).sum(dim=-1)
        weights = torch.softmax(sim / temperature, dim=1).to(torch.float32)
        weights_chunks.append(weights.to(torch.float16))

        normal_positions = torch.as_tensor(
            [[normal_global_to_pos[int(i)] for i in row.tolist()] for row in batch_topk],
            dtype=torch.long,
        )
        neighbor_window = normal_window[normal_positions].to(torch.float32)
        reference = (weights[:, :, None, None, None] * neighbor_window).sum(dim=1)
        raw = patient_window[start:end].to(torch.float32)
        residual = raw - reference
        abs_residual = residual.abs()

        r_total_chunks.append(abs_residual.mean(dim=(1, 2, 3)).cpu())
        r_joint_chunks.append(abs_residual.mean(dim=(1, 3)).cpu())
        r_window_chunks.append(abs_residual.mean(dim=(2, 3)).cpu())
        part_values = []
        for indices in parts.values():
            part_values.append(abs_residual[:, :, indices, :].mean(dim=(1, 2, 3)))
        r_part_chunks.append(torch.stack(part_values, dim=1).cpu())
        if opts.save_residual_tensors:
            residual_chunks.append(residual.detach().cpu().to(torch.float16))

    r_total = torch.cat(r_total_chunks, dim=0).float()
    r_joint = torch.cat(r_joint_chunks, dim=0).float()
    r_window = torch.cat(r_window_chunks, dim=0).float()
    r_part = torch.cat(r_part_chunks, dim=0).float()
    topk_weights = torch.cat(weights_chunks, dim=0)

    output = {
        "patient_indices": patient_indices,
        "topk_normal_indices": topk_normal_indices,
        "topk_weights": topk_weights,
        "r_total": r_total,
        "r_joint": r_joint,
        "r_window": r_window,
        "r_part": r_part,
        "joint_names": joint_names,
        "part_names": part_names,
        "body_parts": parts,
        "features_dir": opts.features_dir,
        "residual_dir": opts.residual_dir,
        "patient_window_file": opts.patient_window_file,
        "normal_window_file": opts.normal_window_file,
    }
    if opts.save_residual_tensors:
        output["r_signed"] = torch.cat(residual_chunks, dim=0)

    aggregate_path = output_dir / "window_joint_residual_aggregates.pt"
    torch.save(output, aggregate_path)

    sample_csv = output_dir / "window_joint_residual_samples.csv"
    write_sample_csv(
        sample_csv,
        rows,
        patient_indices,
        r_total.numpy(),
        r_joint.numpy(),
        r_window.numpy(),
        r_part.numpy(),
        joint_names,
        part_names,
    )
    sample_rows = read_csv_rows(sample_csv)
    stats_csv = output_dir / "window_joint_residual_group_stats.csv"
    write_group_stats(stats_csv, sample_rows, joint_names, part_names)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "n_patients": int(n),
                "n_topk_unique_normals": int(len(normal_window_indices)),
                "window_shape": [int(w), int(j), int(c)],
                "part_names": part_names,
                "temperature": temperature,
                "saved_residual_tensors": bool(opts.save_residual_tensors),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {aggregate_path}")
    print(f"Wrote {sample_csv}")
    print(f"Wrote {stats_csv}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
