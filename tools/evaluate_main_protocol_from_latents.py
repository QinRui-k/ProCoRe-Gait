import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


PATIENT_SUBSETS = {"pdgait", "3dgait"}
DEFAULT_TASKS = [
    "pdgait_binary",
    "pdgait_score_3class",
    "3dgait_binary",
    "3dgait_score_4class",
    "3dgait_subtype_3class",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Formal frozen-latent protocol for raw vs normal-reference residual features."
    )
    parser.add_argument("--features-dir", required=True, help="Directory containing clip_features.pt and metadata.csv")
    parser.add_argument("--output-dir", default="eval/main_protocol_e1")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--normal-subsets", nargs="+", default=["AMASS_GAIT_CMU", "H36M-SH"])
    parser.add_argument("--normal-limit-per-subset", type=int, default=60000)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--query-batch-size", type=int, default=512)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def load_inputs(features_dir):
    features_dir = Path(features_dir)
    feature_data = torch.load(features_dir / "clip_features.pt", map_location="cpu")
    features = feature_data["clip_features"].float()
    with (features_dir / "metadata.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(features) != len(rows):
        raise RuntimeError(f"Feature/metadata mismatch: {len(features)} vs {len(rows)}")
    return features, rows


def safe_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except ValueError:
        return default


def score_label(row):
    return safe_int(row.get("score_label", row.get("label")))


def subtype_label(row):
    subtype = safe_int(row.get("subtype_label"))
    if subtype is not None:
        return subtype
    diag = safe_int(row.get("diag"))
    if diag is None:
        return None
    return {0: 0, 1: 1, 2: 2, 3: 1, 4: 2}.get(diag)


def split_group(row):
    subset = row.get("subset", "")
    if subset == "pdgait":
        return f"pdgait_subject:{row.get('subject') or row.get('id') or row.get('video_name') or row.get('path')}"
    if subset == "3dgait":
        return f"3dgait_patient:{row.get('id') or row.get('path')}"
    return f"{subset}:{row.get('source') or row.get('dataset') or row.get('path')}"


def eval_group(row):
    subset = row.get("subset", "")
    if subset == "pdgait":
        return f"pdgait_video:{row.get('video_name') or row.get('subject') or row.get('id') or row.get('path')}"
    if subset == "3dgait":
        return f"3dgait_patient:{row.get('id') or row.get('path')}"
    return split_group(row)


def task_label(row, task):
    subset = row.get("subset", "")
    score = score_label(row)
    if task == "pdgait_binary":
        return None if subset != "pdgait" or score is None else int(score > 0)
    if task == "pdgait_score_3class":
        return score if subset == "pdgait" and score in {0, 1, 2} else None
    if task == "3dgait_binary":
        return None if subset != "3dgait" or score is None else int(score > 0)
    if task == "3dgait_score_4class":
        return score if subset == "3dgait" and score in {0, 1, 2, 3} else None
    if task == "3dgait_subtype_3class":
        subtype = subtype_label(row)
        return subtype if subset == "3dgait" and subtype in {0, 1, 2} else None
    if task == "combined_binary":
        return None if subset not in PATIENT_SUBSETS or score is None else int(score > 0)
    raise ValueError(f"Unknown task: {task}")


def is_binary_task(task):
    return task.endswith("_binary")


def get_device(name):
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def l2_normalize_torch(x, eps=1e-12):
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


def select_indices(rows, normal_subsets, normal_limit_per_subset, seed):
    rng = np.random.default_rng(seed)
    patient_idx = np.asarray(
        [i for i, row in enumerate(rows) if row.get("subset") in PATIENT_SUBSETS],
        dtype=np.int64,
    )
    normal_parts = []
    normal_counts = {}
    for subset in normal_subsets:
        idxs = np.asarray([i for i, row in enumerate(rows) if row.get("subset") == subset], dtype=np.int64)
        normal_counts[subset] = int(len(idxs))
        if normal_limit_per_subset > 0 and len(idxs) > normal_limit_per_subset:
            idxs = rng.choice(idxs, size=normal_limit_per_subset, replace=False)
        normal_parts.append(idxs)
    normal_idx = np.concatenate(normal_parts) if normal_parts else np.asarray([], dtype=np.int64)
    normal_idx = np.asarray(sorted(set(normal_idx.tolist())), dtype=np.int64)
    return patient_idx, normal_idx, normal_counts


def compute_soft_reference(features, patient_idx, normal_idx, top_k, temperature, batch_size, device):
    normal_bank = l2_normalize_torch(features[normal_idx]).to(device)
    top_k = min(top_k, normal_bank.shape[0])
    raw_chunks = []
    reference_chunks = []
    residual_chunks = []
    distance_chunks = []
    topk_chunks = []

    with torch.no_grad():
        for start in range(0, len(patient_idx), batch_size):
            idx = patient_idx[start : start + batch_size]
            query = l2_normalize_torch(features[idx]).to(device)
            sim = query @ normal_bank.T
            top_sim, top_pos = torch.topk(sim, k=top_k, dim=1)
            weights = torch.softmax(top_sim / temperature, dim=1)
            reference = (weights.unsqueeze(-1) * normal_bank[top_pos]).sum(dim=1)
            residual = query - reference
            soft_distance = 1.0 - (weights * top_sim).sum(dim=1)

            raw_chunks.append(query.cpu())
            reference_chunks.append(reference.cpu())
            residual_chunks.append(residual.cpu())
            distance_chunks.append(soft_distance.cpu())
            topk_chunks.append(torch.as_tensor(normal_idx, dtype=torch.long)[top_pos.cpu()])

    residual = torch.cat(residual_chunks, dim=0)
    return {
        "raw": torch.cat(raw_chunks, dim=0),
        "reference": torch.cat(reference_chunks, dim=0),
        "residual": residual,
        "abs_residual": residual.abs(),
        "soft_distance": torch.cat(distance_chunks, dim=0),
        "topk_normal_indices": torch.cat(topk_chunks, dim=0),
    }


def feature_views(residual_data):
    raw = residual_data["raw"].numpy()
    residual = residual_data["residual"].numpy()
    abs_residual = residual_data["abs_residual"].numpy()
    distance = residual_data["soft_distance"].numpy()[:, None]
    return {
        "raw": raw,
        "residual": residual,
        "abs_residual": abs_residual,
        "raw_plus_abs_residual": np.concatenate([raw, abs_residual], axis=1),
        "raw_plus_residual_abs_distance": np.concatenate([raw, residual, abs_residual, distance], axis=1),
        "soft_distance": distance,
    }


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def iter_group_splits(y, groups, n_splits, seed):
    from sklearn.model_selection import GroupKFold

    n_splits = min(n_splits, len(set(groups)))
    if n_splits < 2:
        return []
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(y)), y, groups))
    except Exception:
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(np.zeros(len(y)), y, groups))


def metric_dict(y_true, probs, preds, binary, prefix):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

    out = {
        f"{prefix}_accuracy": accuracy_score(y_true, preds),
        f"{prefix}_balanced_acc": balanced_accuracy_score(y_true, preds),
        f"{prefix}_macro_f1": f1_score(y_true, preds, average="macro"),
        f"{prefix}_weighted_f1": f1_score(y_true, preds, average="weighted"),
    }
    try:
        if binary:
            score = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.reshape(-1)
            out[f"{prefix}_auroc"] = roc_auc_score(y_true, score)
        else:
            out[f"{prefix}_macro_auroc_ovr"] = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except ValueError:
        key = f"{prefix}_auroc" if binary else f"{prefix}_macro_auroc_ovr"
        out[key] = float("nan")
    return out


def aggregate_groups(y_true, probs, preds, groups):
    buckets = defaultdict(lambda: {"probs": [], "preds": [], "labels": []})
    for y, prob, pred, group in zip(y_true, probs, preds, groups):
        item = buckets[(group, int(y))]
        item["probs"].append(prob)
        item["preds"].append(int(pred))
        item["labels"].append(int(y))

    agg_y = []
    agg_probs = []
    agg_preds = []
    for item in buckets.values():
        label = int(round(sum(item["labels"]) / len(item["labels"])))
        prob = np.mean(np.stack(item["probs"], axis=0), axis=0)
        pred = int(np.argmax(prob))
        agg_y.append(label)
        agg_probs.append(prob)
        agg_preds.append(pred)
    return np.asarray(agg_y), np.stack(agg_probs, axis=0), np.asarray(agg_preds)


def run_probe(task, feature_name, x_view, rows, patient_idx, n_splits, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    local_positions = []
    labels = []
    split_groups = []
    groups = []
    for local_i, global_i in enumerate(patient_idx):
        row = rows[int(global_i)]
        label = task_label(row, task)
        if label is None:
            continue
        local_positions.append(local_i)
        labels.append(label)
        split_groups.append(split_group(row))
        groups.append(eval_group(row))

    if not local_positions:
        return []
    y = np.asarray(labels, dtype=np.int64)
    local_positions = np.asarray(local_positions, dtype=np.int64)
    split_groups = np.asarray(split_groups)
    groups = np.asarray(groups)
    classes = np.asarray(sorted(set(y.tolist())), dtype=np.int64)
    if len(classes) < 2 or len(set(split_groups.tolist())) < 2:
        return []

    binary = is_binary_task(task)
    fold_rows = []
    for fold, (train_rel, test_rel) in enumerate(iter_group_splits(y, split_groups, n_splits, seed), start=1):
        y_train = y[train_rel]
        y_test = y[test_rel]
        if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
            continue

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_view[local_positions[train_rel]])
        x_test = scaler.transform(x_view[local_positions[test_rel]])
        clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
        clf.fit(x_train, y_train)

        probs_raw = clf.predict_proba(x_test)
        prob_classes = clf.classes_
        probs = np.zeros((len(test_rel), len(classes)), dtype=np.float64)
        for src_col, cls in enumerate(prob_classes):
            dst_cols = np.where(classes == cls)[0]
            if len(dst_cols):
                probs[:, dst_cols[0]] = probs_raw[:, src_col]
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = probs / np.maximum(row_sums, 1e-12)
        preds = classes[np.argmax(probs, axis=1)]

        row = {
            "task": task,
            "feature": feature_name,
            "fold": fold,
            "classes": "|".join(str(int(c)) for c in classes),
            "n_train": len(train_rel),
            "n_test": len(test_rel),
            "n_train_groups": len(set(split_groups[train_rel])),
            "n_test_groups": len(set(split_groups[test_rel])),
        }
        row.update(metric_dict(y_test, probs, preds, binary, "clip"))
        gy, gp, gpred = aggregate_groups(y_test, probs, preds, groups[test_rel])
        row.update(metric_dict(gy, gp, gpred, binary, "group"))
        fold_rows.append(row)
    return fold_rows


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


def summarize_folds(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["feature"])].append(row)
    metric_keys = sorted(k for k in rows[0] if k.endswith(("accuracy", "balanced_acc", "macro_f1", "weighted_f1", "auroc", "macro_auroc_ovr")))
    summaries = []
    for (task, feature), items in grouped.items():
        summary = {"task": task, "feature": feature, "n_folds": len(items)}
        for metric in metric_keys:
            vals = []
            for item in items:
                value = item.get(metric)
                if value in ("", None):
                    continue
                value = float(value)
                if not math.isnan(value):
                    vals.append(value)
            if vals:
                mean, std = mean_std(vals)
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_std"] = std
        summaries.append(summary)
    return summaries


def main():
    opts = parse_args()
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)

    output_dir = Path(opts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(opts.device)

    features, rows = load_inputs(opts.features_dir)
    patient_idx, normal_idx, normal_counts = select_indices(
        rows,
        opts.normal_subsets,
        opts.normal_limit_per_subset,
        opts.seed,
    )
    print(
        json.dumps(
            {
                "features_dir": opts.features_dir,
                "feature_shape": list(features.shape),
                "patient_count": int(len(patient_idx)),
                "normal_count": int(len(normal_idx)),
                "normal_counts_before_limit": normal_counts,
                "subset_counts": dict(Counter(row.get("subset", "") for row in rows)),
                "tasks": opts.tasks,
                "device": str(device),
                "top_k": opts.top_k,
                "temperature": opts.temperature,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    residual_data = compute_soft_reference(
        features,
        patient_idx,
        normal_idx,
        opts.top_k,
        opts.temperature,
        opts.query_batch_size,
        device,
    )
    torch.save(
        {
            "patient_indices": torch.as_tensor(patient_idx, dtype=torch.long),
            "normal_indices": torch.as_tensor(normal_idx, dtype=torch.long),
            **residual_data,
            "top_k": opts.top_k,
            "temperature": opts.temperature,
        },
        output_dir / "residual_features.pt",
    )

    views = feature_views(residual_data)
    fold_rows = []
    for task in opts.tasks:
        for feature_name, view in views.items():
            rows_out = run_probe(task, feature_name, view, rows, patient_idx, opts.n_splits, opts.seed)
            fold_rows.extend(rows_out)
            if rows_out:
                print(summarize_folds(rows_out)[0])

    if not fold_rows:
        raise RuntimeError("No fold rows produced. Check tasks, metadata, and split groups.")
    write_csv(output_dir / "main_protocol_folds.csv", fold_rows)
    summary_rows = summarize_folds(fold_rows)
    write_csv(output_dir / "main_protocol_summary.csv", summary_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "features_dir": opts.features_dir,
                "patient_count": int(len(patient_idx)),
                "normal_count": int(len(normal_idx)),
                "normal_counts_before_limit": normal_counts,
                "tasks": opts.tasks,
                "top_k": opts.top_k,
                "temperature": opts.temperature,
                "device": str(device),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
