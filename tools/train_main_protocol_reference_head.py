import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


PATIENT_SUBSETS = {"pdgait", "3dgait"}
DEFAULT_TASKS = [
    "pdgait_binary",
    "pdgait_score_3class",
    "3dgait_binary",
    "3dgait_score_4class",
    "3dgait_subtype_3class",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train DPPD-aligned normal-reference heads on frozen MotionBERT latents.")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--residual-dir", required=True, help="Directory containing residual_features.pt from main protocol eval.")
    parser.add_argument("--output-dir", default="eval/main_protocol_reference_head_e1")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--modes", nargs="+", default=["raw_mlp", "fixed_residual_mlp", "learnable_ref_mlp"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--attn-dim", type=int, default=64)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


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


def get_device(name):
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_normalize_torch(x, eps=1e-12):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def load_metadata(features_dir):
    with (Path(features_dir) / "metadata.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_tensors(features_dir, residual_dir):
    feature_data = torch.load(Path(features_dir) / "clip_features.pt", map_location="cpu")
    all_features = l2_normalize_torch(feature_data["clip_features"].float())
    residual_data = torch.load(Path(residual_dir) / "residual_features.pt", map_location="cpu")
    patient_indices = residual_data["patient_indices"].long()
    topk_indices = residual_data["topk_normal_indices"].long()
    raw = residual_data["raw"].float()
    residual = residual_data["residual"].float()
    abs_residual = residual_data["abs_residual"].float()
    soft_distance = residual_data["soft_distance"].float().unsqueeze(1)
    fixed_features = torch.cat([raw, residual, abs_residual, soft_distance], dim=1)
    neighbor_features = all_features[topk_indices.reshape(-1)].reshape(topk_indices.shape[0], topk_indices.shape[1], -1)
    return {
        "patient_indices": patient_indices,
        "raw": raw,
        "fixed_features": fixed_features,
        "neighbor_features": neighbor_features,
        "topk_indices": topk_indices,
    }


class TensorDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y.long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class ReferenceDataset(Dataset):
    def __init__(self, raw, neighbors, y):
        self.raw = raw
        self.neighbors = neighbors
        self.y = y.long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.raw[index], self.neighbors[index], self.y[index]


class FeatureMLP(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=128, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, num_classes)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, num_classes), num_classes),
        )

    def forward(self, x):
        return self.net(x)


class LearnableReferenceHead(nn.Module):
    def __init__(self, dim, num_classes, attn_dim=64, hidden_dim=128, dropout=0.15):
        super().__init__()
        self.q_proj = nn.Linear(dim, attn_dim)
        self.k_proj = nn.Linear(dim, attn_dim)
        self.scale = math.sqrt(attn_dim)
        self.classifier = FeatureMLP(dim * 3 + 1, num_classes, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, raw, neighbors, return_entropy=False):
        q = self.q_proj(raw)
        k = self.k_proj(neighbors)
        logits = (k * q.unsqueeze(1)).sum(dim=-1) / self.scale
        weights = torch.softmax(logits, dim=1)
        reference = (weights.unsqueeze(-1) * neighbors).sum(dim=1)
        residual = raw - reference
        cosine = (raw.unsqueeze(1) * neighbors).sum(dim=-1)
        soft_distance = 1.0 - (weights * cosine).sum(dim=1, keepdim=True)
        features = torch.cat([raw, residual, residual.abs(), soft_distance], dim=1)
        out = self.classifier(features)
        if not return_entropy:
            return out
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1).mean()
        return out, entropy


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


def make_inner_train_val(y, groups, train_rel, seed):
    y_train = y[train_rel]
    groups_train = groups[train_rel]
    if len(set(groups_train)) < 3 or len(set(y_train)) < 2:
        return train_rel, train_rel
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        n_splits = min(5, len(set(groups_train)))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        inner_train, val = next(splitter.split(np.zeros(len(train_rel)), y_train, groups_train))
        return train_rel[inner_train], train_rel[val]
    except Exception:
        return train_rel, train_rel


def class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)


def build_task_indices(rows, patient_indices, task):
    local_indices = []
    labels = []
    split_groups = []
    eval_groups = []
    for local_i, global_i in enumerate(patient_indices):
        row = rows[int(global_i)]
        label = task_label(row, task)
        if label is None:
            continue
        local_indices.append(local_i)
        labels.append(label)
        split_groups.append(split_group(row))
        eval_groups.append(eval_group(row))
    classes = sorted(set(labels))
    class_to_local = {label: i for i, label in enumerate(classes)}
    y = np.asarray([class_to_local[label] for label in labels], dtype=np.int64)
    return (
        np.asarray(local_indices, dtype=np.int64),
        y,
        np.asarray(split_groups),
        np.asarray(eval_groups),
        classes,
    )


def make_dataset(mode, tensors, indices, y):
    y_tensor = torch.as_tensor(y, dtype=torch.long)
    if mode == "raw_mlp":
        return TensorDataset(tensors["raw"][indices], y_tensor)
    if mode == "fixed_residual_mlp":
        return TensorDataset(tensors["fixed_features"][indices], y_tensor)
    if mode == "learnable_ref_mlp":
        return ReferenceDataset(tensors["raw"][indices], tensors["neighbor_features"][indices], y_tensor)
    raise ValueError(f"Unknown mode: {mode}")


def build_model(mode, tensors, indices, num_classes, opts):
    if mode == "raw_mlp":
        in_dim = tensors["raw"][indices].shape[1]
        return FeatureMLP(in_dim, num_classes, hidden_dim=opts.hidden_dim, dropout=opts.dropout)
    if mode == "fixed_residual_mlp":
        in_dim = tensors["fixed_features"][indices].shape[1]
        return FeatureMLP(in_dim, num_classes, hidden_dim=opts.hidden_dim, dropout=opts.dropout)
    if mode == "learnable_ref_mlp":
        dim = tensors["raw"][indices].shape[1]
        return LearnableReferenceHead(dim, num_classes, attn_dim=opts.attn_dim, hidden_dim=opts.hidden_dim, dropout=opts.dropout)
    raise ValueError(f"Unknown mode: {mode}")


def run_batch(model, mode, batch, device, entropy_weight=0.0, criterion=None):
    if mode == "learnable_ref_mlp":
        raw, neighbors, target = batch
        if entropy_weight > 0:
            logits, entropy = model(raw.to(device), neighbors.to(device), return_entropy=True)
        else:
            logits = model(raw.to(device), neighbors.to(device))
            entropy = None
    else:
        feat, target = batch
        logits = model(feat.to(device))
        entropy = None
    target = target.to(device)
    if criterion is None:
        return logits, target, entropy
    loss = criterion(logits, target)
    if entropy is not None:
        loss = loss + entropy_weight * entropy
    return loss


def evaluate_loss(model, mode, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for batch in loader:
            loss = run_batch(model, mode, batch, device, criterion=criterion)
            if mode == "learnable_ref_mlp":
                n = len(batch[2])
            else:
                n = len(batch[1])
            total_loss += float(loss.item()) * n
            total_n += n
    return total_loss / max(total_n, 1)


def train_model(mode, tensors, indices, y, train_rel, val_rel, num_classes, opts, device):
    train_ds = make_dataset(mode, tensors, indices[train_rel], y[train_rel])
    val_ds = make_dataset(mode, tensors, indices[val_rel], y[val_rel])
    model = build_model(mode, tensors, indices[train_rel], num_classes, opts).to(device)
    weights = class_weights(y[train_rel], num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    train_loader = DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=opts.batch_size, shuffle=False, num_workers=0)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    for _ in range(opts.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = run_batch(model, mode, batch, device, opts.entropy_weight, criterion)
            loss.backward()
            optimizer.step()
        val_loss = evaluate_loss(model, mode, val_loader, criterion, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= opts.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model, mode, tensors, indices, batch_size, device):
    dummy_y = np.zeros(len(indices), dtype=np.int64)
    ds = make_dataset(mode, tensors, indices, dummy_y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits, _, _ = run_batch(model, mode, batch, device)
            chunks.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(chunks, dim=0).numpy()


def metric_dict(y_true, probs, preds, prefix):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

    out = {
        f"{prefix}_accuracy": accuracy_score(y_true, preds),
        f"{prefix}_balanced_acc": balanced_accuracy_score(y_true, preds),
        f"{prefix}_macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
        f"{prefix}_weighted_f1": f1_score(y_true, preds, average="weighted", zero_division=0),
    }
    try:
        if probs.shape[1] == 2:
            out[f"{prefix}_auroc"] = roc_auc_score(y_true, probs[:, 1])
        else:
            out[f"{prefix}_macro_auroc_ovr"] = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except ValueError:
        key = f"{prefix}_auroc" if probs.shape[1] == 2 else f"{prefix}_macro_auroc_ovr"
        out[key] = float("nan")
    return out


def aggregate_groups(y_true, probs, preds, groups):
    buckets = defaultdict(lambda: {"probs": [], "labels": []})
    for y, prob, group in zip(y_true, probs, groups):
        item = buckets[(group, int(y))]
        item["probs"].append(prob)
        item["labels"].append(int(y))
    agg_y, agg_probs, agg_preds = [], [], []
    for item in buckets.values():
        label = int(round(sum(item["labels"]) / len(item["labels"])))
        prob = np.mean(np.stack(item["probs"], axis=0), axis=0)
        agg_y.append(label)
        agg_probs.append(prob)
        agg_preds.append(int(np.argmax(prob)))
    return np.asarray(agg_y), np.stack(agg_probs, axis=0), np.asarray(agg_preds)


def run_experiment(task, mode, rows, tensors, opts, device):
    patient_indices = tensors["patient_indices"].numpy()
    indices, y, split_groups, groups, classes = build_task_indices(rows, patient_indices, task)
    if len(indices) == 0 or len(set(y.tolist())) < 2 or len(set(split_groups.tolist())) < 2:
        return []
    num_classes = len(classes)
    fold_rows = []
    for fold, (train_rel, test_rel) in enumerate(iter_group_splits(y, split_groups, opts.n_splits, opts.seed), start=1):
        if len(set(y[train_rel].tolist())) < 2 or len(set(y[test_rel].tolist())) < 2:
            continue
        inner_train, val_rel = make_inner_train_val(y, split_groups, train_rel, opts.seed + fold)
        model = train_model(mode, tensors, indices, y, inner_train, val_rel, num_classes, opts, device)
        probs = predict(model, mode, tensors, indices[test_rel], opts.batch_size, device)
        preds = probs.argmax(axis=1)
        row = {
            "task": task,
            "mode": mode,
            "fold": fold,
            "classes": "|".join(str(c) for c in classes),
            "n_train": len(inner_train),
            "n_val": len(val_rel),
            "n_test": len(test_rel),
            "n_train_groups": len(set(split_groups[inner_train])),
            "n_test_groups": len(set(split_groups[test_rel])),
        }
        row.update(metric_dict(y[test_rel], probs, preds, "clip"))
        gy, gp, gpred = aggregate_groups(y[test_rel], probs, preds, groups[test_rel])
        row.update(metric_dict(gy, gp, gpred, "group"))
        fold_rows.append(row)
    return fold_rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = set()
    for row in rows:
        keys.update(row.keys())
    fieldnames = sorted(keys)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["mode"])].append(row)
    metric_keys = sorted(k for k in rows[0] if k.endswith(("accuracy", "balanced_acc", "macro_f1", "weighted_f1", "auroc", "macro_auroc_ovr")))
    out = []
    for (task, mode), items in grouped.items():
        row = {"task": task, "mode": mode, "n_folds": len(items)}
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
                m, s = mean_std(vals)
                row[f"{metric}_mean"] = m
                row[f"{metric}_std"] = s
        out.append(row)
    return out


def main():
    opts = parse_args()
    set_seed(opts.seed)
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(opts.device)

    rows = load_metadata(opts.features_dir)
    tensors = load_tensors(opts.features_dir, opts.residual_dir)
    patient_indices = tensors["patient_indices"].numpy()
    print(
        json.dumps(
            {
                "features_dir": opts.features_dir,
                "residual_dir": opts.residual_dir,
                "patient_count": int(len(patient_indices)),
                "label_counts": dict(
                    Counter(
                        f"{rows[int(i)].get('subset', '')}:{rows[int(i)].get('score_label', rows[int(i)].get('label', ''))}"
                        for i in patient_indices
                    )
                ),
                "topk": int(tensors["topk_indices"].shape[1]),
                "device": str(device),
                "tasks": opts.tasks,
                "modes": opts.modes,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    all_rows = []
    for task in opts.tasks:
        for mode in opts.modes:
            print(f"Running {task} / {mode}", flush=True)
            rows_out = run_experiment(task, mode, rows, tensors, opts, device)
            all_rows.extend(rows_out)
            if rows_out:
                for item in summarize(rows_out):
                    print(item, flush=True)

    if not all_rows:
        raise RuntimeError("No fold rows were produced.")
    write_csv(output_dir / "reference_head_folds.csv", all_rows)
    summary_rows = summarize(all_rows)
    write_csv(output_dir / "reference_head_summary.csv", summary_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "features_dir": opts.features_dir,
                "residual_dir": opts.residual_dir,
                "tasks": opts.tasks,
                "modes": opts.modes,
                "epochs": opts.epochs,
                "batch_size": opts.batch_size,
                "lr": opts.lr,
                "weight_decay": opts.weight_decay,
                "dropout": opts.dropout,
                "hidden_dim": opts.hidden_dim,
                "attn_dim": opts.attn_dim,
                "entropy_weight": opts.entropy_weight,
                "device": str(device),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
