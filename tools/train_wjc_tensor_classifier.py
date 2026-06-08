import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_TASKS = ["pdgait_binary", "3dgait_binary", "3dgait_subtype_3class"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train W x J x C tensor classifiers on raw and residual gait latents.")
    parser.add_argument("--features-dir", default="features/e1_patient_dynamic_anchor_clip")
    parser.add_argument("--patient-window-file", default="features/window_joint_e1/patient_window_joint.pt")
    parser.add_argument("--prototype-file", default="eval/strict_window_prototype_e1/window_prototypes_full_healthy.pt")
    parser.add_argument("--full-topk-window-file", default="features/window_joint_e1/topk_normal_window_joint.pt")
    parser.add_argument("--full-topk-aggregate-file", default="eval/window_joint_residual_e1/window_joint_residual_aggregates.pt")
    parser.add_argument("--output-dir", default="eval/wjc_tensor_classifier_e1")
    parser.add_argument("--banks", nargs="+", default=["prototype_M16", "prototype_M32"])
    parser.add_argument("--feature-modes", nargs="+", default=["abs_residual", "raw_plus_abs_residual"])
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--prototype-top-k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260518)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_metadata(features_dir):
    with (Path(features_dir) / "metadata.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def score_label(row):
    score = safe_int(row.get("score_label"))
    if score is None:
        score = safe_int(row.get("label"))
    return score


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
    raise ValueError(f"Unknown task: {task}")


def split_group(row):
    subset = row.get("subset", "")
    if subset == "pdgait":
        return f"pdgait_subject:{row.get('subject') or row.get('id') or row.get('video_name') or row.get('path')}"
    if subset == "3dgait":
        return f"3dgait_patient:{row.get('id') or row.get('path')}"
    return f"{subset}:{row.get('path')}"


def eval_group(row):
    subset = row.get("subset", "")
    if subset == "pdgait":
        return f"pdgait_video:{row.get('video_name') or row.get('subject') or row.get('id') or row.get('path')}"
    if subset == "3dgait":
        return f"3dgait_patient:{row.get('id') or row.get('path')}"
    return split_group(row)


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


def validation_split(y, groups, seed):
    splits = iter_group_splits(y, groups, min(5, len(set(groups.tolist()))), seed)
    all_classes = set(y.tolist())
    for train_idx, val_idx in splits:
        if set(y[train_idx].tolist()) == all_classes and len(set(y[val_idx].tolist())) > 1:
            return train_idx, val_idx
    rng = np.random.default_rng(seed)
    uniq = np.asarray(sorted(set(groups.tolist())))
    rng.shuffle(uniq)
    val_groups = set(uniq[: max(1, int(round(len(uniq) * 0.2)))].tolist())
    val_idx = np.asarray([i for i, g in enumerate(groups) if g in val_groups], dtype=np.int64)
    train_idx = np.asarray([i for i, g in enumerate(groups) if g not in val_groups], dtype=np.int64)
    return train_idx, val_idx


def metric_dict(y_true, probs, preds, classes, prefix):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

    out = {
        f"{prefix}_accuracy": accuracy_score(y_true, preds),
        f"{prefix}_balanced_acc": balanced_accuracy_score(y_true, preds),
        f"{prefix}_macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
        f"{prefix}_weighted_f1": f1_score(y_true, preds, average="weighted", zero_division=0),
    }
    try:
        if len(classes) == 2:
            out[f"{prefix}_auroc"] = roc_auc_score(y_true, probs[:, 1])
        else:
            out[f"{prefix}_macro_auroc_ovr"] = roc_auc_score(
                y_true, probs, labels=classes, multi_class="ovr", average="macro"
            )
    except ValueError:
        out[f"{prefix}_{'auroc' if len(classes) == 2 else 'macro_auroc_ovr'}"] = float("nan")
    return out


def aggregate_groups(y_true, probs, groups, classes):
    buckets = defaultdict(lambda: {"probs": [], "labels": []})
    for y, prob, group in zip(y_true, probs, groups):
        item = buckets[(group, int(y))]
        item["labels"].append(int(y))
        item["probs"].append(prob)
    agg_y, agg_probs = [], []
    for item in buckets.values():
        agg_y.append(int(round(sum(item["labels"]) / len(item["labels"]))))
        agg_probs.append(np.mean(np.stack(item["probs"], axis=0), axis=0))
    agg_y = np.asarray(agg_y, dtype=np.int64)
    agg_probs = np.stack(agg_probs, axis=0)
    agg_preds = np.asarray([classes[i] for i in np.argmax(agg_probs, axis=1)], dtype=np.int64)
    return agg_y, agg_probs, agg_preds


class IndexDataset(Dataset):
    def __init__(self, local_indices, encoded_labels):
        self.local_indices = torch.as_tensor(local_indices, dtype=torch.long)
        self.encoded_labels = torch.as_tensor(encoded_labels, dtype=torch.long)

    def __len__(self):
        return int(self.local_indices.numel())

    def __getitem__(self, index):
        return self.local_indices[index], self.encoded_labels[index]


class WJCTensorClassifier(nn.Module):
    def __init__(self, mode, out_dim, in_channels=512, dim=96, heads=4, layers=1, dropout=0.15):
        super().__init__()
        self.mode = mode
        self.raw_proj = nn.Linear(in_channels, dim)
        self.res_proj = nn.Linear(in_channels, dim)
        self.fuse = nn.Linear(dim * 2, dim)
        self.window_pos = nn.Parameter(torch.zeros(9, dim))
        self.joint_pos = nn.Parameter(torch.zeros(17, dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.attn_score = nn.Linear(dim, 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, out_dim))
        nn.init.trunc_normal_(self.window_pos, std=0.02)
        nn.init.trunc_normal_(self.joint_pos, std=0.02)

    def forward(self, raw, residual):
        if self.mode == "raw":
            x = self.raw_proj(raw)
        elif self.mode == "abs_residual":
            x = self.res_proj(residual.abs())
        elif self.mode == "signed_residual":
            x = self.res_proj(residual)
        elif self.mode == "raw_plus_abs_residual":
            x = self.fuse(torch.cat([self.raw_proj(raw), self.res_proj(residual.abs())], dim=-1))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        x = x + self.window_pos[None, :, None, :] + self.joint_pos[None, None, :, :]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        x = self.encoder(x)
        x = self.norm(x)
        weights = torch.softmax(self.attn_score(x).squeeze(-1), dim=1)
        pooled = (weights.unsqueeze(-1) * x).sum(dim=1)
        return self.classifier(pooled)


class PrototypeReferenceProvider:
    def __init__(self, prototype_file, bank_name, top_k, temperature, device):
        data = torch.load(prototype_file, map_location="cpu")
        key = bank_name.replace("prototype_M", "")
        bank = data["banks"][key]
        self.prototypes = bank["prototypes"].to(device).float()
        self.key_centers = F.normalize(bank["key_centers"].to(device).float(), dim=-1)
        self.top_k = min(int(top_k), int(self.prototypes.shape[1]))
        self.temperature = float(temperature)

    def reference(self, raw, local_indices=None):
        raw_key = F.normalize(raw.mean(dim=2).float(), dim=-1)
        refs = []
        for win in range(raw.shape[1]):
            sim = raw_key[:, win] @ self.key_centers[win].T
            top_sim, top_idx = torch.topk(sim, k=self.top_k, dim=1)
            weights = torch.softmax(top_sim / self.temperature, dim=1)
            ref = (weights[:, :, None, None] * self.prototypes[win, top_idx]).sum(dim=1)
            refs.append(ref)
        return torch.stack(refs, dim=1)


class FullTopkReferenceProvider:
    def __init__(self, topk_window_file, aggregate_file, device):
        topk = torch.load(topk_window_file, map_location="cpu")
        aggr = torch.load(aggregate_file, map_location="cpu")
        self.normal_window = topk["window_joint_features"].float()
        self.global_to_pos = {int(g): pos for pos, g in enumerate(topk["global_indices"].long().tolist())}
        self.topk_indices = aggr["topk_normal_indices"].long()
        self.weights = aggr["topk_weights"].float()
        self.device = device

    def reference(self, raw, local_indices):
        batch_topk = self.topk_indices[local_indices.cpu()].tolist()
        positions = torch.as_tensor(
            [[self.global_to_pos[int(g)] for g in row] for row in batch_topk],
            dtype=torch.long,
        )
        normal = self.normal_window[positions].to(self.device)
        weights = self.weights[local_indices.cpu()].to(self.device)
        return (weights[:, :, None, None, None] * normal).sum(dim=1)


def class_weights(y_encoded, n_classes):
    counts = np.bincount(y_encoded, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def prepare_task(rows, patient_global_indices, task):
    records = []
    for local_i, global_i in enumerate(patient_global_indices.tolist()):
        row = rows[int(global_i)]
        label = task_label(row, task)
        if label is None:
            continue
        records.append(
            {
                "local_i": local_i,
                "label": int(label),
                "split_group": split_group(row),
                "eval_group": eval_group(row),
            }
        )
    classes = np.asarray(sorted({r["label"] for r in records}), dtype=np.int64)
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    local_indices = np.asarray([r["local_i"] for r in records], dtype=np.int64)
    y_raw = np.asarray([r["label"] for r in records], dtype=np.int64)
    y_encoded = np.asarray([class_to_idx[int(y)] for y in y_raw], dtype=np.int64)
    groups = np.asarray([r["split_group"] for r in records])
    eval_groups = np.asarray([r["eval_group"] for r in records])
    return local_indices, y_raw, y_encoded, groups, eval_groups, classes


def evaluate_model(model, provider, patient_window, loader, classes, device):
    model.eval()
    probs_all, idx_all, y_all = [], [], []
    with torch.no_grad():
        for local_idx, y in loader:
            raw = patient_window[local_idx].to(device).float()
            ref = provider.reference(raw, local_idx)
            logits = model(raw, raw - ref)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_all.append(probs)
            idx_all.append(local_idx.numpy())
            y_all.append(y.numpy())
    probs = np.concatenate(probs_all, axis=0)
    local_idx = np.concatenate(idx_all, axis=0)
    y_encoded = np.concatenate(y_all, axis=0)
    preds_encoded = probs.argmax(axis=1)
    y_raw = np.asarray([classes[i] for i in y_encoded], dtype=np.int64)
    preds_raw = np.asarray([classes[i] for i in preds_encoded], dtype=np.int64)
    return local_idx, y_raw, probs, preds_raw


def train_one_fold(opts, provider, patient_window, train_idx, val_idx, test_idx, local_indices, y_encoded, classes, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WJCTensorClassifier(
        opts.current_mode,
        len(classes),
        dim=opts.dim,
        heads=opts.heads,
        layers=opts.layers,
        dropout=opts.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_encoded[train_idx], len(classes)).to(device))
    train_loader = DataLoader(
        IndexDataset(local_indices[train_idx], y_encoded[train_idx]),
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        IndexDataset(local_indices[val_idx], y_encoded[val_idx]),
        batch_size=opts.eval_batch_size,
        shuffle=False,
        num_workers=0,
    )
    best_state = None
    best_score = -1.0
    wait = 0
    from sklearn.metrics import f1_score

    for epoch in range(opts.epochs):
        model.train()
        total_loss = 0.0
        for local_idx, y in train_loader:
            raw = patient_window[local_idx].to(device).float()
            ref = provider.reference(raw, local_idx)
            logits = model(raw, raw - ref)
            loss = criterion(logits, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * int(y.numel())
        _, val_y_raw, val_probs, val_preds = evaluate_model(model, provider, patient_window, val_loader, classes, device)
        val_score = f1_score(val_y_raw, val_preds, average="macro", zero_division=0)
        if val_score > best_score:
            best_score = float(val_score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= opts.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loader = DataLoader(
        IndexDataset(local_indices[test_idx], y_encoded[test_idx]),
        batch_size=opts.eval_batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_local_idx, test_y_raw, test_probs, test_preds = evaluate_model(
        model, provider, patient_window, test_loader, classes, device
    )
    return best_score, test_local_idx, test_y_raw, test_probs, test_preds


def run_config(opts, rows, patient_window, patient_indices, bank_name, mode, task):
    local_indices, y_raw, y_encoded, groups, eval_groups, classes = prepare_task(rows, patient_indices, task)
    splits = iter_group_splits(y_raw, groups, opts.n_splits, opts.seed)
    if opts.max_folds is not None:
        splits = splits[: opts.max_folds]
    if bank_name == "full_topk_unique_normals":
        provider = FullTopkReferenceProvider(opts.full_topk_window_file, opts.full_topk_aggregate_file, torch.device("cuda"))
    elif bank_name.startswith("prototype_M"):
        provider = PrototypeReferenceProvider(
            opts.prototype_file, bank_name, opts.prototype_top_k, opts.temperature, torch.device("cuda")
        )
    else:
        raise ValueError(f"Unknown bank: {bank_name}")

    fold_rows = []
    for fold, (outer_train, test_idx) in enumerate(splits, start=1):
        train_local, val_local = validation_split(y_raw[outer_train], groups[outer_train], opts.seed + fold)
        train_idx = outer_train[train_local]
        val_idx = outer_train[val_local]
        if len(set(y_raw[train_idx].tolist())) < len(classes) or len(set(y_raw[test_idx].tolist())) < 2:
            continue
        best_val, test_local_idx, test_y_raw, test_probs, test_preds = train_one_fold(
            opts, provider, patient_window, train_idx, val_idx, test_idx, local_indices, y_encoded, classes, opts.seed + fold
        )
        row = {
            "bank": bank_name,
            "feature_mode": mode,
            "task": task,
            "fold": fold,
            "best_val_macro_f1": best_val,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "classes": "|".join(str(int(c)) for c in classes),
        }
        row.update(metric_dict(test_y_raw, test_probs, test_preds, classes, "clip"))
        test_eval_groups = eval_groups[test_idx]
        gy, gp, gpred = aggregate_groups(test_y_raw, test_probs, test_eval_groups, classes)
        row.update(metric_dict(gy, gp, gpred, classes, "group"))
        fold_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return fold_rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["bank"], row["feature_mode"], row["task"])].append(row)
    metric_keys = sorted(
        key
        for key in rows[0]
        if key.endswith(("accuracy", "balanced_acc", "macro_f1", "weighted_f1", "auroc", "macro_auroc_ovr"))
    )
    out_rows = []
    for (bank, mode, task), items in grouped.items():
        out = {"bank": bank, "feature_mode": mode, "task": task, "n_folds": len(items)}
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
                arr = np.asarray(vals, dtype=np.float64)
                out[f"{metric}_mean"] = float(arr.mean())
                out[f"{metric}_std"] = float(arr.std(ddof=1) if len(arr) > 1 else 0.0)
        out_rows.append(out)
    return out_rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    lines = [
        "# WJC Tensor Classifier",
        "",
        "Cells are group-level `Macro-F1 / Balanced Accuracy`; binary tasks include AUROC.",
        "",
    ]
    for task in sorted({r["task"] for r in rows}):
        lines.append(f"## {task}")
        lines.append("")
        lines.append("| Bank | Mode | Macro-F1 | BalAcc | AUROC |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        sub = sorted([r for r in rows if r["task"] == task], key=lambda r: r.get("group_macro_f1_mean", -1), reverse=True)
        for row in sub:
            auroc = row.get("group_auroc_mean", row.get("group_macro_auroc_ovr_mean", ""))
            auroc_text = "" if auroc == "" else f"{float(auroc):.4f}"
            lines.append(
                f"| {row['bank']} | {row['feature_mode']} | "
                f"{float(row.get('group_macro_f1_mean', float('nan'))):.4f} | "
                f"{float(row.get('group_balanced_acc_mean', float('nan'))):.4f} | {auroc_text} |"
            )
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    opts = parse_args()
    set_seed(opts.seed)
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_metadata(opts.features_dir)
    patient_data = torch.load(opts.patient_window_file, map_location="cpu")
    patient_window = patient_data["window_joint_features"]
    patient_indices = patient_data["global_indices"].long()
    print(json.dumps({"patient_window_shape": list(patient_window.shape), "tasks": opts.tasks, "banks": opts.banks}, indent=2))

    all_rows = []
    for bank in opts.banks:
        for mode in opts.feature_modes:
            opts.current_mode = mode
            for task in opts.tasks:
                print(f"Running bank={bank} mode={mode} task={task}", flush=True)
                all_rows.extend(run_config(opts, rows, patient_window, patient_indices, bank, mode, task))
                summary_rows = summarize(all_rows)
                write_csv(output_dir / "wjc_tensor_folds.csv", all_rows)
                write_csv(output_dir / "wjc_tensor_summary.csv", summary_rows)
                write_markdown(output_dir / "wjc_tensor_summary.md", summary_rows)
    (output_dir / "summary.json").write_text(json.dumps(vars(opts), ensure_ascii=False, indent=2, default=str) + "\n")
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
