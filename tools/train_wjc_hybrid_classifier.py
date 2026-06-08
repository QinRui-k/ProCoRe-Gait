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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.train_wjc_gated_classifier import GatedWJCClassifier  # noqa: E402
from tools.train_wjc_tensor_classifier import (  # noqa: E402
    DEFAULT_TASKS,
    IndexDataset,
    PrototypeReferenceProvider,
    aggregate_groups,
    class_weights,
    iter_group_splits,
    metric_dict,
    prepare_task,
    read_metadata,
    validation_split,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train hybrid gated-WJC + aggregate residual classifiers.")
    parser.add_argument("--features-dir", default="features/e1_patient_dynamic_anchor_clip")
    parser.add_argument("--patient-window-file", default="features/window_joint_e1/patient_window_joint.pt")
    parser.add_argument("--prototype-file", default="eval/frozen_bank_source_ablation_e1/patient_control/window_prototypes.pt")
    parser.add_argument("--aggregate-file", default="eval/frozen_bank_source_ablation_e1/patient_control/prototype_M64_samples.csv")
    parser.add_argument("--output-dir", default="eval/wjc_hybrid_patient_control_M64_e1")
    parser.add_argument("--bank", default="prototype_M64")
    parser.add_argument("--models", nargs="+", default=["hybrid_token_all", "hybrid_structured_all"])
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--agg-dim", type=int, default=64)
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


def aggregate_columns(fieldnames):
    return (
        ["r_total"]
        + [c for c in fieldnames if c.startswith("joint_")]
        + [c for c in fieldnames if c.startswith("window_")]
        + [c for c in fieldnames if c.startswith("part_")]
    )


def load_aggregate_features(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No aggregate rows in {path}")
    cols = aggregate_columns(rows[0].keys())
    n = max(int(r["local_index"]) for r in rows) + 1
    x = np.zeros((n, len(cols)), dtype=np.float32)
    present = np.zeros(n, dtype=bool)
    for row in rows:
        idx = int(row["local_index"])
        x[idx] = np.asarray([float(row[c]) for c in cols], dtype=np.float32)
        present[idx] = True
    if not np.all(present):
        missing = int((~present).sum())
        raise RuntimeError(f"Aggregate feature file is missing {missing} local indices.")
    return torch.from_numpy(x), cols


class HybridWJCClassifier(nn.Module):
    def __init__(self, model_name, out_dim, agg_in_dim, in_channels=512, dim=96, heads=4, layers=1, agg_dim=64, dropout=0.15):
        super().__init__()
        if model_name == "hybrid_token_all":
            gated_name = "gated_token"
            wjc_dim = dim
        elif model_name == "hybrid_structured_all":
            gated_name = "gated_structured"
            wjc_dim = dim * 3
        else:
            raise ValueError(f"Unknown model: {model_name}")
        self.model_name = model_name
        self.backbone = GatedWJCClassifier(
            gated_name,
            out_dim,
            in_channels=in_channels,
            dim=dim,
            heads=heads,
            layers=layers,
            dropout=dropout,
        )
        # Replace the backbone head with identity-style feature extraction via its internal modules.
        self.backbone.classifier = nn.Identity()
        self.agg_proj = nn.Sequential(
            nn.LayerNorm(agg_in_dim),
            nn.Linear(agg_in_dim, agg_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(wjc_dim + agg_dim),
            nn.Dropout(dropout),
            nn.Linear(wjc_dim + agg_dim, max(dim, out_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(dim, out_dim), out_dim),
        )

    def forward(self, raw, residual, aggregate, return_aux=False):
        features, aux = self.extract_wjc_features(raw, residual)
        agg = self.agg_proj(aggregate)
        logits = self.classifier(torch.cat([features, agg], dim=1))
        if return_aux:
            return logits, aux
        return logits

    def extract_wjc_features(self, raw, residual):
        raw_tokens = self.backbone.encode_branch(raw, self.backbone.raw_proj, self.backbone.raw_encoder)
        res_tokens = self.backbone.encode_branch(residual.abs(), self.backbone.res_proj, self.backbone.res_encoder)
        b, w, j, d = raw_tokens.shape
        h_raw, raw_attn = self.backbone.raw_pool(raw_tokens.reshape(b, w * j, d))
        h_res, res_attn = self.backbone.res_pool(res_tokens.reshape(b, w * j, d))
        h_joint, joint_attn = self.backbone.joint_pool(res_tokens.mean(dim=1))
        h_window, window_attn = self.backbone.window_pool(res_tokens.mean(dim=2))
        gate = self.backbone.gate(torch.cat([h_raw, h_res, h_joint, h_window], dim=1))
        fused = gate * h_res + (1.0 - gate) * h_raw
        if self.model_name == "hybrid_structured_all":
            features = torch.cat([fused, h_joint, h_window], dim=1)
        else:
            features = fused
        return features, {
            "gate": gate.detach(),
            "raw_attn": raw_attn.detach(),
            "res_attn": res_attn.detach(),
            "joint_attn": joint_attn.detach(),
            "window_attn": window_attn.detach(),
        }


def normalize_aggregate(aggregate_features, train_local_indices):
    train_x = aggregate_features[torch.as_tensor(train_local_indices, dtype=torch.long)].float()
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp_min(1e-6)
    return mean, std


def evaluate_model(model, provider, patient_window, aggregate_features, agg_mean, agg_std, loader, classes, device, collect_aux=False):
    model.eval()
    probs_all, idx_all, y_all = [], [], []
    gate_means, joint_attn, window_attn = [], [], []
    with torch.no_grad():
        for local_idx, y in loader:
            raw = patient_window[local_idx].to(device).float()
            ref = provider.reference(raw, local_idx)
            aggregate = aggregate_features[local_idx].to(device).float()
            aggregate = (aggregate - agg_mean.to(device)) / agg_std.to(device)
            if collect_aux:
                logits, aux = model(raw, raw - ref, aggregate, return_aux=True)
                gate_means.append(aux["gate"].mean(dim=1).cpu().numpy())
                joint_attn.append(aux["joint_attn"].cpu().numpy())
                window_attn.append(aux["window_attn"].cpu().numpy())
            else:
                logits = model(raw, raw - ref, aggregate)
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
    aux_summary = {}
    if collect_aux and gate_means:
        gate = np.concatenate(gate_means, axis=0)
        ja = np.concatenate(joint_attn, axis=0)
        wa = np.concatenate(window_attn, axis=0)
        aux_summary["gate_mean"] = float(gate.mean())
        aux_summary["gate_std"] = float(gate.std())
        aux_summary.update({f"joint_attn_{i:02d}": float(ja[:, i].mean()) for i in range(ja.shape[1])})
        aux_summary.update({f"window_attn_{i:02d}": float(wa[:, i].mean()) for i in range(wa.shape[1])})
    return local_idx, y_raw, probs, preds_raw, aux_summary


def train_one_fold(opts, provider, patient_window, aggregate_features, train_idx, val_idx, test_idx, local_indices, y_encoded, classes, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_local_indices = local_indices[train_idx]
    agg_mean, agg_std = normalize_aggregate(aggregate_features, train_local_indices)
    model = HybridWJCClassifier(
        opts.current_model,
        len(classes),
        agg_in_dim=aggregate_features.shape[1],
        dim=opts.dim,
        heads=opts.heads,
        layers=opts.layers,
        agg_dim=opts.agg_dim,
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
    from sklearn.metrics import f1_score

    best_state = None
    best_score = -1.0
    wait = 0
    for _epoch in range(opts.epochs):
        model.train()
        for local_idx, y in train_loader:
            raw = patient_window[local_idx].to(device).float()
            ref = provider.reference(raw, local_idx)
            aggregate = aggregate_features[local_idx].to(device).float()
            aggregate = (aggregate - agg_mean.to(device)) / agg_std.to(device)
            logits = model(raw, raw - ref, aggregate)
            loss = criterion(logits, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        _, val_y_raw, val_probs, val_preds, _ = evaluate_model(
            model, provider, patient_window, aggregate_features, agg_mean, agg_std, val_loader, classes, device
        )
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
    test_local_idx, test_y_raw, test_probs, test_preds, aux = evaluate_model(
        model, provider, patient_window, aggregate_features, agg_mean, agg_std, test_loader, classes, device, collect_aux=True
    )
    return best_score, test_local_idx, test_y_raw, test_probs, test_preds, aux


def run_config(opts, rows, patient_window, patient_indices, aggregate_features, task):
    local_indices, y_raw, y_encoded, groups, eval_groups, classes = prepare_task(rows, patient_indices, task)
    splits = iter_group_splits(y_raw, groups, opts.n_splits, opts.seed)
    if opts.max_folds is not None:
        splits = splits[: opts.max_folds]
    provider = PrototypeReferenceProvider(
        opts.prototype_file,
        opts.bank,
        opts.prototype_top_k,
        opts.temperature,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    fold_rows = []
    for fold, (outer_train, test_idx) in enumerate(splits, start=1):
        train_local, val_local = validation_split(y_raw[outer_train], groups[outer_train], opts.seed + fold)
        train_idx = outer_train[train_local]
        val_idx = outer_train[val_local]
        if len(set(y_raw[train_idx].tolist())) < len(classes) or len(set(y_raw[test_idx].tolist())) < 2:
            continue
        best_val, test_local_idx, test_y_raw, test_probs, test_preds, aux = train_one_fold(
            opts,
            provider,
            patient_window,
            aggregate_features,
            train_idx,
            val_idx,
            test_idx,
            local_indices,
            y_encoded,
            classes,
            opts.seed + fold,
        )
        row = {
            "bank": opts.bank,
            "model": opts.current_model,
            "task": task,
            "fold": fold,
            "best_val_macro_f1": best_val,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "classes": "|".join(str(int(c)) for c in classes),
        }
        row.update(metric_dict(test_y_raw, test_probs, test_preds, classes, "clip"))
        gy, gp, gpred = aggregate_groups(test_y_raw, test_probs, eval_groups[test_idx], classes)
        row.update(metric_dict(gy, gp, gpred, classes, "group"))
        row.update(aux)
        fold_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return fold_rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["bank"], row["model"], row["task"])].append(row)
    metric_keys = sorted(
        key
        for key in rows[0]
        if key.endswith(("accuracy", "balanced_acc", "macro_f1", "weighted_f1", "auroc", "macro_auroc_ovr"))
        or key in {"gate_mean", "gate_std"}
        or key.startswith("joint_attn_")
        or key.startswith("window_attn_")
    )
    out_rows = []
    for (bank, model, task), items in grouped.items():
        out = {"bank": bank, "model": model, "task": task, "n_folds": len(items)}
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


def write_markdown(path, rows):
    lines = [
        "# Hybrid Gated WJC + Aggregate Classifier",
        "",
        "Cells are group-level `Macro-F1 / Balanced Accuracy`; binary tasks include AUROC.",
        "",
    ]
    for task in sorted({r["task"] for r in rows}):
        lines.append(f"## {task}")
        lines.append("")
        lines.append("| Bank | Model | Macro-F1 | BalAcc | AUROC | Gate mean |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        sub = sorted([r for r in rows if r["task"] == task], key=lambda r: r.get("group_macro_f1_mean", -1), reverse=True)
        for row in sub:
            auroc = row.get("group_auroc_mean", row.get("group_macro_auroc_ovr_mean", ""))
            auroc_text = "" if auroc == "" else f"{float(auroc):.4f}"
            gate = row.get("gate_mean_mean", "")
            gate_text = "" if gate == "" else f"{float(gate):.4f}"
            lines.append(
                f"| {row['bank']} | {row['model']} | "
                f"{float(row.get('group_macro_f1_mean', float('nan'))):.4f} | "
                f"{float(row.get('group_balanced_acc_mean', float('nan'))):.4f} | {auroc_text} | {gate_text} |"
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
    aggregate_features, aggregate_cols = load_aggregate_features(opts.aggregate_file)
    print(
        json.dumps(
            {
                "patient_window_shape": list(patient_window.shape),
                "aggregate_shape": list(aggregate_features.shape),
                "aggregate_cols": len(aggregate_cols),
                "bank": opts.bank,
                "models": opts.models,
                "tasks": opts.tasks,
            },
            indent=2,
        ),
        flush=True,
    )
    (output_dir / "aggregate_columns.json").write_text(
        json.dumps(aggregate_cols, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    all_rows = []
    for model_name in opts.models:
        opts.current_model = model_name
        for task in opts.tasks:
            print(f"Running bank={opts.bank} model={model_name} task={task}", flush=True)
            all_rows.extend(run_config(opts, rows, patient_window, patient_indices, aggregate_features, task))
            summary_rows = summarize(all_rows)
            write_csv(output_dir / "wjc_hybrid_folds.csv", all_rows)
            write_csv(output_dir / "wjc_hybrid_summary.csv", summary_rows)
            write_markdown(output_dir / "wjc_hybrid_summary.md", summary_rows)
    (output_dir / "summary.json").write_text(json.dumps(vars(opts), ensure_ascii=False, indent=2, default=str) + "\n")
    print(f"Wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
