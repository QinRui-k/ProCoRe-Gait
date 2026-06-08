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

from tools.train_aggregate_residual_classifier import feature_sets  # noqa: E402
from tools.train_wjc_gated_classifier import GatedWJCClassifier, evaluate_model as evaluate_gated  # noqa: E402
from tools.train_wjc_hybrid_classifier import (  # noqa: E402
    HybridWJCClassifier,
    evaluate_model as evaluate_hybrid,
    load_aggregate_features,
    normalize_aggregate,
)
from tools.train_wjc_tensor_classifier import (  # noqa: E402
    IndexDataset,
    PrototypeReferenceProvider,
    WJCTensorClassifier,
    aggregate_groups,
    class_weights,
    metric_dict,
    prepare_task,
    read_metadata,
    write_csv,
)


TASK = "pdgait_score_3class"


def parse_args():
    parser = argparse.ArgumentParser(description="DPPD-aligned PDGait 3-class scoring experiment.")
    parser.add_argument("--features-dir", default="features/e1_patient_dynamic_anchor_clip")
    parser.add_argument("--patient-window-file", default="features/window_joint_e1/patient_window_joint.pt")
    parser.add_argument("--prototype-file", default="eval/frozen_bank_source_ablation_e1/patient_control/window_prototypes.pt")
    parser.add_argument("--aggregate-file", default="eval/frozen_bank_source_ablation_e1/patient_control/prototype_M64_samples.csv")
    parser.add_argument("--output-dir", default="eval/pdgait_score3_dppd_aligned_e1")
    parser.add_argument("--bank", default="prototype_M64")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260518, 20260519, 20260520, 20260521, 20260522])
    parser.add_argument("--aggregate-features", nargs="+", default=["r_total", "joint_part", "all_interpretable"])
    parser.add_argument("--tensor-modes", nargs="+", default=["raw", "abs_residual", "raw_plus_abs_residual"])
    parser.add_argument("--gated-models", nargs="+", default=["gated_token", "gated_structured"])
    parser.add_argument("--hybrid-models", nargs="+", default=["hybrid_token_all"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
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
    parser.add_argument("--c-values", nargs="+", type=float, default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    parser.add_argument("--max-iter", type=int, default=8000)
    parser.add_argument(
        "--protocol",
        choices=["loso22", "with0_testonly", "with0_release", "random5"],
        default="loso22",
        help="Subject split protocol. random5 uses StratifiedGroupKFold over subjects.",
    )
    parser.add_argument("--split-seed", type=int, default=20260520)
    parser.add_argument("--n-random-splits", type=int, default=5)
    parser.add_argument("--state-filter", choices=["all", "on", "off"], default="all")
    parser.add_argument("--max-folds", type=int, default=None)
    return parser.parse_args()


def active_items(items):
    return [item for item in items if str(item).lower() not in {"none", "skip", "off"}]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_state(row):
    text = " ".join(str(row.get(key, "")) for key in ("state", "state_label", "video_name", "path", "source")).lower()
    if "_off_" in text or "/off/" in text:
        return "off"
    if "_on_" in text or "/on/" in text:
        return "on"
    tokens = text.replace("/", "_").replace("-", "_").split("_")
    if "off" in tokens:
        return "off"
    if "on" in tokens:
        return "on"
    return "unknown"


def apply_state_filter(opts, rows, patient_indices, local_indices, y_raw, groups, eval_groups):
    if opts.state_filter == "all":
        return local_indices, y_raw, groups, eval_groups
    keep = []
    for i, local_i in enumerate(local_indices):
        global_i = int(patient_indices[int(local_i)])
        keep.append(infer_state(rows[global_i]) == opts.state_filter)
    keep = np.asarray(keep, dtype=bool)
    return local_indices[keep], y_raw[keep], groups[keep], eval_groups[keep]


def encode_labels(y_raw):
    classes = np.asarray(sorted(set(y_raw.tolist())), dtype=np.int64)
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    y_encoded = np.asarray([class_to_idx[int(y)] for y in y_raw], dtype=np.int64)
    return y_encoded, classes


def group_accuracy(y_true, probs, groups, classes):
    from sklearn.metrics import accuracy_score

    gy, gp, gpred = aggregate_groups(y_true, probs, groups, classes)
    return float(accuracy_score(gy, gpred))


def aggregate_groups_named(y_true, probs, groups, classes):
    buckets = defaultdict(lambda: {"probs": [], "labels": []})
    for y, prob, group in zip(y_true, probs, groups):
        item = buckets[(group, int(y))]
        item["labels"].append(int(y))
        item["probs"].append(prob)
    out_groups, agg_y, agg_probs = [], [], []
    for (group, _label), item in buckets.items():
        out_groups.append(group)
        agg_y.append(int(round(sum(item["labels"]) / len(item["labels"]))))
        agg_probs.append(np.mean(np.stack(item["probs"], axis=0), axis=0))
    agg_y = np.asarray(agg_y, dtype=np.int64)
    agg_probs = np.stack(agg_probs, axis=0)
    agg_preds = np.asarray([classes[i] for i in np.argmax(agg_probs, axis=1)], dtype=np.int64)
    return np.asarray(out_groups), agg_y, agg_probs, agg_preds


WITH0_SUBJECTS = ["SUB18", "SUB20", "SUB22", "SUB24"]


def pd_group(subject):
    return f"pdgait_subject:{subject}"


def dppd_loso_splits(y_raw, groups, heldout_groups=None, train_pool_groups=None):
    uniq = sorted(set(groups.tolist()))
    heldouts = sorted(set(heldout_groups or uniq))
    train_pool = set(train_pool_groups or uniq)
    all_classes = set(y_raw.tolist())
    splits = []
    for group in heldouts:
        if group not in set(uniq):
            continue
        test_idx = np.asarray([i for i, g in enumerate(groups) if g == group], dtype=np.int64)
        train_idx = np.asarray([i for i, g in enumerate(groups) if g != group and g in train_pool], dtype=np.int64)
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        if len(set(y_raw[train_idx].tolist())) < len(all_classes):
            continue
        splits.append((group, train_idx, test_idx))
    return splits


def random_group_splits(y_raw, groups, n_splits, split_seed):
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    all_classes = set(y_raw.tolist())
    splits = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y_raw)), y_raw, groups), start=1):
        if len(set(y_raw[train_idx].tolist())) < len(all_classes):
            continue
        splits.append((f"random{split_seed}_fold{fold}", train_idx.astype(np.int64), test_idx.astype(np.int64)))
    return splits


def build_splits(opts, y_raw, groups):
    if opts.protocol == "loso22":
        splits = dppd_loso_splits(y_raw, groups)
    elif opts.protocol == "with0_testonly":
        splits = dppd_loso_splits(y_raw, groups, heldout_groups=[pd_group(s) for s in WITH0_SUBJECTS])
    elif opts.protocol == "with0_release":
        with0 = [pd_group(s) for s in WITH0_SUBJECTS]
        splits = dppd_loso_splits(y_raw, groups, heldout_groups=with0, train_pool_groups=with0)
    elif opts.protocol == "random5":
        splits = random_group_splits(y_raw, groups, opts.n_random_splits, opts.split_seed)
    else:
        raise ValueError(f"Unknown protocol: {opts.protocol}")
    if opts.max_folds is not None:
        splits = splits[: opts.max_folds]
    return splits


def global_report(y_true, probs, groups, classes):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    gy, gp, gpred = aggregate_groups(y_true, probs, groups, classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gy, gpred, average="weighted", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        gy, gpred, average="macro", zero_division=0
    )
    out = {
        "n_videos": int(len(gy)),
        "accuracy": float(accuracy_score(gy, gpred)),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "confusion_matrix": json.dumps(confusion_matrix(gy, gpred, labels=classes).tolist()),
    }
    out.update(metric_dict(gy, gp, gpred, classes, "group"))
    return out, gy, gp, gpred


def align_probs(probs_raw, prob_classes, classes):
    probs = np.zeros((len(probs_raw), len(classes)), dtype=np.float64)
    for src_col, cls in enumerate(prob_classes):
        dst = np.where(classes == cls)[0]
        if len(dst):
            probs[:, dst[0]] = probs_raw[:, src_col]
    return probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)


def load_aggregate_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    by_local = {int(row["local_index"]): row for row in rows}
    return rows, by_local


def run_aggregate_feature(opts, rows_by_local, local_indices, y_raw, groups, eval_groups, classes, feature_name, cols, seed, splits):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(
        [[float(rows_by_local[int(local_i)][c]) for c in cols] for local_i in local_indices],
        dtype=np.float64,
    )
    fold_rows, pred_rows = [], []
    for fold, (heldout_group, train_idx, test_idx) in enumerate(splits, start=1):
        best_c, best_acc, best_probs = opts.c_values[0], -1.0, None
        for c in opts.c_values:
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c, max_iter=opts.max_iter, class_weight="balanced", solver="lbfgs"),
            )
            clf.fit(x[train_idx], y_raw[train_idx])
            probs = align_probs(clf.predict_proba(x[test_idx]), clf.named_steps["logisticregression"].classes_, classes)
            acc = group_accuracy(y_raw[test_idx], probs, eval_groups[test_idx], classes)
            if acc > best_acc:
                best_acc = acc
                best_c = c
                best_probs = probs
        preds = np.asarray([classes[i] for i in np.argmax(best_probs, axis=1)], dtype=np.int64)
        row = {
            "family": "aggregate",
            "model": feature_name,
            "protocol": opts.protocol,
            "split_seed": opts.split_seed,
            "state_filter": opts.state_filter,
            "seed": seed,
            "fold": fold,
            "heldout_group": heldout_group,
            "best_val_group_accuracy": best_acc,
            "best_c": best_c,
            "n_features": len(cols),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
        row.update(metric_dict(y_raw[test_idx], best_probs, preds, classes, "clip"))
        gnames, gy, gp, gpred = aggregate_groups_named(y_raw[test_idx], best_probs, eval_groups[test_idx], classes)
        row.update(metric_dict(gy, gp, gpred, classes, "group"))
        fold_rows.append(row)
        for group, yy, pp, pred in zip(gnames, gy, gp, gpred):
            pred_rows.append(prediction_row(opts.protocol, opts.split_seed, opts.state_filter, "aggregate", feature_name, seed, fold, group, yy, pp, pred, classes))
    return fold_rows, pred_rows


def prediction_row(protocol, split_seed, state_filter, family, model, seed, fold, group, y_true, probs, pred, classes):
    row = {
        "protocol": protocol,
        "split_seed": split_seed,
        "state_filter": state_filter,
        "family": family,
        "model": model,
        "seed": seed,
        "fold": fold,
        "eval_group": group,
        "y_true": int(y_true),
        "y_pred": int(pred),
    }
    for i, cls in enumerate(classes):
        row[f"prob_{int(cls)}"] = float(probs[i])
    return row


def train_neural_fold(
    opts,
    model,
    provider,
    patient_window,
    aggregate_features,
    agg_mean,
    agg_std,
    train_idx,
    test_idx,
    local_indices,
    y_encoded,
    y_raw,
    eval_groups,
    classes,
    device,
    family,
    is_hybrid,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_encoded[train_idx], len(classes)).to(device))
    train_loader = DataLoader(
        IndexDataset(local_indices[train_idx], y_encoded[train_idx]),
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        IndexDataset(local_indices[test_idx], y_encoded[test_idx]),
        batch_size=opts.eval_batch_size,
        shuffle=False,
        num_workers=0,
    )
    best_state, best_acc, wait = None, -1.0, 0
    for _epoch in range(opts.epochs):
        model.train()
        for batch_local_idx, y in train_loader:
            raw = patient_window[batch_local_idx].to(device).float()
            ref = provider.reference(raw, batch_local_idx)
            if is_hybrid:
                aggregate = aggregate_features[batch_local_idx].to(device).float()
                aggregate = (aggregate - agg_mean.to(device)) / agg_std.to(device)
                logits = model(raw, raw - ref, aggregate)
            else:
                logits = model(raw, raw - ref)
            loss = criterion(logits, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if is_hybrid:
            _, val_y, val_probs, _, _ = evaluate_hybrid(
                model, provider, patient_window, aggregate_features, agg_mean, agg_std, test_loader, classes, device
            )
        elif family == "tensor":
            _, val_y, val_probs, _ = evaluate_tensor(model, provider, patient_window, test_loader, classes, device)
        else:
            _, val_y, val_probs, _, _ = evaluate_gated(model, provider, patient_window, test_loader, classes, device)
        val_acc = group_accuracy(val_y, val_probs, eval_groups[test_idx], classes)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= opts.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    if is_hybrid:
        _, test_y, test_probs, test_preds, aux = evaluate_hybrid(
            model, provider, patient_window, aggregate_features, agg_mean, agg_std, test_loader, classes, device, collect_aux=True
        )
    elif family == "tensor":
        _, test_y, test_probs, test_preds = evaluate_tensor(model, provider, patient_window, test_loader, classes, device)
        aux = {}
    else:
        _, test_y, test_probs, test_preds, aux = evaluate_gated(
            model, provider, patient_window, test_loader, classes, device, collect_aux=True
        )
    return best_acc, test_y, test_probs, test_preds, aux


def evaluate_tensor(model, provider, patient_window, loader, classes, device):
    model.eval()
    probs_all, idx_all, y_all = [], [], []
    with torch.no_grad():
        for local_idx, y in loader:
            raw = patient_window[local_idx].to(device).float()
            ref = provider.reference(raw, local_idx)
            logits = model(raw, raw - ref)
            probs_all.append(torch.softmax(logits, dim=1).cpu().numpy())
            idx_all.append(local_idx.numpy())
            y_all.append(y.numpy())
    probs = np.concatenate(probs_all, axis=0)
    local_idx = np.concatenate(idx_all, axis=0)
    y_encoded = np.concatenate(y_all, axis=0)
    preds_encoded = probs.argmax(axis=1)
    y_raw = np.asarray([classes[i] for i in y_encoded], dtype=np.int64)
    preds_raw = np.asarray([classes[i] for i in preds_encoded], dtype=np.int64)
    return local_idx, y_raw, probs, preds_raw


def run_neural_model(
    opts,
    family,
    model_name,
    provider,
    patient_window,
    aggregate_features,
    local_indices,
    y_raw,
    y_encoded,
    groups,
    eval_groups,
    classes,
    seed,
    splits,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_rows, pred_rows = [], []
    for fold, (heldout_group, train_idx, test_idx) in enumerate(splits, start=1):
        set_seed(seed + fold)
        if family == "tensor":
            model = WJCTensorClassifier(
                model_name, len(classes), dim=opts.dim, heads=opts.heads, layers=opts.layers, dropout=opts.dropout
            ).to(device)
            agg_mean = agg_std = None
            is_hybrid = False
        elif family == "gated":
            model = GatedWJCClassifier(
                model_name, len(classes), dim=opts.dim, heads=opts.heads, layers=opts.layers, dropout=opts.dropout
            ).to(device)
            agg_mean = agg_std = None
            is_hybrid = False
        elif family == "hybrid":
            train_local_indices = local_indices[train_idx]
            agg_mean, agg_std = normalize_aggregate(aggregate_features, train_local_indices)
            model = HybridWJCClassifier(
                model_name,
                len(classes),
                agg_in_dim=aggregate_features.shape[1],
                dim=opts.dim,
                heads=opts.heads,
                layers=opts.layers,
                agg_dim=opts.agg_dim,
                dropout=opts.dropout,
            ).to(device)
            is_hybrid = True
        else:
            raise ValueError(family)
        best_acc, test_y, test_probs, test_preds, aux = train_neural_fold(
            opts,
            model,
            provider,
            patient_window,
            aggregate_features,
            agg_mean,
            agg_std,
            train_idx,
            test_idx,
            local_indices,
            y_encoded,
            y_raw,
            eval_groups,
            classes,
            device,
            family,
            is_hybrid,
        )
        row = {
            "family": family,
            "model": model_name,
            "protocol": opts.protocol,
            "split_seed": opts.split_seed,
            "state_filter": opts.state_filter,
            "seed": seed,
            "fold": fold,
            "heldout_group": heldout_group,
            "best_val_group_accuracy": best_acc,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
        row.update(metric_dict(test_y, test_probs, test_preds, classes, "clip"))
        gnames, gy, gp, gpred = aggregate_groups_named(test_y, test_probs, eval_groups[test_idx], classes)
        row.update(metric_dict(gy, gp, gpred, classes, "group"))
        row.update(aux)
        fold_rows.append(row)
        for group, yy, pp, pred in zip(gnames, gy, gp, gpred):
            pred_rows.append(prediction_row(opts.protocol, opts.split_seed, opts.state_filter, family, model_name, seed, fold, group, yy, pp, pred, classes))
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return fold_rows, pred_rows


def summarize_from_predictions(pred_rows, classes):
    grouped = defaultdict(list)
    for row in pred_rows:
        grouped[(row.get("state_filter", "all"), row["protocol"], row["split_seed"], row["family"], row["model"], row["seed"])].append(row)
    seed_rows = []
    for (state_filter, protocol, split_seed, family, model, seed), rows in grouped.items():
        y_true = np.asarray([int(r["y_true"]) for r in rows], dtype=np.int64)
        probs = np.asarray([[float(r[f"prob_{int(c)}"]) for c in classes] for r in rows], dtype=np.float64)
        eval_groups = np.asarray([r["eval_group"] for r in rows])
        report, _, _, _ = global_report(y_true, probs, eval_groups, classes)
        report.update({"state_filter": state_filter, "protocol": protocol, "split_seed": split_seed, "family": family, "model": model, "seed": seed})
        seed_rows.append(report)

    summary = []
    grouped_seed = defaultdict(list)
    for row in seed_rows:
        grouped_seed[(row.get("state_filter", "all"), row["protocol"], row["split_seed"], row["family"], row["model"])].append(row)
    metric_keys = [k for k in seed_rows[0].keys() if k not in {"state_filter", "protocol", "split_seed", "family", "model", "seed", "confusion_matrix"}]
    for (state_filter, protocol, split_seed, family, model), rows in grouped_seed.items():
        item = {"state_filter": state_filter, "protocol": protocol, "split_seed": split_seed, "family": family, "model": model, "n_seeds": len(rows)}
        for key in metric_keys:
            vals = [float(r[key]) for r in rows if r.get(key) not in ("", None) and not math.isnan(float(r[key]))]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                item[f"{key}_mean"] = float(arr.mean())
                item[f"{key}_std"] = float(arr.std(ddof=1) if len(arr) > 1 else 0.0)
        summary.append(item)
    return seed_rows, summary


def write_markdown(path, summary_rows):
    rows = sorted(summary_rows, key=lambda r: r.get("weighted_f1_mean", -1), reverse=True)
    lines = [
        "# PDGait Score 3-Class, DPPD-Aligned",
        "",
        "Protocol: leave-one-subject-out, video-level aggregation, and test fold used as validation for best epoch / hyperparameter selection.",
        "",
        "| State | Protocol | Split seed | Family | Model | Acc | Weighted P | Weighted R | Weighted F1 | Macro F1 | BalAcc |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('state_filter', 'all')} | {row.get('protocol', '')} | {row.get('split_seed', '')} | {row['family']} | {row['model']} | "
            f"{row.get('accuracy_mean', float('nan')):.4f} | "
            f"{row.get('weighted_precision_mean', float('nan')):.4f} | "
            f"{row.get('weighted_recall_mean', float('nan')):.4f} | "
            f"{row.get('weighted_f1_mean', float('nan')):.4f} | "
            f"{row.get('macro_f1_mean', float('nan')):.4f} | "
            f"{row.get('group_balanced_acc_mean', float('nan')):.4f} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    opts = parse_args()
    opts.aggregate_features = active_items(opts.aggregate_features)
    opts.tensor_modes = active_items(opts.tensor_modes)
    opts.gated_models = active_items(opts.gated_models)
    opts.hybrid_models = active_items(opts.hybrid_models)
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_metadata(opts.features_dir)
    patient_data = torch.load(opts.patient_window_file, map_location="cpu")
    patient_window = patient_data["window_joint_features"]
    patient_indices = patient_data["global_indices"].long()
    local_indices, y_raw, y_encoded, groups, eval_groups, classes = prepare_task(rows, patient_indices, TASK)
    local_indices, y_raw, groups, eval_groups = apply_state_filter(
        opts, rows, patient_indices, local_indices, y_raw, groups, eval_groups
    )
    y_encoded, classes = encode_labels(y_raw)
    splits = build_splits(opts, y_raw, groups)

    aggregate_rows, rows_by_local = load_aggregate_rows(opts.aggregate_file)
    sets = feature_sets(aggregate_rows[0].keys())
    aggregate_features, aggregate_cols = load_aggregate_features(opts.aggregate_file)
    provider = PrototypeReferenceProvider(
        opts.prototype_file,
        opts.bank,
        opts.prototype_top_k,
        opts.temperature,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    print(
        json.dumps(
            {
                "task": TASK,
                "state_filter": opts.state_filter,
                "protocol": opts.protocol,
                "split_seed": opts.split_seed,
                "samples": int(len(y_raw)),
                "subjects": int(len(set(groups.tolist()))),
                "videos": int(len(set(eval_groups.tolist()))),
                "classes": [int(c) for c in classes],
                "patient_window_shape": list(patient_window.shape),
                "aggregate_cols": len(aggregate_cols),
                "folds": len(splits),
                "fold_names": [name for name, _, _ in splits],
                "seeds": opts.seeds,
            },
            indent=2,
        ),
        flush=True,
    )

    all_fold_rows, all_pred_rows = [], []
    for seed in opts.seeds:
        for feature_name in opts.aggregate_features:
            cols = sets.get(feature_name)
            if not cols:
                raise ValueError(f"Unknown aggregate feature set: {feature_name}")
            print(f"Running aggregate/{feature_name} seed={seed}", flush=True)
            fold_rows, pred_rows = run_aggregate_feature(
                opts, rows_by_local, local_indices, y_raw, groups, eval_groups, classes, feature_name, cols, seed, splits
            )
            all_fold_rows.extend(fold_rows)
            all_pred_rows.extend(pred_rows)
            flush_outputs(output_dir, all_fold_rows, all_pred_rows, classes)
        for mode in opts.tensor_modes:
            print(f"Running tensor/{mode} seed={seed}", flush=True)
            fold_rows, pred_rows = run_neural_model(
                opts, "tensor", mode, provider, patient_window, aggregate_features, local_indices, y_raw,
                y_encoded, groups, eval_groups, classes, seed, splits
            )
            all_fold_rows.extend(fold_rows)
            all_pred_rows.extend(pred_rows)
            flush_outputs(output_dir, all_fold_rows, all_pred_rows, classes)
        for model_name in opts.gated_models:
            print(f"Running gated/{model_name} seed={seed}", flush=True)
            fold_rows, pred_rows = run_neural_model(
                opts, "gated", model_name, provider, patient_window, aggregate_features, local_indices, y_raw,
                y_encoded, groups, eval_groups, classes, seed, splits
            )
            all_fold_rows.extend(fold_rows)
            all_pred_rows.extend(pred_rows)
            flush_outputs(output_dir, all_fold_rows, all_pred_rows, classes)
        for model_name in opts.hybrid_models:
            print(f"Running hybrid/{model_name} seed={seed}", flush=True)
            fold_rows, pred_rows = run_neural_model(
                opts, "hybrid", model_name, provider, patient_window, aggregate_features, local_indices, y_raw,
                y_encoded, groups, eval_groups, classes, seed, splits
            )
            all_fold_rows.extend(fold_rows)
            all_pred_rows.extend(pred_rows)
            flush_outputs(output_dir, all_fold_rows, all_pred_rows, classes)

    flush_outputs(output_dir, all_fold_rows, all_pred_rows, classes)
    (output_dir / "args.json").write_text(json.dumps(vars(opts), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote results to {output_dir}", flush=True)


def flush_outputs(output_dir, fold_rows, pred_rows, classes):
    if fold_rows:
        write_csv(output_dir / "dppd_aligned_folds.csv", fold_rows)
    if pred_rows:
        write_csv(output_dir / "dppd_aligned_video_predictions.csv", pred_rows)
        seed_rows, summary_rows = summarize_from_predictions(pred_rows, classes)
        write_csv(output_dir / "dppd_aligned_seed_summary.csv", seed_rows)
        write_csv(output_dir / "dppd_aligned_summary.csv", summary_rows)
        write_markdown(output_dir / "dppd_aligned_summary.md", summary_rows)


if __name__ == "__main__":
    main()
