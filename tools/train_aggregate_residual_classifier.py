import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_TASKS = ["pdgait_binary", "pdgait_score_3class", "3dgait_binary", "3dgait_subtype_3class"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train no-pandas aggregate residual classifiers.")
    parser.add_argument("--samples", default="eval/frozen_bank_source_ablation_e1/patient_control/prototype_M64_samples.csv")
    parser.add_argument("--output-dir", default="eval/task_specific_head_suite_e1/aggregate_seed_20260518")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--c-values", nargs="+", type=float, default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    return parser.parse_args()


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


def feature_sets(columns):
    joint = [c for c in columns if c.startswith("joint_")]
    window = [c for c in columns if c.startswith("window_")]
    part = [c for c in columns if c.startswith("part_")]
    return {
        "r_total": ["r_total"],
        "joint": joint,
        "window": window,
        "part": part,
        "joint_part": joint + part,
        "window_part": window + part,
        "joint_window_part": joint + window + part,
        "all_interpretable": ["r_total"] + joint + window + part,
    }


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


def align_probs(probs_raw, prob_classes, classes):
    probs = np.zeros((len(probs_raw), len(classes)), dtype=np.float64)
    for src_col, cls in enumerate(prob_classes):
        dst = np.where(classes == cls)[0]
        if len(dst):
            probs[:, dst[0]] = probs_raw[:, src_col]
    return probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)


def choose_c_by_inner_cv(x_train, y_train, groups_train, c_values, max_iter, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score

    if len(set(groups_train.tolist())) < 3 or len(set(y_train.tolist())) < 2:
        return 1.0
    splits = iter_group_splits(y_train, groups_train, min(3, len(set(groups_train.tolist()))), seed + 13)
    best_c, best_score = c_values[0], -1.0
    for c in c_values:
        scores = []
        for inner_train, inner_val in splits:
            if len(set(y_train[inner_train].tolist())) < 2 or len(set(y_train[inner_val].tolist())) < 2:
                continue
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c, max_iter=max_iter, class_weight="balanced", solver="lbfgs"),
            )
            clf.fit(x_train[inner_train], y_train[inner_train])
            scores.append(balanced_accuracy_score(y_train[inner_val], clf.predict(x_train[inner_val])))
        if scores and float(np.mean(scores)) > best_score:
            best_score = float(np.mean(scores))
            best_c = c
    return best_c


def run_task_feature(rows, task, feature_name, feature_cols, opts):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    records = []
    for row in rows:
        label = task_label(row, task)
        if label is None:
            continue
        records.append({"row": row, "label": int(label), "split_group": split_group(row), "eval_group": eval_group(row)})
    if not records:
        return []
    y = np.asarray([r["label"] for r in records], dtype=np.int64)
    groups = np.asarray([r["split_group"] for r in records])
    eval_groups = np.asarray([r["eval_group"] for r in records])
    classes = np.asarray(sorted(set(y.tolist())), dtype=np.int64)
    if len(classes) < 2 or len(set(groups.tolist())) < 2:
        return []
    x = np.asarray([[float(r["row"][c]) for c in feature_cols] for r in records], dtype=np.float64)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(iter_group_splits(y, groups, opts.n_splits, opts.seed), start=1):
        if len(set(y[train_idx].tolist())) < 2 or len(set(y[test_idx].tolist())) < 2:
            continue
        best_c = choose_c_by_inner_cv(
            x[train_idx], y[train_idx], groups[train_idx], opts.c_values, opts.max_iter, opts.seed + fold
        )
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=best_c, max_iter=opts.max_iter, class_weight="balanced", solver="lbfgs"),
        )
        clf.fit(x[train_idx], y[train_idx])
        probs = align_probs(clf.predict_proba(x[test_idx]), clf.named_steps["logisticregression"].classes_, classes)
        preds = np.asarray([classes[i] for i in np.argmax(probs, axis=1)], dtype=np.int64)
        row = {
            "task": task,
            "feature": feature_name,
            "fold": fold,
            "best_c": best_c,
            "n_features": len(feature_cols),
            "classes": "|".join(str(int(c)) for c in classes),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
        row.update(metric_dict(y[test_idx], probs, preds, classes, "clip"))
        gy, gp, gpred = aggregate_groups(y[test_idx], probs, eval_groups[test_idx], classes)
        row.update(metric_dict(gy, gp, gpred, classes, "group"))
        fold_rows.append(row)
    return fold_rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["feature"])].append(row)
    metric_keys = sorted(
        key
        for key in rows[0]
        if key.endswith(("accuracy", "balanced_acc", "macro_f1", "weighted_f1", "auroc", "macro_auroc_ovr"))
    )
    out = []
    for (task, feature), items in grouped.items():
        item = {"task": task, "feature": feature, "n_folds": len(items), "n_features": items[0]["n_features"]}
        for metric in metric_keys:
            vals = []
            for row in items:
                value = row.get(metric)
                if value in ("", None):
                    continue
                value = float(value)
                if not math.isnan(value):
                    vals.append(value)
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                item[f"{metric}_mean"] = float(arr.mean())
                item[f"{metric}_std"] = float(arr.std(ddof=1) if len(arr) > 1 else 0.0)
        out.append(item)
    return out


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


def write_markdown(path, summary_rows):
    lines = [
        "# Aggregate Residual Classifier",
        "",
        "Cells are group-level `Macro-F1 / Balanced Accuracy`; binary tasks include AUROC.",
        "",
    ]
    for task in DEFAULT_TASKS:
        rows = [r for r in summary_rows if r["task"] == task]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r.get("group_macro_f1_mean", -1), reverse=True)
        lines.append(f"## {task}")
        lines.append("")
        lines.append("| Feature | n_features | Macro-F1 | BalAcc | AUROC |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in rows:
            auroc = row.get("group_auroc_mean", row.get("group_macro_auroc_ovr_mean", ""))
            auroc_text = "" if auroc == "" else f"{float(auroc):.4f}"
            lines.append(
                f"| {row['feature']} | {row['n_features']} | "
                f"{float(row.get('group_macro_f1_mean', float('nan'))):.4f} | "
                f"{float(row.get('group_balanced_acc_mean', float('nan'))):.4f} | {auroc_text} |"
            )
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    opts = parse_args()
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(opts.samples).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sets = feature_sets(rows[0].keys())
    all_rows = []
    for task in opts.tasks:
        for feature_name, cols in sets.items():
            if not cols:
                continue
            print(f"Running {task} / {feature_name} ({len(cols)} features)", flush=True)
            fold_rows = run_task_feature(rows, task, feature_name, cols, opts)
            all_rows.extend(fold_rows)
            if fold_rows:
                print(summarize(fold_rows)[0], flush=True)
    if not all_rows:
        raise RuntimeError("No fold rows produced.")
    write_csv(output_dir / "stageC_interpretable_folds.csv", all_rows)
    summary_rows = summarize(all_rows)
    write_csv(output_dir / "stageC_interpretable_summary.csv", summary_rows)
    write_markdown(output_dir / "stageC_interpretable_summary.md", summary_rows)
    (output_dir / "summary.json").write_text(
        json.dumps({"samples": opts.samples, "tasks": opts.tasks, "n_splits": opts.n_splits, "seed": opts.seed}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
