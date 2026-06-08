import argparse
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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.train_pdgait_score3_dppd_aligned import (  # noqa: E402
    TASK,
    aggregate_groups_named,
    apply_state_filter,
    build_splits,
    encode_labels,
    global_report,
    prediction_row,
    read_metadata,
    summarize_from_predictions,
    write_markdown,
)
from tools.train_wjc_gated_classifier import GatedWJCClassifier  # noqa: E402
from tools.train_wjc_tensor_classifier import IndexDataset, class_weights, metric_dict, prepare_task, write_csv  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="PDGait scoring with constrained counterfactual prototype attention.")
    parser.add_argument("--features-dir", default="features/e1_patient_dynamic_anchor_clip")
    parser.add_argument("--patient-window-file", default="features/window_joint_e1/patient_window_joint.pt")
    parser.add_argument("--prototype-file", default="eval/frozen_bank_source_ablation_e1/patient_control/window_prototypes.pt")
    parser.add_argument("--output-dir", default="eval/pdgait_cf_attention_mixed_loso22_s7_20260521")
    parser.add_argument("--bank", default="prototype_M64")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260518, 20260519, 20260520, 20260521, 20260522, 20260523, 20260524])
    parser.add_argument(
        "--task",
        choices=["pdgait_score_3class", "3dgait_binary", "3dgait_score_4class", "3dgait_subtype_3class"],
        default="pdgait_score_3class",
    )
    parser.add_argument(
        "--protocol",
        choices=["loso22", "with0_testonly", "with0_release", "random5", "dppd_random10", "dppd_code_random10", "group10"],
        default="loso22",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--n-random-splits", type=int, default=10)
    parser.add_argument(
        "--eval-unit",
        choices=["default", "dppd_clip", "split_group"],
        default="default",
        help="For 3DGait DPPD alignment, dppd_clip evaluates each pkl/clip independently, matching the original video_idx behavior.",
    )
    parser.add_argument("--state-filter", choices=["all", "on", "off"], default="all")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--ref-dim", type=int, default=64)
    parser.add_argument("--prototype-top-k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.12)
    parser.add_argument("--lambda-align", type=float, default=0.05)
    parser.add_argument("--lambda-entropy", type=float, default=0.005)
    parser.add_argument("--lambda-smooth", type=float, default=0.0)
    parser.add_argument("--model-name", default="cf_attn_gated_token")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save best lightweight classifier checkpoint for each seed/fold.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PrototypeAttentionReference(nn.Module):
    """Constrained Z_cf generator: values are frozen normal prototypes, only attention is learned."""

    def __init__(self, prototype_file, bank_name, ref_dim=64, prototype_top_k=4, temperature=0.12):
        super().__init__()
        data = torch.load(prototype_file, map_location="cpu")
        key = bank_name.replace("prototype_M", "")
        bank = data["banks"][key]
        prototypes = bank["prototypes"].float()
        self.register_buffer("prototypes", prototypes)
        in_channels = int(prototypes.shape[-1])
        self.q_proj = nn.Linear(in_channels, ref_dim, bias=False)
        self.k_proj = nn.Linear(in_channels, ref_dim, bias=False)
        self.prototype_top_k = min(int(prototype_top_k), int(prototypes.shape[1]))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(float(temperature)), dtype=torch.float32))

    def forward(self, raw, return_aux=False):
        prototypes = self.prototypes.to(raw.device)
        q = F.normalize(self.q_proj(raw.float()), dim=-1)
        k = F.normalize(self.k_proj(prototypes), dim=-1)
        temperature = self.log_temperature.exp().clamp(0.02, 1.0)
        logits = torch.einsum("bwjd,wmjd->bwjm", q, k) / temperature
        if self.prototype_top_k < logits.shape[-1]:
            top_logits, top_idx = torch.topk(logits, k=self.prototype_top_k, dim=-1)
            weights = torch.softmax(top_logits, dim=-1)
            proto_by_joint = prototypes.permute(0, 2, 1, 3).unsqueeze(0).expand(
                raw.shape[0], -1, -1, -1, -1
            )
            top_proto = torch.gather(
                proto_by_joint,
                dim=3,
                index=top_idx.unsqueeze(-1).expand(-1, -1, -1, -1, prototypes.shape[-1]),
            )
            ref = (weights.unsqueeze(-1) * top_proto).sum(dim=3)
        else:
            weights = torch.softmax(logits, dim=-1)
            ref = torch.einsum("bwjm,wmjc->bwjc", weights, prototypes)
        if not return_aux:
            return ref
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / math.log(weights.shape[-1])
        aux = {
            "cf_entropy": entropy.mean(),
            "cf_top1": weights.max(dim=-1).values.mean(),
            "cf_temperature": temperature.detach(),
        }
        return ref, aux


class CounterfactualGatedClassifier(nn.Module):
    def __init__(self, opts, out_dim):
        super().__init__()
        self.reference = PrototypeAttentionReference(
            opts.prototype_file,
            opts.bank,
            ref_dim=opts.ref_dim,
            prototype_top_k=opts.prototype_top_k,
            temperature=opts.temperature,
        )
        self.classifier = GatedWJCClassifier(
            "gated_token",
            out_dim,
            dim=opts.dim,
            heads=opts.heads,
            layers=opts.layers,
            dropout=opts.dropout,
        )

    def forward(self, raw, return_aux=False):
        ref, ref_aux = self.reference(raw, return_aux=True)
        if return_aux:
            logits, cls_aux = self.classifier(raw, raw - ref, return_aux=True)
            aux = {}
            aux.update(cls_aux)
            aux.update(ref_aux)
            return logits, ref, aux
        logits = self.classifier(raw, raw - ref)
        return logits, ref, ref_aux


def normal_align_loss(raw, ref, y_encoded, classes):
    raw_classes = torch.as_tensor(classes, device=y_encoded.device, dtype=torch.long)[y_encoded]
    normal = raw_classes == 0
    if not torch.any(normal):
        return raw.new_tensor(0.0)
    return F.mse_loss(ref[normal], raw[normal])


def smooth_loss(ref):
    if ref.shape[1] < 2:
        return ref.new_tensor(0.0)
    return (ref[:, 1:] - ref[:, :-1]).pow(2).mean()


def evaluate_model(model, patient_window, loader, classes, device, collect_aux=False):
    model.eval()
    probs_all, idx_all, y_all = [], [], []
    aux_accum = defaultdict(list)
    with torch.no_grad():
        for local_idx, y in loader:
            raw = patient_window[local_idx].to(device).float()
            if collect_aux:
                logits, _ref, aux = model(raw, return_aux=True)
                if "gate" in aux:
                    aux_accum["gate_mean"].append(aux["gate"].mean(dim=1).detach().cpu().numpy())
                if "joint_attn" in aux:
                    aux_accum["joint_attn"].append(aux["joint_attn"].detach().cpu().numpy())
                if "window_attn" in aux:
                    aux_accum["window_attn"].append(aux["window_attn"].detach().cpu().numpy())
                aux_accum["cf_entropy"].append(np.asarray([float(aux["cf_entropy"].detach().cpu())]))
                aux_accum["cf_top1"].append(np.asarray([float(aux["cf_top1"].detach().cpu())]))
                aux_accum["cf_temperature"].append(np.asarray([float(aux["cf_temperature"].detach().cpu())]))
            else:
                logits, _ref, _aux = model(raw)
            probs_all.append(torch.softmax(logits, dim=1).cpu().numpy())
            idx_all.append(local_idx.numpy())
            y_all.append(y.numpy())
    probs = np.concatenate(probs_all, axis=0)
    local_idx = np.concatenate(idx_all, axis=0)
    y_encoded = np.concatenate(y_all, axis=0)
    preds_encoded = probs.argmax(axis=1)
    y_raw = np.asarray([classes[i] for i in y_encoded], dtype=np.int64)
    preds_raw = np.asarray([classes[i] for i in preds_encoded], dtype=np.int64)
    aux_out = {}
    if collect_aux:
        if aux_accum.get("gate_mean"):
            aux_out["gate_mean"] = float(np.concatenate(aux_accum["gate_mean"]).mean())
            aux_out["gate_std"] = float(np.concatenate(aux_accum["gate_mean"]).std())
        if aux_accum.get("joint_attn"):
            joint = np.concatenate(aux_accum["joint_attn"], axis=0).mean(axis=0)
            for i, value in enumerate(joint.tolist()):
                aux_out[f"joint_attn_{i:02d}"] = float(value)
        if aux_accum.get("window_attn"):
            window = np.concatenate(aux_accum["window_attn"], axis=0).mean(axis=0)
            for i, value in enumerate(window.tolist()):
                aux_out[f"window_attn_{i:02d}"] = float(value)
        for key in ("cf_entropy", "cf_top1", "cf_temperature"):
            if aux_accum.get(key):
                aux_out[key] = float(np.mean(np.concatenate(aux_accum[key])))
    return local_idx, y_raw, probs, preds_raw, aux_out


def group_accuracy(y_true, probs, groups, classes):
    from sklearn.metrics import accuracy_score
    from tools.train_wjc_tensor_classifier import aggregate_groups

    gy, _gp, gpred = aggregate_groups(y_true, probs, groups, classes)
    return float(accuracy_score(gy, gpred))


def dppd_random_splits(y_raw, n_splits, split_seed):
    """DPPD-style repeated random 90/10 clip splits, without train/test overlap."""
    all_idx = np.arange(len(y_raw), dtype=np.int64)
    test_size = max(1, int(round(len(all_idx) / 10)))
    all_classes = set(y_raw.tolist())
    splits = []
    for fold in range(1, n_splits + 1):
        rng = np.random.default_rng(int(split_seed) + fold)
        test_idx = np.sort(rng.choice(all_idx, size=test_size, replace=False)).astype(np.int64)
        mask = np.ones(len(all_idx), dtype=bool)
        mask[test_idx] = False
        train_idx = all_idx[mask].astype(np.int64)
        if len(set(y_raw[train_idx].tolist())) < len(all_classes):
            continue
        splits.append((f"dppd_random10_fold{fold}", train_idx, test_idx))
    return splits


def dppd_code_random_splits(y_raw, n_splits, split_seed):
    """Reproduce the original 3DGait loader behavior: train/test sample independently.

    The original code samples a random 10% test list inside each dataset object.
    If the train and test dataset objects are built separately, the held-out
    shadow list used to remove train clips and the actual test list can differ.
    This deterministic variant mirrors that risk and reports the overlap.
    """
    all_idx = np.arange(len(y_raw), dtype=np.int64)
    test_size = max(1, int(round(len(all_idx) / 10)))
    all_classes = set(y_raw.tolist())
    splits = []
    for fold in range(1, n_splits + 1):
        rng = random.Random(int(split_seed) + fold)
        shadow_test = np.asarray(sorted(rng.sample(all_idx.tolist(), test_size)), dtype=np.int64)
        test_idx = np.asarray(sorted(rng.sample(all_idx.tolist(), test_size)), dtype=np.int64)
        shadow_mask = np.ones(len(all_idx), dtype=bool)
        shadow_mask[shadow_test] = False
        train_idx = all_idx[shadow_mask].astype(np.int64)
        if len(set(y_raw[train_idx].tolist())) < len(all_classes):
            continue
        overlap = np.intersect1d(train_idx, test_idx, assume_unique=False)
        split_meta = {
            "shadow_test_n": int(len(shadow_test)),
            "train_test_overlap_n": int(len(overlap)),
            "train_test_overlap_ratio": float(len(overlap) / max(1, len(test_idx))),
        }
        splits.append((f"dppd_code_random10_fold{fold}", train_idx, test_idx, split_meta))
    return splits


def group10_splits(y_raw, groups, split_seed):
    from sklearn.model_selection import StratifiedGroupKFold

    uniq = sorted(set(groups.tolist()))
    n_splits = min(10, len(uniq))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(split_seed))
    splits = []
    all_classes = set(y_raw.tolist())
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y_raw)), y_raw, groups), start=1):
        train_idx = train_idx.astype(np.int64)
        test_idx = test_idx.astype(np.int64)
        if len(set(y_raw[train_idx].tolist())) < len(all_classes):
            continue
        splits.append((f"group10_fold{fold}", train_idx, test_idx))
    return splits


def build_counterfactual_splits(opts, y_raw, groups):
    if opts.protocol == "dppd_random10":
        return dppd_random_splits(y_raw, opts.n_random_splits, opts.split_seed)
    if opts.protocol == "dppd_code_random10":
        return dppd_code_random_splits(y_raw, opts.n_random_splits, opts.split_seed)
    if opts.protocol == "group10":
        return group10_splits(y_raw, groups, opts.split_seed)
    return build_splits(opts, y_raw, groups)


def apply_eval_unit(opts, rows, patient_indices, local_indices, groups, eval_groups):
    if opts.eval_unit == "default":
        return eval_groups
    if opts.eval_unit == "split_group":
        return groups
    out = []
    for local_i in local_indices:
        row = rows[int(patient_indices[int(local_i)])]
        subset = row.get("subset", "")
        if opts.eval_unit == "dppd_clip" and subset == "3dgait":
            clip = Path(row.get("path", "")).name[:8] or row.get("path", "")
            out.append(f"3dgait_clip:{clip}")
        else:
            out.append(eval_groups[len(out)])
    return np.asarray(out)


def train_fold(opts, model, patient_window, train_idx, test_idx, local_indices, y_encoded, eval_groups, classes, device):
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
            y = y.to(device)
            raw = patient_window[batch_local_idx].to(device).float()
            logits, ref, aux = model(raw)
            loss = criterion(logits, y)
            loss = loss + opts.lambda_align * normal_align_loss(raw, ref, y, classes)
            loss = loss + opts.lambda_entropy * aux["cf_entropy"]
            if opts.lambda_smooth > 0:
                loss = loss + opts.lambda_smooth * smooth_loss(ref)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        _idx, val_y, val_probs, _preds, _aux = evaluate_model(model, patient_window, test_loader, classes, device)
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
    return best_acc, best_state, evaluate_model(model, patient_window, test_loader, classes, device, collect_aux=True)


def run_seed(opts, seed, patient_window, local_indices, y_raw, y_encoded, eval_groups, classes, splits, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_rows, pred_rows = [], []
    for fold, split in enumerate(splits, start=1):
        if len(split) == 4:
            heldout_group, train_idx, test_idx, split_meta = split
        else:
            heldout_group, train_idx, test_idx = split
            split_meta = {}
        set_seed(seed + fold)
        model = CounterfactualGatedClassifier(opts, len(classes)).to(device)
        best_acc, best_state, (_local, test_y, test_probs, test_preds, aux) = train_fold(
            opts, model, patient_window, train_idx, test_idx, local_indices, y_encoded, eval_groups, classes, device
        )
        if opts.save_checkpoints and best_state is not None:
            checkpoint_dir = output_dir / "checkpoints" / f"seed_{seed}" / f"fold_{fold:02d}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": best_state,
                    "seed": seed,
                    "fold": fold,
                    "heldout_group": heldout_group,
                    "classes": [int(c) for c in classes],
                    "best_val_group_accuracy": float(best_acc),
                    "args": vars(opts),
                },
                checkpoint_dir / "best_classifier.pt",
            )
        row = {
            "family": "counterfactual",
            "model": opts.model_name,
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
        row.update(split_meta)
        row.update(metric_dict(test_y, test_probs, test_preds, classes, "clip"))
        gnames, gy, gp, gpred = aggregate_groups_named(test_y, test_probs, eval_groups[test_idx], classes)
        row.update(metric_dict(gy, gp, gpred, classes, "group"))
        row.update(aux)
        fold_rows.append(row)
        for group, yy, pp, pred in zip(gnames, gy, gp, gpred):
            pred_rows.append(
                prediction_row(
                    opts.protocol,
                    opts.split_seed,
                    opts.state_filter,
                    "counterfactual",
                    opts.model_name,
                    seed,
                    fold,
                    group,
                    yy,
                    pp,
                    pred,
                    classes,
                )
            )
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return fold_rows, pred_rows


def flush_outputs(output_dir, fold_rows, pred_rows, classes):
    if fold_rows:
        write_csv(output_dir / "dppd_aligned_folds.csv", fold_rows)
    if pred_rows:
        write_csv(output_dir / "dppd_aligned_video_predictions.csv", pred_rows)
        seed_rows, summary_rows = summarize_from_predictions(pred_rows, classes)
        write_csv(output_dir / "dppd_aligned_seed_summary.csv", seed_rows)
        write_csv(output_dir / "dppd_aligned_summary.csv", summary_rows)
        write_markdown(output_dir / "dppd_aligned_summary.md", summary_rows)


def main():
    opts = parse_args()
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_metadata(opts.features_dir)
    patient_data = torch.load(opts.patient_window_file, map_location="cpu")
    patient_window = patient_data["window_joint_features"]
    patient_indices = patient_data["global_indices"].long()
    local_indices, y_raw, _y_encoded, groups, eval_groups, classes = prepare_task(rows, patient_indices, opts.task)
    local_indices, y_raw, groups, eval_groups = apply_state_filter(
        opts, rows, patient_indices, local_indices, y_raw, groups, eval_groups
    )
    eval_groups = apply_eval_unit(opts, rows, patient_indices, local_indices, groups, eval_groups)
    y_encoded, classes = encode_labels(y_raw)
    splits = build_counterfactual_splits(opts, y_raw, groups)

    print(
        json.dumps(
            {
                "task": opts.task,
                "state_filter": opts.state_filter,
                "protocol": opts.protocol,
                "samples": int(len(y_raw)),
                "subjects": int(len(set(groups.tolist()))),
                "videos": int(len(set(eval_groups.tolist()))),
                "classes": [int(c) for c in classes],
                "patient_window_shape": list(patient_window.shape),
                "folds": len(splits),
                "seeds": opts.seeds,
                "model": opts.model_name,
                "prototype_file": opts.prototype_file,
                "bank": opts.bank,
                "prototype_top_k": opts.prototype_top_k,
                "lambda_align": opts.lambda_align,
                "lambda_entropy": opts.lambda_entropy,
                "lambda_smooth": opts.lambda_smooth,
            },
            indent=2,
        ),
        flush=True,
    )

    all_fold_rows, all_pred_rows = [], []
    for seed in opts.seeds:
        print(f"Running counterfactual/{opts.model_name} seed={seed}", flush=True)
        fold_rows, pred_rows = run_seed(opts, seed, patient_window, local_indices, y_raw, y_encoded, eval_groups, classes, splits, output_dir)
        all_fold_rows.extend(fold_rows)
        all_pred_rows.extend(pred_rows)
        flush_outputs(output_dir, all_fold_rows, all_pred_rows, classes)

    flush_outputs(output_dir, all_fold_rows, all_pred_rows, classes)
    (output_dir / "args.json").write_text(json.dumps(vars(opts), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
