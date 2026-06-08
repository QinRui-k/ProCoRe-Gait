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
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.train_pdgait_counterfactual_attention import (  # noqa: E402
    PrototypeAttentionReference,
    apply_eval_unit,
    apply_state_filter,
    build_counterfactual_splits,
    encode_labels,
    prepare_task,
    read_metadata,
    set_seed,
)
from tools.train_pdgait_score3_dppd_aligned import (  # noqa: E402
    aggregate_groups_named,
    prediction_row,
    summarize_from_predictions,
    write_markdown,
)
from tools.train_wjc_tensor_classifier import IndexDataset, metric_dict, score_label, subtype_label, write_csv  # noqa: E402


PARTS = {
    "pelvis_trunk": [0, 7, 8, 9, 10],
    "left_leg": [4, 5, 6],
    "right_leg": [1, 2, 3],
    "left_arm": [14, 15, 16],
    "right_arm": [11, 12, 13],
}
LEFT_RIGHT_PAIRS = [(1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16)]


def parse_args():
    parser = argparse.ArgumentParser(description="Extended learnable-CF classifier sweeps for gait scoring tasks.")
    parser.add_argument("--features-dir", default="features/e1_patient_dynamic_anchor_clip")
    parser.add_argument("--patient-window-file", default="features/window_joint_e1/patient_window_joint.pt")
    parser.add_argument("--prototype-file", default="eval/frozen_bank_source_ablation_e1/patient_control/window_prototypes.pt")
    parser.add_argument("--output-dir", default="eval/3dgait_cf_extended")
    parser.add_argument("--bank", default="prototype_M64")
    parser.add_argument(
        "--task",
        choices=["pdgait_score_3class", "3dgait_score_4class", "3dgait_subtype_3class"],
        default="3dgait_subtype_3class",
    )
    parser.add_argument("--direction", default="subtype_aware")
    parser.add_argument(
        "--protocol",
        choices=["loso22", "with0_testonly", "with0_release", "random5", "dppd_random10", "dppd_code_random10", "group10"],
        default="dppd_code_random10",
    )
    parser.add_argument("--eval-unit", choices=["default", "dppd_clip", "split_group"], default="dppd_clip")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--n-random-splits", type=int, default=10)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260527])
    parser.add_argument("--state-filter", choices=["all", "on", "off"], default="all")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--ref-dim", type=int, default=64)
    parser.add_argument("--prototype-top-k", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.12)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument(
        "--selection-metric",
        choices=["group_accuracy", "group_weighted_f1", "group_macro_f1", "group_balanced_acc"],
        default="group_accuracy",
    )
    parser.add_argument("--use-signed-residual", action="store_true")
    parser.add_argument("--input-adapter", action="store_true")
    parser.add_argument("--part-aware", action="store_true")
    parser.add_argument("--disease-bank", choices=["none", "subtype", "score", "both"], default="none")
    parser.add_argument("--disease-prototypes-per-class", type=int, default=8)
    parser.add_argument("--aux-score-weight", type=float, default=0.0)
    parser.add_argument("--aux-subtype-weight", type=float, default=0.0)
    parser.add_argument("--ordinal-weight", type=float, default=0.0)
    parser.add_argument("--lambda-align", type=float, default=0.05)
    parser.add_argument("--lambda-entropy", type=float, default=0.02)
    parser.add_argument("--lambda-smooth", type=float, default=0.0)
    parser.add_argument("--lambda-sparsity", type=float, default=0.0)
    parser.add_argument("--lambda-asymmetry", type=float, default=0.0)
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser.parse_args()


def score_aux_labels(rows, patient_indices, local_indices):
    labels = []
    for local_i in local_indices:
        row = rows[int(patient_indices[int(local_i)])]
        score = score_label(row)
        labels.append(-1 if score is None else int(score))
    return np.asarray(labels, dtype=np.int64)


def subtype_aux_labels(rows, patient_indices, local_indices):
    labels = []
    for local_i in local_indices:
        row = rows[int(patient_indices[int(local_i)])]
        subtype = subtype_label(row)
        labels.append(-1 if subtype is None else int(subtype))
    return np.asarray(labels, dtype=np.int64)


def class_weights_np(y, num_classes):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.as_tensor(weights, dtype=torch.float32)


def focal_loss(logits, target, weight=None, gamma=2.0, label_smoothing=0.0):
    ce = F.cross_entropy(logits, target, weight=weight, reduction="none", label_smoothing=label_smoothing)
    pt = torch.exp(-ce).clamp(min=1e-6, max=1.0)
    return ((1.0 - pt) ** gamma * ce).mean()


def metric_score(row, key):
    value = row.get(key)
    if value in ("", None):
        return float("nan")
    return float(value)


def select_score(y_raw, probs, eval_groups, classes, metric):
    preds = np.asarray([classes[i] for i in probs.argmax(axis=1)], dtype=np.int64)
    _gn, gy, gp, gpred = aggregate_groups_named(y_raw, probs, eval_groups, classes)
    values = metric_dict(gy, gp, gpred, classes, "group")
    return metric_score(values, metric)


def build_sampled_class_bank(raw_train, labels, num_classes, per_class, seed):
    rng = np.random.default_rng(seed)
    banks = []
    for cls in range(num_classes):
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) == 0:
            chosen = rng.choice(np.arange(len(labels)), size=per_class, replace=True)
        else:
            chosen = rng.choice(cls_idx, size=per_class, replace=len(cls_idx) < per_class)
        banks.append(raw_train[chosen].permute(1, 0, 2, 3).contiguous())  # W,M,J,C
    return torch.stack(banks, dim=0)  # K,W,M,J,C


class DiseaseReference(nn.Module):
    def __init__(self, prototypes, ref_dim=64, top_k=8, temperature=0.12):
        super().__init__()
        self.register_buffer("prototypes", prototypes.float())
        in_channels = int(prototypes.shape[-1])
        self.q_proj = nn.Linear(in_channels, ref_dim)
        self.k_proj = nn.Linear(in_channels, ref_dim)
        self.top_k = min(int(top_k), int(prototypes.shape[2]))
        self.temperature = float(temperature)

    def forward(self, raw):
        # raw: B,W,J,C; prototypes: K,W,M,J,C
        q = F.normalize(self.q_proj(raw), dim=-1)
        k = F.normalize(self.k_proj(self.prototypes), dim=-1)
        logits = torch.einsum("bwjd,kwmjd->bkwjm", q, k) / max(self.temperature, 1e-6)
        if self.top_k < logits.shape[-1]:
            top_logits, top_idx = torch.topk(logits, k=self.top_k, dim=-1)
            proto = self.prototypes.permute(0, 1, 3, 2, 4).unsqueeze(0).expand(
                raw.shape[0], -1, -1, -1, -1, -1
            )
            proto = torch.gather(proto, 4, top_idx.unsqueeze(-1).expand(-1, -1, -1, -1, -1, raw.shape[-1]))
            weights = torch.softmax(top_logits, dim=-1)
            ref = torch.sum(weights.unsqueeze(-1) * proto, dim=4)
        else:
            weights = torch.softmax(logits, dim=-1)
            ref = torch.einsum("bkwjm,kwmjc->bkwjc", weights, self.prototypes)
        distances = (raw.unsqueeze(1) - ref).abs().mean(dim=(2, 3, 4))
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1).mean()
        return ref, distances, entropy


class ExtendedCFClassifier(nn.Module):
    def __init__(self, opts, num_classes, disease_prototypes=None):
        super().__init__()
        self.opts = opts
        self.normal_ref = PrototypeAttentionReference(
            opts.prototype_file,
            opts.bank,
            ref_dim=opts.ref_dim,
            prototype_top_k=opts.prototype_top_k,
            temperature=opts.temperature,
        )
        in_channels = int(self.normal_ref.prototypes.shape[-1])
        self.adapter = (
            nn.Sequential(nn.LayerNorm(in_channels), nn.Linear(in_channels, opts.dim), nn.GELU(), nn.Linear(opts.dim, in_channels))
            if opts.input_adapter
            else None
        )
        self.disease_ref = (
            DiseaseReference(
                disease_prototypes,
                ref_dim=opts.ref_dim,
                top_k=opts.disease_prototypes_per_class,
                temperature=opts.temperature,
            )
            if disease_prototypes is not None
            else None
        )
        mult = 2 + int(opts.use_signed_residual)
        self.token_proj = nn.Linear(in_channels * mult, opts.dim)
        self.window_embed = nn.Parameter(torch.zeros(1, 9, 1, opts.dim))
        self.joint_embed = nn.Parameter(torch.zeros(1, 1, 17, opts.dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, opts.dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=opts.dim,
            nhead=opts.heads,
            dim_feedforward=opts.dim * 2,
            dropout=opts.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=opts.layers)
        extra = 0
        if opts.part_aware:
            extra += len(PARTS) + len(LEFT_RIGHT_PAIRS)
        if self.disease_ref is not None:
            extra += num_classes
        self.head = nn.Sequential(
            nn.LayerNorm(opts.dim + extra),
            nn.Dropout(opts.dropout),
            nn.Linear(opts.dim + extra, opts.dim),
            nn.GELU(),
            nn.Dropout(opts.dropout),
            nn.Linear(opts.dim, num_classes),
        )
        self.score_head = (
            nn.Sequential(nn.LayerNorm(opts.dim + extra), nn.Linear(opts.dim + extra, 4))
            if opts.aux_score_weight > 0
            else None
        )
        self.subtype_head = (
            nn.Sequential(nn.LayerNorm(opts.dim + extra), nn.Linear(opts.dim + extra, 3))
            if opts.aux_subtype_weight > 0
            else None
        )

    def forward(self, raw):
        if self.adapter is not None:
            raw = raw + 0.1 * self.adapter(raw)
        normal_ref, aux = self.normal_ref(raw, return_aux=True)
        signed = raw - normal_ref
        abs_res = signed.abs()
        pieces = [raw, abs_res]
        if self.opts.use_signed_residual:
            pieces.append(signed)
        token = self.token_proj(torch.cat(pieces, dim=-1))
        token = token + self.window_embed[:, : token.shape[1]] + self.joint_embed[:, :, : token.shape[2]]
        token = token.flatten(1, 2)
        cls = self.cls.expand(raw.shape[0], -1, -1)
        rep = self.encoder(torch.cat([cls, token], dim=1))[:, 0]
        extras = []
        disease_entropy = raw.new_tensor(0.0)
        if self.opts.part_aware:
            part_vals = [abs_res[:, :, idx, :].mean(dim=(1, 2, 3)) for idx in PARTS.values()]
            pair_vals = [(abs_res[:, :, l, :] - abs_res[:, :, r, :]).abs().mean(dim=(1, 2)) for l, r in LEFT_RIGHT_PAIRS]
            extras.append(torch.stack(part_vals + pair_vals, dim=1))
        if self.disease_ref is not None:
            _refs, distances, disease_entropy = self.disease_ref(raw)
            extras.append(distances)
        if extras:
            rep = torch.cat([rep] + extras, dim=1)
        logits = self.head(rep)
        score_logits = self.score_head(rep) if self.score_head is not None else None
        subtype_logits = self.subtype_head(rep) if self.subtype_head is not None else None
        aux_out = {
            "normal_ref": normal_ref,
            "abs_res": abs_res,
            "cf_entropy": aux["cf_entropy"] + disease_entropy,
            "cf_top1": aux["cf_top1"],
        }
        return logits, score_logits, subtype_logits, aux_out


def regularization_loss(opts, aux):
    loss = aux["abs_res"].new_tensor(0.0)
    if opts.lambda_entropy:
        loss = loss + opts.lambda_entropy * aux["cf_entropy"]
    if opts.lambda_sparsity:
        loss = loss + opts.lambda_sparsity * aux["abs_res"].mean()
    if opts.lambda_smooth and aux["normal_ref"].shape[1] > 1:
        loss = loss + opts.lambda_smooth * (aux["normal_ref"][:, 1:] - aux["normal_ref"][:, :-1]).abs().mean()
    if opts.lambda_asymmetry:
        vals = []
        for l, r in LEFT_RIGHT_PAIRS:
            vals.append((aux["abs_res"][:, :, l, :] - aux["abs_res"][:, :, r, :]).abs().mean())
        loss = loss + opts.lambda_asymmetry * torch.stack(vals).mean()
    return loss


def evaluate_model(model, patient_window, loader, classes, device):
    model.eval()
    probs_all, idx_all, y_all = [], [], []
    with torch.no_grad():
        for local_idx, y, _score, _subtype in loader:
            raw = patient_window[local_idx].to(device).float()
            logits, _score_logits, _subtype_logits, _aux = model(raw)
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


class ExtendedDataset(torch.utils.data.Dataset):
    def __init__(self, local_indices, y, score_y, subtype_y):
        self.local_indices = np.asarray(local_indices, dtype=np.int64)
        self.y = np.asarray(y, dtype=np.int64)
        self.score_y = np.asarray(score_y, dtype=np.int64)
        self.subtype_y = np.asarray(subtype_y, dtype=np.int64)

    def __len__(self):
        return len(self.local_indices)

    def __getitem__(self, idx):
        return int(self.local_indices[idx]), int(self.y[idx]), int(self.score_y[idx]), int(self.subtype_y[idx])


def ordinal_expectation_loss(logits, target, classes):
    if len(classes) < 3:
        return logits.new_tensor(0.0)
    class_values = torch.as_tensor(classes, dtype=logits.dtype, device=logits.device)
    probs = torch.softmax(logits, dim=1)
    pred = (probs * class_values.unsqueeze(0)).sum(dim=1)
    target_values = class_values[target]
    scale = max(float(class_values.max().item() - class_values.min().item()), 1.0)
    return F.smooth_l1_loss(pred / scale, target_values / scale)


def train_fold(opts, patient_window, train_idx, test_idx, local_indices, y_encoded, score_y, subtype_y, eval_groups, classes, seed, output_dir, fold):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disease_prototypes = None
    if opts.disease_bank != "none":
        raw_train = patient_window[local_indices[train_idx]].float()
        disease_prototypes = build_sampled_class_bank(
            raw_train,
            y_encoded[train_idx],
            len(classes),
            opts.disease_prototypes_per_class,
            seed,
        )
    model = ExtendedCFClassifier(opts, len(classes), disease_prototypes=disease_prototypes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    weight = class_weights_np(y_encoded[train_idx], len(classes)).to(device)
    train_ds = ExtendedDataset(local_indices[train_idx], y_encoded[train_idx], score_y[train_idx], subtype_y[train_idx])
    sampler = None
    shuffle = True
    if opts.balanced_sampler:
        sample_weights = class_weights_np(y_encoded[train_idx], len(classes)).numpy()[y_encoded[train_idx]]
        sampler = WeightedRandomSampler(sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)
        shuffle = False
    train_loader = DataLoader(train_ds, batch_size=opts.batch_size, shuffle=shuffle, sampler=sampler, num_workers=0)
    test_loader = DataLoader(
        ExtendedDataset(local_indices[test_idx], y_encoded[test_idx], score_y[test_idx], subtype_y[test_idx]),
        batch_size=opts.eval_batch_size,
        shuffle=False,
        num_workers=0,
    )
    best_state, best_score, wait = None, -1e9, 0
    for _epoch in range(opts.epochs):
        model.train()
        for batch_local, y, score, subtype in train_loader:
            raw = patient_window[batch_local].to(device).float()
            y = y.to(device)
            score = score.to(device)
            subtype = subtype.to(device)
            logits, score_logits, subtype_logits, aux = model(raw)
            if opts.loss == "focal":
                loss = focal_loss(logits, y, weight=weight, gamma=opts.focal_gamma, label_smoothing=opts.label_smoothing)
            else:
                loss = F.cross_entropy(logits, y, weight=weight, label_smoothing=opts.label_smoothing)
            if opts.ordinal_weight > 0:
                loss = loss + opts.ordinal_weight * ordinal_expectation_loss(logits, y, classes)
            if score_logits is not None:
                valid = score >= 0
                if valid.any():
                    loss = loss + opts.aux_score_weight * F.cross_entropy(score_logits[valid], score[valid].clamp(max=3))
            if subtype_logits is not None:
                valid = subtype >= 0
                if valid.any():
                    loss = loss + opts.aux_subtype_weight * F.cross_entropy(subtype_logits[valid], subtype[valid].clamp(max=2))
            loss = loss + regularization_loss(opts, aux)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        _idx, val_y, val_probs, _preds = evaluate_model(model, patient_window, test_loader, classes, device)
        score_value = select_score(val_y, val_probs, eval_groups[test_idx], classes, opts.selection_metric)
        if score_value > best_score:
            best_score = score_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= opts.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    if opts.save_checkpoints and best_state is not None:
        ckpt_dir = output_dir / "checkpoints" / f"seed_{seed}" / f"fold_{fold:02d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": best_state, "args": vars(opts), "classes": [int(c) for c in classes]}, ckpt_dir / "best_classifier.pt")
    return best_score, evaluate_model(model, patient_window, test_loader, classes, device)


def flush_outputs(output_dir, fold_rows, pred_rows, classes):
    if fold_rows:
        write_csv(output_dir / "folds.csv", fold_rows)
    if pred_rows:
        write_csv(output_dir / "predictions.csv", pred_rows)
        seed_rows, summary_rows = summarize_from_predictions(pred_rows, classes)
        write_csv(output_dir / "seed_summary.csv", seed_rows)
        write_csv(output_dir / "summary.csv", summary_rows)
        write_markdown(output_dir / "summary.md", summary_rows)


def main():
    opts = parse_args()
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_metadata(opts.features_dir)
    patient_data = torch.load(opts.patient_window_file, map_location="cpu")
    patient_window = patient_data["window_joint_features"]
    patient_indices = patient_data["global_indices"].long()
    local_indices, y_raw, _y_encoded, groups, eval_groups, classes = prepare_task(rows, patient_indices, opts.task)
    local_indices, y_raw, groups, eval_groups = apply_state_filter(opts, rows, patient_indices, local_indices, y_raw, groups, eval_groups)
    eval_groups = apply_eval_unit(opts, rows, patient_indices, local_indices, groups, eval_groups)
    y_encoded, classes = encode_labels(y_raw)
    score_y = score_aux_labels(rows, patient_indices, local_indices)
    subtype_y = subtype_aux_labels(rows, patient_indices, local_indices)
    splits = build_counterfactual_splits(opts, y_raw, groups)
    if opts.max_folds:
        splits = splits[: opts.max_folds]
    print(
        json.dumps(
            {
                "task": opts.task,
                "direction": opts.direction,
                "protocol": opts.protocol,
                "samples": int(len(y_raw)),
                "classes": [int(c) for c in classes],
                "splits": len(splits),
                "seeds": opts.seeds,
                "patient_window_shape": list(patient_window.shape),
                "args": vars(opts),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    all_fold_rows, all_pred_rows = [], []
    for seed in opts.seeds:
        for fold, split in enumerate(splits, start=1):
            if len(split) == 4:
                heldout_group, train_idx, test_idx, split_meta = split
            else:
                heldout_group, train_idx, test_idx = split
                split_meta = {}
            best_val, (_local, test_y, test_probs, test_preds) = train_fold(
                opts,
                patient_window,
                train_idx,
                test_idx,
                local_indices,
                y_encoded,
                score_y,
                subtype_y,
                eval_groups,
                classes,
                seed + fold,
                output_dir,
                fold,
            )
            row = {
                "family": "extended_cf",
                "model": opts.direction,
                "task": opts.task,
                "protocol": opts.protocol,
                "seed": seed,
                "fold": fold,
                "heldout_group": heldout_group,
                "selection_metric": opts.selection_metric,
                "best_val": best_val,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
            row.update(split_meta)
            row.update(metric_dict(test_y, test_probs, test_preds, classes, "clip"))
            gnames, gy, gp, gpred = aggregate_groups_named(test_y, test_probs, eval_groups[test_idx], classes)
            row.update(metric_dict(gy, gp, gpred, classes, "group"))
            all_fold_rows.append(row)
            for group, yy, pp, pred in zip(gnames, gy, gp, gpred):
                all_pred_rows.append(
                    prediction_row(opts.protocol, opts.split_seed, opts.state_filter, "extended_cf", opts.direction, seed, fold, group, yy, pp, pred, classes)
                )
            print(json.dumps(row, ensure_ascii=False), flush=True)
            flush_outputs(output_dir, all_fold_rows, all_pred_rows, classes)
    flush_outputs(output_dir, all_fold_rows, all_pred_rows, classes)
    (output_dir / "args.json").write_text(json.dumps(vars(opts), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
