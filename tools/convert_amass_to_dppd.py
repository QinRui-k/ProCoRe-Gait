import argparse
import csv
import os
import pickle
import sys
import tarfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
HBP_SRC = REPO_ROOT / "human_body_prior" / "src"
if HBP_SRC.exists():
    sys.path.insert(0, str(HBP_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


SMPL_TO_H36M17 = np.array(
    [
        0,   # pelvis
        2,   # right hip
        5,   # right knee
        8,   # right ankle
        1,   # left hip
        4,   # left knee
        7,   # left ankle
        6,   # spine
        9,   # thorax
        12,  # neck
        15,  # head
        16,  # left shoulder
        18,  # left elbow
        20,  # left wrist
        17,  # right shoulder
        19,  # right elbow
        21,  # right wrist
    ],
    dtype=np.int64,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert downloaded AMASS archives/npz files to DPPD MotionDataset3D pkl clips."
    )
    parser.add_argument("--archives-root", default="data/AMASS/downloads/data/AMASS")
    parser.add_argument("--raw-root", default="data/AMASS/amass_raw")
    parser.add_argument("--output-root", default="Dataset/motion3d/MB3D_f81s9/AMASS/0")
    parser.add_argument("--body-model-root", default="data/AMASS/body_models")
    parser.add_argument("--joint-regressor", default="data/AMASS/J_regressor_h36m_correct.npy")
    parser.add_argument("--target-fps", type=float, default=60.0)
    parser.add_argument("--clip-len", type=int, default=81)
    parser.add_argument("--stride", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extract", action="store_true", help="Extract *.tar.bz2 files before conversion.")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--overwrite-extract", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Validate paths and print a conversion summary.")
    return parser.parse_args()


def iter_archives(archives_root):
    return sorted(Path(archives_root).glob("*.tar.bz2"))


def extract_archives(archives_root, raw_root, overwrite=False):
    raw_root.mkdir(parents=True, exist_ok=True)
    archives = iter_archives(archives_root)
    if not archives:
        raise FileNotFoundError(f"No AMASS archives found in {archives_root}")

    for archive in tqdm(archives, desc="Extracting AMASS archives", unit="archive"):
        top_level = archive.name.replace(".tar.bz2", "")
        target_dir = raw_root / top_level
        if target_dir.exists() and not overwrite:
            continue
        with tarfile.open(archive, "r:bz2") as tar:
            for member in tar.getmembers():
                member_path = (raw_root / member.name).resolve()
                if not str(member_path).startswith(str(raw_root.resolve())):
                    raise RuntimeError(f"Refusing to extract unsafe archive member: {member.name}")
            tar.extractall(raw_root)


def iter_npz_files(raw_root):
    return sorted(Path(raw_root).rglob("*_poses.npz"))


def crop_scale_3d(motion):
    result = np.array(motion, dtype=np.float32, copy=True)
    result[:, :, 2] = result[:, :, 2] - result[0, 0, 2]
    xmin = np.min(motion[..., 0])
    xmax = np.max(motion[..., 0])
    ymin = np.min(motion[..., 1])
    ymax = np.max(motion[..., 1])
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros(motion.shape, dtype=np.float32)
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., 2] = result[..., 2] / scale
    return (result - 0.5) * 2


def resample(ori_len, target_len):
    if ori_len > target_len:
        start = np.random.randint(ori_len - target_len)
        return np.arange(start, start + target_len)
    return np.arange(target_len) % ori_len


def split_clips(vid_list, n_frames, data_stride):
    result = []
    start = 0
    i = 0
    saved = set()
    while i < len(vid_list):
        i += 1
        if i - start == n_frames:
            result.append(range(start, i))
            saved.add(vid_list[i - 1])
            start += data_stride
        if i == len(vid_list):
            break
        if vid_list[i] != vid_list[i - 1]:
            if vid_list[i - 1] not in saved:
                result.append(resample(i - start, n_frames) + start)
                saved.add(vid_list[i - 1])
            start = i
    return result


def normalize_gender(gender):
    if isinstance(gender, np.ndarray):
        gender = gender.item()
    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")
    gender = str(gender).lower()
    if gender not in {"female", "male"}:
        return "female"
    return gender


def find_body_model_files(body_model_root, gender):
    root = Path(body_model_root)
    bm_candidates = [
        root / "smplh" / gender / "model.npz",
        root / "smplh" / gender.upper() / "model.npz",
        root / "smplh" / f"{gender}.npz",
        root / "smplh" / f"SMPLH_{gender.upper()}.npz",
    ]
    bm_path = next((path for path in bm_candidates if path.exists()), None)
    if bm_path is None:
        raise FileNotFoundError(
            "Missing SMPLH body model for "
            f"{gender}. Expected one of: {', '.join(str(p) for p in bm_candidates)}"
        )

    dmpl_candidates = [
        root / "dmpls" / gender / "model.npz",
        root / "dmpls" / gender.upper() / "model.npz",
        root / "dmpls" / f"{gender}.npz",
        root / "dmpls" / f"DMPLS_{gender.upper()}.npz",
    ]
    dmpl_path = next((path for path in dmpl_candidates if path.exists()), None)
    return bm_path, dmpl_path


def load_body_model(cache, body_model_root, gender, device):
    from human_body_prior.body_model.body_model import BodyModel

    if gender in cache:
        return cache[gender]

    bm_path, dmpl_path = find_body_model_files(body_model_root, gender)
    if dmpl_path is not None:
        model = BodyModel(
            bm_fname=str(bm_path),
            num_betas=16,
            num_dmpls=8,
            dmpl_fname=str(dmpl_path),
        ).to(device)
    else:
        model = BodyModel(bm_fname=str(bm_path), num_betas=16, num_dmpls=None).to(device)
    model.eval()
    cache[gender] = (model, dmpl_path is not None)
    return cache[gender]


def frame_indices(length, source_fps, target_fps):
    if length <= 0:
        return np.array([], dtype=np.int64)
    if source_fps <= 0 or target_fps <= 0:
        return np.arange(length, dtype=np.int64)
    step = max(source_fps / target_fps, 1.0)
    return np.unique(np.floor(np.arange(0, length, step)).astype(np.int64))


def run_body_model(sequence, model, use_dmpl, joint_regressor, device, batch_size):
    import torch

    poses = np.asarray(sequence["poses"], dtype=np.float32)
    trans = np.asarray(sequence["trans"], dtype=np.float32)
    betas = np.asarray(sequence.get("betas", np.zeros(16, dtype=np.float32)), dtype=np.float32)
    if betas.ndim > 1:
        betas = betas[0]
    betas = betas[:16]
    if len(betas) < 16:
        betas = np.pad(betas, (0, 16 - len(betas)))
    dmpls = np.asarray(sequence.get("dmpls", np.zeros((len(trans), 8), dtype=np.float32)), dtype=np.float32)

    joints_chunks = []
    for start in range(0, len(trans), batch_size):
        end = min(start + batch_size, len(trans))
        pose_hand = np.zeros((end - start, 90), dtype=np.float32)
        hand_values = poses[start:end, 66:156]
        pose_hand[:, : hand_values.shape[1]] = hand_values
        body_parms = {
            "root_orient": torch.as_tensor(poses[start:end, :3], device=device),
            "pose_body": torch.as_tensor(poses[start:end, 3:66], device=device),
            "pose_hand": torch.as_tensor(pose_hand, device=device),
            "trans": torch.as_tensor(trans[start:end], device=device),
            "betas": torch.as_tensor(np.repeat(betas[None], end - start, axis=0), device=device),
        }
        if use_dmpl:
            body_parms["dmpls"] = torch.as_tensor(dmpls[start:end, :8], device=device)

        with torch.no_grad():
            body = model(**body_parms)
            if joint_regressor is not None:
                verts = body.v.detach().cpu().numpy()
                joints = np.einsum("jv,tvc->tjc", joint_regressor, verts)
            else:
                joints = body.Jtr[:, SMPL_TO_H36M17, :].detach().cpu().numpy()
        joints_chunks.append(joints.astype(np.float32))

    return np.concatenate(joints_chunks, axis=0)


def to_motionbert_camera(joints):
    real2cam = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
    return (joints @ real2cam).astype(np.float32)


def save_clips(joints, output_dir, next_id, args, source):
    vid_list = [0] * len(joints)
    clip_indices = split_clips(vid_list, n_frames=args.clip_len, data_stride=args.stride)
    saved = 0
    for local_ids in clip_indices:
        if args.max_clips is not None and next_id + saved >= args.max_clips:
            break
        clip = joints[np.asarray(list(local_ids), dtype=np.int64)]
        clip = crop_scale_3d(clip).astype(np.float32)
        if not np.isfinite(clip).all():
            continue
        record = {
            "id": next_id + saved,
            "pose": clip,
            "label": args.label,
            "score_label": args.label,
            "dataset": "AMASS",
            "source": source,
            "target_fps": args.target_fps,
        }
        with open(output_dir / f"{next_id + saved:08d}.pkl", "wb") as f:
            pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
        saved += 1
    return saved


def validate_inputs(args, npz_files):
    missing = []
    if not Path(args.raw_root).exists():
        missing.append(args.raw_root)
    if not npz_files:
        missing.append(f"{args.raw_root}/**/*_poses.npz")
    for gender in ("female", "male"):
        try:
            find_body_model_files(args.body_model_root, gender)
        except FileNotFoundError as exc:
            missing.append(str(exc))
    regressor_path = Path(args.joint_regressor)
    regressor_status = "found" if regressor_path.exists() else "missing; will fall back to SMPLH joint mapping"

    print(f"AMASS npz files: {len(npz_files)}")
    print(f"Body model root: {args.body_model_root}")
    print(f"H36M joint regressor: {regressor_status}")
    print(f"Output root: {args.output_root}")
    if missing:
        print("Missing prerequisites:")
        for item in missing:
            print(f"  - {item}")
        return False
    return True


def main():
    args = parse_args()
    np.random.seed(args.seed)

    archives_root = Path(args.archives_root)
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_root)

    if args.extract:
        extract_archives(archives_root, raw_root, overwrite=args.overwrite_extract)
    if args.extract_only:
        return

    npz_files = iter_npz_files(raw_root)
    if args.check_only:
        ok = validate_inputs(args, npz_files)
        raise SystemExit(0 if ok else 1)
    if not npz_files:
        archive_hint = " Run again with --extract first." if iter_archives(archives_root) else ""
        raise FileNotFoundError(f"No *_poses.npz files found under {raw_root}.{archive_hint}")

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*.pkl"))
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already contains {len(existing)} pkl files. Pass --overwrite to replace them."
        )
    for path in existing:
        path.unlink()

    regressor_path = Path(args.joint_regressor)
    joint_regressor = np.load(regressor_path).astype(np.float32) if regressor_path.exists() else None
    if joint_regressor is None:
        print("WARNING: H36M joint regressor not found; using the built-in SMPLH joint mapping fallback.")

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    body_model_cache = {}
    next_id = 0
    manifest_path = output_dir.parent / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(["source", "gender", "source_fps", "frames", "sampled_frames", "clips"])

        iterator = tqdm(npz_files[: args.max_sequences], desc="Converting AMASS", unit="seq")
        for npz_path in iterator:
            try:
                raw = np.load(npz_path, allow_pickle=True)
                sequence = {key: raw[key] for key in raw.files}
                gender = normalize_gender(sequence.get("gender", "female"))
                source_fps = float(np.asarray(sequence.get("mocap_framerate", args.target_fps)).item())
                indices = frame_indices(len(sequence["trans"]), source_fps, args.target_fps)
                if len(indices) < 2:
                    continue

                sequence["poses"] = sequence["poses"][indices]
                sequence["trans"] = sequence["trans"][indices]
                if "dmpls" in sequence:
                    sequence["dmpls"] = sequence["dmpls"][indices]

                model, use_dmpl = load_body_model(body_model_cache, args.body_model_root, gender, device)
                joints = run_body_model(sequence, model, use_dmpl, joint_regressor, device, args.batch_size)
                joints = to_motionbert_camera(joints)
                saved = save_clips(joints, output_dir, next_id, args, str(npz_path.relative_to(raw_root)))
                writer.writerow([str(npz_path.relative_to(raw_root)), gender, source_fps, len(indices), len(joints), saved])
                next_id += saved
                if args.max_clips is not None and next_id >= args.max_clips:
                    break
            except Exception as exc:
                print(f"WARNING: skipped {npz_path}: {exc}")

    print(f"Saved {next_id} DPPD-compatible AMASS clips to {output_dir}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
