import argparse
import os
import pickle
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Human3.6M h36m_annot h5 files to DPPD MotionDataset3D pkl clips."
    )
    parser.add_argument("--h36m-root", default="data/Human3.6/h36m/annot")
    parser.add_argument("--output-root", default="Dataset/motion3d/MB3D_f81s9/H36M-SH/0")
    parser.add_argument("--clip-len", type=int, default=81)
    parser.add_argument("--stride", type=int, default=9)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_image_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def video_id_from_image_name(image_name):
    # Example: S1_Directions_1.54138969_000001.jpg -> S1_Directions_1.54138969
    stem = os.path.splitext(image_name)[0]
    return stem.rsplit("_", 1)[0]


def normalize_pose(clip):
    # DPPD's gait pkl files are scale-normalized. Reuse the repository helper so
    # H36M clips land in the same rough coordinate range as the existing data.
    return crop_scale_3d(clip.astype(np.float32)).astype(np.float32)


def convert_split(split_name, h5_path, image_list_path, output_dir, args, start_index):
    image_names = load_image_names(image_list_path)
    with h5py.File(h5_path, "r") as h5_file:
        poses = h5_file["S"]
        if len(image_names) != len(poses):
            raise ValueError(
                f"{image_list_path} has {len(image_names)} image names, but {h5_path} has {len(poses)} poses."
            )

        indices = np.arange(len(poses))[:: args.sample_stride]
        video_ids = np.array([video_id_from_image_name(image_names[i]) for i in indices], dtype=object)
        clip_indices = split_clips(video_ids, args.clip_len, data_stride=args.stride)

        saved = 0
        iterator = tqdm(clip_indices, desc=f"Converting {split_name}", unit="clip")
        for local_clip_id, local_ids in enumerate(iterator):
            if args.max_clips is not None and saved >= args.max_clips:
                break

            source_indices = indices[np.asarray(list(local_ids), dtype=np.int64)]
            clip = np.asarray(poses[source_indices], dtype=np.float32)
            clip = normalize_pose(clip)

            source_name = video_ids[list(local_ids)[0]]
            global_id = start_index + saved
            record = {
                "id": global_id,
                "pose": clip,
                "label": args.label,
                "score_label": args.label,
                "dataset": "H36M-SH",
                "split": split_name,
                "source": source_name,
            }

            output_path = output_dir / f"{global_id:08d}.pkl"
            with open(output_path, "wb") as f:
                pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
            saved += 1

    return saved


def main():
    args = parse_args()
    np.random.seed(args.seed)
    h36m_root = Path(args.h36m_root)
    output_dir = Path(args.output_root)

    required = [
        h36m_root / "train.h5",
        h36m_root / "valid.h5",
        h36m_root / "train_images.txt",
        h36m_root / "valid_images.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Human3.6M annotation files: " + ", ".join(missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*.pkl"))
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already contains {len(existing)} pkl files. "
            "Pass --overwrite to replace this generated subset."
        )
    if existing:
        for path in existing:
            path.unlink()

    total = 0
    total += convert_split(
        "train",
        h36m_root / "train.h5",
        h36m_root / "train_images.txt",
        output_dir,
        args,
        start_index=total,
    )
    total += convert_split(
        "valid",
        h36m_root / "valid.h5",
        h36m_root / "valid_images.txt",
        output_dir,
        args,
        start_index=total,
    )

    print(f"Saved {total} DPPD-compatible H36M clips to {output_dir}")


if __name__ == "__main__":
    main()
