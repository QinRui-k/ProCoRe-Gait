import argparse
import csv
import os
from pathlib import Path


DEFAULT_TERMS = (
    "walk",
    "walking",
    "treadmill",
    "gait",
    "stride",
    "stair",
    "stairs",
    "jog",
    "jogging",
    "run",
    "running",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a DPPD AMASS subset with all CMU clips plus gait/locomotion clips."
    )
    parser.add_argument("--amass-root", default="Dataset/motion3d/MB3D_f81s9/AMASS")
    parser.add_argument("--output-root", default="Dataset/motion3d/MB3D_f81s9/AMASS_GAIT_CMU")
    parser.add_argument("--terms", nargs="*", default=DEFAULT_TERMS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def selected(source, terms):
    low = source.lower()
    return source.startswith("CMU/") or any(term in low for term in terms)


def main():
    args = parse_args()
    amass_root = Path(args.amass_root)
    source_dir = amass_root / "0"
    manifest_path = amass_root / "manifest.csv"
    output_root = Path(args.output_root)
    output_dir = output_root / "0"
    output_manifest = output_root / "manifest.csv"

    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to rebuild it.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in output_dir.glob("*.pkl"):
            path.unlink()

    next_id = 0
    selected_sequences = 0
    selected_clips = 0
    by_dataset = {}

    with manifest_path.open(newline="", encoding="utf-8") as src_file, output_manifest.open(
        "w", newline="", encoding="utf-8"
    ) as dst_file:
        reader = csv.DictReader(src_file)
        fieldnames = list(reader.fieldnames or [])
        writer = csv.DictWriter(dst_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            clips = int(row["clips"])
            source = row["source"]
            start = next_id
            next_id += clips
            if clips <= 0 or not selected(source, args.terms):
                continue

            writer.writerow(row)
            selected_sequences += 1
            selected_clips += clips
            dataset_name = source.split("/", 1)[0]
            by_dataset[dataset_name] = by_dataset.get(dataset_name, 0) + clips

            for clip_id in range(start, start + clips):
                src = source_dir / f"{clip_id:08d}.pkl"
                dst = output_dir / src.name
                if not src.exists():
                    raise FileNotFoundError(src)
                if dst.exists():
                    continue
                os.link(src, dst)

    print(f"Selected sequences: {selected_sequences}")
    print(f"Selected clips: {selected_clips}")
    print(f"Output: {output_dir}")
    print(f"Manifest: {output_manifest}")
    for name, count in sorted(by_dataset.items(), key=lambda item: item[1], reverse=True):
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
