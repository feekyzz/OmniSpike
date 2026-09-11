from __future__ import annotations

import argparse
import os
from pathlib import Path


COCO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
}


def write_junction(link: Path, target: Path) -> None:
    if link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    os.system(f'mklink /J "{link}" "{target}"')


def remap_label(src: Path, dst: Path, class_map: dict[int, int]) -> int:
    if not src.exists():
        return 0
    kept: list[str] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        cls = int(float(parts[0]))
        if cls in class_map:
            kept.append(" ".join([str(class_map[cls]), *parts[1:]]))
    if kept:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return len(kept)


def build_split(src_root: Path, dst_root: Path, split: str, classes: list[int]) -> tuple[int, int]:
    class_map = {old: new for new, old in enumerate(classes)}
    image_src = src_root / "images" / split
    image_dst = dst_root / "images" / split
    label_src = src_root / "labels" / split
    label_dst = dst_root / "labels" / split
    list_file = dst_root / f"{split}.txt"

    write_junction(image_dst, image_src)
    label_dst.mkdir(parents=True, exist_ok=True)

    image_paths: list[str] = []
    objects = 0
    for src_label in sorted(label_src.glob("*.txt")):
        image_name = src_label.with_suffix(".jpg").name
        dst_label = label_dst / src_label.name
        count = remap_label(src_label, dst_label, class_map)
        if count:
            image_paths.append(str(image_dst / image_name))
            objects += count

    list_file.write_text("\n".join(image_paths) + "\n", encoding="utf-8")
    return len(image_paths), objects


def write_yaml(dst_root: Path, yaml_path: Path, classes: list[int]) -> None:
    names = "\n".join(f"  {i}: {COCO_NAMES[c]}" for i, c in enumerate(classes))
    yaml_path.write_text(
        "\n".join(
            [
                f"path: '{dst_root}'",
                "train: train2017.txt",
                "val: val2017.txt",
                "names:",
                names,
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--dst-root", required=True)
    parser.add_argument("--yaml", required=True)
    parser.add_argument("--classes", nargs="+", type=int, required=True)
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    classes = args.classes

    train_images, train_objects = build_split(src_root, dst_root, "train2017", classes)
    val_images, val_objects = build_split(src_root, dst_root, "val2017", classes)
    write_yaml(dst_root, Path(args.yaml), classes)

    print(f"classes={classes}")
    print(f"train: {train_images} images, {train_objects} objects")
    print(f"val: {val_images} images, {val_objects} objects")


if __name__ == "__main__":
    main()
