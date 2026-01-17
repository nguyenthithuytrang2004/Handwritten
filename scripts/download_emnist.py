#!/usr/bin/env python3
"""
Download EMNIST (via torchvision) and export images to a folder structure.

Usage:
  python scripts/download_emnist.py --split byclass --out data/EMNIST/byclass/images

Defaults:
  split: byclass
  out: data/EMNIST/byclass/images

Notes:
 - This uses torchvision.datasets.EMNIST (will download raw data to data/EMNIST).
 - Labels are saved as integer subfolders (use mapping if you need character map).
 - Do NOT commit the downloaded data folder to git.
"""
from pathlib import Path
import argparse
from PIL import Image

try:
    from torchvision.datasets import EMNIST
    from torchvision import transforms
except Exception as exc:
    raise SystemExit("Missing dependency: install torchvision and pillow. e.g. pip install torchvision pillow") from exc


def export_emnist(split: str, out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)
    to_pil = transforms.ToPILImage()

    print(f"Downloading EMNIST split='{split}' (if not present) and exporting images to {out_root} ...")
    dataset = EMNIST(root=str(out_root.parent), split=split, download=True, transform=transforms.ToTensor())

    for idx, (img_tensor, label) in enumerate(dataset):
        label_dir = out_root / str(int(label))
        label_dir.mkdir(parents=True, exist_ok=True)
        img_pil = to_pil(img_tensor)
        img_pil.save(label_dir / f"{idx:06d}.png")
        if (idx + 1) % 5000 == 0:
            print(f"  saved {idx+1} images...")

    print("Done. Images saved under:", out_root)


def main():
    p = argparse.ArgumentParser(description="Download EMNIST and export images.")
    p.add_argument("--split", default="byclass", help="EMNIST split (byclass, bymerge, balanced, letters, digits, mnist)")
    p.add_argument("--out", default="data/EMNIST/byclass/images", help="Output folder for images")
    args = p.parse_args()

    out_root = Path(args.out)
    export_emnist(args.split, out_root)


if __name__ == "__main__":
    main()


