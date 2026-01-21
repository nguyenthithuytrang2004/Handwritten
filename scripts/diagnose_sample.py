#!/usr/bin/env python3
"""
Quick diagnostic for a sample image: show preprocessing steps and top-3 predictions.

Usage:
  python scripts/diagnose_sample.py --image path/to/image.png

Outputs saved to ./diagnostic_out/
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
from PIL import Image

# ensure src is importable
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.hwr.preprocessing import segment_characters, prepare_char_array
from src.hwr.model import load_model, CLASSES


def save_image(arr, path, scale=1):
    img = Image.fromarray(arr.astype(np.uint8))
    if scale != 1:
        img = img.resize((int(img.width*scale), int(img.height*scale)), Image.NEAREST)
    img.save(path)


def diagnose(image_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path).convert('L')
    print(f"Loaded image: {image_path} size={img.size}")
    img.save(out_dir / 'original.png')

    # Run segmentation
    char_images, segments, char_arrays = segment_characters(img)

    # Save binarized view
    arr = np.array(img)
    thresh = (arr > 30).astype(np.uint8) * 255
    save_image(thresh, out_dir / 'binarized.png', scale=4)

    if not char_images:
        print("No characters detected by segmentation.")
        return

    model = load_model()

    for idx, (prepared, arr28) in enumerate(zip(char_images, char_arrays)):
        # prepared is (1,28,28,1) normalized
        probs = model.predict(prepared, verbose=0)[0]
        top3_idx = probs.argsort()[-3:][::-1]
        top3 = [(CLASSES[i], float(probs[i])) for i in top3_idx]

        print(f"Char #{idx}: segments={segments[idx]}")
        print("  Top-3:")
        for c, p in top3:
            print(f"    {c}: {p:.4f}")

        # save prepared 28x28
        img28 = Image.fromarray(arr28.astype(np.uint8)).resize((280,280), Image.NEAREST)
        img28.save(out_dir / f'char_{idx}_28x28.png')

        # save the actual input used (after normalization) as visualization
        vis = (prepared.reshape(28,28) * 255.0).astype(np.uint8)
        Image.fromarray(vis).resize((280,280), Image.NEAREST).save(out_dir / f'char_{idx}_input.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Path to image', required=False)
    args = parser.parse_args()

    # try find an image in data/user_data/*
    user_data = project_root / 'data' / 'user_data'
    if args.image:
        img_path = Path(args.image)
    else:
        # pick first png under user_data
        img_path = None
        if user_data.exists():
            for lbl in user_data.iterdir():
                if lbl.is_dir():
                    img = next(lbl.glob('*.png'), None)
                    if img:
                        img_path = img
                        break
    if img_path is None:
        print('No image found in data/user_data and --image not provided.')
        return

    out_dir = project_root / 'diagnostic_out'
    diagnose(img_path, out_dir)
    print(f"Saved diagnostic outputs to: {out_dir}")


if __name__ == '__main__':
    main()



