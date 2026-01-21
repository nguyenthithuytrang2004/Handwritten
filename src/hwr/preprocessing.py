"""
Image preprocessing utilities for character recognition
"""

import numpy as np
from PIL import Image, ImageFilter


def prepare_char_array(crop, size=(28, 28)):
    """
    Prepare character image array for model input

    Args:
        crop: Cropped character image array
        size: Target size (width, height)

    Returns:
        np.ndarray: Prepared character array or None if invalid
    """
    if crop.size == 0:
        return None

    height, width = crop.shape

    # Apply a small median filter to remove isolated noise
    try:
        pil_crop = Image.fromarray(crop)
        pil_crop = pil_crop.filter(ImageFilter.MedianFilter(size=3))
        arr = np.array(pil_crop)
    except Exception:
        arr = crop

    # Compute Otsu threshold for the crop for robust binarization
    def _otsu_threshold(a):
        # a: 2D uint8 array 0..255
        hist, _ = np.histogram(a.flatten(), bins=256, range=(0, 256))
        total = a.size
        sumB = 0.0
        wB = 0.0
        maximum = 0.0
        sum1 = np.dot(np.arange(256), hist)
        for i in range(256):
            wB += hist[i]
            if wB == 0:
                continue
            wF = total - wB
            if wF == 0:
                break
            sumB += i * hist[i]
            mB = sumB / wB
            mF = (sum1 - sumB) / wF
            between = wB * wF * (mB - mF) ** 2
            if between >= maximum:
                level = i
                maximum = between
        return int(level)

    try:
        otsu = _otsu_threshold(arr)
    except Exception:
        otsu = 127

    # Normalize crop to binary (ensure ink=255, background=0)
    bw = (arr > otsu).astype(np.uint8) * 255

    # Compute target square size with a small margin so strokes are not clipped
    margin = max(4, int(max(height, width) * 0.08))
    target_size = max(height, width) + 2 * margin

    # Create padded square and center the character by its centroid (center of mass)
    padded = np.zeros((target_size, target_size), dtype=np.uint8)
    y_off = (target_size - height) // 2
    x_off = (target_size - width) // 2
    padded[y_off : y_off + height, x_off : x_off + width] = bw

    # Compute centroid of ink in padded coordinates
    ys, xs = np.where(padded > 0)
    if ys.size == 0:
        # empty after threshold
        img = Image.fromarray(padded).resize(size, resample=Image.LANCZOS)
        return np.array(img)

    cy = int(np.mean(ys))
    cx = int(np.mean(xs))

    # Desired center
    center = target_size // 2
    shift_y = center - cy
    shift_x = center - cx

    # Apply shift by rolling and zeroing wrapped regions
    shifted = np.roll(padded, shift_y, axis=0)
    if shift_y > 0:
        shifted[:shift_y, :] = 0
    elif shift_y < 0:
        shifted[shift_y:, :] = 0

    shifted = np.roll(shifted, shift_x, axis=1)
    if shift_x > 0:
        shifted[:, :shift_x] = 0
    elif shift_x < 0:
        shifted[:, shift_x:] = 0

    # Resize to target size with high-quality resampling
    img = Image.fromarray(shifted).resize(size, resample=Image.LANCZOS)
    return np.array(img)


def prepare_char_image(crop, size=(28, 28)):
    """
    Prepare character image for model prediction

    Args:
        crop: Cropped character image array
        size: Target size (width, height)

    Returns:
        np.ndarray: Prepared image tensor (1, H, W, 1) / 255.0 or None
    """
    arr = prepare_char_array(crop, size)
    if arr is None:
        return None
    return arr.reshape((1, size[0], size[1], 1)) / 255.0


def segment_characters(img_gray, threshold=30, min_width=2, merge_gap=2, pad=2):
    """
    Segment characters from grayscale image

    Args:
        img_gray: PIL grayscale image
        threshold: Binarization threshold
        min_width: Minimum character width
        merge_gap: Gap to merge adjacent segments
        pad: Padding around segments

    Returns:
        tuple: (char_images, segments, char_arrays)
    """
    arr = np.array(img_gray)
    binary = arr > threshold

    if not binary.any():
        return [], [], []

    # Find column boundaries
    col_has_ink = binary.any(axis=0)
    segments = []

    start = None
    for idx, has_ink in enumerate(col_has_ink):
        if has_ink and start is None:
            start = idx
        elif not has_ink and start is not None:
            if idx - start >= min_width:
                segments.append((start, idx - 1))
            start = None

    if start is not None and len(col_has_ink) - start >= min_width:
        segments.append((start, len(col_has_ink) - 1))

    # Merge close segments
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        prev_start, prev_end = merged[-1]
        seg_start, seg_end = seg
        if seg_start - prev_end <= merge_gap + 1:
            merged[-1] = (prev_start, seg_end)
        else:
            merged.append(seg)

    # Extract character images
    char_images = []
    char_arrays = []
    kept_segments = []
    height, width = arr.shape

    for seg_start, seg_end in merged:
        x0 = max(0, seg_start - pad)
        x1 = min(width - 1, seg_end + pad)
        sub = binary[:, x0:x1 + 1]
        rows = np.where(sub.any(axis=1))[0]

        if rows.size == 0:
            continue

        y0 = max(0, rows.min() - pad)
        y1 = min(height - 1, rows.max() + pad)
        crop = arr[y0:y1 + 1, x0:x1 + 1]

        prepared_array = prepare_char_array(crop)
        if prepared_array is None:
            continue

        prepared = prepared_array.reshape((1, 28, 28, 1)) / 255.0
        char_images.append(prepared)
        char_arrays.append(prepared_array)
        kept_segments.append((seg_start, seg_end))

    return char_images, kept_segments, char_arrays
