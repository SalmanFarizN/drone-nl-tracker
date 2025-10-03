import pandas as pd
import numpy as np
import cv2


def encode_crop_for_qwen(orig_img, bbox, min_side=28, margin=8):
    """Extract and encode image crop for Qwen vision model processing.

    Crops the original image using the provided bounding box, adds margin for context,
    ensures minimum dimensions for model compatibility, and encodes as JPEG bytes.

    Args:
        orig_img (np.ndarray): Original image array in BGR format.
        bbox (list or np.ndarray): Bounding box coordinates [x1, y1, x2, y2].
        min_side (int, optional): Minimum pixel size for shortest side. Defaults to 28.
        margin (int, optional): Pixel margin to add around bounding box. Defaults to 8.

    Returns:
        bytes or None: JPEG-encoded image bytes, or None if crop is invalid/too small.
    """
    # bbox: [x1, y1, x2, y2]
    x1, y1, x2, y2 = map(int, bbox)
    # expand bbox a bit for context
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(orig_img.shape[1], x2 + margin)
    y2 = min(orig_img.shape[0], y2 + margin)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = orig_img[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return None

    # Ensure shortest side >= min_side
    if min(h, w) < min_side:
        scale = float(min_side) / max(1, min(h, w))
        new_w = max(int(round(w * scale)), min_side)
        new_h = max(int(round(h * scale)), min_side)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    ok, enc = cv2.imencode(".jpg", crop)
    if not ok:
        return None
    return enc.tobytes()


def encode_crop_for_gemma(orig_img, bbox):
    """Extract and encode image crop for Gemma vision model processing.

    Simple cropping function that extracts the bounding box region from the
    original image and encodes it as JPEG bytes for Gemma model input.

    Args:
        orig_img (np.ndarray): Original image array in BGR format.
        bbox (list or np.ndarray): Bounding box coordinates [x1, y1, x2, y2].

    Returns:
        bytes or None: JPEG-encoded image bytes, or None if crop is invalid.
    """
    # bbox: [x1, y1, x2, y2]
    x1, y1, x2, y2 = map(int, bbox)
    crop = orig_img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    ok, enc = cv2.imencode(".jpg", crop)
    if not ok:
        return None
    return enc.tobytes()
