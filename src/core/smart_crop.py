"""Smart Crop module — face-centric training crops using MediaPipe face detection."""

import os
from PIL import Image
from typing import List, Tuple

# Model path — auto-downloaded if missing
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "blaze_face_short_range.tflite")
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"


def _ensure_model():
    """Download BlazeFace model if not present."""
    if os.path.exists(_MODEL_PATH):
        return
    os.makedirs(_MODEL_DIR, exist_ok=True)
    import urllib.request
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)


def _detect_largest_face(img: Image.Image):
    """Run face detection and return (face_cx, face_cy, face_h) as ratios [0-1], or None."""
    import mediapipe as mp
    import numpy as np

    _ensure_model()

    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        min_detection_confidence=0.5,
    )

    w, h = img.size
    rgb = np.array(img)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    with FaceDetector.create_from_options(options) as detector:
        result = detector.detect(mp_image)

    if not result.detections:
        return None

    best = max(
        result.detections,
        key=lambda d: d.bounding_box.width * d.bounding_box.height,
    )
    bb = best.bounding_box

    # Return as ratios so they're resolution-independent
    return (
        (bb.origin_x + bb.width / 2) / w,
        (bb.origin_y + bb.height / 2) / h,
        bb.height / h,
    )


def process_training_image(
    image: Image.Image, min_resolution: int = 512
) -> List[Tuple[str, Image.Image]]:
    """Generate face-centric training crops from an image.

    Uses a two-pass approach: upscales aggressively (4096px) for face
    detection to catch distant/small faces, then maps coordinates back
    to a 2048px crop source for reasonable output sizes.

    Returns a list of (label, cropped_image) tuples.
    If no face is found, returns the (possibly resized) original.
    """
    img = image.copy()

    # Prepare crop source: shortest side = 2048px (or original if already larger)
    w, h = img.size
    if min(w, h) < 2048:
        scale = 2048 / min(w, h)
        crop_img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    else:
        crop_img = img

    # Prepare detection image: shortest side = 4096px for better small-face detection
    if min(w, h) < 4096:
        det_scale = 4096 / min(w, h)
        det_img = img.resize((int(w * det_scale), int(h * det_scale)), Image.LANCZOS)
    else:
        det_img = img

    # Run face detection on the high-res detection image
    face = _detect_largest_face(det_img)

    if face is None:
        return [("original", crop_img)]

    # Map ratio-based coordinates to crop source dimensions
    cw, ch = crop_img.size
    face_x = face[0] * cw
    face_y = face[1] * ch
    face_h = face[2] * ch

    crops: List[Tuple[str, Image.Image]] = []

    def _square_crop(cx, cy, size):
        """Compute a square crop clamped to image bounds. Returns (l,t,r,b)."""
        size = min(size, cw, ch)  # can't exceed image dimensions
        l = int(cx - size / 2)
        t = int(cy - size / 2)
        l = max(0, min(l, cw - size))
        t = max(0, min(t, ch - size))
        return l, t, l + size, t + size

    # --- face_focus: square, 3.5x face height, centred on face ---
    size_ff = int(3.5 * face_h)
    l, t, r, b = _square_crop(face_x, face_y, size_ff)
    if min(r - l, b - t) >= min_resolution:
        crops.append(("face_focus", crop_img.crop((l, t, r, b))))

    # --- upper_body: square, 6x face height, shifted down 0.5x face height ---
    size_ub = int(6.0 * face_h)
    l, t, r, b = _square_crop(face_x, face_y + 0.5 * face_h, size_ub)
    if min(r - l, b - t) >= min_resolution:
        crops.append(("upper_body", crop_img.crop((l, t, r, b))))

    # --- full_body: vertical strip, full height, centred on face X ---
    strip_w = max(int(0.75 * ch), 1)
    strip_w = min(strip_w, cw)  # can't exceed image width
    half_w = strip_w // 2
    left = int(face_x - half_w)
    left = max(0, min(left, cw - strip_w))
    right = left + strip_w
    if min(right - left, ch) >= min_resolution:
        crops.append(("full_body", crop_img.crop((left, 0, right, ch))))

    # If all crops were discarded by quality control, return original
    if not crops:
        return [("original", crop_img)]

    return crops
