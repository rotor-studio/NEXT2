"""
Simple real-time background subtraction demo.
Captures frames from the default camera, downsizes to max 480p, and composes
the views (original, mask, composite) on a single 1280x720 window.
The code is modular so the segmentation block can be swapped for a model later.
"""

from __future__ import annotations

import sys
import time
from typing import Tuple

import cv2
import numpy as np


def resize_frame(frame, max_height: int = 480):
    """Resize frame to a maximum height while keeping aspect ratio."""
    h, w = frame.shape[:2]
    if h <= max_height:
        return frame
    scale = max_height / float(h)
    new_size: Tuple[int, int] = (int(w * scale), int(h * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def compute_mask(frame, subtractor, kernel):
    """Compute a cleaned foreground mask using background subtraction."""
    raw_mask = subtractor.apply(frame)

    # Ensure binary mask.
    _, mask = cv2.threshold(raw_mask, 127, 255, cv2.THRESH_BINARY)

    # Morphological cleanup to reduce noise and close small gaps.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Light blur to smooth edges, then binarize again.
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return mask


def colorize_silhouettes(mask, palette, min_area: int = 500):
    """
    Color each detected silhouette with a different color using contours.
    - mask: binary mask (uint8 0/255).
    - palette: list of BGR tuples used cyclically.
    """
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        color = palette[idx % len(palette)]
        cv2.drawContours(colored, [cnt], -1, color, thickness=cv2.FILLED)

    return colored


def create_composite(mask):
    """Create a silhouette-style composite from the binary mask."""
    palette = [
        (255, 255, 255),  # white
        (0, 200, 255),    # yellow-ish
        (0, 255, 0),      # green
        (255, 0, 0),      # blue
        (0, 128, 255),    # orange-ish
        (255, 0, 255),    # magenta
    ]
    return colorize_silhouettes(mask, palette)


def fit_to_box(image, box_size: Tuple[int, int]):
    """Resize image to fit inside box_size keeping aspect ratio, letterboxed on black."""
    box_w, box_h = box_size
    h, w = image.shape[:2]
    scale = min(box_w / w, box_h / h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    y_off = (box_h - new_size[1]) // 2
    x_off = (box_w - new_size[0]) // 2
    canvas[y_off : y_off + new_size[1], x_off : x_off + new_size[0]] = resized
    return canvas


def add_overlay(image, label: str, fps_text: str | None = None):
    """Add label and optional FPS overlay to an image."""
    overlay = image.copy()
    legend = label if fps_text is None else f"{label} | {fps_text}"
    cv2.putText(overlay, legend, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return overlay


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara (índice 0).", file=sys.stderr)
        return 1

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=False
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    prev_time = time.perf_counter()
    fps = 0.0

    window_name = "NEXT2"
    target_canvas = (1280, 720)  # width, height
    thumb_size = (target_canvas[0] // 3, 240)  # width, height for previews on top row

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame no capturado. Saliendo.", file=sys.stderr)
            break

        resized = resize_frame(frame)
        mask = compute_mask(resized, subtractor, kernel)
        composite = create_composite(mask)

        # FPS calculation.
        now = time.perf_counter()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 1.0 / dt

        fps_text = f"FPS: {fps:0.1f}"

        # Prepare previews with legend; only the first shows FPS.
        previews = []
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        preview_data = (
            ("Original", resized, True),
            ("Mask", mask_bgr, False),
            ("Composite", composite, False),
        )
        for label, img, show_fps in preview_data:
            boxed = fit_to_box(img, thumb_size)
            overlay_text = fps_text if show_fps else None
            boxed = add_overlay(boxed, label, overlay_text)
            previews.append(boxed)

        # Build 720p canvas and place previews on the top row.
        canvas = np.zeros((target_canvas[1], target_canvas[0], 3), dtype=np.uint8)
        for idx, view in enumerate(previews):
            x0 = idx * thumb_size[0]
            canvas[0 : thumb_size[1], x0 : x0 + thumb_size[0]] = view

        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
