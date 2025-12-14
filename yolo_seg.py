"""
Real-time human segmentation using YOLOv8 (segment task) with OpenCV.
Pipeline: capture -> segment -> mask -> composite.
Windows shown: original (con FPS), mask binaria, composite (siluetas blancas sobre negro).
"""

from __future__ import annotations

import sys
import time
from typing import Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# --------------- utils de captura y redimensionado -----------------
def resize_keep_aspect(frame, max_height: int = 480):
    """Resize frame to a max height, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if h <= max_height:
        return frame
    scale = max_height / float(h)
    new_size: Tuple[int, int] = (int(w * scale), int(h * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def capture_frame(cap: cv2.VideoCapture):
    """Capture and resize a frame; returns None if capture fails."""
    ok, frame = cap.read()
    if not ok:
        return None
    return resize_keep_aspect(frame, max_height=480)


# --------------- segmentación --------------------------------------
def segment_people(frame: np.ndarray, model: YOLO) -> np.ndarray:
    """
    Run YOLOv8 segmentation on the frame and return a global person mask (0-255, uint8).
    - Combines all person masks with bitwise OR.
    - Returns all-black mask if no persons are found.
    """
    h, w = frame.shape[:2]
    result = model(frame, verbose=False)[0]

    # If the model returns no masks, bail early.
    if result.masks is None or result.boxes is None:
        return np.zeros((h, w), dtype=np.uint8)

    masks = result.masks.data.cpu().numpy()  # shape: (N, H, W) in model space
    classes = result.boxes.cls.cpu().numpy().astype(int)  # shape: (N,)

    person_mask = np.zeros((h, w), dtype=np.uint8)
    for m, cls_id in zip(masks, classes):
        if cls_id != 0:  # 0 is "person" in COCO
            continue
        # Resize mask from model space to frame space.
        m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m_bin = (m_resized > 0.5).astype(np.uint8) * 255
        person_mask = cv2.bitwise_or(person_mask, m_bin.astype(np.uint8))

    # Suavizado para silueta más orgánica.
    person_mask = cv2.GaussianBlur(person_mask, (5, 5), 0)
    _, person_mask = cv2.threshold(person_mask, 127, 255, cv2.THRESH_BINARY)
    return person_mask


# --------------- composición y overlays ----------------------------
def make_composite(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return black background with white silhouettes."""
    composite = np.zeros_like(frame)
    composite[mask > 0] = (255, 255, 255)
    return composite


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Overlay FPS onto the frame."""
    out = frame.copy()
    text = f"FPS: {fps:.1f}"
    cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


# --------------- loop principal ------------------------------------
def main():
    # Carga modelo YOLOv8 de segmentación (usa uno ligero por defecto).
    model_path = "yolov8n-seg.pt"
    try:
        model = YOLO(model_path)
    except Exception as exc:
        print(f"No se pudo cargar el modelo {model_path}: {exc}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara (índice 0).", file=sys.stderr)
        return 1

    prev_time = time.time()

    while True:
        frame = capture_frame(cap)
        if frame is None:
            print("Frame no capturado. Saliendo.", file=sys.stderr)
            break

        mask = segment_people(frame, model)
        composite = make_composite(frame, mask)

        # FPS.
        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 1.0 / dt if dt > 0 else 0.0
        print(f"FPS: {fps:0.2f}", end="\r", flush=True)

        original_with_fps = draw_fps(frame, fps)

        cv2.imshow("original", original_with_fps)
        cv2.imshow("mask", mask)
        cv2.imshow("composite", composite)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
