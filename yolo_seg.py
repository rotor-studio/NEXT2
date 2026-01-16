"""
Real-time human segmentation using YOLOv8 (segment task) with OpenCV.
Pipeline: captura -> segmentación -> máscara -> composite.
Ventana única 1280x720: vista original anotada (FPS + boxes) y máscara binaria.
"""

from __future__ import annotations

import sys
import time
import math
import threading
from typing import Tuple, Any, Optional
import os
import multiprocessing as mp
from pathlib import Path
from fractions import Fraction
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Configuración de rendimiento
RES_OPTIONS = [160, 240, 320, 360, 480]  # opciones de altura máxima
CURRENT_MAX_HEIGHT = 240  # valor inicial (se sobreescribe si hay guardado)
IMG_SIZE_OPTIONS = [320, 480, 640]  # resoluciones de inferencia YOLO
IMG_SIZE_IDX = 0  # índice inicial -> 320
PROCESS_EVERY_N = 2  # procesa 1 de cada N frames (2 = mitad)
CAP_WIDTH = 640   # resolución solicitada a la cámara (puede ajustarse)
CAP_HEIGHT = 480
DEVICE = None
RES_SAVE_FILE = Path(__file__).with_name("resolution.txt")
MODEL_SAVE_FILE = Path(__file__).with_name("model.txt")
MODEL_OPTIONS = {
    ord("a"): ("yolov8n-seg.pt", "yolov8n-seg (rapido)"),
    ord("s"): ("yolov8s-seg.pt", "yolov8s-seg (balance)"),
    ord("d"): ("yolov8m-seg.pt", "yolov8m-seg (mas pesado)"),
}
CURRENT_MODEL_KEY = ord("a")
CURRENT_MODEL_PATH = MODEL_OPTIONS[CURRENT_MODEL_KEY][0]
PEOPLE_LIMIT_OPTIONS = list(range(1, 21))
CURRENT_PEOPLE_LIMIT = 10
def env_flag(name: str, default: bool = True) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in {"0", "false", "no", "off"}


ENABLE_NDI = env_flag("NEXT_ENABLE_NDI", True)  # Permite desactivar NDI si está inestable
ENABLE_NDI_INPUT = env_flag("NEXT_ENABLE_NDI_INPUT", True)
ENABLE_NDI_OUTPUT = env_flag("NEXT_ENABLE_NDI_OUTPUT", True)  # Publicar máscara por NDI
ENABLE_NDI_TRANSLATIONS_OUTPUT = env_flag("NEXT_ENABLE_NDI_TRANSLATIONS_OUTPUT", False)
ENABLE_RTSP_INPUT = env_flag("NEXT_ENABLE_RTSP_INPUT", True)
RTSP_URL = os.environ.get(
    "NEXT_RTSP_URL",
    "rtsp://192.168.0.215:5543/c9fa2e29ba99618fc28088ccae18076b/live/channel0",
).strip()
RTSP_TRANSPORT = os.environ.get("NEXT_RTSP_TRANSPORT", "udp").strip().lower()
RTSP_RECONNECT_SEC = float(os.environ.get("NEXT_RTSP_RECONNECT_SEC", "2.0"))
USE_RTSP_THREAD = False
DATA_DIR = Path(__file__).with_name("DATA")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
VIDEO_FILES = sorted([p for p in DATA_DIR.glob("*") if p.suffix.lower() in VIDEO_EXTS])
DEFAULT_SOURCE = "camera"  # se ajusta en runtime tras probar NDI
CURRENT_SOURCE = DEFAULT_SOURCE  # "camera", "video", "ndi" o "rtsp"
CURRENT_VIDEO_INDEX = 0
NDI_PREFERRED_SOURCE = "MadMapper"
BLUR_KERNEL_OPTIONS = [1, 3, 5, 7, 9, 11, 13]
BLUR_KERNEL_IDX = 2  # valor inicial -> kernel 5
MASK_THRESH = 127
BLUR_ENABLED = True
HIGH_PRECISION_MODE = False  # modo alta precisión (imgsz alto y sin blur en mask_detail)
PERSIST_HOLD_SEC = 0.35  # segundos para mantener silueta tras perder detección
PERSIST_RISE_TAU = 0.12  # segundos para fade-in
PERSIST_FALL_TAU = 0.25  # segundos para fade-out
FOOTER_GAP_PX = 40  # separación entre vistas y GUI inferior
FOOTER_UI_OFFSET_Y = 40  # baja el GUI dentro del footer
UI_MARGIN_LEFT = 20  # margen izquierdo para textos y GUI
VIEW_LABEL_H = 24  # altura del label sobre cada vista
VIEW_OFFSET_Y = 20  # baja las vistas (label + imagen) dentro del canvas
CANVAS_WIDTH = 1536
CANVAS_HEIGHT = 816  # extra altura para GUI inferior
NDI_TR_OUTPUT_W = 1080
NDI_TR_OUTPUT_H = 1920
UI_BG_COLOR = (30, 30, 30)
VIEW_BG_COLOR = (0, 0, 0)
RIGHT_PANEL_BG_COLOR = (45, 45, 45)
ROI_WORK_BG_COLOR = (0, 0, 0)
RIGHT_PANEL_FOOTER_H = 70  # espacio bajo la tercera ventana para botones
RIGHT_PANEL_MARGIN_X = 20
RIGHT_PANEL_ROW_GAP = 10
FIT_CACHE = {}
FIT_CACHE_ORDER = []
FIT_CACHE_MAX = 6
USE_INFERENCE_THREAD = True
VIEW_HITBOXES = []
ACTIVE_VIEW_TAB = "mask"
ROI_LIST = []
ROI_ACTIVE_IDX = None
ROI_DRAG_OFFSET = (0, 0)
ROI_PANEL_BOUNDS = None  # (w, h) for right panel image area (local)
ROI_PANEL_BOUNDS_ABS = None  # (x0, y0, w, h) absolute for mouse
ROI_MAX = 10
ROI_STATE = []  # per-ROI assignment state (centroids/persistence)
ROI_DIRTY = False

# UI state (footer GUI)
UI_HITBOXES = []
UI_ACTIVE_SLIDER = None
UI_PENDING_MODEL_KEY = None
UI_PENDING_SOURCE = None
NDI_OUTPUT_MASK = "soft"  # "soft" o "detail"
FOOTER_CACHE = None
FOOTER_CACHE_KEY = None
FOOTER_CACHE_HITBOXES = []
FOOTER_PAD_Y = 10
SHOW_DETAIL_DEFAULT = False
SHOW_DETAIL = SHOW_DETAIL_DEFAULT
FLIP_INPUT = False

# --------------- utils de captura y redimensionado -----------------
def resize_keep_aspect(frame, max_height: int = 480):
    """Resize frame to a max height, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if h <= max_height:
        return frame
    scale = max_height / float(h)
    new_size: Tuple[int, int] = (int(w * scale), int(h * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def maybe_flip_input(frame: np.ndarray) -> np.ndarray:
    if not FLIP_INPUT:
        return frame
    return cv2.flip(frame, 1)


def capture_frame(cap: cv2.VideoCapture):
    """Capture and resize a frame; returns (frame, is_new)."""
    if cap is None:
        return None, False
    if hasattr(cap, "read_latest"):
        frame, is_new = cap.read_latest()
        if frame is None:
            return None, False
        frame = resize_keep_aspect(frame, max_height=CURRENT_MAX_HEIGHT)
        return maybe_flip_input(frame), is_new
    if isinstance(cap, NDIReceiver):
        frame = cap.capture()
        if frame is None:
            return None, False
        # Para NDI, limitamos a la resolución de entrada seleccionada (altura) y al imgsz actual.
        target_h = min(CURRENT_MAX_HEIGHT, IMG_SIZE_OPTIONS[IMG_SIZE_IDX])
        if HIGH_PRECISION_MODE:
            target_h = max(target_h, IMG_SIZE_OPTIONS[-1])
        frame = resize_keep_aspect(frame, max_height=target_h)
        return maybe_flip_input(frame), True
    ok, frame = cap.read()
    if not ok:
        return None, False
    frame = resize_keep_aspect(frame, max_height=CURRENT_MAX_HEIGHT)
    return maybe_flip_input(frame), True


# --------------- segmentación --------------------------------------
def segment_people(
    frame: np.ndarray,
    model: YOLO,
    people_limit: int,
    mask_thresh: int,
    blur_enabled: bool,
    compute_person_masks: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Run YOLOv8 segmentation and return:
    - mask_soft: smoothed binary mask (uint8) suitable for NDI/output.
    - mask_detail: sharper binary mask (uint8) preserving fine contours.
    - boxes: ndarray of person boxes (N, 4) in xyxy format.
    mask_detail generation:
      * Resize masks with linear interpolation for smoother edges.
      * Aggregate per-pixel max over persons (filtered by limit).
      * Optional light morphology to clean speckles.
      * Threshold with user-defined value.
    mask_soft generation:
      * Derived from mask_detail, optional Gaussian blur, re-threshold, optional light closing.
    """
    h, w = frame.shape[:2]
    imgsz = max(IMG_SIZE_OPTIONS) if HIGH_PRECISION_MODE else IMG_SIZE_OPTIONS[IMG_SIZE_IDX]
    result = model(frame, imgsz=imgsz, verbose=False, device=DEVICE)[0]

    # If the model returns no masks, bail early.
    if result.masks is None or result.boxes is None:
        empty = np.zeros((h, w), dtype=np.uint8)
        return empty, empty, np.empty((0, 4)), []

    masks = result.masks.data.cpu().numpy()  # shape: (N, H, W) in model space
    classes = result.boxes.cls.cpu().numpy().astype(int)  # shape: (N,)
    boxes_all = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    # Filtrar solo personas y aplicar límite ordenando por confianza.
    person_indices = [i for i, cls_id in enumerate(classes) if cls_id == 0]
    if not person_indices:
        empty = np.zeros((h, w), dtype=np.uint8)
        return empty, empty, np.empty((0, 4)), []
    person_indices = sorted(person_indices, key=lambda i: confs[i], reverse=True)[:people_limit]

    # Aggregate soft mask (float) to keep fine contours before thresholding.
    person_mask = np.zeros((h, w), dtype=np.float32)
    person_masks = [] if compute_person_masks else None
    person_boxes = []
    for idx in person_indices:
        m = masks[idx]
        # Resize mask from model space to frame space; keep float precision for later threshold.
        m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        person_mask = np.maximum(person_mask, m_resized)
        if compute_person_masks:
            person_masks.append(m_resized)
        person_boxes.append(boxes_all[idx])

    # Convert to uint8 scale 0-255 for morphology/thresholding.
    person_mask_u8 = np.clip(person_mask * 255.0, 0, 255).astype(np.uint8)

    # Morphology to clean speckles while preserving contour.
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    person_mask_u8 = cv2.morphologyEx(person_mask_u8, cv2.MORPH_OPEN, morph_kernel, iterations=1)
    person_mask_u8 = cv2.morphologyEx(person_mask_u8, cv2.MORPH_CLOSE, morph_kernel, iterations=1)

    # mask_detail: threshold with minimal blur (none by default).
    mask_detail = person_mask_u8.copy()
    _, mask_detail = cv2.threshold(mask_detail, mask_thresh, 255, cv2.THRESH_BINARY)

    # mask_soft: derived from mask_detail, optional blur + closing to smooth edges.
    mask_soft = mask_detail.copy()
    ksize = BLUR_KERNEL_OPTIONS[BLUR_KERNEL_IDX]
    if blur_enabled and ksize > 1:
        mask_soft = cv2.GaussianBlur(mask_soft, (ksize, ksize), 0)
        _, mask_soft = cv2.threshold(mask_soft, mask_thresh, 255, cv2.THRESH_BINARY)
    # Light closing to reduce jaggies without over-rounding.
    mask_soft = cv2.morphologyEx(mask_soft, cv2.MORPH_CLOSE, morph_kernel, iterations=1)

    person_masks_soft = []
    if compute_person_masks and person_masks:
        for m in person_masks:
            m_u8 = np.clip(m * 255.0, 0, 255).astype(np.uint8)
            _, m_u8 = cv2.threshold(m_u8, mask_thresh, 255, cv2.THRESH_BINARY)
            if ksize > 1:
                m_u8 = cv2.GaussianBlur(m_u8, (ksize, ksize), 0)
                _, m_u8 = cv2.threshold(m_u8, mask_thresh, 255, cv2.THRESH_BINARY)
            m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
            person_masks_soft.append(m_u8)
    return mask_soft, mask_detail, np.array(person_boxes), person_masks_soft


# --------------- composición y overlays ----------------------------
def make_composite(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return black background with white silhouettes."""
    composite = np.zeros_like(frame)
    composite[mask > 0] = (255, 255, 255)
    return composite


def fit_to_box(image: np.ndarray, box_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to fit inside box_size keeping aspect ratio; no extra padding if width matches."""
    box_w, box_h = box_size
    h, w = image.shape[:2]
    cache_key = (id(image), box_w, box_h, w, h)
    cached = FIT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    scale = min(box_w / w, box_h / h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    if new_size[0] == box_w:
        # Height is already within box_h; crop if needed to avoid left padding.
        if new_size[1] > box_h:
            resized = resized[0:box_h, 0:box_w]
        canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
        canvas[0 : resized.shape[0], 0 : resized.shape[1]] = resized
        FIT_CACHE[cache_key] = canvas
        FIT_CACHE_ORDER.append(cache_key)
        if len(FIT_CACHE_ORDER) > FIT_CACHE_MAX:
            old_key = FIT_CACHE_ORDER.pop(0)
            FIT_CACHE.pop(old_key, None)
        return canvas

    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    y_off = 0  # align to top
    x_off = (box_w - new_size[0]) // 2
    canvas[y_off : y_off + new_size[1], x_off : x_off + new_size[0]] = resized
    FIT_CACHE[cache_key] = canvas
    FIT_CACHE_ORDER.append(cache_key)
    if len(FIT_CACHE_ORDER) > FIT_CACHE_MAX:
        old_key = FIT_CACHE_ORDER.pop(0)
        FIT_CACHE.pop(old_key, None)
    return canvas


def add_overlay(image: np.ndarray, label: str) -> np.ndarray:
    """Backwards-compatible wrapper (deprecated)."""
    overlay = image.copy()
    cv2.putText(overlay, label, (UI_MARGIN_LEFT, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return overlay


def make_labeled_view(image: np.ndarray, label: str, box_size: Tuple[int, int]) -> np.ndarray:
    """Place label above the image; image is fitted below the label strip."""
    box_w, box_h = box_size
    label_h = min(VIEW_LABEL_H, max(16, box_h // 6))
    view = np.full((box_h, box_w, 3), VIEW_BG_COLOR, dtype=np.uint8)
    cv2.rectangle(view, (0, 0), (box_w - 1, label_h), UI_BG_COLOR, -1)
    cv2.putText(view, label, (UI_MARGIN_LEFT, label_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1)
    if box_h - label_h > 0:
        fitted = fit_to_box(image, (box_w, box_h - label_h))
        view[label_h : label_h + fitted.shape[0], 0 : fitted.shape[1]] = fitted
    return view


def make_tabbed_view(
    image: np.ndarray, box_size: Tuple[int, int], tabs: Tuple[Tuple[str, str], ...], active_key: str
) -> Tuple[np.ndarray, list]:
    """Create a view with a tab bar on top; returns view and tab rects (relative)."""
    box_w, box_h = box_size
    label_h = min(VIEW_LABEL_H, max(16, box_h // 6))
    view = np.full((box_h, box_w, 3), VIEW_BG_COLOR, dtype=np.uint8)
    cv2.rectangle(view, (0, 0), (box_w - 1, label_h), UI_BG_COLOR, -1)

    tab_rects = []
    x = UI_MARGIN_LEFT
    for key, label in tabs:
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        tab_w = max(70, text_size[0] + 20)
        rect = (x, 2, min(x + tab_w, box_w - 4), label_h - 2)
        is_active = key == active_key
        fill = (70, 70, 70) if is_active else (45, 45, 45)
        cv2.rectangle(view, (rect[0], rect[1]), (rect[2], rect[3]), fill, -1)
        cv2.rectangle(view, (rect[0], rect[1]), (rect[2], rect[3]), (90, 90, 90), 1)
        ty = rect[1] + (rect[3] - rect[1] + text_size[1]) // 2
        tx = rect[0] + (rect[2] - rect[0] - text_size[0]) // 2
        cv2.putText(view, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1)
        tab_rects.append((key, rect))
        x = rect[2] + 8

    if box_h - label_h > 0:
        fitted = fit_to_box(image, (box_w, box_h - label_h))
        view[label_h : label_h + fitted.shape[0], 0 : fitted.shape[1]] = fitted
    return view, tab_rects


def _roi_rect(roi: dict) -> Tuple[int, int, int, int]:
    half = int(roi["half"])
    cx = int(roi["cx"])
    cy = int(roi["cy"])
    return cx - half, cy - half, cx + half, cy + half


def _clamp_roi(roi: dict, w: int, h: int) -> None:
    half = max(20, int(roi["half"]))
    max_half = max(20, min(w, h) // 2)
    half = min(half, max_half)
    cx = int(roi["cx"])
    cy = int(roi["cy"])
    cx = max(half, min(w - half, cx))
    cy = max(half, min(h - half, cy))
    roi["half"] = half
    roi["cx"] = cx
    roi["cy"] = cy


def add_roi() -> None:
    global ROI_LIST, ROI_DIRTY
    if ROI_PANEL_BOUNDS is None or len(ROI_LIST) >= ROI_MAX:
        return
    w, h = ROI_PANEL_BOUNDS
    half = max(20, min(w, h) // 6)
    roi = {"cx": w // 2, "cy": h // 2, "half": half}
    _clamp_roi(roi, w, h)
    ROI_LIST.append(roi)
    ROI_DIRTY = True


def translate_masks_to_rois(
    masks: list, roi_list: list, roi_size: Tuple[int, int], now: float, dt: float
) -> np.ndarray:
    """Place each mask into a ROI (one per ROI) with stable assignment and fade."""
    w, h = roi_size
    out = np.zeros((h, w), dtype=np.uint8)
    if not masks or not roi_list:
        return out

    global ROI_STATE
    if len(ROI_STATE) != len(roi_list):
        ROI_STATE = [
            {"centroid": None, "persist_mask": None, "last_detect_time": None, "last_detect_mask": None}
            for _ in roi_list
        ]

    mask_info = []
    for m in masks:
        if m is None or not np.any(m):
            continue
        ys, xs = np.where(m > 0)
        if ys.size == 0 or xs.size == 0:
            continue
        cx = int(xs.mean())
        cy = int(ys.mean())
        mask_info.append((m, (cx, cy)))
    if not mask_info:
        return out

    mask_info.sort(key=lambda item: item[1][0])
    unused = set(range(len(mask_info)))
    assign = [None] * len(roi_list)
    dist_thresh = max(30, min(w, h) // 10)

    for i, state in enumerate(ROI_STATE):
        if state["centroid"] is None:
            continue
        best = None
        best_d = None
        for idx in list(unused):
            _, (cx, cy) = mask_info[idx]
            pcx, pcy = state["centroid"]
            d = (cx - pcx) ** 2 + (cy - pcy) ** 2
            if best_d is None or d < best_d:
                best_d = d
                best = idx
        if best is not None and best_d is not None and best_d <= dist_thresh * dist_thresh:
            assign[i] = best
            unused.remove(best)

    for i in range(len(roi_list)):
        if assign[i] is None and unused:
            assign[i] = min(unused)
            unused.remove(assign[i])

    for roi_idx, mask_idx in enumerate(assign):
        if mask_idx is None:
            empty = np.zeros((h, w), dtype=np.uint8)
            state = ROI_STATE[roi_idx]
            persist_mask, last_time, last_mask = update_persistent_mask(
                state["persist_mask"], empty, state["last_detect_time"], state["last_detect_mask"], now, dt
            )
            state["persist_mask"] = persist_mask
            state["last_detect_time"] = last_time
            state["last_detect_mask"] = last_mask
            out = np.maximum(out, np.clip(persist_mask * 255.0, 0, 255).astype(np.uint8))
            continue
        m, centroid = mask_info[mask_idx]
        ys, xs = np.where(m > 0)
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        crop = m[y1 : y2 + 1, x1 : x2 + 1]
        ch, cw = crop.shape[:2]
        if ch < 2 or cw < 2:
            continue
        roi = roi_list[roi_idx]
        half = int(roi["half"])
        roi_w = half * 2
        roi_h = half * 2
        scale = min(roi_w / float(cw), roi_h / float(ch), 1.0)
        if scale < 1.0:
            new_w = max(2, int(cw * scale))
            new_h = max(2, int(ch * scale))
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            ch, cw = crop.shape[:2]

        state = ROI_STATE[roi_idx]
        state["centroid"] = centroid

        dest_x1 = roi["cx"] - (cw // 2)
        dest_y1 = roi["cy"] - (ch // 2)
        roi_x1 = roi["cx"] - half
        roi_y1 = roi["cy"] - half
        roi_x2 = roi["cx"] + half
        roi_y2 = roi["cy"] + half
        dest_x1 = max(roi_x1, min(roi_x2 - cw, dest_x1))
        dest_y1 = max(roi_y1, min(roi_y2 - ch, dest_y1))
        dest_x2 = min(dest_x1 + cw, w)
        dest_y2 = min(dest_y1 + ch, h)
        src_w = dest_x2 - dest_x1
        src_h = dest_y2 - dest_y1
        if src_w <= 0 or src_h <= 0:
            continue

        placed = np.zeros((h, w), dtype=np.uint8)
        placed[dest_y1:dest_y2, dest_x1:dest_x2] = crop[:src_h, :src_w]
        persist_mask, last_time, last_mask = update_persistent_mask(
            state["persist_mask"], placed, state["last_detect_time"], state["last_detect_mask"], now, dt
        )
        state["persist_mask"] = persist_mask
        state["last_detect_time"] = last_time
        state["last_detect_mask"] = last_mask
        out = np.maximum(out, np.clip(persist_mask * 255.0, 0, 255).astype(np.uint8))
    return out


def update_persistent_mask(
    persist_mask: Optional[np.ndarray],
    mask_binary: np.ndarray,
    last_detect_time: Optional[float],
    last_detect_mask: Optional[np.ndarray],
    now: float,
    dt: float,
) -> Tuple[np.ndarray, Optional[float], Optional[np.ndarray]]:
    """Temporal smoothing + hold to reduce flicker; returns float mask in [0,1]."""
    if persist_mask is None or persist_mask.shape != mask_binary.shape:
        persist_mask = np.zeros(mask_binary.shape, dtype=np.float32)
        last_detect_mask = None

    has_detection = np.any(mask_binary > 0)
    if has_detection:
        last_detect_time = now
        last_detect_mask = (mask_binary > 0).astype(np.float32)

    if last_detect_mask is not None and last_detect_mask.shape != persist_mask.shape:
        last_detect_mask = None
    if last_detect_time is not None and (now - last_detect_time) <= PERSIST_HOLD_SEC:
        target_on = last_detect_mask if last_detect_mask is not None else np.zeros_like(persist_mask)
    else:
        target_on = np.zeros_like(persist_mask)

    rise_rate = 1.0 - math.exp(-dt / max(PERSIST_RISE_TAU, 1e-6))
    fall_rate = 1.0 - math.exp(-dt / max(PERSIST_FALL_TAU, 1e-6))

    on_mask = target_on > 0.0
    persist_mask[on_mask] = (1.0 - rise_rate) * persist_mask[on_mask] + rise_rate * 1.0
    persist_mask[~on_mask] = (1.0 - fall_rate) * persist_mask[~on_mask]
    return persist_mask, last_detect_time, last_detect_mask


def draw_boxes(frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Draw bounding boxes on the frame."""
    out = frame.copy()
    if boxes is None:
        return out
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return out


def add_header(
    canvas: np.ndarray,
    fps: float,
    device: str,
    res_text: str,
    people_count: int,
    source_label: str,
    cap_fps: float | None = None,
) -> None:
    """Draw a single header line with source, resolution, FPS, device, model and people count at the top of the canvas."""
    current_model_label = MODEL_OPTIONS.get(CURRENT_MODEL_KEY, (CURRENT_MODEL_PATH, CURRENT_MODEL_PATH))[1]
    fps_str = f"{fps:06.1f}"  # ancho fijo para evitar saltos en el texto
    cap_fps_text = f"{cap_fps:0.1f}" if cap_fps is not None and cap_fps > 0 else "n/a"
    text = (
        f"SRC: {source_label} | RES: {res_text} | MAXH: {CURRENT_MAX_HEIGHT} | FPS: {fps_str} | "
        f"CAP_FPS: {cap_fps_text} | GPU: {device} | MODEL: {current_model_label} | "
        f"PEOPLE NOW: {people_count} | PREC: {'HIGH' if HIGH_PRECISION_MODE else 'NORM'}"
    )
    font_scale = 0.5
    max_w = canvas.shape[1] - UI_MARGIN_LEFT - 10
    render_text = text
    while True:
        size, _ = cv2.getTextSize(render_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        if size[0] <= max_w or len(render_text) <= 8:
            break
        render_text = render_text[:-4] + "..."
    cv2.putText(canvas, render_text, (UI_MARGIN_LEFT, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)


def _draw_label(canvas: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)


def _draw_button(canvas: np.ndarray, rect: Tuple[int, int, int, int], text: str, active: bool, enabled: bool) -> None:
    x1, y1, x2, y2 = rect
    if not enabled:
        fill = (40, 40, 40)
        text_color = (120, 120, 120)
    else:
        fill = (70, 120, 70) if active else (60, 60, 60)
        text_color = (235, 235, 235)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (90, 90, 90), 1)
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    tx = x1 + max(4, (x2 - x1 - text_size[0]) // 2)
    ty = y1 + (y2 - y1 + text_size[1]) // 2
    cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)


def _draw_slider(
    canvas: np.ndarray,
    rect: Tuple[int, int, int, int],
    value: float,
    min_val: float,
    max_val: float,
    label: str,
    value_text: str,
) -> None:
    x1, y1, x2, y2 = rect
    _draw_label(canvas, label, x1, y1 - 6)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (50, 50, 50), -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (90, 90, 90), 1)
    if max_val > min_val:
        ratio = (value - min_val) / float(max_val - min_val)
    else:
        ratio = 0.0
    ratio = max(0.0, min(1.0, ratio))
    knob_x = x1 + int(ratio * (x2 - x1))
    knob_y = (y1 + y2) // 2
    cv2.circle(canvas, (knob_x, knob_y), 6, (190, 190, 190), -1)
    cv2.putText(canvas, value_text, (x2 + 8, y2 + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)


def add_footer(canvas: np.ndarray, current_res: int, footer_h: int | None = None, footer_top: int | None = None) -> None:
    """Draw footer GUI with mouse-interactive controls."""
    global UI_HITBOXES
    UI_HITBOXES = []
    if footer_h is None:
        footer_h = 380
    if footer_top is None:
        footer_top = canvas.shape[0] - footer_h
    cv2.rectangle(canvas, (0, footer_top), (canvas.shape[1], canvas.shape[0]), UI_BG_COLOR, -1)

    row1_y = footer_top + FOOTER_UI_OFFSET_Y + FOOTER_PAD_Y
    row2_y = row1_y + 70 + 50
    row3_y = row2_y + 70
    row4_y = row3_y + 70
    btn_h = 24
    gap = 8
    x = UI_MARGIN_LEFT
    max_w = canvas.shape[1] - UI_MARGIN_LEFT

    def _wrap_if_needed(next_w: int) -> None:
        nonlocal x, row1_y
        if x + next_w > max_w:
            row1_y += 30
            x = UI_MARGIN_LEFT

    # RES buttons
    _draw_label(canvas, "RES (1-5)", x, row1_y - 6)
    for idx, res in enumerate(RES_OPTIONS):
        rect = (x, row1_y, x + 52, row1_y + btn_h)
        _draw_button(canvas, rect, f"{idx+1}:{res}", current_res == res, True)
        UI_HITBOXES.append({"type": "button", "id": "res", "rect": rect, "value": idx, "enabled": True})
        x += 52 + gap
    x += 12

    # MODEL buttons
    _wrap_if_needed(230)
    _draw_label(canvas, "MODEL (a/s/d)", x, row1_y - 6)
    model_items = [(ord("a"), "a:n"), (ord("s"), "s:s"), (ord("d"), "d:m")]
    for key, label in model_items:
        rect = (x, row1_y, x + 56, row1_y + btn_h)
        _draw_button(canvas, rect, label, CURRENT_MODEL_KEY == key, True)
        UI_HITBOXES.append({"type": "button", "id": "model", "rect": rect, "value": key, "enabled": True})
        x += 56 + gap
    x += 12

    # SOURCE buttons
    _wrap_if_needed(220)
    _draw_label(canvas, "SRC (c/v/n)", x, row1_y - 6)
    source_items = [
        ("camera", "c:cam", True),
        ("video", "v:vid", bool(VIDEO_FILES)),
        ("ndi", "n:ndi", bool(ENABLE_NDI and ENABLE_NDI_INPUT)),
        ("rtsp", "r:rtsp", bool(ENABLE_RTSP_INPUT and RTSP_URL)),
    ]
    for src, label, enabled in source_items:
        rect = (x, row1_y, x + 60, row1_y + btn_h)
        _draw_button(canvas, rect, label, CURRENT_SOURCE == src, enabled)
        UI_HITBOXES.append({"type": "button", "id": "source", "rect": rect, "value": src, "enabled": enabled})
        x += 60 + gap

    # Toggle buttons row (use remaining width)
    x = UI_MARGIN_LEFT
    toggle_y = row1_y + btn_h + 40
    blur_rect = (x, toggle_y, x + 72, toggle_y + btn_h)
    _draw_button(canvas, blur_rect, "b:blur", BLUR_ENABLED, True)
    UI_HITBOXES.append({"type": "toggle", "id": "blur_enabled", "rect": blur_rect, "enabled": True})
    x += 72 + gap + 8

    hi_rect = (x, toggle_y, x + 90, toggle_y + btn_h)
    _draw_button(canvas, hi_rect, "h:hi", HIGH_PRECISION_MODE, True)
    UI_HITBOXES.append({"type": "toggle", "id": "high_prec", "rect": hi_rect, "enabled": True})
    x += 90 + gap + 12

    _draw_label(canvas, "MASK (m)", x, toggle_y - 6)
    soft_rect = (x, toggle_y, x + 70, toggle_y + btn_h)
    _draw_button(canvas, soft_rect, "soft", not SHOW_DETAIL, True)
    UI_HITBOXES.append({"type": "button", "id": "mask_view", "rect": soft_rect, "value": "soft", "enabled": True})
    x += 70 + gap
    detail_rect = (x, toggle_y, x + 70, toggle_y + btn_h)
    _draw_button(canvas, detail_rect, "detail", SHOW_DETAIL, True)
    UI_HITBOXES.append({"type": "button", "id": "mask_view", "rect": detail_rect, "value": "detail", "enabled": True})
    x += 70 + gap + 12

    flip_rect = (x, toggle_y, x + 90, toggle_y + btn_h)
    _draw_button(canvas, flip_rect, "f:flip", FLIP_INPUT, True)
    UI_HITBOXES.append({"type": "toggle", "id": "flip", "rect": flip_rect, "enabled": True})
    x += 90 + gap + 12


    # NDI toggles moved to per-view bottom-left corners.

    # Row 2: PEOPLE slider + BLUR kernel
    x = UI_MARGIN_LEFT
    people_rect = (x, row2_y + 20, x + 260, row2_y + 32)
    _draw_slider(
        canvas,
        people_rect,
        CURRENT_PEOPLE_LIMIT,
        PEOPLE_LIMIT_OPTIONS[0],
        PEOPLE_LIMIT_OPTIONS[-1],
        "PEOPLE (+/-)",
        str(CURRENT_PEOPLE_LIMIT),
    )
    UI_HITBOXES.append({"type": "slider", "id": "people", "rect": people_rect})
    x += 260 + 60

    kernel_rect = (x, row2_y + 20, x + 220, row2_y + 32)
    _draw_slider(
        canvas,
        kernel_rect,
        BLUR_KERNEL_IDX,
        0,
        len(BLUR_KERNEL_OPTIONS) - 1,
        "KERN (o/p)",
        str(BLUR_KERNEL_OPTIONS[BLUR_KERNEL_IDX]),
    )
    UI_HITBOXES.append({"type": "slider_steps", "id": "blur_kernel", "rect": kernel_rect, "steps": BLUR_KERNEL_OPTIONS})

    # Row 3: THRESH + IMG SIZE
    x = UI_MARGIN_LEFT
    thresh_rect = (x, row3_y + 20, x + 340, row3_y + 32)
    _draw_slider(canvas, thresh_rect, MASK_THRESH, 0, 255, "THRESH (j/k)", str(MASK_THRESH))
    UI_HITBOXES.append({"type": "slider", "id": "mask_thresh", "rect": thresh_rect})
    x += 340 + 60

    imgsz_rect = (x, row3_y + 20, x + 220, row3_y + 32)
    _draw_slider(
        canvas,
        imgsz_rect,
        IMG_SIZE_IDX,
        0,
        len(IMG_SIZE_OPTIONS) - 1,
        "IMG (,/.)",
        str(IMG_SIZE_OPTIONS[IMG_SIZE_IDX]),
    )
    UI_HITBOXES.append({"type": "slider_steps", "id": "img_size", "rect": imgsz_rect, "steps": IMG_SIZE_OPTIONS})

    # Row 4: persistence sliders
    x = UI_MARGIN_LEFT
    hold_rect = (x, row4_y + 20, x + 260, row4_y + 32)
    _draw_slider(
        canvas,
        hold_rect,
        PERSIST_HOLD_SEC,
        0.1,
        1.0,
        "PERSIST HOLD",
        f"{PERSIST_HOLD_SEC:.2f}s",
    )
    UI_HITBOXES.append({"type": "slider", "id": "persist_hold", "rect": hold_rect})
    x += 260 + 60

    rise_rect = (x, row4_y + 20, x + 220, row4_y + 32)
    _draw_slider(
        canvas,
        rise_rect,
        PERSIST_RISE_TAU,
        0.05,
        0.5,
        "PERSIST RISE",
        f"{PERSIST_RISE_TAU:.2f}s",
    )
    UI_HITBOXES.append({"type": "slider", "id": "persist_rise", "rect": rise_rect})
    x += 220 + 60

    fall_rect = (x, row4_y + 20, x + 220, row4_y + 32)
    _draw_slider(
        canvas,
        fall_rect,
        PERSIST_FALL_TAU,
        0.1,
        0.8,
        "PERSIST FALL",
        f"{PERSIST_FALL_TAU:.2f}s",
    )
    UI_HITBOXES.append({"type": "slider", "id": "persist_fall", "rect": fall_rect})

    # Bottom row for right panel moved to main canvas.


def _point_in_rect(x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def _apply_button_action(item: dict) -> None:
    global UI_PENDING_MODEL_KEY, UI_PENDING_SOURCE, SHOW_DETAIL, FLIP_INPUT, HIGH_PRECISION_MODE
    global BLUR_ENABLED, ROI_LIST, ROI_STATE, ROI_DIRTY
    global ENABLE_NDI_INPUT, ENABLE_NDI_OUTPUT, ENABLE_NDI_TRANSLATIONS_OUTPUT, CURRENT_SOURCE
    global ENABLE_RTSP_INPUT, RTSP_URL
    item_id = item["id"]
    if item_id == "res":
        set_resolution_by_index(int(item["value"]))
        save_settings()
        return
    if item_id == "model":
        UI_PENDING_MODEL_KEY = item["value"]
        return
    if item_id == "source":
        UI_PENDING_SOURCE = item["value"]
        return
    if item_id == "mask_view":
        SHOW_DETAIL = item["value"] == "detail"
        save_settings()
        return
    if item_id == "flip":
        FLIP_INPUT = not FLIP_INPUT
        save_settings()
        return
    if item_id == "high_prec":
        HIGH_PRECISION_MODE = not HIGH_PRECISION_MODE
        save_settings()
        return
    if item_id == "blur_enabled":
        BLUR_ENABLED = not BLUR_ENABLED
        save_settings()
        return
    if item_id == "roi_add":
        add_roi()
        return
    if item_id == "roi_remove":
        if ROI_LIST:
            ROI_LIST.pop()
            ROI_STATE = ROI_STATE[: len(ROI_LIST)]
            ROI_DIRTY = True
        return
    if item_id == "roi_reset":
        ROI_LIST.clear()
        ROI_STATE.clear()
        ROI_DIRTY = True
        return
    if item_id == "ndi_input":
        ENABLE_NDI_INPUT = not ENABLE_NDI_INPUT
        if not ENABLE_NDI_INPUT and CURRENT_SOURCE == "ndi":
            UI_PENDING_SOURCE = "camera"
        save_settings()
        return
    if item_id == "ndi_output":
        ENABLE_NDI_OUTPUT = not ENABLE_NDI_OUTPUT
        save_settings()
        return
    if item_id == "ndi_trans_output":
        ENABLE_NDI_TRANSLATIONS_OUTPUT = not ENABLE_NDI_TRANSLATIONS_OUTPUT
        save_settings()
        return
    if item_id == "rtsp_input":
        ENABLE_RTSP_INPUT = not ENABLE_RTSP_INPUT
        if not ENABLE_RTSP_INPUT and CURRENT_SOURCE == "rtsp":
            UI_PENDING_SOURCE = "camera"
        save_settings()
        return
    if item_id == "rtsp_cfg":
        load_rtsp_url_from_settings()
        save_settings()
        UI_PENDING_SOURCE = "rtsp"
        return


def _apply_slider_action(item: dict, x: int) -> None:
    global CURRENT_PEOPLE_LIMIT, BLUR_KERNEL_IDX, MASK_THRESH, IMG_SIZE_IDX
    x1, _, x2, _ = item["rect"]
    if x2 <= x1:
        return
    ratio = (x - x1) / float(x2 - x1)
    ratio = max(0.0, min(1.0, ratio))
    item_id = item["id"]

    if item["type"] == "slider_steps":
        steps = item["steps"]
        idx = int(round(ratio * (len(steps) - 1))) if steps else 0
        idx = max(0, min(idx, len(steps) - 1))
        if item_id == "blur_kernel" and idx != BLUR_KERNEL_IDX:
            BLUR_KERNEL_IDX = idx
            save_settings()
        elif item_id == "img_size" and idx != IMG_SIZE_IDX:
            IMG_SIZE_IDX = idx
            save_settings()
        return

    if item_id == "people":
        min_val, max_val = PEOPLE_LIMIT_OPTIONS[0], PEOPLE_LIMIT_OPTIONS[-1]
        val = int(round(min_val + ratio * (max_val - min_val)))
        val = max(min_val, min(max_val, val))
        if val != CURRENT_PEOPLE_LIMIT:
            CURRENT_PEOPLE_LIMIT = val
            save_settings()
        return
    if item_id == "mask_thresh":
        val = int(round(ratio * 255))
        val = max(0, min(255, val))
        if val != MASK_THRESH:
            MASK_THRESH = val
            save_settings()
        return
    if item_id == "persist_hold":
        min_val, max_val = 0.1, 1.0
        val = min_val + ratio * (max_val - min_val)
        if abs(val - PERSIST_HOLD_SEC) > 1e-3:
            globals()["PERSIST_HOLD_SEC"] = val
        return
    if item_id == "persist_rise":
        min_val, max_val = 0.05, 0.5
        val = min_val + ratio * (max_val - min_val)
        if abs(val - PERSIST_RISE_TAU) > 1e-3:
            globals()["PERSIST_RISE_TAU"] = val
        return
    if item_id == "persist_fall":
        min_val, max_val = 0.1, 0.8
        val = min_val + ratio * (max_val - min_val)
        if abs(val - PERSIST_FALL_TAU) > 1e-3:
            globals()["PERSIST_FALL_TAU"] = val
        return


def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
    global UI_ACTIVE_SLIDER, ROI_ACTIVE_IDX, ROI_DRAG_OFFSET, ROI_DIRTY

    if event == cv2.EVENT_LBUTTONDOWN:
        if ROI_PANEL_BOUNDS_ABS is not None:
            bx, by, bw, bh = ROI_PANEL_BOUNDS_ABS
            if bx <= x <= bx + bw and by <= y <= by + bh:
                local_x = x - bx
                local_y = y - by
                for idx, roi in enumerate(reversed(ROI_LIST)):
                    real_idx = len(ROI_LIST) - 1 - idx
                    rx1, ry1, rx2, ry2 = _roi_rect(roi)
                    if rx1 <= local_x <= rx2 and ry1 <= local_y <= ry2:
                        ROI_ACTIVE_IDX = real_idx
                        ROI_DRAG_OFFSET = (roi["cx"] - local_x, roi["cy"] - local_y)
                        return

        for item in VIEW_HITBOXES + UI_HITBOXES:
            if not item.get("enabled", True):
                continue
            if _point_in_rect(x, y, item["rect"]):
                if item.get("type") == "view_tab":
                    globals()["ACTIVE_VIEW_TAB"] = item["value"]
                    return
                if item["type"] in {"slider", "slider_steps"}:
                    UI_ACTIVE_SLIDER = item
                    _apply_slider_action(item, x)
                else:
                    _apply_button_action(item)
                return

    if event == cv2.EVENT_MOUSEMOVE:
        if ROI_ACTIVE_IDX is not None and ROI_PANEL_BOUNDS_ABS is not None and flags & cv2.EVENT_FLAG_LBUTTON:
            bx, by, bw, bh = ROI_PANEL_BOUNDS_ABS
            local_x = x - bx
            local_y = y - by
            dx, dy = ROI_DRAG_OFFSET
            roi = ROI_LIST[ROI_ACTIVE_IDX]
            roi["cx"] = int(local_x + dx)
            roi["cy"] = int(local_y + dy)
            _clamp_roi(roi, bw, bh)
            ROI_DIRTY = True
            return
        if UI_ACTIVE_SLIDER is not None and flags & cv2.EVENT_FLAG_LBUTTON:
            _apply_slider_action(UI_ACTIVE_SLIDER, x)
            return

    if event == cv2.EVENT_MOUSEWHEEL:
        if ROI_PANEL_BOUNDS_ABS is None:
            return
        bx, by, bw, bh = ROI_PANEL_BOUNDS_ABS
        if not (bx <= x <= bx + bw and by <= y <= by + bh):
            return
        delta = (flags >> 16) & 0xFFFF
        if delta & 0x8000:
            delta = delta - 0x10000
        if delta == 0:
            return
        local_x = x - bx
        local_y = y - by
        target_idx = None
        for idx in range(len(ROI_LIST) - 1, -1, -1):
            rx1, ry1, rx2, ry2 = _roi_rect(ROI_LIST[idx])
            if rx1 <= local_x <= rx2 and ry1 <= local_y <= ry2:
                target_idx = idx
                break
        if target_idx is None:
            return
        roi = ROI_LIST[target_idx]
        step = 6 if delta > 0 else -6
        roi["half"] = int(roi["half"]) + step
        _clamp_roi(roi, bw, bh)
        ROI_DIRTY = True
        return

    if event == cv2.EVENT_LBUTTONUP:
        UI_ACTIVE_SLIDER = None
        ROI_ACTIVE_IDX = None


def set_resolution_by_index(idx: int):
    """Set CURRENT_MAX_HEIGHT by index."""
    global CURRENT_MAX_HEIGHT
    idx = max(0, min(idx, len(RES_OPTIONS) - 1))
    CURRENT_MAX_HEIGHT = RES_OPTIONS[idx]
    try:
        RES_SAVE_FILE.write_text(str(CURRENT_MAX_HEIGHT), encoding="utf-8")
    except OSError:
        pass


def load_saved_resolution():
    """Load saved resolution if present and valid."""
    global CURRENT_MAX_HEIGHT
    try:
        val = int(RES_SAVE_FILE.read_text(encoding="utf-8").strip())
        if val in RES_OPTIONS:
            CURRENT_MAX_HEIGHT = val
    except (OSError, ValueError):
        pass


def load_saved_model():
    """Load saved model path if present and valid, updating current key/path."""
    global CURRENT_MODEL_KEY, CURRENT_MODEL_PATH
    try:
        val = MODEL_SAVE_FILE.read_text(encoding="utf-8").strip()
        for k, (path, _) in MODEL_OPTIONS.items():
            if path == val:
                CURRENT_MODEL_KEY = k
                CURRENT_MODEL_PATH = path
                return
    except OSError:
        pass


def save_current_model():
    """Persist current model path."""
    try:
        MODEL_SAVE_FILE.write_text(CURRENT_MODEL_PATH, encoding="utf-8")
    except OSError:
        pass


def load_model(path: str):
    """Load YOLO model on the configured device."""
    mdl = YOLO(path)
    mdl.to(DEVICE)
    return mdl


def load_rtsp_url_from_settings() -> None:
    """Reload RTSP_URL from settings.json if present."""
    global RTSP_URL
    try:
        import json

        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        url = str(data.get("rtsp_url", RTSP_URL)).strip()
        if url:
            RTSP_URL = url
    except Exception:
        pass


def open_capture(source: str):
    """Open capture for camera, video file, or NDI."""
    global CURRENT_VIDEO_INDEX
    if source == "camera":
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
        return cap
    if source == "ndi":
        return NDIReceiver(source_name=NDI_PREFERRED_SOURCE)
    if source == "rtsp":
        if not RTSP_URL:
            print("RTSP URL vacío.", file=sys.stderr)
            return None
        if "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
            transport = "tcp" if RTSP_TRANSPORT == "tcp" else "udp"
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{transport}|fflags;nobuffer|flags;low_delay"
            )
        cap = cv2.VideoCapture(RTSP_URL)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if USE_RTSP_THREAD:
            return LatestFrameCapture(cap)
        return cap
    if source == "video":
        if not VIDEO_FILES:
            print("No hay videos en DATA/", file=sys.stderr)
            return None
        CURRENT_VIDEO_INDEX %= len(VIDEO_FILES)
        path = VIDEO_FILES[CURRENT_VIDEO_INDEX]
        cap = cv2.VideoCapture(str(path))
        return cap
    return None


def source_label():
    if CURRENT_SOURCE == "camera":
        return "Camera"
    if CURRENT_SOURCE == "ndi":
        return "NDI"
    if CURRENT_SOURCE == "rtsp":
        return "RTSP"
    if CURRENT_SOURCE == "video" and VIDEO_FILES:
        return f"Video: {VIDEO_FILES[CURRENT_VIDEO_INDEX].name}"
    return CURRENT_SOURCE


def capture_ready(cap):
    if cap is None:
        return False
    if isinstance(cap, NDIReceiver):
        return cap.ready
    if hasattr(cap, "isOpened"):
        try:
            return cap.isOpened()
        except Exception:
            return False
    return cap.isOpened()


def release_capture(cap):
    if cap is None:
        return
    if hasattr(cap, "release"):
        try:
            cap.release()
        except Exception:
            pass


def apply_model_change(model: YOLO, new_key: int, load: bool = True) -> YOLO:
    """Swap YOLO model if key differs; keeps current on failure."""
    global CURRENT_MODEL_KEY, CURRENT_MODEL_PATH
    if new_key not in MODEL_OPTIONS or new_key == CURRENT_MODEL_KEY:
        return model
    CURRENT_MODEL_KEY = new_key
    CURRENT_MODEL_PATH = MODEL_OPTIONS[new_key][0]
    save_current_model()
    save_settings()
    if not load:
        return model
    try:
        return load_model(CURRENT_MODEL_PATH)
    except Exception as exc:
        print(f"No se pudo cargar el modelo {CURRENT_MODEL_PATH}: {exc}", file=sys.stderr)
        return model


def apply_source_change(new_source: str, cap):
    """Swap capture source; keeps previous on failure."""
    global CURRENT_SOURCE, CURRENT_VIDEO_INDEX
    if new_source == CURRENT_SOURCE:
        return cap
    if new_source == "video" and not VIDEO_FILES:
        return cap
    if new_source == "ndi" and (not ENABLE_NDI or not ENABLE_NDI_INPUT):
        return cap
    if new_source == "rtsp" and (not ENABLE_RTSP_INPUT or not RTSP_URL):
        return cap

    prev_cap, prev_src = cap, CURRENT_SOURCE
    CURRENT_SOURCE = new_source
    if new_source == "video":
        CURRENT_VIDEO_INDEX = 0
    release_capture(cap)
    cap = open_capture(CURRENT_SOURCE)
    if not capture_ready(cap):
        print(f"No se pudo abrir la fuente {new_source}, se mantiene la anterior.", file=sys.stderr)
        CURRENT_SOURCE = prev_src
        cap = prev_cap
    else:
        save_settings()
    return cap


# --------------- helpers NDI --------------------------------------
def get_ndi_module():
    """Intentar importar cyndilib (preferido). Evitamos NDIlib por segfaults reportados."""
    try:
        import importlib.resources as ir
        # Aseguramos ruta de runtime NDI para cyndilib (incluye libndi.dylib empaquetado).
        try:
            bin_path = ir.files("cyndilib.wrapper").joinpath("bin")
            bin_str = str(bin_path)
            os.environ.setdefault("NDI_RUNTIME_DIR", bin_str)
            os.environ["DYLD_LIBRARY_PATH"] = f"{bin_str}:{os.environ.get('DYLD_LIBRARY_PATH','')}"
        except Exception:
            pass
        import cyndilib as ndi  # type: ignore

        return ndi
    except Exception:
        return None


# --------------- Probing seguro de NDI ----------------------------
def _ndi_probe_worker(result_queue: mp.Queue):
    """Intento seguro de cargar NDIlib en un proceso aislado."""
    try:
        ndi = get_ndi_module()
        if ndi is None:
            result_queue.put(False)
            return
        # NDI salida no depende de fuentes detectadas; basta con cargar runtime.
        result_queue.put(True)
    except Exception:
        result_queue.put(False)


def probe_ndi_available(timeout: float = 3.0) -> bool:
    """Carga NDIlib en un proceso aparte para evitar segfaults en el proceso principal."""
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_ndi_probe_worker, args=(q,), daemon=True)
    p.start()
    p.join(timeout)
    ok = False
    if p.exitcode is None:
        p.terminate()
    elif p.exitcode == 0 and not q.empty():
        ok = bool(q.get())
    return ok


# --------------- persistencia de settings ---------------------------
SETTINGS_FILE = Path(__file__).with_name("settings.json")


def _clamp(val: int, low: int, high: int) -> int:
    return max(low, min(high, val))


def load_settings():
    global CURRENT_PEOPLE_LIMIT, BLUR_KERNEL_IDX, BLUR_ENABLED, MASK_THRESH
    global IMG_SIZE_IDX, HIGH_PRECISION_MODE, NDI_OUTPUT_MASK, CURRENT_SOURCE, SHOW_DETAIL, FLIP_INPUT
    global ENABLE_NDI_INPUT, ENABLE_NDI_OUTPUT, ENABLE_NDI_TRANSLATIONS_OUTPUT, ENABLE_RTSP_INPUT, RTSP_URL
    try:
        import json

        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return
    try:
        CURRENT_PEOPLE_LIMIT = _clamp(int(data.get("people_limit", CURRENT_PEOPLE_LIMIT)), PEOPLE_LIMIT_OPTIONS[0], PEOPLE_LIMIT_OPTIONS[-1])
    except Exception:
        pass
    try:
        BLUR_KERNEL_IDX = _clamp(int(data.get("blur_kernel_idx", BLUR_KERNEL_IDX)), 0, len(BLUR_KERNEL_OPTIONS) - 1)
    except Exception:
        pass
    try:
        BLUR_ENABLED = bool(data.get("blur_enabled", BLUR_ENABLED))
    except Exception:
        pass
    try:
        MASK_THRESH = _clamp(int(data.get("mask_thresh", MASK_THRESH)), 0, 255)
    except Exception:
        pass
    try:
        IMG_SIZE_IDX = _clamp(int(data.get("img_size_idx", IMG_SIZE_IDX)), 0, len(IMG_SIZE_OPTIONS) - 1)
    except Exception:
        pass
    try:
        HIGH_PRECISION_MODE = bool(data.get("high_precision_mode", HIGH_PRECISION_MODE))
    except Exception:
        pass
    try:
        if data.get("ndi_output_mask") in {"soft", "detail"}:
            NDI_OUTPUT_MASK = data["ndi_output_mask"]
    except Exception:
        pass
    try:
        ENABLE_NDI_INPUT = bool(data.get("ndi_input_enabled", ENABLE_NDI_INPUT))
    except Exception:
        pass
    try:
        ENABLE_NDI_OUTPUT = bool(data.get("ndi_output_enabled", ENABLE_NDI_OUTPUT))
    except Exception:
        pass
    try:
        ENABLE_NDI_TRANSLATIONS_OUTPUT = bool(
            data.get("ndi_translations_output_enabled", ENABLE_NDI_TRANSLATIONS_OUTPUT)
        )
    except Exception:
        pass
    try:
        src = str(data.get("source", CURRENT_SOURCE)).lower()
        if src in {"camera", "video", "ndi"}:
            CURRENT_SOURCE = src
    except Exception:
        pass
    try:
        SHOW_DETAIL = bool(data.get("show_detail", SHOW_DETAIL_DEFAULT))
    except Exception:
        pass
    try:
        FLIP_INPUT = bool(data.get("flip_input", FLIP_INPUT))
    except Exception:
        pass
    try:
        ENABLE_RTSP_INPUT = bool(data.get("rtsp_input_enabled", ENABLE_RTSP_INPUT))
    except Exception:
        pass
    try:
        url = str(data.get("rtsp_url", RTSP_URL)).strip()
        if url:
            RTSP_URL = url
    except Exception:
        pass


def save_settings(extra: dict[str, Any] | None = None):
    base = {
        "people_limit": CURRENT_PEOPLE_LIMIT,
        "blur_kernel_idx": BLUR_KERNEL_IDX,
        "blur_enabled": BLUR_ENABLED,
        "mask_thresh": MASK_THRESH,
        "img_size_idx": IMG_SIZE_IDX,
        "high_precision_mode": HIGH_PRECISION_MODE,
        "ndi_output_mask": NDI_OUTPUT_MASK,
        "ndi_input_enabled": ENABLE_NDI_INPUT,
        "ndi_output_enabled": ENABLE_NDI_OUTPUT,
        "ndi_translations_output_enabled": ENABLE_NDI_TRANSLATIONS_OUTPUT,
        "rtsp_input_enabled": ENABLE_RTSP_INPUT,
        "rtsp_url": RTSP_URL,
        "source": CURRENT_SOURCE,
        "show_detail": SHOW_DETAIL,
        "flip_input": FLIP_INPUT,
    }
    if extra:
        base.update(extra)
    try:
        import json

        SETTINGS_FILE.write_text(json.dumps(base, indent=2), encoding="utf-8")
    except Exception:
        pass


class InferenceWorker:
    def __init__(self, model_path: str):
        self.desired_model_path = model_path
        self.loaded_model_path = model_path
        self.model = load_model(model_path)
        self.frame_lock = threading.Lock()
        self.result_lock = threading.Lock()
        self.event = threading.Event()
        self.stop_event = threading.Event()
        self.pending = None
        self.result = None
        self.result_seq = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def update_frame(self, frame: np.ndarray, settings: dict[str, Any]) -> None:
        with self.frame_lock:
            self.pending = (frame, settings)
        self.event.set()

    def set_model_path(self, model_path: str) -> None:
        with self.frame_lock:
            self.desired_model_path = model_path
        self.event.set()

    def get_latest(self) -> Tuple[Any, int]:
        with self.result_lock:
            return self.result, self.result_seq

    def stop(self) -> None:
        self.stop_event.set()
        self.event.set()
        self.thread.join(timeout=1.0)

    def _maybe_reload(self) -> None:
        if self.desired_model_path == self.loaded_model_path:
            return
        self.model = load_model(self.desired_model_path)
        self.loaded_model_path = self.desired_model_path

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            self.event.wait(0.01)
            if self.stop_event.is_set():
                break
            with self.frame_lock:
                req = self.pending
                self.pending = None
            self.event.clear()
            if req is None:
                self._maybe_reload()
                continue
            self._maybe_reload()
            frame, settings = req
            masks = segment_people(
                frame,
                self.model,
                settings["people_limit"],
                settings["mask_thresh"],
                settings["blur_enabled"],
                compute_person_masks=settings["compute_person_masks"],
            )
            with self.result_lock:
                self.result = masks
                self.result_seq += 1


class LatestFrameCapture:
    """Background reader that always exposes the latest frame (low-latency)."""

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.new = False
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = frame
                self.new = True

    def read_latest(self):
        with self.lock:
            frame = self.frame
            is_new = self.new
            self.new = False
        return frame, is_new

    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        self.stop_event.set()
        self.thread.join(timeout=0.5)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass


# --------------- Syphon (opcional) ----------------------------------
class SyphonPublisher:
    """Wrapper para publicar frames por Syphon si la librería está disponible."""

    def __init__(self, name: str):
        self.server = None
        self.name = name

    def publish(self, frame: np.ndarray):
        return


class NDIPublisher:
    """Wrapper para publicar frames por NDI si la librería está disponible."""

    def __init__(self, name: str):
        self.name = name
        self.ndi = None
        self.sender = None
        self.ready = False
        self._announced = False
        self.vf = None
        if not ENABLE_NDI:
            return
        try:
            ndi = get_ndi_module()
            if ndi is None or not hasattr(ndi, "Sender"):
                raise RuntimeError("No se pudo importar cyndilib Sender")

            self.ndi = ndi
            self.sender = ndi.Sender(ndi_name=name)
        except Exception as exc:
            print(f"[NDI] No disponible: {exc}", file=sys.stderr)
            self.sender = None
            self.ndi = None

    def publish(self, frame: np.ndarray):
        if self.sender is None or self.ndi is None:
            return
        try:
            ndi = self.ndi
            VideoSendFrame = ndi.video_frame.VideoSendFrame if hasattr(ndi, "video_frame") else None
            # Convertimos a BGRA para NDI.
            if frame.ndim == 2:
                frame_bgra = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGRA)
            else:
                frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            if not frame_bgra.flags["C_CONTIGUOUS"]:
                frame_bgra = frame_bgra.copy()

            h, w = frame_bgra.shape[:2]
            # Inicializar VideoSendFrame si hace falta o si cambia la resolución.
            if self.vf is None or self.vf.get_resolution() != (w, h):
                if VideoSendFrame is None:
                    print("[NDI] cyndilib sin VideoSendFrame; no se envía.", file=sys.stderr)
                    return
                if self.ready:
                    # Si el sender está abierto y cambia la resolución, cerrar y reconfigurar.
                    try:
                        self.sender.close()
                    except Exception:
                        pass
                    self.ready = False
                vf = VideoSendFrame()
                vf.set_resolution(w, h)
                vf.set_fourcc(ndi.FourCC.BGRA)
                # Usa 60fps como valor por defecto; NDI ajusta internamente.
                vf.set_frame_rate(Fraction(60, 1))
                self.sender.set_video_frame(vf)
                self.vf = vf
                if not self.ready:
                    # Abrimos el sender una vez tengamos frame configurado.
                    self.sender.open()
                    self.ready = True
            # write_video_async admite memoryview 1D
            self.sender.write_video_async(frame_bgra.ravel(order="C"))
            if not self._announced:
                print(f"[NDI] Enviando salida NDI '{self.name}' ({w}x{h})", file=sys.stderr)
                self._announced = True
        except Exception as exc:
            print(f"[NDI] Error al publicar: {exc}", file=sys.stderr)


class NDIReceiver:
    """Wrapper para recibir frames por NDI."""

    def __init__(self, source_name: str | None = None):
        self.ndi = None
        self.recv = None
        self.ready = False
        self.source_name = source_name
        if not ENABLE_NDI:
            return
        try:
            ndi = get_ndi_module()
            if ndi is None or not hasattr(ndi, "Receiver"):
                raise RuntimeError("No se pudo importar cyndilib Receiver")

            finder = ndi.Finder()
            finder.open()
            finder.wait_for_sources(2.0)
            sources = list(finder.iter_sources())
            finder.close()

            if not sources:
                print("[NDI] No se encontraron fuentes NDI.", file=sys.stderr)
                return

            preferred = (source_name or "").lower().strip()
            selected = None
            for src in sources:
                name = getattr(src, "name", "") or getattr(src, "stream_name", "") or ""
                if preferred and preferred in name.lower():
                    selected = src
                    break
            if selected is None:
                selected = sources[0]

            self.recv = ndi.Receiver(
                source_name=getattr(selected, "name", ""),
                color_format=ndi.RecvColorFormat.BGRX_BGRA,
                bandwidth=ndi.RecvBandwidth.highest,
                allow_video_fields=False,
                recv_name="NEXT2-Recv",
            )
            # Usar FrameSync para simplificar la lectura de vídeo.
            vf_sync = ndi.video_frame.VideoFrameSync()
            self.recv.frame_sync.set_video_frame(vf_sync)
            self.recv.connect_to(selected)
            try:
                self.recv._wait_for_connect(2.0)
            except Exception:
                pass
            self.ndi = ndi
            self.ready = True
            print(f"[NDI] Conectado a '{getattr(selected, 'name', 'NDI')}'", file=sys.stderr)
        except Exception as exc:
            print(f"[NDI] No se pudo preparar la entrada NDI: {exc}", file=sys.stderr)
            self.ndi = None
            self.recv = None
            self.ready = False

    def capture(self):
        """Returns BGR frame or None if not available."""
        if not self.ready or self.recv is None or self.ndi is None:
            return None
        ndi = self.ndi
        try:
            fs = self.recv.frame_sync
            fs.capture_video()
            vf = fs.video_frame
            w, h = vf.get_resolution()
            stride = vf.get_line_stride()
            mv = memoryview(vf)
            flat = np.frombuffer(mv, dtype=np.uint8).copy()
            mv.release()
            if flat.size < stride * h or w <= 0 or h <= 0:
                return None
            frame_bgra = flat[: stride * h].reshape((h, stride))[:, : w * 4].reshape((h, w, 4))
            frame_bgr = frame_bgra[:, :, :3].copy()
            return frame_bgr
        except Exception as exc:
            print(f"[NDI] Error al recibir: {exc}", file=sys.stderr)
            return None

    def release(self):
        # No-op for NDI receiver; placeholder for interface compatibility.
        return


# --------------- loop principal ------------------------------------
def main():
    # Carga modelo YOLOv8 de segmentación (usa uno ligero por defecto).
    model_path = "yolov8n-seg.pt"
    global DEVICE, CURRENT_MODEL_PATH, CURRENT_MODEL_KEY, CURRENT_PEOPLE_LIMIT, CURRENT_SOURCE
    global BLUR_KERNEL_IDX, MASK_THRESH, BLUR_ENABLED, IMG_SIZE_IDX, HIGH_PRECISION_MODE, ENABLE_NDI, DEFAULT_SOURCE, SHOW_DETAIL, FLIP_INPUT
    global ENABLE_NDI_INPUT, ENABLE_NDI_OUTPUT, ENABLE_NDI_TRANSLATIONS_OUTPUT, ENABLE_RTSP_INPUT
    global UI_PENDING_MODEL_KEY, UI_PENDING_SOURCE, VIEW_HITBOXES, ROI_PANEL_BOUNDS, ROI_PANEL_BOUNDS_ABS
    global FOOTER_CACHE, FOOTER_CACHE_KEY, FOOTER_CACHE_HITBOXES, UI_HITBOXES
    load_saved_resolution()
    load_saved_model()
    load_settings()
    env_device = os.environ.get("NEXT_DEVICE", "").strip().lower()
    if env_device and env_device != "auto":
        # Permite forzar cpu/mps/cuda manualmente.
        DEVICE = env_device
    else:
        # Auto: intenta mps -> cuda -> cpu.
        if torch.backends.mps.is_available():
            DEVICE = "mps"
        elif torch.cuda.is_available():
            DEVICE = "cuda"
        else:
            DEVICE = "cpu"

    env_source = os.environ.get("NEXT_SOURCE", "").strip().lower()
    # Probar NDI en proceso aislado para evitar segfault en main.
    ndi_ok = False
    if ENABLE_NDI:
        ndi_ok = probe_ndi_available()
        if not ndi_ok:
            print("[NDI] No se pudo inicializar NDIlib.", file=sys.stderr)
            # No forzamos el apagado global para permitir reactivar salida NDI manualmente.

    # Decide fuente por env o por disponibilidad (por defecto cámara; NDI se selecciona a mano).
    if env_source in {"ndi", "camera", "video", "rtsp"}:
        CURRENT_SOURCE = env_source
    else:
        CURRENT_SOURCE = "camera"

    if CURRENT_SOURCE == "ndi" and (not ENABLE_NDI or not ENABLE_NDI_INPUT):
        print("[NDI] Fuente NDI solicitada pero NDI no está disponible, usando cámara.", file=sys.stderr)
        CURRENT_SOURCE = "camera"
    if CURRENT_SOURCE == "rtsp" and (not ENABLE_RTSP_INPUT or not RTSP_URL):
        print("[RTSP] Fuente RTSP solicitada pero no está disponible, usando cámara.", file=sys.stderr)
        CURRENT_SOURCE = "camera"

    infer_worker = None
    model = None
    if USE_INFERENCE_THREAD:
        try:
            infer_worker = InferenceWorker(CURRENT_MODEL_PATH)
        except Exception as exc:
            print(f"No se pudo cargar el modelo {CURRENT_MODEL_PATH}: {exc}", file=sys.stderr)
            return 1
    else:
        try:
            model = load_model(CURRENT_MODEL_PATH)
        except Exception as exc:
            print(f"No se pudo cargar el modelo {CURRENT_MODEL_PATH}: {exc}", file=sys.stderr)
            return 1

    cap = open_capture(CURRENT_SOURCE)
    if not capture_ready(cap):
        print("No se pudo abrir la fuente de video.", file=sys.stderr)
        return 1

    prev_time = time.time()
    window_name = "NEXT2 VISION - ROTOR STUDIO"
    canvas_size = (CANVAS_WIDTH, CANVAS_HEIGHT)  # width, height (3 panels + gap)
    header_h = 40
    footer_h = 380
    cv2.setUseOptimized(True)
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    frame_idx = 0
    last_mask_soft = None
    last_mask_detail = None
    last_boxes = None
    last_boxed = None
    last_person_masks = []
    last_frame = None
    last_rtsp_frame_time = time.time()
    last_mask_to_show = None
    cached_middle_view = None
    cached_middle_key = None
    cached_translated = None
    cached_left_view = None
    cached_left_key = None
    cached_left_labeled = None
    cached_left_labeled_key = None
    cached_right_bgr = None
    cached_right_key = None
    persist_mask = None
    last_detect_time = None
    last_detect_mask = None
    show_detail = SHOW_DETAIL
    ndi_pub = NDIPublisher("NEXT2 Mask NDI") if ENABLE_NDI and ENABLE_NDI_OUTPUT else None
    ndi_trans_pub = (
        NDIPublisher("NEXT2 Translations NDI") if ENABLE_NDI and ENABLE_NDI_TRANSLATIONS_OUTPUT else None
    )

    last_result_seq = 0
    while True:
        frame_start = time.perf_counter()
        if UI_PENDING_MODEL_KEY is not None:
            model = apply_model_change(model, UI_PENDING_MODEL_KEY, load=not USE_INFERENCE_THREAD)
            if infer_worker is not None:
                infer_worker.set_model_path(CURRENT_MODEL_PATH)
            UI_PENDING_MODEL_KEY = None
        if UI_PENDING_SOURCE is not None:
            cap = apply_source_change(UI_PENDING_SOURCE, cap)
            UI_PENDING_SOURCE = None
        show_detail = SHOW_DETAIL

        t0 = time.perf_counter()
        frame, got_new_frame = capture_frame(cap)
        cap_ms = (time.perf_counter() - t0) * 1000.0
        if frame is None:
            if CURRENT_SOURCE == "video" and cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame, got_new_frame = capture_frame(cap)
            elif CURRENT_SOURCE in {"ndi", "rtsp"}:
                # Mantener último frame o negro mientras esperamos señal.
                if last_frame is not None:
                    frame = last_frame.copy()
                else:
                    fallback_h = min(CURRENT_MAX_HEIGHT, IMG_SIZE_OPTIONS[IMG_SIZE_IDX])
                    fallback_w = int(fallback_h * 16 / 9)
                    frame = np.zeros((fallback_h, fallback_w, 3), dtype=np.uint8)
                got_new_frame = False
            if frame is None:
                print("Frame no capturado. Saliendo.", file=sys.stderr)
                break
        if CURRENT_SOURCE == "rtsp":
            if got_new_frame:
                last_rtsp_frame_time = time.time()
            elif time.time() - last_rtsp_frame_time > RTSP_RECONNECT_SEC:
                release_capture(cap)
                cap = open_capture("rtsp")
                last_rtsp_frame_time = time.time()

        do_process = (frame_idx % PROCESS_EVERY_N == 0 or last_mask_soft is None) and got_new_frame
        if do_process:
            need_person_masks = len(ROI_LIST) > 0
            if infer_worker is not None:
                infer_worker.update_frame(
                    frame,
                    {
                        "people_limit": CURRENT_PEOPLE_LIMIT,
                        "mask_thresh": MASK_THRESH,
                        "blur_enabled": BLUR_ENABLED,
                        "compute_person_masks": need_person_masks,
                    },
                )
            else:
                mask_soft, mask_detail, boxes, person_masks = segment_people(
                    frame, model, CURRENT_PEOPLE_LIMIT, MASK_THRESH, BLUR_ENABLED, compute_person_masks=need_person_masks
                )
                last_mask_soft, last_mask_detail, last_boxes, last_person_masks = (
                    mask_soft,
                    mask_detail,
                    boxes,
                    person_masks,
                )
                last_frame = frame

        if infer_worker is not None:
            result, seq = infer_worker.get_latest()
            if result is not None and seq != last_result_seq:
                last_result_seq = seq
                mask_soft, mask_detail, boxes, person_masks = result
                last_mask_soft, last_mask_detail, last_boxes, last_person_masks = (
                    mask_soft,
                    mask_detail,
                    boxes,
                    person_masks,
                )
                last_frame = frame
            else:
                mask_soft, mask_detail, boxes, person_masks = (
                    last_mask_soft,
                    last_mask_detail,
                    last_boxes,
                    last_person_masks,
                )
        elif not do_process:
            mask_soft, mask_detail, boxes, person_masks = (
                last_mask_soft,
                last_mask_detail,
                last_boxes,
                last_person_masks,
            )
        frame_idx += 1

        # Fallback en caso de no tener máscaras aún (evita crash en cvtColor).
        if mask_soft is None or mask_detail is None:
            h, w = frame.shape[:2]
            fallback_mask = np.zeros((h, w), dtype=np.uint8)
            if mask_soft is None:
                mask_soft = fallback_mask
            if mask_detail is None:
                mask_detail = fallback_mask

        if do_process or last_boxed is None:
            boxed = draw_boxes(frame, boxes)
            last_boxed = boxed
        else:
            boxed = last_boxed

        # FPS + persistencia.
        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 1.0 / dt if dt > 0 else 0.0
        persist_mask, last_detect_time, last_detect_mask = update_persistent_mask(
            persist_mask, mask_detail, last_detect_time, last_detect_mask, now, max(dt, 1e-6)
        )
        persist_u8 = np.clip(persist_mask * 255.0, 0, 255).astype(np.uint8)
        if frame_idx % 10 == 0:
            print(f"FPS: {fps:0.2f}", end="\r", flush=True)
        if frame_idx % 30 == 0:
            frame_ms = (time.perf_counter() - frame_start) * 1000.0
            print(
                f"FRAME_MS: {frame_ms:0.2f} | CAP_MS: {cap_ms:0.2f} | WAIT_MS: {wait_ms:0.2f}",
                end="\r",
                flush=True,
            )

        # Composición en ventana única: tres vistas (izquierda + panel tabbed + derecha).
        canvas = np.full((canvas_size[1], canvas_size[0], 3), UI_BG_COLOR, dtype=np.uint8)
        view_h = canvas_size[1] - header_h - footer_h - FOOTER_GAP_PX - VIEW_OFFSET_Y
        third_w = canvas_size[0] // 3
        right_view_h = canvas_size[1] - header_h - VIEW_OFFSET_Y - RIGHT_PANEL_FOOTER_H
        view_y = header_h + VIEW_OFFSET_Y
        VIEW_HITBOXES = []

        left_key = (id(boxed), third_w, view_h)
        if cached_left_view is None or cached_left_key != left_key:
            left_view = fit_to_box(boxed, (third_w, view_h))
            cached_left_view = left_view
            cached_left_key = left_key
        else:
            left_view = cached_left_view
        if do_process or last_mask_to_show is None:
            mask_to_show = mask_detail if show_detail else mask_soft
            last_mask_to_show = mask_to_show
        else:
            mask_to_show = last_mask_to_show
        if ACTIVE_VIEW_TAB == "mask":
            middle_key = (show_detail, id(mask_soft), id(mask_detail), ACTIVE_VIEW_TAB)
            if cached_middle_view is None or cached_middle_key != middle_key:
                middle_image = cv2.cvtColor(mask_to_show, cv2.COLOR_GRAY2BGR)
                cached_middle_view = make_tabbed_view(
                    middle_image,
                    (third_w, view_h),
                    (("mask", "Mask"), ("mod", "Suavizado")),
                    ACTIVE_VIEW_TAB,
                )
                cached_middle_key = middle_key
            middle_view, tab_rects = cached_middle_view
        else:
            middle_image = cv2.cvtColor(persist_u8, cv2.COLOR_GRAY2BGR)
            middle_view, tab_rects = make_tabbed_view(
                middle_image,
                (third_w, view_h),
                (("mask", "Mask"), ("mod", "Suavizado")),
                ACTIVE_VIEW_TAB,
            )
        roi_avail_w = third_w
        roi_avail_h = max(1, right_view_h - VIEW_LABEL_H)
        roi_scale = min(
            roi_avail_w / float(NDI_TR_OUTPUT_W),
            roi_avail_h / float(NDI_TR_OUTPUT_H),
        )
        roi_w = max(1, int(NDI_TR_OUTPUT_W * roi_scale))
        roi_h = max(1, int(NDI_TR_OUTPUT_H * roi_scale))
        roi_x_off = max(0, (third_w - roi_w) // 2)
        roi_y_off = VIEW_LABEL_H  # align to top under label, like other views
        if cached_translated is None or ROI_DIRTY or do_process:
            translated = translate_masks_to_rois(
                person_masks, ROI_LIST, (roi_w, roi_h), now, max(dt, 1e-6)
            )
            cached_translated = translated
            ROI_DIRTY = False
        else:
            translated = cached_translated
        right_key = (id(translated), roi_w, roi_h)
        if cached_right_bgr is None or cached_right_key != right_key:
            right_image = cv2.cvtColor(translated, cv2.COLOR_GRAY2BGR)
            cached_right_bgr = right_image
            cached_right_key = right_key
        else:
            right_image = cached_right_bgr

        left_labeled_key = (id(left_view), third_w, view_h)
        if cached_left_labeled is None or cached_left_labeled_key != left_labeled_key:
            left_view = make_labeled_view(left_view, "Original + Boxes", (third_w, view_h))
            cached_left_labeled = left_view
            cached_left_labeled_key = left_labeled_key
        else:
            left_view = cached_left_labeled
        right_view = np.full((right_view_h, third_w, 3), RIGHT_PANEL_BG_COLOR, dtype=np.uint8)
        cv2.rectangle(right_view, (0, 0), (third_w - 1, VIEW_LABEL_H), UI_BG_COLOR, -1)
        cv2.putText(
            right_view,
            "Traslaciones",
            (UI_MARGIN_LEFT, VIEW_LABEL_H - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (235, 235, 235),
            1,
        )
        roi_view = np.full((roi_h, roi_w, 3), ROI_WORK_BG_COLOR, dtype=np.uint8)
        if translated is not None:
            mask = translated > 0
            roi_view[mask] = right_image[mask]
        right_view[roi_y_off : roi_y_off + roi_h, roi_x_off : roi_x_off + roi_w] = roi_view

        ROI_PANEL_BOUNDS = (roi_w, roi_h)
        ROI_PANEL_BOUNDS_ABS = (
            2 * third_w + roi_x_off,
            view_y + roi_y_off,
            roi_w,
            roi_h,
        )
        rx0, ry0, rw, rh = ROI_PANEL_BOUNDS_ABS
        if ROI_LIST:
            for roi in ROI_LIST:
                _clamp_roi(roi, rw, rh)
                x1, y1, x2, y2 = _roi_rect(roi)
                x1 = max(0, min(rw - 1, x1))
                y1 = max(0, min(rh - 1, y1))
                x2 = max(0, min(rw - 1, x2))
                y2 = max(0, min(rh - 1, y2))
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(
                        right_view,
                        (roi_x_off + x1, roi_y_off + y1),
                        (roi_x_off + x2, roi_y_off + y2),
                        (255, 0, 0),
                        2,
                    )

        canvas[view_y : view_y + left_view.shape[0], 0:third_w] = left_view
        canvas[
            view_y : view_y + middle_view.shape[0], third_w : third_w + middle_view.shape[1]
        ] = middle_view
        canvas[
            view_y : view_y + right_view.shape[0], 2 * third_w : 2 * third_w + right_view.shape[1]
        ] = right_view

        for key, rect in tab_rects:
            abs_rect = (
                third_w + rect[0],
                view_y + rect[1],
                third_w + rect[2],
                view_y + rect[3],
            )
            VIEW_HITBOXES.append({"type": "view_tab", "rect": abs_rect, "value": key, "enabled": True})

        border_color = (200, 200, 200)
        border_top = view_y + VIEW_LABEL_H
        border_bottom = view_y + view_h - 1
        cv2.rectangle(canvas, (0, border_top), (third_w - 1, border_bottom), border_color, 1)
        cv2.rectangle(canvas, (third_w, border_top), (2 * third_w - 1, border_bottom), border_color, 1)
        right_border_bottom = view_y + right_view_h - 1
        cv2.rectangle(canvas, (2 * third_w, border_top), (3 * third_w - 1, right_border_bottom), border_color, 1)

        # Right panel controls (ROI +/-, reset, NDI TR) under the third window.
        right_x0 = 2 * third_w
        btn_h = 24
        gap = 8
        x = right_x0 + RIGHT_PANEL_MARGIN_X
        y = view_y + right_view_h + RIGHT_PANEL_ROW_GAP
        max_x = right_x0 + third_w - RIGHT_PANEL_MARGIN_X

        def _place_right_button(label: str, width: int, active: bool, enabled: bool, item_id: str) -> None:
            nonlocal x, y
            if x + width > max_x:
                y += btn_h + RIGHT_PANEL_ROW_GAP
                x = right_x0 + RIGHT_PANEL_MARGIN_X
            rect = (x, y, x + width, y + btn_h)
            _draw_button(canvas, rect, label, active, enabled)
            VIEW_HITBOXES.append({"type": "button" if item_id.startswith("roi") else "toggle", "id": item_id, "rect": rect, "enabled": enabled})
            x += width + gap

        _place_right_button("ROI +", 90, False, len(ROI_LIST) < ROI_MAX, "roi_add")
        _place_right_button("ROI -", 70, False, len(ROI_LIST) > 0, "roi_remove")
        _place_right_button("ROI reset", 90, False, len(ROI_LIST) > 0, "roi_reset")
        _place_right_button("NDI TR", 120, ENABLE_NDI_TRANSLATIONS_OUTPUT, ENABLE_NDI, "ndi_trans_output")

        btn_h = 24
        pad = 8
        btn_w = 90
        btn_y = view_y + view_h + 10

        left_btn = (pad, btn_y, pad + btn_w, btn_y + btn_h)
        _draw_button(canvas, left_btn, "NDI IN", ENABLE_NDI_INPUT, ENABLE_NDI)
        VIEW_HITBOXES.append({"type": "toggle", "id": "ndi_input", "rect": left_btn, "enabled": ENABLE_NDI})
        rtsp_btn = (pad + btn_w + 8, btn_y, pad + btn_w + 8 + btn_w, btn_y + btn_h)
        _draw_button(canvas, rtsp_btn, "RTSP IN", ENABLE_RTSP_INPUT, True)
        VIEW_HITBOXES.append({"type": "toggle", "id": "rtsp_input", "rect": rtsp_btn, "enabled": True})
        rtsp_cfg_btn = (pad + (btn_w + 8) * 2, btn_y, pad + (btn_w + 8) * 2 + btn_w, btn_y + btn_h)
        _draw_button(canvas, rtsp_cfg_btn, "RTSP CFG", False, True)
        VIEW_HITBOXES.append({"type": "button", "id": "rtsp_cfg", "rect": rtsp_cfg_btn, "enabled": True})

        mid_x0 = third_w
        mid_btn = (mid_x0 + pad, btn_y, mid_x0 + pad + 100, btn_y + btn_h)
        _draw_button(canvas, mid_btn, "NDI OUT", ENABLE_NDI_OUTPUT, ENABLE_NDI)
        VIEW_HITBOXES.append({"type": "toggle", "id": "ndi_output", "rect": mid_btn, "enabled": ENABLE_NDI})


        res_text = f"{frame.shape[1]}x{frame.shape[0]}"
        people_count = len(boxes) if boxes is not None else 0
        cap_fps = None
        if isinstance(cap, cv2.VideoCapture):
            cap_fps = cap.get(cv2.CAP_PROP_FPS)
        add_header(canvas, fps, DEVICE, res_text, people_count, source_label(), cap_fps=cap_fps)

        footer_top = canvas.shape[0] - footer_h
        left_footer_w = third_w * 2
        footer_key = (
            left_footer_w,
            footer_h,
            CURRENT_MAX_HEIGHT,
            CURRENT_PEOPLE_LIMIT,
            BLUR_KERNEL_IDX,
            BLUR_ENABLED,
            MASK_THRESH,
            IMG_SIZE_IDX,
            HIGH_PRECISION_MODE,
            SHOW_DETAIL,
            FLIP_INPUT,
            CURRENT_SOURCE,
            bool(VIDEO_FILES),
            ENABLE_NDI,
            ENABLE_NDI_INPUT,
            ENABLE_NDI_OUTPUT,
            ENABLE_NDI_TRANSLATIONS_OUTPUT,
            len(ROI_LIST),
            round(PERSIST_HOLD_SEC, 3),
            round(PERSIST_RISE_TAU, 3),
            round(PERSIST_FALL_TAU, 3),
        )
        if FOOTER_CACHE is None or FOOTER_CACHE_KEY != footer_key:
            footer_layer = np.full((footer_h, left_footer_w, 3), UI_BG_COLOR, dtype=np.uint8)
            add_footer(footer_layer, CURRENT_MAX_HEIGHT, footer_h=footer_h, footer_top=0)
            FOOTER_CACHE = footer_layer
            FOOTER_CACHE_KEY = footer_key
            FOOTER_CACHE_HITBOXES = [item.copy() for item in UI_HITBOXES]
        else:
            UI_HITBOXES = [item.copy() for item in FOOTER_CACHE_HITBOXES]

        if FOOTER_CACHE is not None:
            canvas[-footer_h:, :left_footer_w] = FOOTER_CACHE
        # Offset cached hitboxes to absolute canvas coordinates.
        for item in UI_HITBOXES:
            x1, y1, x2, y2 = item["rect"]
            item["rect"] = (x1, y1 + footer_top, x2, y2 + footer_top)

        # Publicación NDI (máscara) - por defecto la suave; cambiar a mask_detail si se desea.
        if ENABLE_NDI and ENABLE_NDI_OUTPUT and ndi_pub is None:
            ndi_pub = NDIPublisher("NEXT2 Mask NDI")
        if not ENABLE_NDI_OUTPUT:
            ndi_pub = None
        if ndi_pub is not None:
            mask_out = mask_soft if NDI_OUTPUT_MASK == "soft" else mask_detail
            ndi_pub.publish(mask_out)
        if ENABLE_NDI and ENABLE_NDI_TRANSLATIONS_OUTPUT and ndi_trans_pub is None:
            ndi_trans_pub = NDIPublisher("NEXT2 Translations NDI")
        if not ENABLE_NDI_TRANSLATIONS_OUTPUT:
            ndi_trans_pub = None
        if ndi_trans_pub is not None:
            trans_out = translated
            if trans_out.shape[:2] != (NDI_TR_OUTPUT_H, NDI_TR_OUTPUT_W):
                trans_out = cv2.resize(
                    trans_out,
                    (NDI_TR_OUTPUT_W, NDI_TR_OUTPUT_H),
                    interpolation=cv2.INTER_NEAREST,
                )
            ndi_trans_pub.publish(trans_out)

        cv2.imshow(window_name, canvas)

        key = 0
        wait_ms = 0.0
        if frame_idx % 2 == 0:
            t_wait = time.perf_counter()
            if hasattr(cv2, "pollKey"):
                key = cv2.pollKey() & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF
            wait_ms = (time.perf_counter() - t_wait) * 1000.0
        if key == ord("q"):
            break
        # Resolución por teclas 1-5
        if key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
            set_resolution_by_index(int(chr(key)) - 1)
            save_settings()
        # Cambio de modelo por teclas configuradas
        if key in MODEL_OPTIONS:
            model = apply_model_change(model, key)
        # Ajuste de límite de personas con +/- (teclado principal)
        if key == ord("+") or key == ord("="):
            CURRENT_PEOPLE_LIMIT = min(CURRENT_PEOPLE_LIMIT + 1, PEOPLE_LIMIT_OPTIONS[-1])
            save_settings()
        if key == ord("-"):
            CURRENT_PEOPLE_LIMIT = max(CURRENT_PEOPLE_LIMIT - 1, PEOPLE_LIMIT_OPTIONS[0])
            save_settings()
        # Ajuste de blur (detallado de silueta)
        if key == ord("o"):
            BLUR_KERNEL_IDX = max(0, BLUR_KERNEL_IDX - 1)
            save_settings()
        if key == ord("p"):
            BLUR_KERNEL_IDX = min(len(BLUR_KERNEL_OPTIONS) - 1, BLUR_KERNEL_IDX + 1)
            save_settings()
        if key == ord("b"):
            BLUR_ENABLED = not BLUR_ENABLED
            save_settings()
        # Ajuste de threshold de binarización
        if key == ord("j"):
            MASK_THRESH = max(0, MASK_THRESH - 5)
            save_settings()
        if key == ord("k"):
            MASK_THRESH = min(255, MASK_THRESH + 5)
            save_settings()
        # Ajuste de imgsz (resolución de inferencia)
        if key == ord(","):
            IMG_SIZE_IDX = max(0, IMG_SIZE_IDX - 1)
            save_settings()
        if key == ord("."):
            IMG_SIZE_IDX = min(len(IMG_SIZE_OPTIONS) - 1, IMG_SIZE_IDX + 1)
            save_settings()
        # Cambio de fuente
        if key == ord("c"):
            cap = apply_source_change("camera", cap)
        if key == ord("v") and VIDEO_FILES:
            cap = apply_source_change("video", cap)
        if key == ord("n") and ENABLE_NDI and ENABLE_NDI_INPUT:
            cap = apply_source_change("ndi", cap)
        if key == ord("r"):
            cap = apply_source_change("rtsp", cap)
        # Modo alta precisión (usa imgsz máximo y detalle sin blur)
        if key == ord("h"):
            HIGH_PRECISION_MODE = not HIGH_PRECISION_MODE
            save_settings()
        # Toggle de máscara mostrada (soft/detail)
        if key == ord("m"):
            show_detail = not show_detail
            SHOW_DETAIL = show_detail
            save_settings()
        # Flip horizontal de la entrada
        if key == ord("f"):
            FLIP_INPUT = not FLIP_INPUT
            save_settings()

    if infer_worker is not None:
        infer_worker.stop()
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
