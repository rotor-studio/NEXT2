"""Minimal camera viewer.
- Shows camera feed in a single window.
- Press 'c' to cycle camera inputs.
- Press 'q' to quit.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
import threading
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

WINDOW_NAME = "NEXT2"
CANVAS_WIDTH = max(320, int(os.environ.get("NEXT_CANVAS_W", "1536")))
CANVAS_HEIGHT = max(240, int(os.environ.get("NEXT_CANVAS_H", "864")))
CAM_MAX_INDEX = max(0, int(os.environ.get("NEXT_CAM_MAX", "4")))
CAM_WIDTH = max(0, int(os.environ.get("NEXT_CAM_WIDTH", "1280")))
CAM_HEIGHT = max(0, int(os.environ.get("NEXT_CAM_HEIGHT", "720")))
YOLO_MODEL_PATH = os.environ.get("NEXT_YOLO_MODEL", "yolov8n-seg.pt")
YOLO_CONF = float(os.environ.get("NEXT_YOLO_CONF", "0.25"))
YOLO_IMGSZ = int(os.environ.get("NEXT_YOLO_IMGSZ", "320"))
YOLO_DEVICE = os.environ.get("NEXT_YOLO_DEVICE", "").strip()
YOLO_SKIP_FRAMES = max(0, int(os.environ.get("NEXT_YOLO_SKIP", "0")))
MAX_SKIP_FRAMES = max(0, int(os.environ.get("NEXT_YOLO_SKIP_MAX", "5")))
MAX_PERSON_LIMIT = int(os.environ.get("NEXT_PERSON_LIMIT_MAX", "30"))
MASK_BLUR = max(0, int(os.environ.get("NEXT_MASK_BLUR", "5")))
MAX_MASK_BLUR = max(1, int(os.environ.get("NEXT_MASK_BLUR_MAX", "15")))
CONF_MIN = float(os.environ.get("NEXT_CONF_MIN", "0.1"))
CONF_MAX = float(os.environ.get("NEXT_CONF_MAX", "0.6"))
IMG_SIZES = [256, 320, 384, 448, 512, 640]
MAX_MORPH = max(0, int(os.environ.get("NEXT_MASK_MORPH_MAX", "5")))
DEFAULT_SETTINGS_PATH = str(Path(__file__).resolve().parent / "settings.json")
SETTINGS_PATH = os.environ.get("NEXT_SETTINGS_PATH", DEFAULT_SETTINGS_PATH)
SETTINGS_SAVE_INTERVAL = float(os.environ.get("NEXT_SETTINGS_SAVE_SEC", "0.75"))

RES_OPTIONS = [
    ("320x180", 320, 180),
    ("480x270", 480, 270),
    ("640x360", 640, 360),
]
MODEL_OPTIONS = [
    ("FAST", "yolov8n-seg.pt"),
    ("MED", "yolov8s-seg.pt"),
    ("HEAVY", "yolov8m-seg.pt"),
]

UI_FONT = cv2.FONT_HERSHEY_SIMPLEX
UI_FONT_SCALE = 0.5
UI_TEXT_THICKNESS = 1
UI_BTN_W = 120
UI_BTN_H = 26
UI_BTN_GAP = 10
UI_SECTION_GAP = 22
UI_LEFT_MARGIN = 20
UI_TOP_MARGIN = 20


def _open_camera(idx: int) -> Optional[cv2.VideoCapture]:
    try:
        if sys.platform == "darwin":
            cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(idx)
    except Exception:
        return None
    if not cap or not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return None
    if CAM_WIDTH > 0 and CAM_HEIGHT > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    return cap


def _cycle_camera(start_idx: int) -> Tuple[Optional[cv2.VideoCapture], int]:
    for step in range(CAM_MAX_INDEX + 1):
        idx = (start_idx + step) % (CAM_MAX_INDEX + 1)
        cap = _open_camera(idx)
        if cap is not None:
            return cap, idx
    return None, start_idx


def _placeholder(frame_size: Tuple[int, int]) -> np.ndarray:
    w, h = frame_size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(
        img,
        "No camera",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
    )
    return img


def _fit_to_box(image: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(box_w / float(w), box_h / float(h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    x_off = 0
    y_off = 0
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def _fit_to_box_with_rect(
    image: np.ndarray, box_w: int, box_h: int
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    scale = min(box_w / float(w), box_h / float(h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    x_off = 0
    y_off = 0
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas, (x_off, y_off, x_off + new_w, y_off + new_h)


def _fit_to_width(image: np.ndarray, target_w: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= 0:
        return image
    scale = target_w / float(w)
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (target_w, new_h), interpolation=cv2.INTER_AREA)

def _mask_placeholder(frame_size: Tuple[int, int]) -> np.ndarray:
    w, h = frame_size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(
        img,
        "No mask",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (80, 80, 80),
        2,
    )
    return img


def _smooth_mask(mask: np.ndarray) -> np.ndarray:
    if MASK_BLUR <= 1:
        return mask
    k = MASK_BLUR if MASK_BLUR % 2 == 1 else MASK_BLUR + 1
    return cv2.GaussianBlur(mask, (k, k), 0)


def _apply_morph(mask: np.ndarray, strength: int) -> np.ndarray:
    if strength == 0:
        return mask
    k = abs(strength) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if strength > 0:
        return cv2.dilate(mask, kernel, iterations=1)
    return cv2.erode(mask, kernel, iterations=1)


@dataclass
class InferenceResult:
    boxes: list[Tuple[int, int, int, int]] = field(default_factory=list)
    combined_mask: Optional[np.ndarray] = None
    persons: int = 0


class YoloWorker:
    def __init__(self, model: YOLO, device: str, conf: float, imgsz: int) -> None:
        self.model = model
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending_frame: Optional[np.ndarray] = None
        self._stop = False
        self._result = InferenceResult()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop = True
            self._cond.notify_all()
        self._thread.join(timeout=1.0)

    def submit(self, frame: np.ndarray) -> None:
        with self._lock:
            self._pending_frame = frame
            self._cond.notify()

    def get_latest(self) -> InferenceResult:
        with self._lock:
            return self._result

    def update_params(self, conf: float, imgsz: int) -> None:
        with self._lock:
            self.conf = conf
            self.imgsz = imgsz

    def _loop(self) -> None:
        while True:
            with self._lock:
                while self._pending_frame is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                frame = self._pending_frame
                self._pending_frame = None
            try:
                result = self.model.predict(
                    source=frame,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )[0]
                boxes_out: list[Tuple[int, int, int, int]] = []
                persons = 0
                combined_mask = None
                mask_canvas = None
                if result.boxes is not None:
                    if result.masks is not None and result.masks.xy is not None:
                        mask_canvas = np.zeros(frame.shape[:2], dtype=np.uint8)
                        polygons = result.masks.xy
                    else:
                        polygons = None
                    for idx, cls_id in enumerate(result.boxes.cls):
                        if int(cls_id) != 0:
                            continue
                        persons += 1
                        x1, y1, x2, y2 = result.boxes.xyxy[idx].int().tolist()
                        boxes_out.append((x1, y1, x2, y2))
                        if polygons is not None and idx < len(polygons):
                            pts = polygons[idx].astype(np.int32)
                            if pts.size > 0:
                                cv2.fillPoly(mask_canvas, [pts], 255)
                    if mask_canvas is not None:
                        combined_mask = mask_canvas
                with self._lock:
                    self._result = InferenceResult(
                        boxes=boxes_out,
                        combined_mask=combined_mask,
                        persons=persons,
                    )
            except Exception as exc:
                print(f"[YOLO] Error de inferencia: {exc}", file=sys.stderr)
                with self._lock:
                    self._result = InferenceResult()


@dataclass
class UIState:
    selected_res: int = 1
    selected_model: int = 0
    person_limit: int = 10
    persons_detected: int = 0
    slider_drag: bool = False
    skip_drag: bool = False
    skip_frames: int = YOLO_SKIP_FRAMES
    conf_drag: bool = False
    imgsz_drag: bool = False
    blur_drag: bool = False
    morph_drag: bool = False
    conf: float = YOLO_CONF
    imgsz_idx: int = 0
    blur: int = MASK_BLUR
    morph: int = 0
    rects: Dict[str, Tuple[int, int, int, int]] = field(default_factory=dict)


def _load_model(model_path: str) -> Optional[YOLO]:
    if not Path(model_path).exists():
        return None
    try:
        return YOLO(model_path)
    except Exception as exc:
        print(f"[YOLO] No se pudo cargar modelo: {exc}", file=sys.stderr)
        return None


def _load_settings(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _save_settings(path: str, data: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=True, indent=2)
    except Exception:
        pass


def _draw_button(
    canvas: np.ndarray, label: str, rect: Tuple[int, int, int, int], active: bool
) -> None:
    x1, y1, x2, y2 = rect
    fill = (70, 70, 70) if not active else (0, 120, 80)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, thickness=-1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 180, 180), thickness=1)
    cv2.putText(
        canvas,
        label,
        (x1 + 10, y2 - 8),
        UI_FONT,
        UI_FONT_SCALE,
        (230, 230, 230),
        UI_TEXT_THICKNESS,
        lineType=cv2.LINE_8,
    )


def _draw_ui(canvas: np.ndarray, ui: UIState, start_x: int) -> None:
    y = UI_TOP_MARGIN
    ui.rects = {}

    res_label = RES_OPTIONS[ui.selected_res][0] if RES_OPTIONS else "N/A"
    cv2.putText(
        canvas,
        "RES",
        (start_x, y + 12),
        UI_FONT,
        UI_FONT_SCALE,
        (200, 200, 200),
        UI_TEXT_THICKNESS,
        lineType=cv2.LINE_8,
    )
    y += UI_SECTION_GAP
    x = start_x
    for idx, (label, _, _) in enumerate(RES_OPTIONS):
        rect = (x, y, x + UI_BTN_W, y + UI_BTN_H)
        ui.rects[f"res_{idx}"] = rect
        _draw_button(canvas, label, rect, idx == ui.selected_res)
        x += UI_BTN_W + UI_BTN_GAP
    y += UI_BTN_H + UI_SECTION_GAP

    cv2.putText(
        canvas,
        "MODEL",
        (start_x, y + 12),
        UI_FONT,
        UI_FONT_SCALE,
        (200, 200, 200),
        UI_TEXT_THICKNESS,
        lineType=cv2.LINE_8,
    )
    y += UI_SECTION_GAP
    x = start_x
    for idx, (label, _) in enumerate(MODEL_OPTIONS):
        rect = (x, y, x + UI_BTN_W, y + UI_BTN_H)
        ui.rects[f"model_{idx}"] = rect
        _draw_button(canvas, label, rect, idx == ui.selected_model)
        x += UI_BTN_W + UI_BTN_GAP
    y += UI_BTN_H + UI_SECTION_GAP

    cv2.putText(
        canvas,
        f"LIMIT | {ui.person_limit:02d} | Detected: {ui.persons_detected:02d}",
        (start_x, y + 12),
        UI_FONT,
        UI_FONT_SCALE,
        (200, 200, 200),
        UI_TEXT_THICKNESS,
        lineType=cv2.LINE_8,
    )
    y += UI_SECTION_GAP
    slider_w = UI_BTN_W * 2 + UI_BTN_GAP
    slider_h = 10
    rect = (start_x, y, start_x + slider_w, y + slider_h)
    ui.rects["slider"] = rect
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (80, 80, 80), -1)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (140, 140, 140), 1)
    if MAX_PERSON_LIMIT > 0:
        ratio = ui.person_limit / float(MAX_PERSON_LIMIT)
    else:
        ratio = 0.0
    knob_x = rect[0] + int(ratio * slider_w)
    cv2.rectangle(
        canvas,
        (knob_x - 4, rect[1] - 4),
        (knob_x + 4, rect[3] + 4),
        (0, 140, 120),
        -1,
    )
    y += slider_h + UI_SECTION_GAP

    cv2.putText(
        canvas,
        f"SKIP | {ui.skip_frames}",
        (start_x, y + 12),
        UI_FONT,
        UI_FONT_SCALE,
        (200, 200, 200),
        UI_TEXT_THICKNESS,
        lineType=cv2.LINE_8,
    )
    y += UI_SECTION_GAP
    slider_w = UI_BTN_W * 2 + UI_BTN_GAP
    slider_h = 10
    rect = (start_x, y, start_x + slider_w, y + slider_h)
    ui.rects["skip_slider"] = rect
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (80, 80, 80), -1)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (140, 140, 140), 1)
    if MAX_SKIP_FRAMES > 0:
        ratio = ui.skip_frames / float(MAX_SKIP_FRAMES)
    else:
        ratio = 0.0
    knob_x = rect[0] + int(ratio * slider_w)
    cv2.rectangle(
        canvas,
        (knob_x - 4, rect[1] - 4),
        (knob_x + 4, rect[3] + 4),
        (0, 140, 120),
        -1,
    )
    y += slider_h + UI_SECTION_GAP

    cv2.putText(
        canvas,
        f"CONF | {ui.conf:.2f}",
        (start_x, y + 12),
        UI_FONT,
        UI_FONT_SCALE,
        (200, 200, 200),
        UI_TEXT_THICKNESS,
        lineType=cv2.LINE_8,
    )
    y += UI_SECTION_GAP
    slider_w = UI_BTN_W * 2 + UI_BTN_GAP
    slider_h = 10
    rect = (start_x, y, start_x + slider_w, y + slider_h)
    ui.rects["conf_slider"] = rect
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (80, 80, 80), -1)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (140, 140, 140), 1)
    conf_range = max(0.0001, CONF_MAX - CONF_MIN)
    ratio = (ui.conf - CONF_MIN) / conf_range
    ratio = min(1.0, max(0.0, ratio))
    knob_x = rect[0] + int(ratio * slider_w)
    cv2.rectangle(
        canvas,
        (knob_x - 4, rect[1] - 4),
        (knob_x + 4, rect[3] + 4),
        (0, 140, 120),
        -1,
    )
    y += slider_h + UI_SECTION_GAP

    imgsz_val = IMG_SIZES[ui.imgsz_idx] if IMG_SIZES else YOLO_IMGSZ
    cv2.putText(
        canvas,
        f"IMGSZ | {imgsz_val}",
        (start_x, y + 12),
        UI_FONT,
        UI_FONT_SCALE,
        (200, 200, 200),
        UI_TEXT_THICKNESS,
        lineType=cv2.LINE_8,
    )
    y += UI_SECTION_GAP
    rect = (start_x, y, start_x + slider_w, y + slider_h)
    ui.rects["imgsz_slider"] = rect
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (80, 80, 80), -1)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (140, 140, 140), 1)
    if IMG_SIZES:
        ratio = ui.imgsz_idx / float(max(1, len(IMG_SIZES) - 1))
    else:
        ratio = 0.0
    knob_x = rect[0] + int(ratio * slider_w)
    cv2.rectangle(
        canvas,
        (knob_x - 4, rect[1] - 4),
        (knob_x + 4, rect[3] + 4),
        (0, 140, 120),
        -1,
    )
    y += slider_h + UI_SECTION_GAP

    cv2.putText(
        canvas,
        f"MASK BLUR | {ui.blur}",
        (start_x, y + 12),
        UI_FONT,
        UI_FONT_SCALE,
        (200, 200, 200),
        UI_TEXT_THICKNESS,
        lineType=cv2.LINE_8,
    )
    y += UI_SECTION_GAP
    rect = (start_x, y, start_x + slider_w, y + slider_h)
    ui.rects["blur_slider"] = rect
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (80, 80, 80), -1)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (140, 140, 140), 1)
    blur_max = MAX_MASK_BLUR
    ratio = min(1.0, max(0.0, ui.blur / float(blur_max)))
    knob_x = rect[0] + int(ratio * slider_w)
    cv2.rectangle(
        canvas,
        (knob_x - 4, rect[1] - 4),
        (knob_x + 4, rect[3] + 4),
        (0, 140, 120),
        -1,
    )
    y += slider_h + UI_SECTION_GAP

    cv2.putText(
        canvas,
        f"MASK MORPH | {ui.morph}",
        (start_x, y + 12),
        UI_FONT,
        UI_FONT_SCALE,
        (200, 200, 200),
        UI_TEXT_THICKNESS,
        lineType=cv2.LINE_8,
    )
    y += UI_SECTION_GAP
    rect = (start_x, y, start_x + slider_w, y + slider_h)
    ui.rects["morph_slider"] = rect
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (80, 80, 80), -1)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (140, 140, 140), 1)
    morph_range = max(1, MAX_MORPH)
    ratio = (ui.morph + morph_range) / float(morph_range * 2)
    ratio = min(1.0, max(0.0, ratio))
    knob_x = rect[0] + int(ratio * slider_w)
    cv2.rectangle(
        canvas,
        (knob_x - 4, rect[1] - 4),
        (knob_x + 4, rect[3] + 4),
        (0, 140, 120),
        -1,
    )
    y += slider_h + UI_SECTION_GAP


def main() -> int:
    cv2.setUseOptimized(True)
    try:
        cv2.setNumThreads(max(1, int(os.environ.get("NEXT_CV_THREADS", "2"))))
    except Exception:
        pass

    model_path = YOLO_MODEL_PATH
    model = _load_model(model_path)
    if model is None:
        fallback = "yolov8n.pt"
        model = _load_model(fallback)
        if model is not None:
            model_path = fallback

    device = YOLO_DEVICE
    if not device:
        try:
            import torch

            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    ui = UIState()
    settings = _load_settings(SETTINGS_PATH)
    cam_index = max(0, int(settings.get("cam_index", 0)))
    if "selected_res" in settings:
        ui.selected_res = int(settings.get("selected_res", ui.selected_res))
    if "selected_model" in settings:
        ui.selected_model = int(settings.get("selected_model", ui.selected_model))
    if "person_limit" in settings:
        ui.person_limit = int(settings.get("person_limit", ui.person_limit))
    if "skip_frames" in settings:
        ui.skip_frames = int(settings.get("skip_frames", ui.skip_frames))
    if "conf" in settings:
        ui.conf = float(settings.get("conf", ui.conf))
    if "imgsz_idx" in settings:
        ui.imgsz_idx = int(settings.get("imgsz_idx", ui.imgsz_idx))
    if "blur" in settings:
        ui.blur = int(settings.get("blur", ui.blur))
    if "morph" in settings:
        ui.morph = int(settings.get("morph", ui.morph))

    cap, cam_index = _cycle_camera(cam_index)
    if cap is None:
        print("[CAM] No cameras available.", file=sys.stderr)
    for idx, (_, w, h) in enumerate(RES_OPTIONS):
        if w == CAM_WIDTH and h == CAM_HEIGHT:
            if "selected_res" not in settings:
                ui.selected_res = idx
            break
    for idx, (_, path) in enumerate(MODEL_OPTIONS):
        if path == model_path and "selected_model" not in settings:
            ui.selected_model = idx
            break
    if IMG_SIZES:
        if "imgsz_idx" not in settings:
            closest = min(range(len(IMG_SIZES)), key=lambda i: abs(IMG_SIZES[i] - YOLO_IMGSZ))
            ui.imgsz_idx = closest

    last_frame_time = time.time()
    prev_time = time.time()
    fps = 0.0

    def apply_resolution(idx: int) -> None:
        nonlocal cap
        global CAM_WIDTH, CAM_HEIGHT
        ui.selected_res = idx
        _, w, h = RES_OPTIONS[idx]
        CAM_WIDTH, CAM_HEIGHT = w, h
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        cap, _ = _cycle_camera(cam_index)
        if cap is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    worker: Optional[YoloWorker] = None

    def reset_worker() -> None:
        nonlocal worker
        if worker is not None:
            worker.stop()
            worker = None
        if model is not None:
            imgsz_val = IMG_SIZES[ui.imgsz_idx] if IMG_SIZES else YOLO_IMGSZ
            worker = YoloWorker(model, device, ui.conf, imgsz_val)

    reset_worker()

    def apply_model(idx: int) -> None:
        nonlocal model, model_path
        ui.selected_model = idx
        _, path = MODEL_OPTIONS[idx]
        model_path = path
        model = _load_model(model_path)
        if model is None:
            print("[YOLO] Modelo no disponible.", file=sys.stderr)
            reset_worker()
            return
        reset_worker()

    if IMG_SIZES:
        ui.imgsz_idx = max(0, min(ui.imgsz_idx, len(IMG_SIZES) - 1))
    if "selected_res" in settings and 0 <= ui.selected_res < len(RES_OPTIONS):
        apply_resolution(ui.selected_res)
    if "selected_model" in settings and 0 <= ui.selected_model < len(MODEL_OPTIONS):
        apply_model(ui.selected_model)

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            for idx in range(len(RES_OPTIONS)):
                rect = ui.rects.get(f"res_{idx}")
                if rect and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                    apply_resolution(idx)
                    mark_dirty()
                    return
            for idx in range(len(MODEL_OPTIONS)):
                rect = ui.rects.get(f"model_{idx}")
                if rect and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                    apply_model(idx)
                    mark_dirty()
                    return
            rect = ui.rects.get("slider")
            if rect and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                ui.slider_drag = True
                if rect[2] > rect[0]:
                    ratio = (x - rect[0]) / float(rect[2] - rect[0])
                    ui.person_limit = int(round(ratio * MAX_PERSON_LIMIT))
                    mark_dirty()
                return
            rect = ui.rects.get("skip_slider")
            if rect and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                ui.skip_drag = True
                if rect[2] > rect[0]:
                    ratio = (x - rect[0]) / float(rect[2] - rect[0])
                    ui.skip_frames = int(round(ratio * MAX_SKIP_FRAMES))
                    mark_dirty()
                return
            rect = ui.rects.get("conf_slider")
            if rect and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                ui.conf_drag = True
                if rect[2] > rect[0]:
                    ratio = (x - rect[0]) / float(rect[2] - rect[0])
                    ui.conf = CONF_MIN + ratio * (CONF_MAX - CONF_MIN)
                    mark_dirty()
                return
            rect = ui.rects.get("imgsz_slider")
            if rect and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                ui.imgsz_drag = True
                if IMG_SIZES and rect[2] > rect[0]:
                    ratio = (x - rect[0]) / float(rect[2] - rect[0])
                    ui.imgsz_idx = int(round(ratio * (len(IMG_SIZES) - 1)))
                    mark_dirty()
                return
            rect = ui.rects.get("blur_slider")
            if rect and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                ui.blur_drag = True
                if rect[2] > rect[0]:
                    ratio = (x - rect[0]) / float(rect[2] - rect[0])
                    ui.blur = int(round(ratio * MAX_MASK_BLUR))
                    mark_dirty()
                return
            rect = ui.rects.get("morph_slider")
            if rect and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                ui.morph_drag = True
                if rect[2] > rect[0]:
                    ratio = (x - rect[0]) / float(rect[2] - rect[0])
                    ui.morph = int(round((ratio * 2 - 1) * MAX_MORPH))
                    mark_dirty()
                return
        if event == cv2.EVENT_LBUTTONUP:
            ui.slider_drag = False
            ui.skip_drag = False
            ui.conf_drag = False
            ui.imgsz_drag = False
            ui.blur_drag = False
            ui.morph_drag = False
        if event == cv2.EVENT_MOUSEMOVE and ui.slider_drag:
            rect = ui.rects.get("slider")
            if rect and rect[2] > rect[0]:
                ratio = (x - rect[0]) / float(rect[2] - rect[0])
                ratio = min(1.0, max(0.0, ratio))
                ui.person_limit = int(round(ratio * MAX_PERSON_LIMIT))
                mark_dirty()
        if event == cv2.EVENT_MOUSEMOVE and ui.skip_drag:
            rect = ui.rects.get("skip_slider")
            if rect and rect[2] > rect[0]:
                ratio = (x - rect[0]) / float(rect[2] - rect[0])
                ratio = min(1.0, max(0.0, ratio))
                ui.skip_frames = int(round(ratio * MAX_SKIP_FRAMES))
                mark_dirty()
        if event == cv2.EVENT_MOUSEMOVE and ui.conf_drag:
            rect = ui.rects.get("conf_slider")
            if rect and rect[2] > rect[0]:
                ratio = (x - rect[0]) / float(rect[2] - rect[0])
                ratio = min(1.0, max(0.0, ratio))
                ui.conf = CONF_MIN + ratio * (CONF_MAX - CONF_MIN)
                mark_dirty()
        if event == cv2.EVENT_MOUSEMOVE and ui.imgsz_drag:
            rect = ui.rects.get("imgsz_slider")
            if IMG_SIZES and rect and rect[2] > rect[0]:
                ratio = (x - rect[0]) / float(rect[2] - rect[0])
                ratio = min(1.0, max(0.0, ratio))
                ui.imgsz_idx = int(round(ratio * (len(IMG_SIZES) - 1)))
                mark_dirty()
        if event == cv2.EVENT_MOUSEMOVE and ui.blur_drag:
            rect = ui.rects.get("blur_slider")
            if rect and rect[2] > rect[0]:
                ratio = (x - rect[0]) / float(rect[2] - rect[0])
                ratio = min(1.0, max(0.0, ratio))
                ui.blur = int(round(ratio * MAX_MASK_BLUR))
                mark_dirty()
        if event == cv2.EVENT_MOUSEMOVE and ui.morph_drag:
            rect = ui.rects.get("morph_slider")
            if rect and rect[2] > rect[0]:
                ratio = (x - rect[0]) / float(rect[2] - rect[0])
                ratio = min(1.0, max(0.0, ratio))
                ui.morph = int(round((ratio * 2 - 1) * MAX_MORPH))
                mark_dirty()

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    frame_idx = 0
    last_boxes: list[Tuple[int, int, int, int]] = []
    last_persons = 0
    last_mask = None
    last_params = (ui.conf, IMG_SIZES[ui.imgsz_idx] if IMG_SIZES else YOLO_IMGSZ)
    last_saved = 0.0
    settings_dirty = False

    def mark_dirty() -> None:
        nonlocal settings_dirty
        settings_dirty = True
    while True:
        if cap is not None:
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = _placeholder((CAM_WIDTH, CAM_HEIGHT))
        else:
            frame = _placeholder((CAM_WIDTH, CAM_HEIGHT))

        persons = 0
        if worker is not None:
            imgsz_val = IMG_SIZES[ui.imgsz_idx] if IMG_SIZES else YOLO_IMGSZ
            params = (ui.conf, imgsz_val)
            if params != last_params:
                worker.update_params(ui.conf, imgsz_val)
                last_params = params
            skip_frames = ui.skip_frames if MAX_SKIP_FRAMES > 0 else YOLO_SKIP_FRAMES
            run_infer = skip_frames == 0 or (frame_idx % (skip_frames + 1) == 0)
            if run_infer:
                worker.submit(frame.copy())
            result = worker.get_latest()
            last_boxes = result.boxes
            last_persons = result.persons
            last_mask = result.combined_mask
            persons = last_persons
            for i, (x1, y1, x2, y2) in enumerate(last_boxes, start=1):
                if ui.person_limit > 0 and i > ui.person_limit:
                    continue
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        ui.persons_detected = persons
        frame_idx += 1

        now = time.time()
        dt = now - prev_time
        if dt > 0:
            fps = 1.0 / dt
        prev_time = now

        third_w = CANVAS_WIDTH // 3
        cam_view = _fit_to_width(frame, third_w)
        cam_h = cam_view.shape[0]
        mask_top = cam_h
        mask_h = cam_h
        total_h = cam_h + mask_h
        canvas = np.zeros((total_h, CANVAS_WIDTH, 3), dtype=np.uint8)
        canvas[:cam_h, :third_w] = cam_view
        if mask_h > 0:
            if last_mask is not None:
                mask = last_mask
                if mask.shape[:2] != frame.shape[:2]:
                    mask = cv2.resize(
                        mask,
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                if ui.blur > 1:
                    k = ui.blur if ui.blur % 2 == 1 else ui.blur + 1
                    mask = cv2.GaussianBlur(mask, (k, k), 0)
                else:
                    mask = _smooth_mask(mask)
                mask = _apply_morph(mask, ui.morph)
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            else:
                mask_bgr = _mask_placeholder((third_w, mask_h))
            mask_view = cv2.resize(mask_bgr, (third_w, mask_h), interpolation=cv2.INTER_NEAREST)
            canvas[mask_top : mask_top + mask_h, :third_w] = mask_view
            cv2.rectangle(
                canvas,
                (0, mask_top),
                (third_w - 1, mask_top + mask_h - 1),
                (90, 90, 90),
                1,
            )
        res_label = RES_OPTIONS[ui.selected_res][0] if RES_OPTIONS else "N/A"
        cv2.putText(
            canvas,
            f"CAM {cam_index} {res_label} | FPS {fps:05.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        _draw_ui(canvas, ui, third_w + UI_LEFT_MARGIN)
        cv2.imshow(WINDOW_NAME, canvas)

        now_time = time.time()
        if settings_dirty and (now_time - last_saved) >= SETTINGS_SAVE_INTERVAL:
            data = {
                "cam_index": cam_index,
                "selected_res": ui.selected_res,
                "selected_model": ui.selected_model,
                "person_limit": ui.person_limit,
                "skip_frames": ui.skip_frames,
                "conf": ui.conf,
                "imgsz_idx": ui.imgsz_idx,
                "blur": ui.blur,
                "morph": ui.morph,
            }
            _save_settings(SETTINGS_PATH, data)
            last_saved = now_time
            settings_dirty = False

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            cap, cam_index = _cycle_camera(cam_index + 1)
            if cap is None:
                print("[CAM] No cameras available.", file=sys.stderr)
            mark_dirty()

        # light sleep to avoid busy loop if camera is missing
        if cap is None and (time.time() - last_frame_time) < 0.05:
            time.sleep(0.01)
        last_frame_time = time.time()

    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    if worker is not None:
        worker.stop()
    cv2.destroyAllWindows()
    if settings_dirty:
        data = {
            "cam_index": cam_index,
            "selected_res": ui.selected_res,
            "selected_model": ui.selected_model,
            "person_limit": ui.person_limit,
            "skip_frames": ui.skip_frames,
            "conf": ui.conf,
            "imgsz_idx": ui.imgsz_idx,
            "blur": ui.blur,
            "morph": ui.morph,
        }
        _save_settings(SETTINGS_PATH, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
