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
from fractions import Fraction
from pathlib import Path
from dataclasses import dataclass, field
import threading
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

WINDOW_NAME = "NEXT2"
CANVAS_WIDTH = max(320, int(os.environ.get("NEXT_CANVAS_W", "1536")))
CANVAS_HEIGHT = max(240, int(os.environ.get("NEXT_CANVAS_H", "800")))
CAM_MAX_INDEX = max(0, int(os.environ.get("NEXT_CAM_MAX", "4")))
CAM_WIDTH = max(0, int(os.environ.get("NEXT_CAM_WIDTH", "1280")))
CAM_HEIGHT = max(0, int(os.environ.get("NEXT_CAM_HEIGHT", "720")))
YOLO_MODEL_PATH = os.environ.get("NEXT_YOLO_MODEL", "yolov8n-seg.pt")
YOLO_CONF = float(os.environ.get("NEXT_YOLO_CONF", "0.25"))
YOLO_IMGSZ = int(os.environ.get("NEXT_YOLO_IMGSZ", "320"))
YOLO_DEVICE = os.environ.get("NEXT_YOLO_DEVICE", "").strip()
YOLO_SKIP_FRAMES = max(0, int(os.environ.get("NEXT_YOLO_SKIP", "0")))
MAX_SKIP_FRAMES = max(0, int(os.environ.get("NEXT_YOLO_SKIP_MAX", "5")))
MAX_PERSON_LIMIT = int(os.environ.get("NEXT_PERSON_LIMIT_MAX", "20"))
MASK_BLUR = max(0, int(os.environ.get("NEXT_MASK_BLUR", "5")))
MAX_MASK_BLUR = max(1, int(os.environ.get("NEXT_MASK_BLUR_MAX", "15")))
CONF_MIN = float(os.environ.get("NEXT_CONF_MIN", "0.1"))
CONF_MAX = float(os.environ.get("NEXT_CONF_MAX", "0.6"))
IMG_SIZES = [256, 320, 384, 448, 512, 640, 768]
MAX_MORPH = max(0, int(os.environ.get("NEXT_MASK_MORPH_MAX", "5")))
PERSIST_HOLD_MAX = float(os.environ.get("NEXT_PERSIST_HOLD_MAX", "2.0"))
PERSIST_RISE_MAX = float(os.environ.get("NEXT_PERSIST_RISE_MAX", "1.0"))
PERSIST_FALL_MAX = float(os.environ.get("NEXT_PERSIST_FALL_MAX", "1.5"))
PERSIST_MIN = float(os.environ.get("NEXT_PERSIST_MIN", "0.02"))
ENABLE_NDI_GALLERY = os.environ.get("NEXT_ENABLE_NDI_GALLERY", "1") == "1"
NDI_GALLERY_NAME = os.environ.get("NEXT_NDI_GALLERY_NAME", "NEXT2 Gallery NDI")
AREA_MIN_MAX = float(os.environ.get("NEXT_AREA_MIN_MAX", "0.2"))
ASPECT_MIN_MIN = float(os.environ.get("NEXT_ASPECT_MIN_MIN", "0.5"))
ASPECT_MIN_MAX = float(os.environ.get("NEXT_ASPECT_MIN_MAX", "2.5"))
ASPECT_MAX_MIN = float(os.environ.get("NEXT_ASPECT_MAX_MIN", "1.5"))
ASPECT_MAX_MAX = float(os.environ.get("NEXT_ASPECT_MAX_MAX", "5.0"))
SOLIDITY_MIN_MAX = float(os.environ.get("NEXT_SOLIDITY_MIN_MAX", "1.0"))
GALLERY_SCALE_MIN = float(os.environ.get("NEXT_GALLERY_SCALE_MIN", "0.05"))
GALLERY_SCALE_MAX = float(os.environ.get("NEXT_GALLERY_SCALE_MAX", "0.6"))
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
UI_FONT_SCALE = 0.42
UI_TEXT_THICKNESS = 1
UI_BTN_W = 96
UI_BTN_H = 22
UI_BTN_GAP = 8
UI_SECTION_GAP = 16
UI_LABEL_GAP = 20
UI_LEFT_MARGIN = 14
UI_TOP_MARGIN = 14
UI_PANEL_W = max(UI_BTN_W * 3 + UI_BTN_GAP * 2, UI_BTN_W * 2 + UI_BTN_GAP)


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


def _fit_to_box_center(image: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(box_w / float(w), box_h / float(h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    x_off = max(0, (box_w - new_w) // 2)
    y_off = max(0, (box_h - new_h) // 2)
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def _fit_to_box_center_mask(image: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(box_w / float(w), box_h / float(h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    x_off = max(0, (box_w - new_w) // 2)
    y_off = max(0, (box_h - new_h) // 2)
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


def _get_ndi_module():
    try:
        import importlib.resources as ir

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


class NDIPublisher:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ndi = None
        self.sender = None
        self.ready = False
        self._announced = False
        self.vf = None
        if not ENABLE_NDI_GALLERY:
            return
        try:
            ndi = _get_ndi_module()
            if ndi is None or not hasattr(ndi, "Sender"):
                raise RuntimeError("NDI no disponible")
            self.ndi = ndi
            self.sender = ndi.Sender(ndi_name=name)
        except Exception as exc:
            print(f"[NDI] No disponible: {exc}", file=sys.stderr)
            self.sender = None
            self.ndi = None

    def publish(self, frame: np.ndarray) -> None:
        if self.sender is None or self.ndi is None:
            return
        try:
            ndi = self.ndi
            VideoSendFrame = ndi.video_frame.VideoSendFrame if hasattr(ndi, "video_frame") else None
            if frame.ndim == 2:
                frame_bgra = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGRA)
            else:
                frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            if not frame_bgra.flags["C_CONTIGUOUS"]:
                frame_bgra = frame_bgra.copy()
            h, w = frame_bgra.shape[:2]
            if self.vf is None or self.vf.get_resolution() != (w, h):
                if VideoSendFrame is None:
                    print("[NDI] cyndilib sin VideoSendFrame.", file=sys.stderr)
                    return
                if self.ready:
                    try:
                        self.sender.close()
                    except Exception:
                        pass
                    self.ready = False
                vf = VideoSendFrame()
                vf.set_resolution(w, h)
                vf.set_fourcc(ndi.FourCC.BGRA)
                vf.set_frame_rate(Fraction(60, 1))
                self.sender.set_video_frame(vf)
                self.vf = vf
                if not self.ready:
                    self.sender.open()
                    self.ready = True
            self.sender.write_video_async(frame_bgra.ravel(order="C"))
            if not self._announced:
                print(f"[NDI] Enviando salida NDI '{self.name}' ({w}x{h})", file=sys.stderr)
                self._announced = True
        except Exception as exc:
            print(f"[NDI] Error al publicar: {exc}", file=sys.stderr)


def _update_persistent_mask(
    persist_mask: Optional[np.ndarray],
    mask_binary: np.ndarray,
    last_detect_time: Optional[float],
    last_detect_mask: Optional[np.ndarray],
    now: float,
    dt: float,
    hold_sec: float,
    rise_tau: float,
    fall_tau: float,
) -> tuple[np.ndarray, Optional[float], Optional[np.ndarray]]:
    if persist_mask is None:
        persist_mask = np.zeros(mask_binary.shape, dtype=np.float32)
        last_detect_mask = None
    elif persist_mask.shape != mask_binary.shape:
        mask_binary = cv2.resize(
            mask_binary, (persist_mask.shape[1], persist_mask.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    has_detection = np.any(mask_binary > 0)
    if has_detection:
        last_detect_time = now
        last_detect_mask = (mask_binary > 0).astype(np.float32)

    if last_detect_mask is not None and last_detect_mask.shape != persist_mask.shape:
        last_detect_mask = None
    if last_detect_time is not None and (now - last_detect_time) <= hold_sec:
        target_on = last_detect_mask if last_detect_mask is not None else np.zeros_like(persist_mask)
    else:
        target_on = np.zeros_like(persist_mask)

    rise_rate = 1.0 - np.exp(-dt / max(rise_tau, 1e-6))
    fall_rate = 1.0 - np.exp(-dt / max(fall_tau, 1e-6))

    on_mask = target_on > 0.0
    persist_mask[on_mask] = (1.0 - rise_rate) * persist_mask[on_mask] + rise_rate * 1.0
    persist_mask[~on_mask] = (1.0 - fall_rate) * persist_mask[~on_mask]
    return persist_mask, last_detect_time, last_detect_mask


@dataclass
class InferenceResult:
    boxes: list[Tuple[int, int, int, int]] = field(default_factory=list)
    combined_mask: Optional[np.ndarray] = None
    person_masks: list[np.ndarray] = field(default_factory=list)
    person_scales: list[float] = field(default_factory=list)
    person_centroids: list[Tuple[float, float]] = field(default_factory=list)
    persons: int = 0


class YoloWorker:
    def __init__(
        self,
        model: YOLO,
        device: str,
        conf: float,
        imgsz: int,
        area_min: float,
        aspect_min: float,
        aspect_max: float,
        solidity_min: float,
    ) -> None:
        self.model = model
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.area_min = area_min
        self.aspect_min = aspect_min
        self.aspect_max = aspect_max
        self.solidity_min = solidity_min
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

    def update_params(
        self,
        conf: float,
        imgsz: int,
        area_min: float,
        aspect_min: float,
        aspect_max: float,
        solidity_min: float,
    ) -> None:
        with self._lock:
            self.conf = conf
            self.imgsz = imgsz
            self.area_min = area_min
            self.aspect_min = aspect_min
            self.aspect_max = aspect_max
            self.solidity_min = solidity_min

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
                person_masks: list[np.ndarray] = []
                person_scales: list[float] = []
                person_centroids: list[Tuple[float, float]] = []
                persons = 0
                combined_mask = None
                mask_canvas = None
                frame_h = max(1, frame.shape[0])
                if result.boxes is not None:
                    if result.masks is not None and result.masks.xy is not None:
                        mask_canvas = np.zeros(frame.shape[:2], dtype=np.uint8)
                        polygons = result.masks.xy
                    else:
                        polygons = None
                    for idx, cls_id in enumerate(result.boxes.cls):
                        if int(cls_id) != 0:
                            continue
                        x1, y1, x2, y2 = result.boxes.xyxy[idx].int().tolist()
                        bx1 = max(0, min(x1, frame.shape[1] - 1))
                        bx2 = max(0, min(x2, frame.shape[1]))
                        by1 = max(0, min(y1, frame.shape[0] - 1))
                        by2 = max(0, min(y2, frame.shape[0]))
                        if bx2 <= bx1 or by2 <= by1:
                            continue
                        box_h = max(1, by2 - by1)
                        size_ratio = box_h / float(frame_h)
                        box_area = float((bx2 - bx1) * (by2 - by1))
                        aspect = (by2 - by1) / float(max(1, bx2 - bx1))
                        mask_area = None
                        local_mask = None
                        pts = None
                        if polygons is not None and idx < len(polygons):
                            pts = polygons[idx].astype(np.int32)
                            if pts.size > 0:
                                local_pts = pts - np.array([bx1, by1], dtype=np.int32)
                                local_mask = np.zeros((by2 - by1, bx2 - bx1), dtype=np.uint8)
                                cv2.fillPoly(local_mask, [local_pts], 255)
                                mask_area = float(np.count_nonzero(local_mask))
                        if mask_area is None:
                            mask_area = box_area
                        area_ratio = mask_area / float(frame.shape[0] * frame.shape[1])
                        solidity = mask_area / max(1.0, box_area)
                        if (
                            area_ratio < self.area_min
                            or aspect < self.aspect_min
                            or aspect > self.aspect_max
                            or solidity < self.solidity_min
                        ):
                            continue
                        if local_mask is None:
                            continue
                        persons += 1
                        boxes_out.append((bx1, by1, bx2, by2))
                        person_scales.append(min(1.0, max(0.0, size_ratio)))
                        person_centroids.append(((bx1 + bx2) / 2.0, (by1 + by2) / 2.0))
                        person_masks.append(local_mask)
                        if mask_canvas is not None and pts is not None:
                            cv2.fillPoly(mask_canvas, [pts], 255)
                    if mask_canvas is not None:
                        combined_mask = mask_canvas
                with self._lock:
                    self._result = InferenceResult(
                        boxes=boxes_out,
                        combined_mask=combined_mask,
                        person_masks=person_masks,
                        person_scales=person_scales,
                        person_centroids=person_centroids,
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
    hold_drag: bool = False
    rise_drag: bool = False
    fall_drag: bool = False
    size_drag: bool = False
    area_drag: bool = False
    aspect_min_drag: bool = False
    aspect_max_drag: bool = False
    solidity_drag: bool = False
    conf: float = YOLO_CONF
    imgsz_idx: int = 0
    blur: int = MASK_BLUR
    morph: int = 0
    hold: float = 0.35
    rise: float = 0.12
    fall: float = 0.25
    size_smooth: float = 0.35
    area_min: float = 0.005
    aspect_min: float = 0.8
    aspect_max: float = 5.5
    solidity_min: float = 0.15
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
    y += UI_LABEL_GAP
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
    y += UI_LABEL_GAP
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
    y += UI_LABEL_GAP
    slider_w = UI_BTN_W * 2 + UI_BTN_GAP
    slider_h = 8
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
    y += UI_LABEL_GAP
    slider_w = UI_BTN_W * 2 + UI_BTN_GAP
    slider_h = 8
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
    y += UI_LABEL_GAP
    slider_w = UI_BTN_W * 2 + UI_BTN_GAP
    slider_h = 8
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
    y += UI_LABEL_GAP
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
    y += UI_LABEL_GAP
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
    y += UI_LABEL_GAP
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

    hold_val = min(PERSIST_HOLD_MAX, max(PERSIST_MIN, ui.hold))
    cv2.putText(
        canvas,
        f"PERSIST HOLD | {hold_val:.2f}s",
        (start_x, y + 12),
        UI_FONT,
        UI_FONT_SCALE,
        (200, 200, 200),
        UI_TEXT_THICKNESS,
        lineType=cv2.LINE_8,
    )
    y += UI_LABEL_GAP
    rect = (start_x, y, start_x + slider_w, y + slider_h)
    ui.rects["hold_slider"] = rect
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (80, 80, 80), -1)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (140, 140, 140), 1)
    ratio = hold_val / max(PERSIST_HOLD_MAX, PERSIST_MIN)
    knob_x = rect[0] + int(ratio * slider_w)
    cv2.rectangle(
        canvas,
        (knob_x - 4, rect[1] - 4),
        (knob_x + 4, rect[3] + 4),
        (0, 140, 120),
        -1,
    )
    y += slider_h + UI_SECTION_GAP

    rise_val = min(PERSIST_RISE_MAX, max(PERSIST_MIN, ui.rise))
    cv2.putText(
        canvas,
        f"PERSIST RISE | {rise_val:.2f}s",
        (start_x, y + 12),
        UI_FONT,
        UI_FONT_SCALE,
        (200, 200, 200),
        UI_TEXT_THICKNESS,
        lineType=cv2.LINE_8,
    )
    y += UI_LABEL_GAP
    rect = (start_x, y, start_x + slider_w, y + slider_h)
    ui.rects["rise_slider"] = rect
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (80, 80, 80), -1)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (140, 140, 140), 1)
    ratio = rise_val / max(PERSIST_RISE_MAX, PERSIST_MIN)
    knob_x = rect[0] + int(ratio * slider_w)
    cv2.rectangle(
        canvas,
        (knob_x - 4, rect[1] - 4),
        (knob_x + 4, rect[3] + 4),
        (0, 140, 120),
        -1,
    )
    y += slider_h + UI_SECTION_GAP

    fall_val = min(PERSIST_FALL_MAX, max(PERSIST_MIN, ui.fall))
    cv2.putText(
        canvas,
        f"PERSIST FALL | {fall_val:.2f}s",
        (start_x, y + 12),
        UI_FONT,
        UI_FONT_SCALE,
        (200, 200, 200),
        UI_TEXT_THICKNESS,
        lineType=cv2.LINE_8,
    )
    y += UI_LABEL_GAP
    rect = (start_x, y, start_x + slider_w, y + slider_h)
    ui.rects["fall_slider"] = rect
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (80, 80, 80), -1)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (140, 140, 140), 1)
    ratio = fall_val / max(PERSIST_FALL_MAX, PERSIST_MIN)
    knob_x = rect[0] + int(ratio * slider_w)
    cv2.rectangle(
        canvas,
        (knob_x - 4, rect[1] - 4),
        (knob_x + 4, rect[3] + 4),
        (0, 140, 120),
        -1,
    )
    y += slider_h + UI_SECTION_GAP

    smooth_range = max(0.0001, GALLERY_SCALE_MAX - GALLERY_SCALE_MIN)
    smooth_val = min(GALLERY_SCALE_MAX, max(GALLERY_SCALE_MIN, ui.size_smooth))
    cv2.putText(
        canvas,
        f"SIZE SMOOTH | {smooth_val:.2f}",
        (start_x, y + 12),
        UI_FONT,
        UI_FONT_SCALE,
        (200, 200, 200),
        UI_TEXT_THICKNESS,
        lineType=cv2.LINE_8,
    )
    y += UI_LABEL_GAP
    rect = (start_x, y, start_x + slider_w, y + slider_h)
    ui.rects["size_slider"] = rect
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (80, 80, 80), -1)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (140, 140, 140), 1)
    ratio = (smooth_val - GALLERY_SCALE_MIN) / smooth_range
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

    area_val = min(AREA_MIN_MAX, max(0.0, ui.area_min))
    cv2.putText(
        canvas,
        f"MIN AREA | {area_val:.3f}",
        (start_x, y + 12),
        UI_FONT,
        UI_FONT_SCALE,
        (200, 200, 200),
        UI_TEXT_THICKNESS,
        lineType=cv2.LINE_8,
    )
    y += UI_LABEL_GAP
    rect = (start_x, y, start_x + slider_w, y + slider_h)
    ui.rects["area_slider"] = rect
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (80, 80, 80), -1)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (140, 140, 140), 1)
    ratio = area_val / max(AREA_MIN_MAX, 1e-6)
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

    aspect_min_val = min(ASPECT_MIN_MAX, max(ASPECT_MIN_MIN, ui.aspect_min))
    cv2.putText(
        canvas,
        f"ASPECT MIN | {aspect_min_val:.2f}",
        (start_x, y + 12),
        UI_FONT,
        UI_FONT_SCALE,
        (200, 200, 200),
        UI_TEXT_THICKNESS,
        lineType=cv2.LINE_8,
    )
    y += UI_LABEL_GAP
    rect = (start_x, y, start_x + slider_w, y + slider_h)
    ui.rects["aspect_min_slider"] = rect
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (80, 80, 80), -1)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (140, 140, 140), 1)
    ratio = (aspect_min_val - ASPECT_MIN_MIN) / max(ASPECT_MIN_MAX - ASPECT_MIN_MIN, 1e-6)
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

    aspect_max_val = min(ASPECT_MAX_MAX, max(ASPECT_MAX_MIN, ui.aspect_max))
    cv2.putText(
        canvas,
        f"ASPECT MAX | {aspect_max_val:.2f}",
        (start_x, y + 12),
        UI_FONT,
        UI_FONT_SCALE,
        (200, 200, 200),
        UI_TEXT_THICKNESS,
        lineType=cv2.LINE_8,
    )
    y += UI_LABEL_GAP
    rect = (start_x, y, start_x + slider_w, y + slider_h)
    ui.rects["aspect_max_slider"] = rect
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (80, 80, 80), -1)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (140, 140, 140), 1)
    ratio = (aspect_max_val - ASPECT_MAX_MIN) / max(ASPECT_MAX_MAX - ASPECT_MAX_MIN, 1e-6)
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

    solidity_val = min(SOLIDITY_MIN_MAX, max(0.0, ui.solidity_min))
    cv2.putText(
        canvas,
        f"SOLIDITY | {solidity_val:.2f}",
        (start_x, y + 12),
        UI_FONT,
        UI_FONT_SCALE,
        (200, 200, 200),
        UI_TEXT_THICKNESS,
        lineType=cv2.LINE_8,
    )
    y += UI_LABEL_GAP
    rect = (start_x, y, start_x + slider_w, y + slider_h)
    ui.rects["solidity_slider"] = rect
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (80, 80, 80), -1)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (140, 140, 140), 1)
    ratio = solidity_val / max(SOLIDITY_MIN_MAX, 1e-6)
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

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_WIDTH, CANVAS_HEIGHT)
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
    if "hold" in settings:
        ui.hold = float(settings.get("hold", ui.hold))
    if "rise" in settings:
        ui.rise = float(settings.get("rise", ui.rise))
    if "fall" in settings:
        ui.fall = float(settings.get("fall", ui.fall))
    if "area_min" in settings:
        ui.area_min = float(settings.get("area_min", ui.area_min))
    if "aspect_min" in settings:
        ui.aspect_min = float(settings.get("aspect_min", ui.aspect_min))
    if "aspect_max" in settings:
        ui.aspect_max = float(settings.get("aspect_max", ui.aspect_max))
    if "solidity_min" in settings:
        ui.solidity_min = float(settings.get("solidity_min", ui.solidity_min))
    if "size_smooth" in settings:
        ui.size_smooth = float(settings.get("size_smooth", ui.size_smooth))

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
            worker = YoloWorker(
                model,
                device,
                ui.conf,
                imgsz_val,
                ui.area_min,
                ui.aspect_min,
                ui.aspect_max,
                ui.solidity_min,
            )

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
            rect = ui.rects.get("hold_slider")
            if rect and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                ui.hold_drag = True
                if rect[2] > rect[0]:
                    ratio = (x - rect[0]) / float(rect[2] - rect[0])
                    ui.hold = PERSIST_MIN + ratio * (PERSIST_HOLD_MAX - PERSIST_MIN)
                    mark_dirty()
                return
            rect = ui.rects.get("rise_slider")
            if rect and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                ui.rise_drag = True
                if rect[2] > rect[0]:
                    ratio = (x - rect[0]) / float(rect[2] - rect[0])
                    ui.rise = PERSIST_MIN + ratio * (PERSIST_RISE_MAX - PERSIST_MIN)
                    mark_dirty()
                return
            rect = ui.rects.get("fall_slider")
            if rect and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                ui.fall_drag = True
                if rect[2] > rect[0]:
                    ratio = (x - rect[0]) / float(rect[2] - rect[0])
                    ui.fall = PERSIST_MIN + ratio * (PERSIST_FALL_MAX - PERSIST_MIN)
                    mark_dirty()
                return
            rect = ui.rects.get("size_slider")
            if rect and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                ui.size_drag = True
                if rect[2] > rect[0]:
                    ratio = (x - rect[0]) / float(rect[2] - rect[0])
                    ui.size_smooth = GALLERY_SCALE_MIN + ratio * (GALLERY_SCALE_MAX - GALLERY_SCALE_MIN)
                    mark_dirty()
                return
            rect = ui.rects.get("area_slider")
            if rect and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                ui.area_drag = True
                if rect[2] > rect[0]:
                    ratio = (x - rect[0]) / float(rect[2] - rect[0])
                    ui.area_min = ratio * AREA_MIN_MAX
                    mark_dirty()
                return
            rect = ui.rects.get("aspect_min_slider")
            if rect and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                ui.aspect_min_drag = True
                if rect[2] > rect[0]:
                    ratio = (x - rect[0]) / float(rect[2] - rect[0])
                    ui.aspect_min = ASPECT_MIN_MIN + ratio * (ASPECT_MIN_MAX - ASPECT_MIN_MIN)
                    mark_dirty()
                return
            rect = ui.rects.get("aspect_max_slider")
            if rect and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                ui.aspect_max_drag = True
                if rect[2] > rect[0]:
                    ratio = (x - rect[0]) / float(rect[2] - rect[0])
                    ui.aspect_max = ASPECT_MAX_MIN + ratio * (ASPECT_MAX_MAX - ASPECT_MAX_MIN)
                    mark_dirty()
                return
            rect = ui.rects.get("solidity_slider")
            if rect and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                ui.solidity_drag = True
                if rect[2] > rect[0]:
                    ratio = (x - rect[0]) / float(rect[2] - rect[0])
                    ui.solidity_min = ratio * SOLIDITY_MIN_MAX
                    mark_dirty()
                return
        if event == cv2.EVENT_LBUTTONUP:
            ui.slider_drag = False
            ui.skip_drag = False
            ui.conf_drag = False
            ui.imgsz_drag = False
            ui.blur_drag = False
            ui.morph_drag = False
            ui.hold_drag = False
            ui.rise_drag = False
            ui.fall_drag = False
            ui.size_drag = False
            ui.area_drag = False
            ui.aspect_min_drag = False
            ui.aspect_max_drag = False
            ui.solidity_drag = False
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
        if event == cv2.EVENT_MOUSEMOVE and ui.hold_drag:
            rect = ui.rects.get("hold_slider")
            if rect and rect[2] > rect[0]:
                ratio = (x - rect[0]) / float(rect[2] - rect[0])
                ratio = min(1.0, max(0.0, ratio))
                ui.hold = PERSIST_MIN + ratio * (PERSIST_HOLD_MAX - PERSIST_MIN)
                mark_dirty()
        if event == cv2.EVENT_MOUSEMOVE and ui.rise_drag:
            rect = ui.rects.get("rise_slider")
            if rect and rect[2] > rect[0]:
                ratio = (x - rect[0]) / float(rect[2] - rect[0])
                ratio = min(1.0, max(0.0, ratio))
                ui.rise = PERSIST_MIN + ratio * (PERSIST_RISE_MAX - PERSIST_MIN)
                mark_dirty()
        if event == cv2.EVENT_MOUSEMOVE and ui.fall_drag:
            rect = ui.rects.get("fall_slider")
            if rect and rect[2] > rect[0]:
                ratio = (x - rect[0]) / float(rect[2] - rect[0])
                ratio = min(1.0, max(0.0, ratio))
                ui.fall = PERSIST_MIN + ratio * (PERSIST_FALL_MAX - PERSIST_MIN)
                mark_dirty()
        if event == cv2.EVENT_MOUSEMOVE and ui.size_drag:
            rect = ui.rects.get("size_slider")
            if rect and rect[2] > rect[0]:
                ratio = (x - rect[0]) / float(rect[2] - rect[0])
                ratio = min(1.0, max(0.0, ratio))
                ui.size_smooth = GALLERY_SCALE_MIN + ratio * (GALLERY_SCALE_MAX - GALLERY_SCALE_MIN)
                mark_dirty()
        if event == cv2.EVENT_MOUSEMOVE and ui.area_drag:
            rect = ui.rects.get("area_slider")
            if rect and rect[2] > rect[0]:
                ratio = (x - rect[0]) / float(rect[2] - rect[0])
                ratio = min(1.0, max(0.0, ratio))
                ui.area_min = ratio * AREA_MIN_MAX
                mark_dirty()
        if event == cv2.EVENT_MOUSEMOVE and ui.aspect_min_drag:
            rect = ui.rects.get("aspect_min_slider")
            if rect and rect[2] > rect[0]:
                ratio = (x - rect[0]) / float(rect[2] - rect[0])
                ratio = min(1.0, max(0.0, ratio))
                ui.aspect_min = ASPECT_MIN_MIN + ratio * (ASPECT_MIN_MAX - ASPECT_MIN_MIN)
                mark_dirty()
        if event == cv2.EVENT_MOUSEMOVE and ui.aspect_max_drag:
            rect = ui.rects.get("aspect_max_slider")
            if rect and rect[2] > rect[0]:
                ratio = (x - rect[0]) / float(rect[2] - rect[0])
                ratio = min(1.0, max(0.0, ratio))
                ui.aspect_max = ASPECT_MAX_MIN + ratio * (ASPECT_MAX_MAX - ASPECT_MAX_MIN)
                mark_dirty()
        if event == cv2.EVENT_MOUSEMOVE and ui.solidity_drag:
            rect = ui.rects.get("solidity_slider")
            if rect and rect[2] > rect[0]:
                ratio = (x - rect[0]) / float(rect[2] - rect[0])
                ratio = min(1.0, max(0.0, ratio))
                ui.solidity_min = ratio * SOLIDITY_MIN_MAX
                mark_dirty()

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    frame_idx = 0
    last_boxes: list[Tuple[int, int, int, int]] = []
    last_persons = 0
    last_mask = None
    last_params = (
        ui.conf,
        IMG_SIZES[ui.imgsz_idx] if IMG_SIZES else YOLO_IMGSZ,
        ui.area_min,
        ui.aspect_min,
        ui.aspect_max,
        ui.solidity_min,
    )
    last_saved = 0.0
    settings_dirty = False
    gallery_slots: list[Optional[np.ndarray]] = [None for _ in range(MAX_PERSON_LIMIT)]
    gallery_scales: list[float] = [0.0 for _ in range(MAX_PERSON_LIMIT)]
    gallery_persist: list[Optional[np.ndarray]] = [None for _ in range(MAX_PERSON_LIMIT)]
    gallery_last_time: list[Optional[float]] = [None for _ in range(MAX_PERSON_LIMIT)]
    gallery_last_mask: list[Optional[np.ndarray]] = [None for _ in range(MAX_PERSON_LIMIT)]
    gallery_centroids: list[Optional[Tuple[float, float]]] = [None for _ in range(MAX_PERSON_LIMIT)]
    ndi_gallery = NDIPublisher(NDI_GALLERY_NAME) if ENABLE_NDI_GALLERY else None

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

        now = time.time()
        dt = now - prev_time
        if dt > 0:
            fps = 1.0 / dt
        prev_time = now

        persons = 0
        if worker is not None:
            imgsz_val = IMG_SIZES[ui.imgsz_idx] if IMG_SIZES else YOLO_IMGSZ
            params = (ui.conf, imgsz_val, ui.area_min, ui.aspect_min, ui.aspect_max, ui.solidity_min)
            if params != last_params:
                worker.update_params(
                    ui.conf,
                    imgsz_val,
                    ui.area_min,
                    ui.aspect_min,
                    ui.aspect_max,
                    ui.solidity_min,
                )
                last_params = params
            skip_frames = ui.skip_frames if MAX_SKIP_FRAMES > 0 else YOLO_SKIP_FRAMES
            run_infer = skip_frames == 0 or (frame_idx % (skip_frames + 1) == 0)
            if run_infer:
                worker.submit(frame.copy())
            result = worker.get_latest()
            last_boxes = result.boxes
            last_persons = result.persons
            last_mask = result.combined_mask
            if ui.person_limit == 0:
                for idx in range(MAX_PERSON_LIMIT):
                    gallery_slots[idx] = None
                    gallery_scales[idx] = 0.0
                    gallery_persist[idx] = None
                    gallery_last_time[idx] = None
                    gallery_last_mask[idx] = None
                    gallery_centroids[idx] = None
            if result.person_masks:
                dets = list(
                    zip(
                        result.person_masks,
                        result.person_scales,
                        result.person_centroids,
                    )
                )
            else:
                dets = []

            assigned = set()
            slot_used = [False] * MAX_PERSON_LIMIT
            for i, det in enumerate(dets):
                mask, scale, cent = det
                best_slot = None
                best_dist = None
                for s in range(MAX_PERSON_LIMIT):
                    if slot_used[s]:
                        continue
                    if gallery_centroids[s] is None:
                        continue
                    sx, sy = gallery_centroids[s]
                    dx = cent[0] - sx
                    dy = cent[1] - sy
                    dist = dx * dx + dy * dy
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_slot = s
                if best_slot is None:
                    for s in range(MAX_PERSON_LIMIT):
                        if slot_used[s]:
                            continue
                        if gallery_centroids[s] is None:
                            best_slot = s
                            break
                if best_slot is None:
                    for s in range(MAX_PERSON_LIMIT):
                        if slot_used[s]:
                            continue
                        best_slot = s
                        break
                if best_slot is None:
                    continue
                assigned.add((i, best_slot))
                slot_used[best_slot] = True

            for s in range(MAX_PERSON_LIMIT):
                det = None
                for i, slot in assigned:
                    if slot == s:
                        det = dets[i]
                        break
                if det is not None:
                    new_mask, raw, cent = det
                    persist, last_t, last_m = _update_persistent_mask(
                        gallery_persist[s],
                        new_mask,
                        gallery_last_time[s],
                        gallery_last_mask[s],
                        now,
                        dt,
                        ui.hold,
                        ui.rise,
                        ui.fall,
                    )
                    gallery_persist[s] = persist
                    gallery_last_time[s] = last_t
                    gallery_last_mask[s] = last_m
                    gallery_slots[s] = np.clip(persist * 255.0, 0, 255).astype(np.uint8)
                    gallery_centroids[s] = cent
                    if gallery_scales[s] <= 0.0 or raw >= gallery_scales[s]:
                        gallery_scales[s] = raw
                    else:
                        alpha = min(GALLERY_SCALE_MAX, max(GALLERY_SCALE_MIN, ui.size_smooth))
                        gallery_scales[s] = (1.0 - alpha) * gallery_scales[s] + alpha * raw
                else:
                    if gallery_persist[s] is not None:
                        empty_mask = np.zeros_like(gallery_persist[s], dtype=np.uint8)
                        persist, last_t, last_m = _update_persistent_mask(
                            gallery_persist[s],
                            empty_mask,
                            gallery_last_time[s],
                            gallery_last_mask[s],
                            now,
                            dt,
                            ui.hold,
                            ui.rise,
                            ui.fall,
                        )
                        gallery_persist[s] = persist
                        gallery_last_time[s] = last_t
                        gallery_last_mask[s] = last_m
                        gallery_slots[s] = np.clip(persist * 255.0, 0, 255).astype(np.uint8)
                    else:
                        gallery_slots[s] = None
                        gallery_scales[s] = 0.0
                        gallery_centroids[s] = None
            persons = last_persons
            for i, (x1, y1, x2, y2) in enumerate(last_boxes, start=1):
                if ui.person_limit > 0 and i > ui.person_limit:
                    continue
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        ui.persons_detected = persons
        frame_idx += 1

        third_w = CANVAS_WIDTH // 3
        cam_view = _fit_to_width(frame, third_w)
        cam_h = cam_view.shape[0]
        mask_top = cam_h
        mask_h = cam_h
        total_h = max(CANVAS_HEIGHT, cam_h + mask_h)
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
        ui_start_x = third_w + UI_LEFT_MARGIN
        _draw_ui(canvas, ui, ui_start_x)
        gallery_x = ui_start_x + UI_PANEL_W + UI_LEFT_MARGIN
        gallery_w = max(0, CANVAS_WIDTH - gallery_x - UI_LEFT_MARGIN)
        gallery_canvas = None
        if gallery_w > 60:
            gap = 0
            inner_margin = 2
            min_scale = 0.4
            min_thumb = 50
            available_h = max(1, total_h - UI_TOP_MARGIN)
            max_slots = MAX_PERSON_LIMIT
            show_slots = min(max_slots, ui.person_limit if ui.person_limit > 0 else max_slots)
            max_cols = max(1, min(show_slots, gallery_w // min_thumb))
            best_cols = 1
            best_thumb = 0
            for cols in range(1, max_cols + 1):
                rows_needed = max(1, int(np.ceil(show_slots / float(cols))))
                thumb = min(
                    gallery_w // cols,
                    max(1, int((available_h - (rows_needed - 1) * gap) / float(rows_needed))),
                )
                if thumb > best_thumb:
                    best_thumb = thumb
                    best_cols = cols
            cols = best_cols
            rows_needed = max(1, int(np.ceil(show_slots / float(cols))))
            thumb = best_thumb
            start_y = UI_TOP_MARGIN
            gallery_canvas = np.zeros((total_h, gallery_w, 3), dtype=np.uint8)
            for i in range(show_slots):
                col = i % cols
                row = i // cols
                x0 = col * (thumb + gap)
                y0 = start_y + row * (thumb + gap)
                slot = gallery_slots[i] if i < len(gallery_slots) else None
                slot_img = np.zeros((thumb, thumb, 3), dtype=np.uint8)
                if slot is not None:
                    slot_bgr = cv2.cvtColor(slot, cv2.COLOR_GRAY2BGR)
                    inner_max = max(1, thumb - inner_margin * 2)
                    scale = gallery_scales[i] if i < len(gallery_scales) else 0.0
                    scale = min(1.0, max(min_scale, scale))
                    target_h = max(1, int(inner_max * scale))
                    target_w = target_h
                    fitted = _fit_to_box_center_mask(slot_bgr, target_w, target_h)
                    x_off = (thumb - target_w) // 2
                    y_off = (thumb - target_h) // 2
                    slot_img[y_off : y_off + target_h, x_off : x_off + target_w] = fitted
                gallery_canvas[y0 : y0 + thumb, x0 : x0 + thumb] = slot_img
                cv2.rectangle(
                    gallery_canvas,
                    (x0, y0),
                    (x0 + thumb - 1, y0 + thumb - 1),
                    (60, 60, 60),
                    1,
                )
            canvas[:total_h, gallery_x : gallery_x + gallery_w] = gallery_canvas
        if ndi_gallery is not None and gallery_canvas is not None:
            ndi_gallery.publish(gallery_canvas)
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
                "hold": ui.hold,
                "rise": ui.rise,
                "fall": ui.fall,
                "size_smooth": ui.size_smooth,
                "area_min": ui.area_min,
                "aspect_min": ui.aspect_min,
                "aspect_max": ui.aspect_max,
                "solidity_min": ui.solidity_min,
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
            "hold": ui.hold,
            "rise": ui.rise,
            "fall": ui.fall,
            "size_smooth": ui.size_smooth,
            "area_min": ui.area_min,
            "aspect_min": ui.aspect_min,
            "aspect_max": ui.aspect_max,
            "solidity_min": ui.solidity_min,
        }
        _save_settings(SETTINGS_PATH, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
