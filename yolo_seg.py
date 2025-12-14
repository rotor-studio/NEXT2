"""
Real-time human segmentation using YOLOv8 (segment task) with OpenCV.
Pipeline: captura -> segmentación -> máscara -> composite.
Ventana única 1280x720: vista original anotada (FPS + boxes) y máscara binaria.
"""

from __future__ import annotations

import sys
import time
from typing import Tuple
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Configuración de rendimiento
RES_OPTIONS = [160, 240, 320, 360, 480]  # opciones de altura máxima
CURRENT_MAX_HEIGHT = 240  # valor inicial (se sobreescribe si hay guardado)
IMG_SIZE = 320    # resolución de inferencia YOLO
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
ENABLE_NDI = True     # Envía la máscara por NDI
DATA_DIR = Path(__file__).with_name("DATA")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
VIDEO_FILES = sorted([p for p in DATA_DIR.glob("*") if p.suffix.lower() in VIDEO_EXTS])
CURRENT_SOURCE = "camera"  # "camera" o "video"
CURRENT_VIDEO_INDEX = 0
BLUR_KERNEL_OPTIONS = [1, 3, 5, 7, 9, 11, 13]
BLUR_KERNEL_IDX = 2  # valor inicial -> kernel 5
MASK_THRESH = 127
BLUR_ENABLED = True

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
    return resize_keep_aspect(frame, max_height=CURRENT_MAX_HEIGHT)


# --------------- segmentación --------------------------------------
def segment_people(frame: np.ndarray, model: YOLO, people_limit: int, mask_thresh: int, blur_enabled: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run YOLOv8 segmentation and return:
    - mask: global person mask (0-255, uint8) resized to frame size.
    - boxes: ndarray of person boxes (N, 4) in xyxy format.
    - Combines all person masks with bitwise OR.
    - Returns all-black mask if no persons are found.
    """
    h, w = frame.shape[:2]
    result = model(frame, imgsz=IMG_SIZE, verbose=False, device=DEVICE)[0]

    # If the model returns no masks, bail early.
    if result.masks is None or result.boxes is None:
        return np.zeros((h, w), dtype=np.uint8), np.empty((0, 4))

    masks = result.masks.data.cpu().numpy()  # shape: (N, H, W) in model space
    classes = result.boxes.cls.cpu().numpy().astype(int)  # shape: (N,)
    boxes_all = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    # Filtrar solo personas y aplicar límite ordenando por confianza.
    person_indices = [i for i, cls_id in enumerate(classes) if cls_id == 0]
    if not person_indices:
        return np.zeros((h, w), dtype=np.uint8), np.empty((0, 4))
    person_indices = sorted(person_indices, key=lambda i: confs[i], reverse=True)[:people_limit]

    person_mask = np.zeros((h, w), dtype=np.uint8)
    person_boxes = []
    for idx in person_indices:
        m = masks[idx]
        # Resize mask from model space to frame space and scale to 0-255 for flexible thresholding.
        m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        m_uint8 = np.clip(m_resized * 255.0, 0, 255).astype(np.uint8)
        person_mask = cv2.bitwise_or(person_mask, m_uint8)
        person_boxes.append(boxes_all[idx])

    # Suavizado configurable.
    ksize = BLUR_KERNEL_OPTIONS[BLUR_KERNEL_IDX]
    if blur_enabled and ksize > 1:
        person_mask = cv2.GaussianBlur(person_mask, (ksize, ksize), 0)
    _, person_mask = cv2.threshold(person_mask, mask_thresh, 255, cv2.THRESH_BINARY)
    return person_mask, np.array(person_boxes)


# --------------- composición y overlays ----------------------------
def make_composite(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return black background with white silhouettes."""
    composite = np.zeros_like(frame)
    composite[mask > 0] = (255, 255, 255)
    return composite


def fit_to_box(image: np.ndarray, box_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to fit inside box_size keeping aspect ratio; letterbox on black (top-aligned)."""
    box_w, box_h = box_size
    h, w = image.shape[:2]
    scale = min(box_w / w, box_h / h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    y_off = 0  # align to top
    x_off = (box_w - new_size[0]) // 2
    canvas[y_off : y_off + new_size[1], x_off : x_off + new_size[0]] = resized
    return canvas


def add_overlay(image: np.ndarray, label: str) -> np.ndarray:
    """Add label overlay to an image (top-left) with small white text."""
    overlay = image.copy()
    cv2.putText(overlay, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return overlay


def draw_boxes(frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Draw bounding boxes on the frame."""
    out = frame.copy()
    if boxes is None:
        return out
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return out


def add_header(canvas: np.ndarray, fps: float, device: str, res_text: str, people_count: int, source_label: str) -> None:
    """Draw a single header line with source, resolution, FPS, device, model and people count at the top of the canvas."""
    current_model_label = MODEL_OPTIONS.get(CURRENT_MODEL_KEY, (CURRENT_MODEL_PATH, CURRENT_MODEL_PATH))[1]
    text = (
        f"SRC: {source_label} | RES: {res_text} | FPS: {fps:.1f} | GPU: {device} | "
        f"MODEL: {current_model_label} | PEOPLE NOW: {people_count}"
    )
    cv2.putText(canvas, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def add_footer(canvas: np.ndarray, current_res: int) -> None:
    """Draw footer with resolution selector hint."""
    footer_y1 = canvas.shape[0] - 190
    footer_y2 = canvas.shape[0] - 160
    footer_y3 = canvas.shape[0] - 130
    footer_y4 = canvas.shape[0] - 100
    footer_y5 = canvas.shape[0] - 70
    footer_y6 = canvas.shape[0] - 40
    res_opts = " | ".join(f"{i+1}:{r}" for i, r in enumerate(RES_OPTIONS))
    model_opts = " | ".join(f"{chr(k)}:{v[0]}" for k, v in MODEL_OPTIONS.items())
    cv2.putText(canvas, f"RES -> {res_opts} ", (10, footer_y1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    current_model_label = MODEL_OPTIONS.get(CURRENT_MODEL_KEY, ("", ""))[1]
    cv2.putText(canvas, f"MODEL -> {model_opts} ", (10, footer_y2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(canvas, f"PEOPLE -> +/- (limit={CURRENT_PEOPLE_LIMIT})", (10, footer_y3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    source_hint = "SRC -> c:camara" + (" | v:video" if VIDEO_FILES else "")
    cv2.putText(canvas, source_hint, (10, footer_y4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    blur_hint = f"BLUR -> o / p (ksize={BLUR_KERNEL_OPTIONS[BLUR_KERNEL_IDX]}) | b: {'ON' if BLUR_ENABLED else 'OFF'}"
    cv2.putText(canvas, blur_hint, (10, footer_y5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    thresh_hint = f"THRESH -> j/k (val={MASK_THRESH})"
    cv2.putText(canvas, thresh_hint, (10, footer_y6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


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


def open_capture(source: str):
    """Open VideoCapture for camera or current video."""
    global CURRENT_VIDEO_INDEX
    if source == "camera":
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
        return cap
    if not VIDEO_FILES:
        print("No hay videos en DATA/", file=sys.stderr)
        return None
    CURRENT_VIDEO_INDEX %= len(VIDEO_FILES)
    path = VIDEO_FILES[CURRENT_VIDEO_INDEX]
    cap = cv2.VideoCapture(str(path))
    return cap


def source_label():
    if CURRENT_SOURCE == "camera":
        return "Camera"
    if VIDEO_FILES:
        return f"Video: {VIDEO_FILES[CURRENT_VIDEO_INDEX].name}"
    return "Video"


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
        if not ENABLE_NDI:
            return
        try:
            import NDIlib as ndi  # type: ignore

            if not ndi.initialize():
                print("[NDI] No se pudo inicializar NDIlib.", file=sys.stderr)
                return
            send_settings = ndi.SendCreate()
            send_settings.ndi_name = name
            self.sender = ndi.send_create(send_settings)
            self.ndi = ndi
        except Exception as exc:
            print(f"[NDI] No disponible: {exc}", file=sys.stderr)
            self.sender = None
            self.ndi = None

    def publish(self, frame: np.ndarray):
        if self.sender is None or self.ndi is None:
            return
        try:
            ndi = self.ndi
            # Convertimos a BGRA para NDI.
            if frame.ndim == 2:
                frame_bgra = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGRA)
            else:
                frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

            h, w = frame_bgra.shape[:2]
            video_frame = ndi.VideoFrameV2()
            video_frame.xres = w
            video_frame.yres = h
            video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRA
            video_frame.frame_rate_N = 60
            video_frame.frame_rate_D = 1
            video_frame.line_stride_in_bytes = frame_bgra.strides[0]
            video_frame.data = frame_bgra

            ndi.send_send_video_v2(self.sender, video_frame)
        except Exception as exc:
            print(f"[NDI] Error al publicar: {exc}", file=sys.stderr)


# --------------- loop principal ------------------------------------
def main():
    # Carga modelo YOLOv8 de segmentación (usa uno ligero por defecto).
    model_path = "yolov8n-seg.pt"
    global DEVICE, CURRENT_MODEL_PATH, CURRENT_MODEL_KEY, CURRENT_PEOPLE_LIMIT, CURRENT_SOURCE, BLUR_KERNEL_IDX, MASK_THRESH, BLUR_ENABLED
    load_saved_resolution()
    load_saved_model()
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    try:
        model = load_model(CURRENT_MODEL_PATH)
    except Exception as exc:
        print(f"No se pudo cargar el modelo {CURRENT_MODEL_PATH}: {exc}", file=sys.stderr)
        return 1

    cap = open_capture(CURRENT_SOURCE)
    if cap is None or not cap.isOpened():
        print("No se pudo abrir la fuente de video.", file=sys.stderr)
        return 1

    prev_time = time.time()
    window_name = "NEXT2 VISION - ROTOR STUDIO"
    canvas_size = (1280, 720)  # width, height
    header_h = 40
    footer_h = 180
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(window_name, canvas_size[0], canvas_size[1])
    frame_idx = 0
    last_mask = None
    last_boxes = None
    ndi_pub = NDIPublisher("NEXT2 Mask NDI") if ENABLE_NDI else None

    while True:
        frame = capture_frame(cap)
        if frame is None:
            if CURRENT_SOURCE == "video" and cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame = capture_frame(cap)
            if frame is None:
                print("Frame no capturado. Saliendo.", file=sys.stderr)
                break

        do_process = frame_idx % PROCESS_EVERY_N == 0 or last_mask is None
        if do_process:
            mask, boxes = segment_people(frame, model, CURRENT_PEOPLE_LIMIT, MASK_THRESH, BLUR_ENABLED)
            last_mask, last_boxes = mask, boxes
        else:
            mask, boxes = last_mask, last_boxes
        frame_idx += 1

        boxed = draw_boxes(frame, boxes)

        # FPS.
        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 1.0 / dt if dt > 0 else 0.0
        print(f"FPS: {fps:0.2f}", end="\r", flush=True)

        # Composición en ventana única: dos vistas lado a lado ocupando todo el ancho.
        canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
        view_h = canvas_size[1] - header_h - footer_h
        half_w = canvas_size[0] // 2

        left_view = fit_to_box(boxed, (half_w, view_h))
        right_view = fit_to_box(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (half_w, view_h))

        left_view = add_overlay(left_view, "Original + Boxes")
        right_view = add_overlay(right_view, "Mask")

        canvas[header_h : header_h + left_view.shape[0], 0:half_w] = left_view
        canvas[header_h : header_h + right_view.shape[0], half_w : half_w + right_view.shape[1]] = right_view

        res_text = f"{frame.shape[1]}x{frame.shape[0]}"
        people_count = len(boxes) if boxes is not None else 0
        add_header(canvas, fps, DEVICE, res_text, people_count, source_label())
        add_footer(canvas, CURRENT_MAX_HEIGHT)

        # Publicación NDI (máscara)
        if ndi_pub is not None:
            ndi_pub.publish(mask)

        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        # Resolución por teclas 1-5
        if key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
            set_resolution_by_index(int(chr(key)) - 1)
        # Cambio de modelo por teclas configuradas
        if key in MODEL_OPTIONS:
            try:
                new_model = load_model(MODEL_OPTIONS[key][0])
                CURRENT_MODEL_KEY = key
                CURRENT_MODEL_PATH = MODEL_OPTIONS[key][0]
                save_current_model()
                model = new_model
            except Exception as exc:
                print(f"No se pudo cargar el modelo {MODEL_OPTIONS[key][0]}: {exc}", file=sys.stderr)
        # Ajuste de límite de personas con +/- (teclado principal)
        if key == ord("+") or key == ord("="):
            CURRENT_PEOPLE_LIMIT = min(CURRENT_PEOPLE_LIMIT + 1, PEOPLE_LIMIT_OPTIONS[-1])
        if key == ord("-"):
            CURRENT_PEOPLE_LIMIT = max(CURRENT_PEOPLE_LIMIT - 1, PEOPLE_LIMIT_OPTIONS[0])
        # Ajuste de blur (detallado de silueta)
        if key == ord("o"):
            BLUR_KERNEL_IDX = max(0, BLUR_KERNEL_IDX - 1)
        if key == ord("p"):
            BLUR_KERNEL_IDX = min(len(BLUR_KERNEL_OPTIONS) - 1, BLUR_KERNEL_IDX + 1)
        if key == ord("b"):
            BLUR_ENABLED = not BLUR_ENABLED
        # Ajuste de threshold de binarización
        if key == ord("j"):
            MASK_THRESH = max(0, MASK_THRESH - 5)
        if key == ord("k"):
            MASK_THRESH = min(255, MASK_THRESH + 5)
        # Cambio de fuente
        if key == ord("c"):
            CURRENT_SOURCE = "camera"
            if cap:
                cap.release()
            cap = open_capture(CURRENT_SOURCE)
        if key == ord("v") and VIDEO_FILES:
            CURRENT_SOURCE = "video"
            CURRENT_VIDEO_INDEX = 0
            if cap:
                cap.release()
            cap = open_capture(CURRENT_SOURCE)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
