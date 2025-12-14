"""
Real-time human segmentation using YOLOv8 (segment task) with OpenCV.
Pipeline: captura -> segmentación -> máscara -> composite.
Ventana única 1280x720: vista original anotada (FPS + boxes) y máscara binaria.
"""

from __future__ import annotations

import sys
import time
from typing import Tuple, Any
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
ENABLE_NDI_OUTPUT = env_flag("NEXT_ENABLE_NDI_OUTPUT", True)  # Publicar máscara por NDI
DATA_DIR = Path(__file__).with_name("DATA")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
VIDEO_FILES = sorted([p for p in DATA_DIR.glob("*") if p.suffix.lower() in VIDEO_EXTS])
DEFAULT_SOURCE = "camera"  # se ajusta en runtime tras probar NDI
CURRENT_SOURCE = DEFAULT_SOURCE  # "camera", "video" o "ndi"
CURRENT_VIDEO_INDEX = 0
NDI_PREFERRED_SOURCE = "MadMapper"
BLUR_KERNEL_OPTIONS = [1, 3, 5, 7, 9, 11, 13]
BLUR_KERNEL_IDX = 2  # valor inicial -> kernel 5
MASK_THRESH = 127
BLUR_ENABLED = True
HIGH_PRECISION_MODE = False  # modo alta precisión (imgsz alto y sin blur en mask_detail)
NDI_OUTPUT_MASK = "soft"  # "soft" o "detail"
SHOW_DETAIL_DEFAULT = False
SHOW_DETAIL = SHOW_DETAIL_DEFAULT

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
    if isinstance(cap, NDIReceiver):
        frame = cap.capture()
        if frame is None:
            return None
        # Para NDI, limitamos a la resolución de entrada seleccionada (altura) y al imgsz actual.
        target_h = min(CURRENT_MAX_HEIGHT, IMG_SIZE_OPTIONS[IMG_SIZE_IDX])
        if HIGH_PRECISION_MODE:
            target_h = max(target_h, IMG_SIZE_OPTIONS[-1])
        return resize_keep_aspect(frame, max_height=target_h)
    ok, frame = cap.read()
    if not ok:
        return None
    return resize_keep_aspect(frame, max_height=CURRENT_MAX_HEIGHT)


# --------------- segmentación --------------------------------------
def segment_people(frame: np.ndarray, model: YOLO, people_limit: int, mask_thresh: int, blur_enabled: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return empty, empty, np.empty((0, 4))

    masks = result.masks.data.cpu().numpy()  # shape: (N, H, W) in model space
    classes = result.boxes.cls.cpu().numpy().astype(int)  # shape: (N,)
    boxes_all = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    # Filtrar solo personas y aplicar límite ordenando por confianza.
    person_indices = [i for i, cls_id in enumerate(classes) if cls_id == 0]
    if not person_indices:
        empty = np.zeros((h, w), dtype=np.uint8)
        return empty, empty, np.empty((0, 4))
    person_indices = sorted(person_indices, key=lambda i: confs[i], reverse=True)[:people_limit]

    # Aggregate soft mask (float) to keep fine contours before thresholding.
    person_mask = np.zeros((h, w), dtype=np.float32)
    person_boxes = []
    for idx in person_indices:
        m = masks[idx]
        # Resize mask from model space to frame space; keep float precision for later threshold.
        m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        person_mask = np.maximum(person_mask, m_resized)
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

    return mask_soft, mask_detail, np.array(person_boxes)


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
    scale = min(box_w / w, box_h / h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    if new_size[0] == box_w:
        # Height is already within box_h; crop if needed to avoid left padding.
        if new_size[1] > box_h:
            resized = resized[0:box_h, 0:box_w]
        canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
        canvas[0 : resized.shape[0], 0 : resized.shape[1]] = resized
        return canvas

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
    fps_str = f"{fps:06.1f}"  # ancho fijo para evitar saltos en el texto
    text = (
        f"SRC: {source_label} | RES: {res_text} | FPS: {fps_str} | GPU: {device} | "
        f"MODEL: {current_model_label} | PEOPLE NOW: {people_count} | PREC: {'HIGH' if HIGH_PRECISION_MODE else 'NORM'}"
    )
    cv2.putText(canvas, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def add_footer(canvas: np.ndarray, current_res: int) -> None:
    """Draw footer with resolution selector hint."""
    footer_y1 = canvas.shape[0] - 240
    footer_y2 = canvas.shape[0] - 210
    footer_y3 = canvas.shape[0] - 180
    footer_y4 = canvas.shape[0] - 150
    footer_y5 = canvas.shape[0] - 120
    footer_y6 = canvas.shape[0] - 90
    footer_y7 = canvas.shape[0] - 60
    footer_y8 = canvas.shape[0] - 30
    res_opts = " | ".join(f"{i+1}:{r}" for i, r in enumerate(RES_OPTIONS))
    model_opts = " | ".join(f"{chr(k)}:{v[0]}" for k, v in MODEL_OPTIONS.items())
    cv2.putText(canvas, f"RES -> {res_opts} ", (10, footer_y1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    current_model_label = MODEL_OPTIONS.get(CURRENT_MODEL_KEY, ("", ""))[1]
    cv2.putText(canvas, f"MODEL -> {model_opts} ", (10, footer_y2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(canvas, f"PEOPLE -> +/- (limit={CURRENT_PEOPLE_LIMIT})", (10, footer_y3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    src_opts = ["c:camara"]
    if VIDEO_FILES:
        src_opts.append("v:video")
    if ENABLE_NDI:
        src_opts.append("n:ndi")
    source_hint = "SRC -> " + " | ".join(src_opts)
    cv2.putText(canvas, source_hint, (10, footer_y4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    blur_hint = f"BLUR -> o / p (ksize={BLUR_KERNEL_OPTIONS[BLUR_KERNEL_IDX]}) | b: {'ON' if BLUR_ENABLED else 'OFF'}"
    cv2.putText(canvas, blur_hint, (10, footer_y5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    thresh_hint = f"THRESH -> j/k (val={MASK_THRESH})"
    cv2.putText(canvas, thresh_hint, (10, footer_y6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    imgsz_hint = f"IMG_SZ -> , / . (imgsz={IMG_SIZE_OPTIONS[IMG_SIZE_IDX]})"
    cv2.putText(canvas, imgsz_hint, (10, footer_y7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    hi_hint = f"HIGH PREC -> h ({'ON' if HIGH_PRECISION_MODE else 'OFF'}) | MASK VIEW -> m (soft/detail)"
    cv2.putText(canvas, hi_hint, (10, footer_y8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


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
    if CURRENT_SOURCE == "video" and VIDEO_FILES:
        return f"Video: {VIDEO_FILES[CURRENT_VIDEO_INDEX].name}"
    return CURRENT_SOURCE


def capture_ready(cap):
    if cap is None:
        return False
    if isinstance(cap, NDIReceiver):
        return cap.ready
    return cap.isOpened()


def release_capture(cap):
    if cap is None:
        return
    if hasattr(cap, "release"):
        try:
            cap.release()
        except Exception:
            pass


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
        import cyndilib as ndi  # type: ignore
        finder = ndi.Finder()
        finder.open()
        finder.wait_for_sources(0.5)
        result_queue.put(finder.num_sources > 0)
        finder.close()
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
    global IMG_SIZE_IDX, HIGH_PRECISION_MODE, NDI_OUTPUT_MASK, CURRENT_SOURCE, SHOW_DETAIL
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
        src = str(data.get("source", CURRENT_SOURCE)).lower()
        if src in {"camera", "video", "ndi"}:
            CURRENT_SOURCE = src
    except Exception:
        pass
    try:
        SHOW_DETAIL = bool(data.get("show_detail", SHOW_DETAIL_DEFAULT))
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
        "source": CURRENT_SOURCE,
        "show_detail": SHOW_DETAIL,
    }
    if extra:
        base.update(extra)
    try:
        import json

        SETTINGS_FILE.write_text(json.dumps(base, indent=2), encoding="utf-8")
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
    global BLUR_KERNEL_IDX, MASK_THRESH, BLUR_ENABLED, IMG_SIZE_IDX, HIGH_PRECISION_MODE, ENABLE_NDI, DEFAULT_SOURCE, SHOW_DETAIL
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
            print("[NDI] No se pudo inicializar NDIlib (se desactiva).", file=sys.stderr)
            ENABLE_NDI = False

    # Decide fuente por env o por disponibilidad (por defecto cámara; NDI se selecciona a mano).
    if env_source in {"ndi", "camera", "video"}:
        CURRENT_SOURCE = env_source
    else:
        CURRENT_SOURCE = "camera"

    if CURRENT_SOURCE == "ndi" and not ENABLE_NDI:
        print("[NDI] Fuente NDI solicitada pero NDI no está disponible, usando cámara.", file=sys.stderr)
        CURRENT_SOURCE = "camera"

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
    canvas_size = (1280, 720)  # width, height
    header_h = 40
    footer_h = 260
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(window_name, canvas_size[0], canvas_size[1])
    frame_idx = 0
    last_mask_soft = None
    last_mask_detail = None
    last_boxes = None
    last_frame = None
    show_detail = SHOW_DETAIL
    ndi_pub = NDIPublisher("NEXT2 Mask NDI") if ENABLE_NDI and ENABLE_NDI_OUTPUT else None

    while True:
        frame = capture_frame(cap)
        got_new_frame = frame is not None
        if frame is None:
            if CURRENT_SOURCE == "video" and cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame = capture_frame(cap)
                got_new_frame = frame is not None
            elif CURRENT_SOURCE == "ndi":
                # Mantener último frame o negro mientras esperamos señal NDI.
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

        do_process = (frame_idx % PROCESS_EVERY_N == 0 or last_mask_soft is None) and got_new_frame
        if do_process:
            mask_soft, mask_detail, boxes = segment_people(
                frame, model, CURRENT_PEOPLE_LIMIT, MASK_THRESH, BLUR_ENABLED
            )
            last_mask_soft, last_mask_detail, last_boxes = mask_soft, mask_detail, boxes
            last_frame = frame
        else:
            mask_soft, mask_detail, boxes = last_mask_soft, last_mask_detail, last_boxes
        frame_idx += 1

        # Fallback en caso de no tener máscaras aún (evita crash en cvtColor).
        if mask_soft is None or mask_detail is None:
            h, w = frame.shape[:2]
            fallback_mask = np.zeros((h, w), dtype=np.uint8)
            if mask_soft is None:
                mask_soft = fallback_mask
            if mask_detail is None:
                mask_detail = fallback_mask

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
        mask_to_show = mask_detail if show_detail else mask_soft
        right_view = fit_to_box(cv2.cvtColor(mask_to_show, cv2.COLOR_GRAY2BGR), (half_w, view_h))

        left_view = add_overlay(left_view, "Original + Boxes")
        right_view = add_overlay(right_view, "Mask Detail" if show_detail else "Mask Soft")

        canvas[header_h : header_h + left_view.shape[0], 0:half_w] = left_view
        canvas[header_h : header_h + right_view.shape[0], half_w : half_w + right_view.shape[1]] = right_view

        res_text = f"{frame.shape[1]}x{frame.shape[0]}"
        people_count = len(boxes) if boxes is not None else 0
        add_header(canvas, fps, DEVICE, res_text, people_count, source_label())
        add_footer(canvas, CURRENT_MAX_HEIGHT)

        # Publicación NDI (máscara) - por defecto la suave; cambiar a mask_detail si se desea.
        if ndi_pub is not None:
            mask_out = mask_soft if NDI_OUTPUT_MASK == "soft" else mask_detail
            ndi_pub.publish(mask_out)

        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        # Resolución por teclas 1-5
        if key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
            set_resolution_by_index(int(chr(key)) - 1)
            save_settings()
        # Cambio de modelo por teclas configuradas
        if key in MODEL_OPTIONS:
            try:
                new_model = load_model(MODEL_OPTIONS[key][0])
                CURRENT_MODEL_KEY = key
                CURRENT_MODEL_PATH = MODEL_OPTIONS[key][0]
                save_current_model()
                save_settings()
                model = new_model
            except Exception as exc:
                print(f"No se pudo cargar el modelo {MODEL_OPTIONS[key][0]}: {exc}", file=sys.stderr)
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
            CURRENT_SOURCE = "camera"
            release_capture(cap)
            cap = open_capture(CURRENT_SOURCE)
            save_settings()
        if key == ord("v") and VIDEO_FILES:
            CURRENT_SOURCE = "video"
            CURRENT_VIDEO_INDEX = 0
            release_capture(cap)
            cap = open_capture(CURRENT_SOURCE)
            save_settings()
        if key == ord("n") and ENABLE_NDI:
            prev_cap, prev_src = cap, CURRENT_SOURCE
            CURRENT_SOURCE = "ndi"
            release_capture(cap)
            cap = open_capture(CURRENT_SOURCE)
            if not capture_ready(cap):
                print("[NDI] No se pudo abrir fuente NDI, se mantiene la anterior.", file=sys.stderr)
                CURRENT_SOURCE = prev_src
                cap = prev_cap
            save_settings()
        # Modo alta precisión (usa imgsz máximo y detalle sin blur)
        if key == ord("h"):
            HIGH_PRECISION_MODE = not HIGH_PRECISION_MODE
            save_settings()
        # Toggle de máscara mostrada (soft/detail)
        if key == ord("m"):
            show_detail = not show_detail
            SHOW_DETAIL = show_detail
            save_settings()

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
