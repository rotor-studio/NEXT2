"""
Quick NDI receive tester.
- Lists sources and connects to the first one (or one matching NDI_SOURCE env var).
- Resizes to max_height for display.
- Shows FPS; exit with 'q'.

Run: conda run -n NEXT python ndi_receive.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

MAX_HEIGHT = 480


def resize_keep_aspect(frame, max_height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h <= max_height:
        return frame
    scale = max_height / float(h)
    new_size: Tuple[int, int] = (int(w * scale), int(h * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def pick_source(ndi):
    finder = ndi.find_create_v2()
    ndi.find_wait_for_sources(finder, 2000)
    sources = ndi.find_get_current_sources(finder)
    if not sources:
        print("No se encontraron fuentes NDI.")
        ndi.find_destroy(finder)
        return None
    env_name = os.environ.get("NDI_SOURCE")
    if env_name:
        for s in sources:
            if env_name in s.ndi_name:
                ndi.find_destroy(finder)
                return s
        print(f"NDI_SOURCE '{env_name}' no encontrada, usando la primera.")
    print("Fuentes NDI:")
    for i, s in enumerate(sources):
        print(f"[{i}] {s.ndi_name}")
    src = sources[0]
    print(f"Conectando a: {src.ndi_name}")
    ndi.find_destroy(finder)
    return src


def main():
    log_path = "/tmp/ndi_receive.log"
    def log(msg: str):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        print(msg, flush=True)

    log("[NDI] Arrancando test...")
    try:
        import cyndilib as ndi  # type: ignore
        import importlib.resources as ir
        try:
            bin_path = ir.files("cyndilib.wrapper").joinpath("bin")
            bin_str = str(bin_path)
            os.environ.setdefault("NDI_RUNTIME_DIR", bin_str)
            os.environ["DYLD_LIBRARY_PATH"] = f"{bin_str}:{os.environ.get('DYLD_LIBRARY_PATH','')}"
        except Exception:
            pass
    except Exception as exc:
        log(f"No se pudo importar cyndilib: {exc}")
        return 1

    log("[NDI] Iniciando test de recepción...")
    finder = ndi.Finder()
    finder.open()
    finder.wait_for_sources(2.0)
    sources = list(finder.iter_sources())
    finder.close()
    if not sources:
        log("No se encontraron fuentes NDI.")
        return 1
    log(f"[NDI] Fuentes encontradas: {[getattr(s,'name','') for s in sources]}")
    env_name = os.environ.get("NDI_SOURCE", "").lower()
    selected = None
    if env_name:
        for s in sources:
            name = getattr(s, "name", "") or ""
            if env_name in name.lower():
                selected = s
                break
    if selected is None:
        selected = sources[0]
    log(f"Conectando a: {getattr(selected, 'name', 'NDI')} (valid={getattr(selected,'valid',None)})")
    try:
        selected.update()
    except Exception:
        pass

    recv = ndi.Receiver(
        source_name=getattr(selected, "name", ""),
        color_format=ndi.RecvColorFormat.BGRX_BGRA,
        bandwidth=ndi.RecvBandwidth.highest,
        allow_video_fields=False,
        recv_name="ndi_receive_test",
    )
    fs = recv.frame_sync
    fs.set_video_frame(ndi.VideoFrameSync())
    # Forzamos connect usando Source para asegurar dirección completa.
    try:
        recv.connect_to(selected)
    except Exception:
        pass
    try:
        recv._wait_for_connect(2.0)
    except Exception:
        pass
    log(f"[NDI] Conectado a {getattr(selected, 'name', 'NDI')} ({getattr(selected, 'stream_name', '')})")

    prev = time.time()
    fps = 0.0
    last_log = time.time()

    while True:
        fs.capture_video()
        vf = fs.video_frame
        w, h = vf.get_resolution()
        stride = vf.get_line_stride()
        mv = memoryview(vf)
        flat = np.frombuffer(mv, dtype=np.uint8).copy()
        mv.release()
        now = time.time()
        if now - last_log > 2.0:
            log(f"[NDI] loop connected={recv.is_connected()} num_conn={recv.get_num_connections()} size={flat.size} w={w} h={h} stride={stride}")
            last_log = now
        if flat.size >= stride * h and w > 0 and h > 0:
            frame_bgra = flat[: stride * h].reshape((h, stride))[:, : w * 4].reshape((h, w, 4))
            frame = frame_bgra[:, :, :3].copy()
            frame = resize_keep_aspect(frame, MAX_HEIGHT)
            now = time.time()
            dt = now - prev
            prev = now
            fps = 1.0 / dt if dt > 0 else fps
            cv2.putText(frame, f"FPS {fps:05.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("NDI Receive Test", frame)
            continue

        # No frame; show black placeholder
        placeholder = np.zeros((MAX_HEIGHT, int(MAX_HEIGHT * 16 / 9), 3), dtype=np.uint8)
        cv2.putText(placeholder, "Esperando NDI...", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("NDI Receive Test", placeholder)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    recv.disconnect()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
