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
    try:
        import NDIlib as ndi  # type: ignore
    except Exception as exc:
        print(f"No se pudo importar NDIlib: {exc}", file=sys.stderr)
        return 1

    if not ndi.initialize():
        print("No se pudo inicializar NDIlib.", file=sys.stderr)
        return 1

    src = pick_source(ndi)
    if src is None:
        return 1

    recv_settings = ndi.RecvCreateV3()
    recv_settings.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
    recv_settings.allow_video_fields = False
    recv_settings.source_to_connect_to = src
    recv = ndi.recv_create_v3(recv_settings)
    if recv is None:
        print("No se pudo crear el receptor NDI.", file=sys.stderr)
        return 1

    prev = time.time()
    fps = 0.0

    while True:
        frame_type, video_frame, _, _ = ndi.recv_capture_v2(recv, 100)
        if frame_type == ndi.FRAME_TYPE_VIDEO:
            h, w = video_frame.yres, video_frame.xres
            data = np.frombuffer(video_frame.data, dtype=np.uint8)
            frame_bgra = data.reshape((h, video_frame.line_stride_in_bytes // 4, 4))
            frame = frame_bgra[:, :w, :3].copy()
            ndi.recv_free_video_v2(recv, video_frame)

            frame = resize_keep_aspect(frame, MAX_HEIGHT)
            now = time.time()
            dt = now - prev
            prev = now
            fps = 1.0 / dt if dt > 0 else fps
            cv2.putText(frame, f"FPS {fps:05.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("NDI Receive Test", frame)
        else:
            # No frame; show black placeholder
            placeholder = np.zeros((MAX_HEIGHT, int(MAX_HEIGHT * 16 / 9), 3), dtype=np.uint8)
            cv2.putText(placeholder, "Esperando NDI...", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("NDI Receive Test", placeholder)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    ndi.recv_destroy(recv)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
