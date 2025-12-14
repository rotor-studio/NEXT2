# NEXT2 Vision – Operator Notes

Pipeline overview
- Single-window UI (1280x720) with two views at the top: left = camera + boxes, right = binary mask.
- Header shows resolution, FPS, device (GPU/MPS/CPU), model, and live people count.
- Footer shows controls to switch resolution, model, and people limit.

Controls
- Quit: `q`
- Resolution (height): `1`=160, `2`=240, `3`=320, `4`=360, `5`=480
- Model: `a`=yolov8n-seg (fast), `s`=yolov8s-seg (balanced), `d`=yolov8m-seg (heavier)
- People limit (in-memory only): `+` / `-`
- Source: `c`=camera, `v`=video (cycles through DATA/ videos, starts at first)
- Mask blur: `o` / `p` to shrink/grow kernel, `b` toggle blur on/off
- Mask threshold: `j` / `k` to lower/raise binarization threshold
- Mask view: `m` toggles between soft/detail
- Precision: `h` toggles high precision (uses max imgsz)
- img size for YOLO: `,` / `.` cycle imgsz (320/480/640)

Persistence
- Resolution persists in `resolution.txt`
- Model choice persists in `model.txt`

Running
- Conda env: `conda run -n NEXT python yolo_seg.py`
- Requires: `ultralytics`, `opencv-python`, `torch` (MPS/CPU auto-selected; CUDA if available)
- NDI: publica la máscara como fuente “NEXT2 Mask NDI” (BGRA) si `ndi-python` está instalado.

Performance tips
- Lower resolution (1–3) for higher FPS.
- Prefer `yolov8n-seg.pt` for speed; `s`/`m` for more robustness.
- Frame skipping is enabled (processes 1 of every 2 frames). Adjust `PROCESS_EVERY_N` in code if needed.
