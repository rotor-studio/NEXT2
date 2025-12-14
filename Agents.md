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

Persistence
- Resolution persists in `resolution.txt`
- Model choice persists in `model.txt`

Running
- Conda env: `conda run -n NEXT python yolo_seg.py`
- Requires: `ultralytics`, `opencv-python`, `torch` (MPS/CPU auto-selected; CUDA if available)

Performance tips
- Lower resolution (1–3) for higher FPS.
- Prefer `yolov8n-seg.pt` for speed; `s`/`m` for more robustness.
- Frame skipping is enabled (processes 1 of every 2 frames). Adjust `PROCESS_EVERY_N` in code if needed.
