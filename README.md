# NEXT2 Vision

## Descripción
Pipeline de segmentación de personas en tiempo real con YOLOv8 (segmentación) y OpenCV. La app muestra en una sola ventana (1280x720) la vista original anotada (FPS + boxes) y la máscara binaria. Soporta entrada desde cámara, videos locales (`DATA/`) y NDI (p. ej. MadMapper). Publica la máscara por NDI como “NEXT2 Mask NDI”.

## Requisitos
- Python 3.10+
- Paquetes: `ultralytics`, `opencv-python`, `torch` (MPS/CPU/CUDA), `cyndilib` (runtime NDI incluido)
- Modelos YOLOv8 segment: `yolov8n-seg.pt`, `yolov8s-seg.pt`, `yolov8m-seg.pt` en el directorio raíz

## Ejecución
```bash
conda run -n NEXT python yolo_seg.py
```
Variables útiles:
- `NEXT_DEVICE`: `mps`/`cuda`/`cpu`/`auto` (por defecto auto: mps→cuda→cpu)
- `NEXT_SOURCE`: `camera`/`video`/`ndi` (por defecto camera)
- `NEXT_ENABLE_NDI_OUTPUT`: `1`/`0` (por defecto 1)

## Controles en la ventana
- Salir: `q`
- Resolución (altura): `1`=160, `2`=240, `3`=320, `4`=360, `5`=480
- Modelos: `a` (yolov8n), `s` (yolov8s), `d` (yolov8m)
- Fuente: `c`=cámara, `v`=video (cicla `DATA/`), `n`=NDI
- Límite de personas: `+` / `-`
- Blur máscara: `o` / `p` (kernel), `b` ON/OFF
- Umbral máscara: `j` / `k`
- img size YOLO: `,` / `.`
- Modo alta precisión: `h`
- Vista máscara: `m` (soft/detail)

## Persistencia
- Resolución: `resolution.txt`
- Modelo: `model.txt`
- Parámetros en runtime: `settings.json` (límite personas, blur, threshold, imgsz, fuente, vista máscara, modo alta precisión, etc.)

## Notas NDI
- Entrada y salida usan `cyndilib` (incluye `libndi.dylib` en `cyndilib/wrapper/bin`).
- Entrada NDI: tecla `n` (se conecta a MadMapper si está disponible).
- Salida NDI: fuente “NEXT2 Mask NDI” (BGRA) siempre activa cuando hay frames.
