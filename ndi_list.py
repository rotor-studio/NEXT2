"""
Utility to list available NDI sources using NDIlib (ndi-python).
Run: conda run -n NEXT python ndi_list.py
"""

from __future__ import annotations

import sys


def main():
    try:
        import NDIlib as ndi  # type: ignore
    except Exception as exc:
        print(f"No se pudo importar NDIlib (ndi-python): {exc}", file=sys.stderr)
        return 1

    if not ndi.initialize():
        print("No se pudo inicializar NDIlib.", file=sys.stderr)
        return 1

    finder = ndi.find_create_v2()
    ndi.find_wait_for_sources(finder, 2000)  # espera hasta 2s
    sources = ndi.find_get_current_sources(finder)
    if not sources:
        print("No se encontraron fuentes NDI.")
        ndi.find_destroy(finder)
        return 0

    print("Fuentes NDI encontradas:")
    for idx, src in enumerate(sources):
        print(f"[{idx}] {src.ndi_name}")

    ndi.find_destroy(finder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
