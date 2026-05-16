#!/usr/bin/env python3
"""
Paso 1: Descripción de imágenes con MiniCPM-V + clasificación NSFW
- Arranca llama-server con MiniCPM automáticamente
- Describe cada imagen con detalle
- Clasifica NSFW con Falconsai/nsfw_image_detection
- Escribe ambos resultados en metadatos XMP con exiftool
- Mata el servidor al terminar para liberar VRAM
"""

import os
import re
import sys
import time
import base64
import signal
import subprocess
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm
from transformers import pipeline

# ── CONFIG ────────────────────────────────────────────────────────────────────
PICTURES_DIR = Path("/home/walid/Pictures")
LLAMA_BIN    = Path("/home/walid/llama.cpp/build/bin/llama-server")
MINICPM_GGUF = Path("/home/walid/.cache/huggingface/hub/models--openbmb--MiniCPM-V-2_6-gguf/snapshots/48fe6436abf57b3df6ec34f73cdc1fb4b740acb0/ggml-model-Q8_0.gguf")
MMPROJ_GGUF  = Path("/home/walid/.cache/huggingface/hub/models--openbmb--MiniCPM-V-2_6-gguf/snapshots/48fe6436abf57b3df6ec34f73cdc1fb4b740acb0/mmproj-model-f16.gguf")
SERVER_PORT  = 8082
SERVER_URL   = f"http://localhost:{SERVER_PORT}"
EXTENSIONS   = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif", ".bmp"}

DESCRIBE_PROMPT = (
    "Describe this image in detail in English. "
    "Include: what type of content it is (anime, manga, meme, artwork, screenshot, photo, etc.), "
    "what is shown, style, colors, any text visible, and mood. "
    "Be specific and concise. Max 3 sentences."
)
# ─────────────────────────────────────────────────────────────────────────────


def start_server() -> subprocess.Popen:
    print("▶ Arrancando MiniCPM-V en GPU...")
    proc = subprocess.Popen(
        [
            str(LLAMA_BIN),
            "-m", str(MINICPM_GGUF),
            "--mmproj", str(MMPROJ_GGUF),
            "--n-gpu-layers", "999",
            "--host", "0.0.0.0",
            "--port", str(SERVER_PORT),
            "--ctx-size", "4096",
            "--batch-size", "512",
            "--flash-attn",
            "--log-disable",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Espera a que el servidor esté listo
    for _ in range(60):
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=2)
            if r.status_code == 200:
                print("✓ Servidor listo\n")
                return proc
        except Exception:
            pass
        time.sleep(2)
    proc.kill()
    raise RuntimeError("El servidor no arrancó en 120s")


def stop_server(proc: subprocess.Popen):
    print("\n■ Parando servidor y liberando VRAM...")
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("✓ VRAM liberada")


def image_to_base64_jpeg(path: Path) -> str:
    import io
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def describe_image(img_path: Path) -> str:
    payload = {
        "model": "minicpm",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64_jpeg(img_path)}"}},
                    {"type": "text", "text": DESCRIBE_PROMPT},
                ],
            }
        ],
        "max_tokens": 256,
        "temperature": 0.1,
    }
    r = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def write_metadata(img_path: Path, description: str, nsfw_label: str, nsfw_score: float):
    nsfw_value = f"{nsfw_label} ({nsfw_score:.2f})"
    subprocess.run(
        [
            "exiftool", "-overwrite_original",
            f"-XMP:Description={description}",
            f"-XMP:Subject={nsfw_value}",
            str(img_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main():
    images = [f for f in PICTURES_DIR.iterdir() if f.is_file() and f.suffix.lower() in EXTENSIONS]
    print(f"Imágenes encontradas: {len(images)}")

    # ── Carga clasificador NSFW (CPU, ligero) ────────────────────────────────
    print("▶ Cargando clasificador NSFW (CPU)...")
    nsfw_classifier = pipeline(
        "image-classification",
        model="Falconsai/nsfw_image_detection",
        device=-1,  # CPU siempre — deja VRAM para MiniCPM
    )
    print("✓ Clasificador NSFW listo\n")

    # ── Arranca MiniCPM ──────────────────────────────────────────────────────
    server_proc = start_server()

    try:
        errors = []
        for img_path in tqdm(images, desc="Procesando"):
            try:
                # Descripción
                description = describe_image(img_path)

                # NSFW
                pil_img = Image.open(img_path).convert("RGB")
                nsfw_results = nsfw_classifier(pil_img)
                top = max(nsfw_results, key=lambda x: x["score"])
                nsfw_label = top["label"]   # "safe" o "nsfw"
                nsfw_score = top["score"]

                # Escribe metadatos
                write_metadata(img_path, description, nsfw_label, nsfw_score)

            except Exception as e:
                errors.append((img_path.name, str(e)))
                tqdm.write(f"  ✗ Error: {img_path.name} — {e}")

    finally:
        stop_server(server_proc)

    print(f"\n✓ Paso 1 completado. Errores: {len(errors)}")
    if errors:
        print("Archivos con error:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print("\nEjecuta ahora: python paso2_clustering.py")


if __name__ == "__main__":
    main()
