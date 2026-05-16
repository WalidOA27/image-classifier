#!/usr/bin/env python3
"""
Paso 3: Clasificación con Qwen + movimiento de archivos
- Arranca Qwen 9B via llama-server
- Lee XMP:Description y XMP:Subject de cada imagen
- Qwen asigna categoría basándose en la descripción
- Mueve cada imagen a su carpeta destino
- Para el servidor al terminar
"""

import json
import shutil
import signal
import subprocess
import time
from pathlib import Path

import requests
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
PICTURES_DIR = Path("/home/walid/Pictures")
DEST_DIR     = Path("/home/walid/Pictures/organized")
LLAMA_BIN    = Path("/home/walid/llama.cpp/build/bin/llama-server")
QWEN_GGUF    = Path("/home/walid/.cache/huggingface/hub/models--mradermacher--Huihui-Qwen3.5-9B-abliterated-GGUF/snapshots/9f646d7eda193ddf2348134f3bff3d49eed7a2c6/Huihui-Qwen3.5-9B-abliterated.Q4_K_M.gguf")
SERVER_PORT  = 8083
SERVER_URL   = f"http://localhost:{SERVER_PORT}"
EXTENSIONS   = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif", ".bmp"}

CATEGORIES = ["anime", "manga", "meme", "artwork", "screenshot", "nsfw", "_revisar"]

SYSTEM_PROMPT = """You are an image classifier. You will receive a description of an image and must assign it to exactly one category.

Categories:
- anime: screenshots or frames from anime series, anime-style illustrations
- manga: black and white manga panels, manga pages, comic pages in japanese style
- meme: internet memes, images with humorous text overlay, reaction images, meme formats
- artwork: digital art, illustrations, concept art, fan art, drawings (colored)
- screenshot: screenshots of software, apps, websites, desktop, phone UI, games UI
- nsfw: sexually explicit or adult content
- _revisar: anything that doesn't clearly fit the above categories

Rules:
- If NSFW score is high (nsfw > 0.7), always assign "nsfw"
- Respond with ONLY the category name, nothing else
- No explanation, no punctuation, just the single word"""
# ─────────────────────────────────────────────────────────────────────────────


def start_server() -> subprocess.Popen:
    print("▶ Arrancando Qwen 9B...")
    proc = subprocess.Popen(
        [
            str(LLAMA_BIN),
            "-m", str(QWEN_GGUF),
            "--n-gpu-layers", "999",
            "--host", "0.0.0.0",
            "--port", str(SERVER_PORT),
            "--ctx-size", "2048",
            "--batch-size", "512",
            "--flash-attn",
            "--log-disable",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
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


def read_metadata(images: list) -> list:
    """Lee metadatos de todas las imágenes en una sola llamada."""
    result = subprocess.run(
        ["exiftool", "-XMP:Description", "-XMP:Subject", "-j"] + [str(i) for i in images],
        capture_output=True, text=True
    )
    try:
        all_meta = json.loads(result.stdout)
    except Exception:
        all_meta = []

    records = []
    for m in all_meta:
        records.append({
            "path": Path(m.get("SourceFile", "")),
            "description": m.get("Description", "").strip(),
            "nsfw": m.get("Subject", "unknown"),
        })
    return records


def classify(description: str, nsfw: str) -> str:
    """Pide a Qwen que clasifique basándose en la descripción."""
    # Si NSFW es alto, clasificar directamente sin llamar al LLM
    if "nsfw" in nsfw.lower():
        try:
            score = float(nsfw.split("(")[-1].rstrip(")"))
            if score > 0.7:
                return "nsfw"
        except Exception:
            pass

    if not description:
        return "_revisar"

    user_msg = f"Description: {description}\nNSFW: {nsfw}\n\nCategory:"

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 50,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.0,
    }
    try:
        r = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=30)
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"].strip().lower()
        # Valida que la respuesta sea una categoría válida
        for cat in CATEGORIES:
            if cat in answer:
                return cat
        return "_revisar"
    except Exception:
        return "_revisar"


def main():
    # Crea carpetas destino
    for cat in CATEGORIES:
        (DEST_DIR / cat).mkdir(parents=True, exist_ok=True)

    # Recopila imágenes (solo raíz, ignora subcarpetas)
    images = [f for f in PICTURES_DIR.iterdir()
              if f.is_file() and f.suffix.lower() in EXTENSIONS]
    print(f"Imágenes encontradas: {len(images)}")

    # Lee todos los metadatos de golpe
    print("Leyendo metadatos...")
    records = read_metadata(images)
    with_desc    = [r for r in records if r["description"]]
    without_desc = [r for r in records if not r["description"]]
    print(f"  Con descripción: {len(with_desc)}")
    print(f"  Sin descripción: {len(without_desc)} → _revisar directamente\n")

    # Mueve las sin descripción a _revisar
    for r in without_desc:
        shutil.copy2(r["path"], DEST_DIR / "_revisar" / r["path"].name)

    # Arranca Qwen
    server_proc = start_server()

    stats = {cat: 0 for cat in CATEGORIES}
    errors = []

    try:
        for r in tqdm(with_desc, desc="Clasificando"):
            category = classify(r["description"], r["nsfw"])
            stats[category] += 1
            try:
                shutil.copy2(r["path"], DEST_DIR / category / r["path"].name)
            except Exception as e:
                errors.append((r["path"].name, str(e)))
    finally:
        stop_server(server_proc)

    print("\n✓ Clasificación completada")
    print("\nResultado:")
    for cat, count in sorted(stats.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {cat}: {count} imágenes")

    if errors:
        print(f"\nErrores ({len(errors)}):")
        for name, err in errors:
            print(f"  - {name}: {err}")

    print(f"\nImágenes en: {DEST_DIR}")
    print("Revisa _revisar/ y ajusta manualmente si es necesario.")


if __name__ == "__main__":
    main()
