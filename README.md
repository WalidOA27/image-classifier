# Image Classifier Pipeline

Clasificador local de imágenes con VLM + LLM. Sin cloud, sin APIs externas.

## Stack

- **MiniCPM-V 2.6 Q8** — describe cada imagen con detalle
- **Falconsai/nsfw_image_detection** — detecta contenido adulto
- **Huihui-Qwen3.5-9B** — asigna categoría basándose en la descripción
- **exiftool** — escribe metadatos en las imágenes (permanente)

## Requisitos

- Arch Linux
- AMD GPU con ROCm (16GB VRAM recomendado)
- Python 3.10+
- llama.cpp (si no existe, el install.sh lo compila)

## Instalación

```bash
git clone https://github.com/WalidOA27/image-classifier
cd image-classifier
chmod +x install.sh
./install.sh
```

Si ya tienes llama.cpp compilado con ROCm, el script lo detecta automáticamente y no lo recompila.

## Uso

```bash
# Todo en uno
~/clasificador/run.sh

# O paso a paso
source ~/clasificador/bin/activate
python ~/clasificador/paso1_describir.py   # describe + NSFW
python ~/clasificador/paso3_clasificar.py  # clasifica + mueve
```

## Categorías de salida

| Carpeta | Contenido |
|---|---|
| `anime/` | Capturas y frames de anime |
| `manga/` | Páginas y paneles de manga |
| `meme/` | Memes e imágenes con texto humorístico |
| `artwork/` | Arte digital, ilustraciones, fanart |
| `screenshot/` | Capturas de pantalla de apps/webs |
| `nsfw/` | Contenido adulto |
| `_revisar/` | No clasificado, revisar manualmente |

## Configuración

Edita las variables en cada script:

- `PICTURES_DIR` — carpeta de imágenes de entrada
- `DEST_DIR` — carpeta de salida (por defecto `Pictures/organized`)
- `THRESHOLD` — umbral NSFW (por defecto 0.7)

## Notas

- Los metadatos (`XMP:Description`, `XMP:Subject`) se escriben permanentemente en cada imagen
- Las imágenes originales no se borran, solo se copian
- Si ya tienes modelos descargados, el install.sh los detecta y no los vuelve a descargar
