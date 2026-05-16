# Image Classifier Pipeline

Local image classifier using VLM + LLM. No cloud, no external APIs.

## Stack

- **MiniCPM-V 2.6 Q8** — describes each image in detail
- **Falconsai/nsfw_image_detection** — detects adult content
- **Huihui-Qwen3.5-9B** — assigns a category based on the description
- **exiftool** — writes metadata directly into images (permanent)

## Requirements

- Arch Linux
- AMD GPU with ROCm (16 GB VRAM recommended)
- Python 3.10+
- llama.cpp (if not present, `install.sh` compiles it automatically)

## Installation

```bash
git clone https://github.com/WalidOA27/image-classifier
cd image-classifier
chmod +x install.sh
./install.sh
```

If you already have llama.cpp compiled with ROCm, the script detects it automatically and skips recompilation.

## Usage

```bash
# All-in-one
~/clasificador/run.sh

# Or step by step
source ~/clasificador/bin/activate
python ~/clasificador/paso1_describir.py   # describe + NSFW detection
python ~/clasificador/paso3_clasificar.py  # classify + move files
```

## Output Categories

| Folder | Contents |
|---|---|
| `anime/` | Anime screenshots and frames |
| `manga/` | Manga pages and panels |
| `meme/` | Memes and images with humorous text |
| `artwork/` | Digital art, illustrations, fanart |
| `screenshot/` | App and website screenshots |
| `nsfw/` | Adult content |
| `_revisar/` | Unclassified — review manually |

## Configuration

Edit the variables at the top of each script:

- `PICTURES_DIR` — input image folder
- `DEST_DIR` — output folder (default: `Pictures/organized`)
- `THRESHOLD` — NSFW threshold (default: `0.7`)

## Notes

- Metadata (`XMP:Description`, `XMP:Subject`) is written permanently into each image file
- Original images are not deleted — they are only copied to the output folder
- If models are already downloaded, `install.sh` detects them and skips re-downloading
