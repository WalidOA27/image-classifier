#!/bin/bash
# ============================================================
# install.sh — Image Classifier Pipeline
# Arch Linux + AMD ROCm
# ============================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}▶ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
err()  { echo -e "${RED}✗ $1${NC}"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HOME/clasificador"
MODELS_DIR="$HOME/.cache/huggingface/hub"

MINICPM_REPO="openbmb/MiniCPM-V-2_6-gguf"
MINICPM_FILE="ggml-model-Q8_0.gguf"
MMPROJ_FILE="mmproj-model-f16.gguf"
QWEN_REPO="mradermacher/Huihui-Qwen3.5-9B-abliterated-GGUF"
QWEN_FILE="Huihui-Qwen3.5-9B-abliterated.Q4_K_M.gguf"

# ── 1. Dependencias del sistema ──────────────────────────────
log "Verificando dependencias del sistema..."

MISSING=()
for pkg in python perl-image-exiftool git wget; do
    if ! pacman -Qi "$pkg" &>/dev/null; then
        MISSING+=("$pkg")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    log "Instalando: ${MISSING[*]}"
    sudo pacman -S --noconfirm "${MISSING[@]}"
fi

# ── 2. ROCm ──────────────────────────────────────────────────
log "Verificando ROCm..."
if ! command -v rocm-smi &>/dev/null && ! [ -f /opt/rocm/bin/rocm-smi ]; then
    warn "ROCm no encontrado. Instalando..."
    sudo pacman -S --noconfirm rocm-opencl-runtime rocm-hip-sdk
else
    log "ROCm encontrado"
fi

if ! echo "$PATH" | grep -q "/opt/rocm/bin"; then
    echo 'export PATH=$PATH:/opt/rocm/bin' >> "$HOME/.bashrc"
    export PATH=$PATH:/opt/rocm/bin
fi

# ── 3. llama.cpp ─────────────────────────────────────────────
log "Verificando llama.cpp..."

LLAMA_BIN=""
for candidate in \
    "$HOME/llama.cpp/build/bin/llama-server" \
    "/usr/local/bin/llama-server" \
    "/usr/bin/llama-server"; do
    if [ -f "$candidate" ]; then
        LLAMA_BIN="$candidate"
        log "llama.cpp encontrado: $LLAMA_BIN"
        break
    fi
done

if [ -z "$LLAMA_BIN" ]; then
    warn "llama.cpp no encontrado. Compilando con ROCm..."
    cd "$HOME"
    if [ ! -d "$HOME/llama.cpp" ]; then
        git clone https://github.com/ggerganov/llama.cpp.git
    fi
    cd "$HOME/llama.cpp"
    cmake -B build \
        -DGGML_HIPBLAS=ON \
        -DAMDGPU_TARGETS=gfx1201 \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j$(nproc)
    LLAMA_BIN="$HOME/llama.cpp/build/bin/llama-server"
    log "llama.cpp compilado: $LLAMA_BIN"
fi

# ── 4. Python venv ───────────────────────────────────────────
log "Configurando entorno Python..."

if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install --quiet --upgrade pip
pip install --quiet \
    torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3
pip install --quiet \
    transformers \
    pillow \
    tqdm \
    requests \
    huggingface-hub

log "Dependencias Python instaladas"

# ── 5. Modelos ───────────────────────────────────────────────
log "Verificando modelos..."

download_model() {
    local repo=$1
    local filename=$2
    local dest="$MODELS_DIR/models--$(echo "$repo" | tr '/' '--')/snapshots/local"
    mkdir -p "$dest"
    if [ ! -f "$dest/$filename" ]; then
        log "Descargando $filename..."
        python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='$repo', filename='$filename', local_dir='$dest')
"
    else
        log "$filename ya existe, omitiendo"
    fi
    echo "$dest/$filename"
}

MINICPM_PATH=$(download_model "$MINICPM_REPO" "$MINICPM_FILE")
MMPROJ_PATH=$(download_model "$MINICPM_REPO" "$MMPROJ_FILE")
QWEN_PATH=$(download_model "$QWEN_REPO" "$QWEN_FILE")

# ── 6. Configura scripts ─────────────────────────────────────
log "Configurando scripts..."

for script in paso1_describir.py paso3_clasificar.py; do
    if [ -f "$SCRIPT_DIR/$script" ]; then
        cp "$SCRIPT_DIR/$script" "$VENV_DIR/$script"
        python3 - "$VENV_DIR/$script" "$LLAMA_BIN" "$MINICPM_PATH" "$MMPROJ_PATH" "$QWEN_PATH" << 'PYEOF'
import re, sys
path, llama, minicpm, mmproj, qwen = sys.argv[1:]
with open(path) as f:
    c = f.read()
c = re.sub(r'LLAMA_BIN\s*=.*',    f'LLAMA_BIN    = Path("{llama}")',   c)
c = re.sub(r'MINICPM_GGUF\s*=.*', f'MINICPM_GGUF = Path("{minicpm}")', c)
c = re.sub(r'MMPROJ_GGUF\s*=.*',  f'MMPROJ_GGUF  = Path("{mmproj}")',  c)
c = re.sub(r'QWEN_GGUF\s*=.*',    f'QWEN_GGUF    = Path("{qwen}")',    c)
with open(path, 'w') as f:
    f.write(c)
print(f"  configurado: {path}")
PYEOF
    else
        warn "$script no encontrado en $SCRIPT_DIR"
    fi
done

# ── 7. Servicio systemd GPU performance ──────────────────────
log "Configurando GPU en high performance..."

ROCM_SMI_BIN=$(command -v rocm-smi 2>/dev/null || echo "/opt/rocm/bin/rocm-smi")

sudo tee /etc/systemd/system/gpu-performance.service > /dev/null << SVCEOF
[Unit]
Description=Set GPU to high performance
After=multi-user.target

[Service]
Type=oneshot
ExecStart=${ROCM_SMI_BIN} --setperflevel high
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
SVCEOF

sudo systemctl daemon-reload
sudo systemctl enable --now gpu-performance.service || warn "Ejecuta manualmente: rocm-smi --setperflevel high"

# ── 8. Script de ejecución ───────────────────────────────────
cat > "$VENV_DIR/run.sh" << RUNEOF
#!/bin/bash
source "$VENV_DIR/bin/activate"
echo "=== Paso 1: Describir imágenes ==="
python "$VENV_DIR/paso1_describir.py"
echo ""
echo "=== Paso 3: Clasificar y mover ==="
python "$VENV_DIR/paso3_clasificar.py"
RUNEOF
chmod +x "$VENV_DIR/run.sh"

# ── Resumen ──────────────────────────────────────────────────
echo ""
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}  Instalación completada${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo ""
echo "  Modelos:"
echo "    MiniCPM : $MINICPM_PATH"
echo "    mmproj  : $MMPROJ_PATH"
echo "    Qwen 9B : $QWEN_PATH"
echo ""
echo "  Para ejecutar:"
echo "    $VENV_DIR/run.sh"
echo ""
