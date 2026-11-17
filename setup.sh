#!/bin/bash
set -e

echo "=== jax-learn setup ==="
echo ""

# check we're in WSL
if [ ! -f /proc/version ] || ! grep -qi microsoft /proc/version 2>/dev/null; then
    echo "ERROR: This script must run inside WSL."
    echo ""
    echo "If you don't have WSL installed, open PowerShell as Admin and run:"
    echo "  wsl --install"
    echo "Then restart your PC, open Ubuntu from the Start menu, and run this script again."
    exit 1
fi

# check nvidia driver is visible from WSL
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found in WSL."
    echo ""
    echo "You need NVIDIA GPU drivers installed on Windows (not inside WSL)."
    echo "1. Download from: https://www.nvidia.com/drivers"
    echo "2. Install the driver, restart your PC"
    echo "3. Open WSL and run this script again"
    echo ""
    echo "Do NOT install CUDA toolkit inside WSL — the driver handles GPU passthrough."
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# install system deps if needed
echo "Checking system dependencies..."
sudo apt update -qq
sudo apt install -y -qq python3 python3-venv python3-pip git curl > /dev/null 2>&1
echo "System dependencies OK."
echo ""

PYTHON=$(command -v python3)
PY_VERSION=$($PYTHON --version 2>&1)
echo "Python: $PY_VERSION"

# check python version is 3.10+
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MINOR" -lt 10 ]; then
    echo "ERROR: Python 3.10+ required, got $PY_VERSION"
    echo "Run: sudo apt install python3.12 python3.12-venv"
    exit 1
fi
echo ""

# figure out where the repo is
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
echo "Project directory: $SCRIPT_DIR"
echo ""

# create venv
if [ ! -d ".venv-wsl" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv .venv-wsl
else
    echo "Virtual environment already exists."
fi

source .venv-wsl/bin/activate
echo "Activated venv: $(which python)"
echo ""

# install python deps
echo "Installing Python packages (JAX + CUDA, Flax, Optax)..."
echo "This may take a few minutes on first run..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "Verifying JAX can see the GPU..."
python -c "
import jax
devices = jax.devices()
gpu = [d for d in devices if d.device_kind == 'gpu']
if gpu:
    print(f'SUCCESS: JAX sees GPU — {gpu[0]}')
else:
    print('WARNING: No GPU detected by JAX. Training will run on CPU (very slow).')
    print('Try:')
    print('  1. Make sure NVIDIA drivers are up to date on Windows')
    print('  2. Run: wsl --shutdown (in PowerShell), then reopen WSL')
    print('  3. Run this script again')
"

# download TinyStories if not present
if [ ! -f "resources/tinystories.txt" ]; then
    echo ""
    echo "Downloading TinyStories dataset (~28MB download, trimmed to ~10MB)..."
    mkdir -p resources
    python -I -c "
import urllib.request
url = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt'
print(f'Downloading from {url}...')
urllib.request.urlretrieve(url, '/tmp/tinystories_raw.txt')
with open('/tmp/tinystories_raw.txt', 'r', encoding='utf-8') as f:
    text = f.read()
text = text[:10_000_000]
last = text.rfind('<|endoftext|>')
if last != -1:
    text = text[:last + len('<|endoftext|>')].strip() + '\n'
with open('resources/tinystories.txt', 'w', encoding='utf-8') as f:
    f.write(text)
print(f'Saved {len(text)/1024/1024:.1f} MB to resources/tinystories.txt')
"
else
    echo "TinyStories dataset already present."
fi

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "To train the model, run:"
echo ""
echo "  cd $SCRIPT_DIR"
echo "  source .venv-wsl/bin/activate"
echo "  XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/train_tinystories.py"
echo ""
echo "Training takes ~1-2 hours on a 5080. Keep the terminal open."
echo "The checkpoint saves to checkpoints/gpt_tinystories.pkl when done."
echo ""
echo "To generate text after training:"
echo ""
echo "  XLA_PYTHON_CLIENT_PREALLOCATE=false python -c \""
echo "  from inference import Generator"
echo "  from flax import nnx"
echo "  g = Generator('checkpoints/gpt_tinystories.pkl', nnx.Rngs(42))"
echo "  print(g.generate('Once upon a time', max_new_tokens=100))"
echo "  \""
