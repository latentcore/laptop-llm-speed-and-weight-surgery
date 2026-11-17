# Keeps downloads, temp files and caches inside the experiment folder (source this before running anything).
ROOT="${HIZ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WROOT="$(cygpath -m "$ROOT" 2>/dev/null || echo "$ROOT")"
mkdir -p "$ROOT"/{tmp,cache/hf,cache/pip,cache/torch,cache/triton,cache/cuda,cache/xdg,models,bin,data,logs,results}
export TEMP="$WROOT/tmp" TMP="$WROOT/tmp" TMPDIR="$WROOT/tmp"
export HF_HOME="$WROOT/cache/hf" HF_HUB_CACHE="$WROOT/cache/hf/hub" PIP_CACHE_DIR="$WROOT/cache/pip"
export TORCH_HOME="$WROOT/cache/torch" TRITON_CACHE_DIR="$WROOT/cache/triton" CUDA_CACHE_PATH="$WROOT/cache/cuda" XDG_CACHE_HOME="$WROOT/cache/xdg"
export PIP_DISABLE_PIP_VERSION_CHECK=1
cd "$ROOT"
