#!/usr/bin/env bash
# Hiz_Deneyi indirme betigi: her sey klasorun icine iner (env.sh sorumlu).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
LOG="$ROOT/logs/indir.log"
: > "$LOG"

dl() {
  local url="$1" dest="$2" name
  name="$(basename "$dest")"
  echo "[$(date +%H:%M:%S)] BASLA  $name" >> "$LOG"
  if curl -sSL --fail --retry 5 --retry-delay 3 -o "$dest.part" \
       -w "   -> %{size_download} bayt, %{speed_download} bayt/sn, %{time_total} sn\n" "$url" >> "$LOG" 2>&1; then
    mv -f "$dest.part" "$dest"
    echo "[$(date +%H:%M:%S)] BITTI  $name ($(stat -c %s "$dest") bayt)" >> "$LOG"
  else
    echo "[$(date +%H:%M:%S)] HATA   $name" >> "$LOG"
  fi
}

REL="https://github.com/ggml-org/llama.cpp/releases/download/b12473"
HF3="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main"
HF1="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main"

mkdir -p "$ROOT/bin/zips" "$ROOT/data" "$ROOT/models"

# Akis 1: llama.cpp ikilileri (resmi ggml-org surumu) + CUDA calisma zamani + veri + 1B taslak model
(
  dl "$REL/llama-b12473-bin-win-cuda-12.4-x64.zip"  "$ROOT/bin/zips/llama-b12473-bin-win-cuda-12.4-x64.zip"
  dl "$REL/cudart-llama-bin-win-cuda-12.4-x64.zip"  "$ROOT/bin/zips/cudart-llama-bin-win-cuda-12.4-x64.zip"
  dl "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip" "$ROOT/data/wikitext-2-raw-v1.zip"
  dl "$HF1/Llama-3.2-1B-Instruct-Q4_K_M.gguf"       "$ROOT/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
) &

# Akis 2: 3B Q4_K_M ve Q8_0
(
  dl "$HF3/Llama-3.2-3B-Instruct-Q4_K_M.gguf"       "$ROOT/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
  dl "$HF3/Llama-3.2-3B-Instruct-Q8_0.gguf"         "$ROOT/models/Llama-3.2-3B-Instruct-Q8_0.gguf"
) &

# Akis 3: 3B f16 (en buyuk)
(
  dl "$HF3/Llama-3.2-3B-Instruct-f16.gguf"          "$ROOT/models/Llama-3.2-3B-Instruct-f16.gguf"
) &

wait
echo "[$(date +%H:%M:%S)] HEPSI BITTI" >> "$LOG"
