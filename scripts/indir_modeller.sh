#!/usr/bin/env bash
# Model dosyalari icin surdurulebilir indirici + SHA-256 dogrulama (kaynak: bartowski GGUF, HF).
# Bitmis dosya varsa yalniz dogrular. Her sey klasorun icinde.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
LOG="$ROOT/logs/indir_modeller.log"
: > "$LOG"
DIG="$ROOT/logs/hf_digests.txt"
HF3="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main"
HF1="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main"

hf_dl() {
  local base="$1" name="$2" dest="$ROOT/models/$2" want sha have got tries=0
  want="$(awk -v n="$name" '$1==n {print $2}' "$DIG")"
  sha="$(awk -v n="$name" '$1==n {print $3}' "$DIG")"
  if [ ! -f "$dest" ]; then
    while [ "$tries" -lt 80 ]; do
      have=0; [ -f "$dest.part" ] && have="$(stat -c %s "$dest.part")"
      [ "$have" -ge "$want" ] && break
      tries=$((tries+1))
      echo "[$(date +%H:%M:%S)] DENEME $tries $name elde=$have/$want" >> "$LOG"
      curl -sSL -C - --speed-limit 500000 --speed-time 10 --connect-timeout 15 \
           -o "$dest.part" "$base/$name" >> "$LOG" 2>&1
    done
    have=0; [ -f "$dest.part" ] && have="$(stat -c %s "$dest.part")"
    if [ "$have" != "$want" ]; then
      echo "[$(date +%H:%M:%S)] HATA   $name boyut $have/$want" >> "$LOG"; return
    fi
    mv -f "$dest.part" "$dest"
  fi
  got="$(sha256sum "$dest" | cut -d' ' -f1)"
  if [ "$got" = "$sha" ]; then
    echo "[$(date +%H:%M:%S)] TAMAM  $name sha256 dogru" >> "$LOG"
  else
    echo "[$(date +%H:%M:%S)] HATA   $name sha256 UYUMSUZ ($got)" >> "$LOG"
  fi
}

hf_dl "$HF3" "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
hf_dl "$HF3" "Llama-3.2-3B-Instruct-Q8_0.gguf"
hf_dl "$HF1" "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
echo "[$(date +%H:%M:%S)] MODELLER BITTI" >> "$LOG"
