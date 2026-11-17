#!/usr/bin/env bash
# GitHub surum zip'leri icin saglam indirici: takilan baglantiyi keser, kaldigi yerden surdurur,
# SHA-256 dogrular, sonra bin/llama-b12473/ altina acar. Her sey klasorun icinde.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
LOG="$ROOT/logs/indir_github.log"
: > "$LOG"
REL="https://github.com/ggml-org/llama.cpp/releases/download/b12473"
DIG="$ROOT/logs/github_digests.txt"

want_size() { awk -v n="$1" '$1==n {print $2}' "$DIG"; }
want_sha()  { awk -v n="$1" '$1==n {sub("sha256:","",$3); print $3}' "$DIG"; }

gh_dl() {
  local name="$1" dest="$ROOT/bin/zips/$1" want have tries=0 got
  want="$(want_size "$name")"
  while [ "$tries" -lt 80 ]; do
    have=0; [ -f "$dest.part" ] && have="$(stat -c %s "$dest.part")"
    [ "$have" -ge "$want" ] && break
    tries=$((tries+1))
    echo "[$(date +%H:%M:%S)] DENEME $tries $name elde=$have/$want" >> "$LOG"
    curl -sSL -C - --speed-limit 1000000 --speed-time 6 --connect-timeout 15 \
         -o "$dest.part" "$REL/$name" >> "$LOG" 2>&1
  done
  have=0; [ -f "$dest.part" ] && have="$(stat -c %s "$dest.part")"
  if [ "$have" = "$want" ]; then
    got="$(sha256sum "$dest.part" | cut -d' ' -f1)"
    if [ "$got" = "$(want_sha "$name")" ]; then
      mv -f "$dest.part" "$dest"
      echo "[$(date +%H:%M:%S)] BITTI  $name sha256 TAMAM" >> "$LOG"
    else
      echo "[$(date +%H:%M:%S)] HATA   $name sha256 UYUMSUZ ($got)" >> "$LOG"
    fi
  else
    echo "[$(date +%H:%M:%S)] HATA   $name boyut $have/$want" >> "$LOG"
  fi
}

mkdir -p "$ROOT/bin/zips"
( gh_dl "llama-b12473-bin-win-cuda-12.4-x64.zip" ) &
( gh_dl "cudart-llama-bin-win-cuda-12.4-x64.zip" ) &
wait

A="$ROOT/bin/zips/llama-b12473-bin-win-cuda-12.4-x64.zip"
B="$ROOT/bin/zips/cudart-llama-bin-win-cuda-12.4-x64.zip"
if [ -f "$A" ] && [ -f "$B" ]; then
  OUT="$ROOT/bin/llama-b12473"
  mkdir -p "$OUT"
  python -I -m zipfile -e "$A" "$OUT" >> "$LOG" 2>&1
  python -I -m zipfile -e "$B" "$OUT" >> "$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] ACILDI  $OUT ($(ls "$OUT" | wc -l) dosya)" >> "$LOG"
fi
echo "[$(date +%H:%M:%S)] GITHUB BITTI" >> "$LOG"
