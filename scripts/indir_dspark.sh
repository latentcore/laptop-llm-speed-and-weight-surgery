#!/usr/bin/env bash
# Llama-3.2-3B-Instruct icin DSpark taslak yan-dosyasi (rasyosef, GGUF Q8_0, 0,8 GB): indir + SHA-256 dogrula.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
LOG="$ROOT/logs/indir_dspark.log"
: > "$LOG"
REPO="rasyosef/Llama-3.2-3B-Instruct-DSpark-GGUF"
NAME="Llama-3.2-3B-Instruct-DSpark-Q8_0.gguf"
DEST="$ROOT/models/$NAME"

read -r WANT SHA < <(curl -sL "https://huggingface.co/api/models/$REPO/tree/main" | python -I -c "
import sys,json
for f in json.load(sys.stdin):
    if f.get('path')==sys.argv[1]:
        print(f['size'], f['lfs']['oid'])
" "$NAME")
echo "[$(date +%H:%M:%S)] beklenen: $WANT bayt, sha256 $SHA" >> "$LOG"
echo "$NAME $WANT $SHA" >> "$ROOT/logs/hf_digests.txt"

tries=0
while [ "$tries" -lt 60 ]; do
  have=0; [ -f "$DEST.part" ] && have="$(stat -c %s "$DEST.part")"
  [ "$have" -ge "$WANT" ] && break
  tries=$((tries+1))
  echo "[$(date +%H:%M:%S)] DENEME $tries elde=$have/$WANT" >> "$LOG"
  curl -sSL -C - --speed-limit 500000 --speed-time 10 --connect-timeout 15 \
       -o "$DEST.part" "https://huggingface.co/$REPO/resolve/main/$NAME" >> "$LOG" 2>&1
done
have=0; [ -f "$DEST.part" ] && have="$(stat -c %s "$DEST.part")"
if [ "$have" = "$WANT" ] && [ "$(sha256sum "$DEST.part" | cut -d' ' -f1)" = "$SHA" ]; then
  mv -f "$DEST.part" "$DEST"
  echo "[$(date +%H:%M:%S)] TAMAM $NAME sha256 dogru" >> "$LOG"
else
  echo "[$(date +%H:%M:%S)] HATA boyut=$have/$WANT veya sha256 uyumsuz" >> "$LOG"
fi
