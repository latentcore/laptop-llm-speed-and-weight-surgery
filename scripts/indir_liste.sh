#!/usr/bin/env bash
# Model adaylari icin genel indirici: HF LFS SHA-256 ile dogrular, kaldigi yerden surdurur. Hepsi models/ icine.
# Kullanim: indir_liste.sh "<repo>|<dosya>|<hedef_ad>" ...
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
LOG="$ROOT/logs/indir_liste.log"
: >> "$LOG"

dl() {
  local repo="$1" name="$2" dest="$ROOT/models/$3" want sha have tries=0
  if [ -f "$dest" ]; then echo "[$(date +%H:%M:%S)] VAR   $3" >> "$LOG"; return; fi
  read -r want sha < <(curl -sL "https://huggingface.co/api/models/$repo/tree/main" | python -I -c "
import sys,json
for f in json.load(sys.stdin):
    if f.get('path')==sys.argv[1]:
        print(f['size'], f['lfs']['oid'])
" "$name" | tr -d '\r')
  echo "[$(date +%H:%M:%S)] BASLA $3 ($want bayt) <- $repo/$name" >> "$LOG"
  while [ "$tries" -lt 60 ]; do
    have=0; [ -f "$dest.part" ] && have="$(stat -c %s "$dest.part")"
    [ "$have" -ge "$want" ] && break
    tries=$((tries+1))
    curl -sSL -C - --speed-limit 500000 --speed-time 10 --connect-timeout 15 \
         -o "$dest.part" "https://huggingface.co/$repo/resolve/main/$name" >> "$LOG" 2>&1
  done
  have=0; [ -f "$dest.part" ] && have="$(stat -c %s "$dest.part")"
  if [ "$have" = "$want" ] && [ "$(sha256sum "$dest.part" | cut -d' ' -f1)" = "$sha" ]; then
    mv -f "$dest.part" "$dest"
    echo "$3 $want $sha" >> "$ROOT/logs/hf_digests.txt"
    echo "[$(date +%H:%M:%S)] TAMAM $3 sha256 dogru" >> "$LOG"
  else
    echo "[$(date +%H:%M:%S)] HATA  $3 boyut=$have/$want veya sha256 uyumsuz" >> "$LOG"
  fi
}

for spec in "$@"; do
  IFS='|' read -r repo name dest <<< "$spec"
  dl "$repo" "$name" "$dest"
done
echo "[$(date +%H:%M:%S)] LISTE BITTI" >> "$LOG"
