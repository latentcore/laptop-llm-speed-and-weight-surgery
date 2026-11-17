#!/usr/bin/env bash
# Hizli tarama: dongulu/kesilen 39 + temiz 40 MATH-500 sorusunda adaptor adaylari (kullanim: wsp_tara.sh "A:1.0" "B:1.0" ...)
# Her aday: --lora-scaled <adaptor> <olcek>. 'taban' = adaptorsuz.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
IDX="$(cat "$ROOT/data/wsp/screen_idx.txt")"

ozet() {  # etiket
  python -I - "$ROOT/results/eval_$1.json" "$1" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8")); s = d["sets"][0]
    print(f"{sys.argv[2]:16s} dogruluk=%{s['acc']:5.1f}  kesilme=%{s['kesilme_yuzde']:4.1f}  dongu=%{s['dongu_yuzde']:4.1f}  son_cevap_yok=%{s['son_cevap_yok_yuzde']:4.1f}  ort_token={s['ort_token']:4d}", flush=True)
except Exception as e:
    print(sys.argv[2], "HATA", e)
EOF
}

kos() {   # etiket, ek bayrak
  EV_IDX="$IDX" EV_EXTRA="$2" python -I "$ROOT/scripts/degerlendir.py" Llama-3.2-3B-Instruct-Q8_0.gguf "$1" 0 1 > /dev/null 2>&1
  ozet "$1"
}

for c in "$@"; do
  if [ "$c" = "taban" ]; then kos "scr_taban" ""; continue; fi
  ad="${c%%:*}"; sc="${c##*:}"
  # bu surumde bicim DOSYA:OLCEK; Windows surucu harfindeki ':' karisir -> sunucu cwd=ROOT, goreli yol kullan
  kos "scr_${ad}_${sc}" "--lora-scaled data/wsp/lora_${ad}.gguf:$sc"
done
