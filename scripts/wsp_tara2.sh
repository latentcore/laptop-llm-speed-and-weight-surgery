#!/usr/bin/env bash
# Piyango taramasi (vekil olcut): her aday adaptor, Qwen3.5-0.8B dusunme modunda ilk 16 MATH-500 sorusunda, 8192 token sinirinda;
# olculen: dongu %, sinira carpmadan biten %, dogruluk. Kullanim: wsp_tara2.sh A1 A2 ...  ('taban' = adaptorsuz)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export EV_CTX=147456 EV_SAMPLER='{"temperature":0.6,"top_p":0.95,"top_k":20}' EV_IDX="$(seq -s, 0 15)"
for a in "$@"; do
  if [ "$a" = "taban" ]; then extra=""; else extra="--lora-scaled data/wsp/lora_$a.gguf:1.0"; fi
  EV_EXTRA="$extra" python -I "$ROOT/scripts/degerlendir.py" Qwen3.5-0.8B-Q8_0.gguf "scan_$a" 0 1 768 8192 > /dev/null 2>&1
  python -I - "$ROOT/results/eval_scan_$a.json" "$a" <<'EOF'
import json, sys
try:
    s = json.load(open(sys.argv[1], encoding="utf-8"))["sets"][0]
    print(f"{sys.argv[2]:8s} dongu=%{s['dongu_yuzde']:5.1f}  kesilme(bitmeyen)=%{s['kesilme_yuzde']:5.1f}  dogruluk=%{s['acc']:5.1f}  ort_token={s['ort_token']}", flush=True)
except Exception as e:
    print(sys.argv[2], "HATA", e)
EOF
done
