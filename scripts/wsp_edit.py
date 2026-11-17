#!/usr/bin/env python3
"""Hiz_Deneyi / wsp_edit: 'Weight-Space Programming' benzeri, egitimsiz agirlik duzenleme (nöron cerrahisi).

1) llama-imatrix ciktilari (dongu metni vs temiz metin) -> her katmanda MLP nöronlarinin (ffn_down girisi) ortalama
   etkinlik farki.
2) Dongüde asiri etkin nöronlari sec (etki = nöronun cikis vektor normu^2 x (karesel etkinlik farki)).
3) Secilen nöronlarin ffn_down cikislarini bastiran LoRA adaptoru yaz (A: tek-sicak secici, B: -W[:, S]).
   llama.cpp'de --lora-scaled <dosya> <olcek> ile calisma zamaninda acilir/kapanir/olceklenir.

Kullanim (venv python ile):
  python wsp_edit.py ozet                              # katman bazli skor ozeti
  python wsp_edit.py yaz <ad> <katmanlar,> <K> [<mod>]  # adaptor yaz, ornek: yaz A 15,19 64
     mod: msq (varsayilan: karesel etkinlik farki) | mean (ortalama etkinlik farki)
"""
import os
import sys

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter
from gguf.quants import dequantize

ROOT = os.environ.get("HIZ_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "Llama-3.2-3B-Instruct-Q8_0.gguf")
LOOP = os.path.join(ROOT, "data", "wsp", "imat_loop.gguf")
CLEAN = os.path.join(ROOT, "data", "wsp", "imat_clean.gguf")
NL = 28


def stats(path):
    r = GGUFReader(path)
    t = {x.name: np.asarray(x.data, dtype=np.float64).copy() for x in r.tensors if "ffn_down" in x.name}
    out = {}
    for L in range(NL):
        cnt = float(t[f"blk.{L}.ffn_down.weight.counts"][0])
        out[L] = {"mean": t[f"blk.{L}.ffn_down.weight.in_sum"] / cnt, "msq": t[f"blk.{L}.ffn_down.weight.in_sum2"] / cnt, "n": cnt}
    return out


def down_weight(reader, L):
    for x in reader.tensors:
        if x.name == f"blk.{L}.ffn_down.weight":
            return dequantize(x.data, x.tensor_type).astype(np.float32)   # (3072, 8192): cikis x girdi
    raise KeyError(L)


def scores(sl, sc, W, mode):
    col2 = (W.astype(np.float64) ** 2).sum(axis=0)               # her nöronun cikis vektor normu^2
    if mode == "mean":
        diff = (sl["mean"] ** 2) - (sc["mean"] ** 2)
    else:
        diff = sl["msq"] - sc["msq"]
    return col2 * diff


def cmd_ozet():
    sl, sc = stats(LOOP), stats(CLEAN)
    rd = GGUFReader(MODEL)
    print("katman | toplam pozitif skor (ilk 64 nöron) | ilk-64 / tum pozitif")
    rows = []
    for L in range(NL):
        W = down_weight(rd, L)
        s = scores(sl[L], sc[L], W, "msq")
        pos = np.clip(s, 0, None)
        top = np.sort(pos)[::-1]
        rows.append((L, top[:64].sum(), top[:64].sum() / max(pos.sum(), 1e-12)))
    for L, a, b in rows:
        print(f"{L:5d} | {a:12.4g} | {b:.2f}")
    best = sorted(rows, key=lambda x: -x[1])[:6]
    print("en yuksek 6 katman:", [x[0] for x in best])


def cmd_yaz(name, layers, K, mode="msq"):
    sl, sc = stats(LOOP), stats(CLEAN)
    rd = GGUFReader(MODEL)
    out = os.path.join(ROOT, "data", "wsp", f"lora_{name}.gguf")
    w = GGUFWriter(out, arch="llama")
    w.add_string("general.type", "adapter")
    w.add_string("adapter.type", "lora")
    w.add_float32("adapter.lora.alpha", float(K))               # alpha = rank -> carpan 1
    chosen = {}
    for L in layers:
        W = down_weight(rd, L)
        s = scores(sl[L], sc[L], W, mode)
        idx = np.argsort(-s)[:K]
        chosen[L] = idx
        A = np.zeros((K, W.shape[1]), dtype=np.float32)
        A[np.arange(K), idx] = 1.0
        B = -W[:, idx].astype(np.float32)                        # (3072, K)
        w.add_tensor(f"blk.{L}.ffn_down.weight.lora_a", A)
        w.add_tensor(f"blk.{L}.ffn_down.weight.lora_b", B)
        print(f"katman {L}: {K} nöron, skor araligi {s[idx[-1]]:.3g}..{s[idx[0]]:.3g}, indeks ornek {idx[:5].tolist()}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("yazildi:", out, f"{os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "ozet":
        cmd_ozet()
    elif c == "yaz":
        cmd_yaz(sys.argv[2], [int(x) for x in sys.argv[3].split(",")], int(sys.argv[4]), sys.argv[5] if len(sys.argv) > 5 else "msq")
