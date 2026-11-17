#!/usr/bin/env python3
"""Hiz_Deneyi / wsp_kes: makaledeki ('Weight-Space Programming') rank-1 duzenlemeyi egitimsiz ve torch'suz yeniden uretir.

Tarif (makalenin Qwen3.5-0.8B duzenlemesinden sokuldu): secili katmanlarda ffn_down icin
    W[:, S] <- W[:, S] - c * u (u^T W[:, S])         (S = |u^T W| en buyuk K nöron; c ~ 6; u birim 'dongu yonu')
Bu, LoRA rank-1: B = -c*u (cikis x 1), A = (u^T W) yalniz S'de dolu (1 x girdi).

'Dongu yonu' u: llama-imatrix ciktilarindan (ffn_up girdisinin ortalamasi) = dongu metni - temiz metin; norm agirligina bolunerek
artik-akim (residual) tabanina cevrilir. Kontrol: rastgele yon.

  python wsp_kes.py yaz <model> <imat_loop> <imat_clean> <ad> <katmanlar,> <K> <c> [baglam|rastgele] [tohum]
  python wsp_kes.py karsilastir <model> <imat_loop> <imat_clean> <katmanlar,> <taban_gguf> <duzenlenmis_gguf>
"""
import os
import sys

import numpy as np
from gguf import GGUFReader, GGUFWriter
from gguf.quants import dequantize

ROOT = os.environ.get("HIZ_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tensors(path):
    r = GGUFReader(path)
    return r, {t.name: t for t in r.tensors}


def imat(path):
    r = GGUFReader(path)
    return {t.name: np.asarray(t.data, dtype=np.float64).copy() for t in r.tensors}


def mean_in(st, L):
    return st[f"blk.{L}.ffn_up.weight.in_sum"] / st[f"blk.{L}.ffn_up.weight.counts"][0]


def loop_dir(model_t, sl, sc, L):
    """dongu - temiz ortalama farki, norm agirligina bolunerek residual tabana, birim vektor."""
    d = mean_in(sl, L) - mean_in(sc, L)
    g = dequantize(model_t[f"blk.{L}.post_attention_norm.weight"].data, model_t[f"blk.{L}.post_attention_norm.weight"].tensor_type).astype(np.float64).reshape(-1)
    g = np.where(np.abs(g) < 1e-3, 1e-3, g)
    u = d / g
    return u / (np.linalg.norm(u) + 1e-12)


def lda_dir(model_t, sl, sc, L):
    """Kose-LDA: ortalama farki / ortak varyans (buyuk-varyansli 'massive' kanallarin baskinligini kirar), residual tabana, birim."""
    def st(s):
        n = s[f"blk.{L}.ffn_up.weight.counts"][0]
        m = s[f"blk.{L}.ffn_up.weight.in_sum"] / n
        e2 = s[f"blk.{L}.ffn_up.weight.in_sum2"] / n
        return m, np.maximum(e2 - m * m, 1e-9)
    ml, vl = st(sl)
    mc, vc = st(sc)
    d = (ml - mc) / (0.5 * (vl + vc))
    t = model_t[f"blk.{L}.post_attention_norm.weight"]
    g = dequantize(t.data, t.tensor_type).astype(np.float64).reshape(-1)
    g = np.where(np.abs(g) < 1e-3, 1e-3, g)
    u = d / g
    return u / (np.linalg.norm(u) + 1e-12)


def down_w(model_t, L):
    t = model_t[f"blk.{L}.ffn_down.weight"]
    return dequantize(t.data, t.tensor_type).astype(np.float64)       # (cikis, girdi)


def yaz(model, il, ic, ad, layers, K, c, mod="baglam", seed=0):
    _, mt = tensors(model)
    sl, sc = imat(il), imat(ic)
    rng = np.random.default_rng(seed)
    out = os.path.join(ROOT, "data", "wsp", f"lora_{ad}.gguf")
    w = GGUFWriter(out, arch=model_arch(model))
    w.add_string("general.type", "adapter")
    w.add_string("adapter.type", "lora")
    w.add_float32("adapter.lora.alpha", 1.0)                          # rank 1 -> carpan 1
    for L in layers:
        W = down_w(mt, L)
        u = loop_dir(mt, sl, sc, L) if mod == "baglam" else (lda_dir(mt, sl, sc, L) if mod == "lda" else None)
        if u is None:
            u = rng.standard_normal(W.shape[0]); u /= np.linalg.norm(u)
        proj = u @ W                                                   # (girdi,)
        S = np.argsort(-np.abs(proj))[:K]
        A = np.zeros((1, W.shape[1]), dtype=np.float32); A[0, S] = proj[S]
        B = (-c * u).astype(np.float32).reshape(-1, 1)
        w.add_tensor(f"blk.{L}.ffn_down.weight.lora_a", A)
        w.add_tensor(f"blk.{L}.ffn_down.weight.lora_b", B)
        print(f"katman {L}: |u^T W| ilk-{K} toplam enerji orani={np.sum(proj[S]**2)/np.sum(proj**2):.2f}  u en buyuk bilesenler={np.argsort(-np.abs(u))[:4].tolist()}")
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("yazildi:", out)


def model_arch(model):
    return GGUFReader(model).fields["general.architecture"].contents()


def karsilastir(model, il, ic, layers, base, edited):
    _, mt = tensors(model)
    sl, sc = imat(il), imat(ic)
    _, ta = tensors(base)
    _, tb = tensors(edited)
    for L in layers:
        n = f"blk.{L}.ffn_down.weight"
        D = dequantize(tb[n].data, tb[n].tensor_type).astype(np.float64) - dequantize(ta[n].data, ta[n].tensor_type).astype(np.float64)
        U, S_, Vt = np.linalg.svd(D, full_matrices=False)
        up = U[:, 0]
        um = loop_dir(mt, sl, sc, L)
        print(f"katman {L}: makalenin u'su ile benim dongu yonum arasi |cos| = {abs(float(up @ um)):.3f}")


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "yaz":
        yaz(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], [int(x) for x in sys.argv[6].split(",")], int(sys.argv[7]),
            float(sys.argv[8]), sys.argv[9] if len(sys.argv) > 9 else "baglam", int(sys.argv[10]) if len(sys.argv) > 10 else 0)
    elif c == "karsilastir":
        karsilastir(sys.argv[2], sys.argv[3], sys.argv[4], [int(x) for x in sys.argv[5].split(",")], sys.argv[6], sys.argv[7])
