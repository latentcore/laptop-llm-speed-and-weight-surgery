import sys, numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
a, b = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])
ta = {t.name: t for t in a.tensors}; tb = {t.name: t for t in b.tensors}
print("tensor sayisi:", len(ta), len(tb), " ortak:", len(set(ta) & set(tb)))
diff = [n for n in ta if n in tb and (ta[n].tensor_type != tb[n].tensor_type or not np.array_equal(ta[n].data, tb[n].data))]
print("farkli tensor sayisi:", len(diff))
for n in diff:
    Wa = dequantize(ta[n].data, ta[n].tensor_type).astype(np.float32); Wb = dequantize(tb[n].data, tb[n].tensor_type).astype(np.float32)
    D = Wb - Wa
    if D.ndim != 2:
        print(n, D.shape, "|D|=%.3g" % np.linalg.norm(D)); continue
    colmax = np.abs(D).max(axis=0); rowmax = np.abs(D).max(axis=1)
    thr = 1e-3 * max(1e-12, np.abs(Wa).max())
    ncol = int((colmax > thr).sum()); nrow = int((rowmax > thr).sum())
    s = np.linalg.svd(D, compute_uv=False)
    print(f"{n:28s} sekil={D.shape} degisen_sutun={ncol} degisen_satir={nrow} |D|/|W|={np.linalg.norm(D)/np.linalg.norm(Wa):.4f} "
          f"singular(ilk5)={np.round(s[:5],3).tolist()} rank1_enerji={s[0]**2/np.sum(s**2):.3f}")
