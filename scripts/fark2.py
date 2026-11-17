import sys, numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
a, b = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])
ta = {t.name: t for t in a.tensors}; tb = {t.name: t for t in b.tensors}
dirs = {}
for L in (15, 19):
    n = f"blk.{L}.ffn_down.weight"
    Wa = dequantize(ta[n].data, ta[n].tensor_type).astype(np.float64); Wb = dequantize(tb[n].data, tb[n].tensor_type).astype(np.float64)
    D = Wb - Wa
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    u, v = U[:, 0], Vt[0]                       # D ~= S0 * u v^T
    absv = np.abs(v); order = np.argsort(-absv)
    cum = np.cumsum(absv[order] ** 2) / np.sum(absv ** 2)
    k90 = int(np.searchsorted(cum, 0.90)) + 1; k99 = int(np.searchsorted(cum, 0.99)) + 1
    wr = Wa.T @ u                                # u yonundeki nöron agirliklari
    top = order[:64]
    corr = np.corrcoef(v[top], wr[top])[0, 1]
    # D ~ -(u u^T) W[:, S] ?  -> u^T D / (u^T W) orani
    ratio = (u @ D)[top] / (u @ Wa)[top]
    print(f"katman {L}: S0={S[0]:.3f}  v'nin %90 enerjisi {k90} nöronda, %99'u {k99} nöronda (toplam 3584)")
    print(f"   ilk-64 nöronda  v ile W^T u korelasyonu = {corr:.3f};  (u^T D)/(u^T W) orani ort={ratio.mean():.3f} std={ratio.std():.3f}")
    print(f"   u'nun en buyuk 5 bileseni: {np.argsort(-np.abs(u))[:5].tolist()}  norm(u)={np.linalg.norm(u):.3f}")
    dirs[L] = u
print("u15 ile u19 arasi |cos| =", abs(float(dirs[15] @ dirs[19])))
