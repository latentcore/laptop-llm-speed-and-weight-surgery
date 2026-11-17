#!/usr/bin/env python3
"""Hiz_Deneyi: llama-server'i ayarlarla baslatir, prompt'lari greedy (temp 0) calistirir,
tek akis hizini ve taslak-kabul oranini olcer, sonucu results/spec_<ad><HIZ_TAG>.json'a yazar.

  python -I hiz_testi.py run q4_base,q4_ds8        # ayar adi: <model>_<yontem>[+<yontem>]
  python -I hiz_testi.py conc q8,q4                # eszamanli istekler (gercek sunucu)
  python -I hiz_testi.py cmp a.json b.json         # iki sonucun token bazli esitligi

Model: q4 (3B Q4_K_M) | q8 (3B Q8_0)
Yontem: base | dN (1B taslak, N token) | dsN[pXX] (DSpark, N token, XX=p-min yuzde) |
        ngm | ngmA | ngmB | ngmC (ngram-mod) | ngs | ngk ; '+' ile birlestirilir (ds8+ngmB)
Ortam: HIZ_PROMPTDIR, HIZ_PROMPTS (virgullu), HIZ_REPS, HIZ_NPRED, HIZ_CTX, HIZ_TAG,
       HIZ_EXTRA (sunucuya ek bayrak), HIZ_REQ (istege ek JSON), HIZ_DSPARK (taslak dosya adi)
"""
import json, os, re, subprocess, sys, time, urllib.request

ROOT = os.environ.get("HIZ_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(ROOT, "bin", "llama-b12473", "llama-server.exe")
M = lambda n: os.path.join(ROOT, "models", n)
Q4 = M("Llama-3.2-3B-Instruct-Q4_K_M.gguf")
Q8 = M("Llama-3.2-3B-Instruct-Q8_0.gguf")
D1 = M("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
DS = M(os.environ.get("HIZ_DSPARK", "Llama-3.2-3B-Instruct-DSpark-Q8_0.gguf"))
MODELS = {"q4": Q4, "q8": Q8, "b1": D1}   # b1 = Llama 3.2 1B Instruct Q4_K_M (olcek noktasi)
PORT = 8089
N_PREDICT = int(os.environ.get("HIZ_NPRED", "256"))
REPS = int(os.environ.get("HIZ_REPS", "2"))
CTX = os.environ.get("HIZ_CTX", "4096")
PROMPT_DIR = os.environ.get("HIZ_PROMPTDIR", os.path.join(ROOT, "data", "prompts"))
PROMPTS = os.environ.get("HIZ_PROMPTS", "chat,code,repeat").split(",")
TAG = os.environ.get("HIZ_TAG", "")


def method(m):
    """-> (spec_turleri, ek_bayraklar)"""
    if m == "base":
        return [], []
    g = re.fullmatch(r"d(\d+)", m)
    if g:
        return ["draft-simple"], ["-md", D1, "-ngld", "99", "--spec-draft-n-max", g[1], "--spec-draft-n-min", "1"]
    g = re.fullmatch(r"ds(\d+)(?:p(\d+))?", m)
    if g:
        a = ["-md", DS, "-ngld", "99", "--spec-draft-n-max", g[1], "--spec-draft-n-min", "0"]
        if g[2]:
            a += ["--spec-draft-p-min", str(int(g[2]) / 100)]
        return ["draft-dspark"], a
    table = {
        "ngm": (["ngram-mod"], []),
        "ngmA": (["ngram-mod"], ["--spec-ngram-mod-n-match", "24", "--spec-ngram-mod-n-min", "48", "--spec-ngram-mod-n-max", "64"]),
        "ngmB": (["ngram-mod"], ["--spec-ngram-mod-n-match", "12", "--spec-ngram-mod-n-min", "16", "--spec-ngram-mod-n-max", "32"]),
        "ngmC": (["ngram-mod"], ["--spec-ngram-mod-n-match", "8", "--spec-ngram-mod-n-min", "8", "--spec-ngram-mod-n-max", "24"]),
        "ngs": (["ngram-simple"], []),
        "ngk": (["ngram-map-k"], []),
    }
    return table[m]


def build(name):
    mk, _, meth = name.partition("_")
    types, args = [], []
    for part in meth.split("+"):
        t, a = method(part)
        types += t
        args += a
    return MODELS[mk], (["--spec-type", ",".join(types)] if types else []) + args


def http(path, payload=None, timeout=600):
    url = f"http://127.0.0.1:{PORT}{path}"
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(proc, limit=180):
    t0 = time.time()
    while time.time() - t0 < limit:
        if proc.poll() is not None:
            return False
        try:
            http("/health", timeout=3)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def complete(prompt, n, extra=None):
    body = {"prompt": prompt, "n_predict": n, "temperature": 0.0, "top_k": 1, "seed": 1,
            "cache_prompt": False, "return_tokens": True, "stream": False}
    if extra:
        body.update(extra)
    return http("/completion", body)


def stop(proc, log):
    proc.terminate()
    try:
        proc.wait(15)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()


def run_config(name):
    model, extra = build(name)
    log = open(os.path.join(ROOT, "logs", f"server_{name}{TAG}.log"), "w", encoding="utf-8")
    cmd = [SERVER, "-m", model, "-ngl", "99", "-fa", "on", "-c", CTX, "-np", "1",
           "--host", "127.0.0.1", "--port", str(PORT), "--no-webui", *extra]
    cmd += os.environ.get("HIZ_EXTRA", "").split()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    req_extra = json.loads(os.environ.get("HIZ_REQ", "null"))
    out = {"config": name, "model": os.path.basename(model), "args": extra, "prompts": {}}
    try:
        if not wait_ready(proc):
            out["error"] = f"sunucu hazir olmadi (bkz logs/server_{name}{TAG}.log)"
            return out
        complete("<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", 16)  # isinma
        for p in PROMPTS:
            prompt = open(os.path.join(PROMPT_DIR, p + ".txt"), encoding="utf-8").read()
            runs = []
            for _ in range(REPS):
                r = complete(prompt, N_PREDICT, req_extra)
                t = r.get("timings", {})
                runs.append({"tps": t.get("predicted_per_second"), "n": t.get("predicted_n"),
                             "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
                             "prompt_n": t.get("prompt_n"),
                             "tokens": r.get("tokens"), "text": r.get("content")})
            out["prompts"][p] = runs
    finally:
        stop(proc, log)
    return out


def summarize(res):
    cells = []
    for p in PROMPTS:
        runs = res["prompts"].get(p)
        if not runs:
            cells.append("-")
            continue
        # Yalniz ILK (taze) calisma sayilir: ikinci calisma ayni istegin tekrari; n-gram
        # yontemleri onceki uretimden ogrendigi icin tekrarda sahte hizlanir.
        r0 = runs[0]
        dn = r0["draft_n"] or 0
        da = r0["draft_acc"] or 0
        acc = f" kabul %{100*da/dn:.0f}" if dn else ""
        cells.append(f"{r0['tps'] or 0:6.1f}{acc}")
    return f"{res['config']:12s} | " + " | ".join(f"{p}: {c}" for p, c in zip(PROMPTS, cells))


def cmd_cmp(a, b):
    A = json.load(open(a, encoding="utf-8"))
    B = json.load(open(b, encoding="utf-8"))
    print(f"{A['config']} (referans)  vs  {B['config']}")
    for p in A["prompts"]:
        if p not in B["prompts"]:
            continue
        ta = A["prompts"][p][0]["tokens"]
        tb = B["prompts"][p][0]["tokens"]
        n = min(len(ta), len(tb))
        k = 0
        while k < n and ta[k] == tb[k]:
            k += 1
        print(f"  {p:10s}: ilk {k}/{n} token ayni" + ("  (TAM ESIT)" if k == n and len(ta) == len(tb) else f"  (ilk fark: {k}. token)"))


def run_conc(tag):
    """Gercek sunucuda eszamanli istekler: toplam token/sn (istek basina 128 token, onbellek kapali)."""
    import concurrent.futures as cf
    model = Q8 if tag == "q8" else Q4
    log = open(os.path.join(ROOT, "logs", f"server_conc_{tag}.log"), "w", encoding="utf-8")
    cmd = [SERVER, "-m", model, "-ngl", "99", "-fa", "on", "-c", "16384", "-np", "16",
           "--host", "127.0.0.1", "--port", str(PORT), "--no-webui"]
    cmd += os.environ.get("HIZ_EXTRA", "").split()
    req_extra = json.loads(os.environ.get("HIZ_REQ", "null"))
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    prompts = [open(os.path.join(ROOT, "data", "prompts", p + ".txt"), encoding="utf-8").read() for p in ("chat", "code", "repeat")]
    out = []
    try:
        if not wait_ready(proc):
            print("sunucu hazir olmadi")
            return
        complete(prompts[0], 16)
        for level in (1, 4, 8, 16):
            t0 = time.time()
            with cf.ThreadPoolExecutor(level) as ex:
                res = list(ex.map(lambda i: complete(prompts[i % 3], 128, req_extra), range(level)))
            wall = time.time() - t0
            tok = sum((r.get("timings") or {}).get("predicted_n", 0) for r in res)
            line = f"{os.path.basename(model)[:-5]:28s} N={level:2d}: toplam {tok/wall:7.1f} tok/sn (istek basina {tok/wall/level:6.1f}), {tok} token, {wall:.2f} sn"
            print(line, flush=True)
            out.append(line)
    finally:
        stop(proc, log)
    with open(os.path.join(ROOT, "results", f"eszamanli_{tag}{TAG}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "cmp":
        cmd_cmp(sys.argv[2], sys.argv[3])
        sys.exit()
    if len(sys.argv) >= 3 and sys.argv[1] == "conc":
        for t in sys.argv[2].split(","):
            run_conc(t)
        sys.exit()
    for nm in sys.argv[2].split(","):
        res = run_config(nm)
        with open(os.path.join(ROOT, "results", f"spec_{nm}{TAG}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
        print(res.get("error") or summarize(res), flush=True)
