#!/usr/bin/env python3
"""Hiz_Deneyi / degerlendir: yerel modeli GSM8K ve MATH-500 alt kumesinde olcer.
llama-server (16 slot, greedy) + basit, her model icin AYNI puanlayici. Dogruluk, kesilme (token siniri),
tekrar dongusu ve 'son cevap yok' oranlarini raporlar.

  python -I degerlendir.py <model_gguf_adi> <etiket> [gsm8k_n=300] [math_n=200] [max_tok_gsm=768] [max_tok_math=1024]
Cikti: results/eval_<etiket>.json ve results/eval_<etiket>_<set>.jsonl
"""
import concurrent.futures as cf
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request

ROOT = os.environ.get("HIZ_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(ROOT, "bin", "llama-b12473", "llama-server.exe")
PORT = int(os.environ.get("EV_PORT", "8091"))
SLOTS = int(os.environ.get("EV_SLOTS", "16"))
EXTRA = os.environ.get("EV_EXTRA", "").split()
SYSTEM = os.environ.get("EV_SYSTEM", "")      # istege bagli sistem istemi
MOFF = int(os.environ.get("EV_MOFF", "0"))    # MATH-500'de baslangic indeksi (200+ = gorulmemis sorular)
SAMPLER = json.loads(os.environ.get("EV_SAMPLER", '{"temperature": 0.0}'))   # or. {"temperature":0.6,"top_p":0.95,"top_k":20,"seed":1}
KWARGS = json.loads(os.environ.get("EV_KWARGS", "null"))                     # or. {"enable_thinking": false}


def start(model_path, tag):
    log = open(os.path.join(ROOT, "logs", f"eval_server_{tag}.log"), "w", encoding="utf-8")
    cmd = [SERVER, "-m", model_path, "-ngl", "99", "-fa", "on", "-c", os.environ.get("EV_CTX", "32768"), "-np", str(SLOTS), "-kvu", "-bs",
           "-ctk", "f16", "-ctv", "f16", "--host", "127.0.0.1", "--port", str(PORT), "--no-webui", *EXTRA]
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 240:
        if p.poll() is not None:
            raise RuntimeError("sunucu kapandi")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                if r.status == 200:
                    return p, log
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    raise RuntimeError("sunucu hazir olmadi")


def chat(user, max_tokens):
    msgs = ([{"role": "system", "content": SYSTEM}] if SYSTEM else []) + [{"role": "user", "content": user}]
    body = {"model": "local", "messages": msgs, "max_tokens": max_tokens, "stream": False, "seed": 1, **SAMPLER}
    if KWARGS:
        body["chat_template_kwargs"] = KWARGS
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    m = d["choices"][0]["message"]
    return (m.get("content") or ""), d["choices"][0].get("finish_reason"), d.get("usage", {}).get("completion_tokens", 0), (m.get("reasoning_content") or "")


# ------------------------------------------------------------ puanlama
NUM = r"-?\d[\d,]*\.?\d*|-?\.\d+"


def gsm_pred(text):
    m = re.findall(r"answer is[:\s]*\$?\s*\**\s*(" + NUM + r")", text, flags=re.I)
    if m:
        return m[-1], True
    nums = re.findall(NUM, text)
    return (nums[-1] if nums else None), False


def to_float(s):
    try:
        return float(s.replace(",", "").rstrip("."))
    except Exception:  # noqa: BLE001
        return None


def boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i + 7:j - 1] if depth == 0 else None


def norm(s):
    s = s.strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    for a, b in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""), ("dfrac", "frac"),
                 ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""), ("%", ""), ("$", ""), ("\\$", "")):
        s = s.replace(a, b)
    s = re.sub(r"\s+", "", s).rstrip(".")
    s = re.sub(r"^[a-zA-Z]=", "", s)
    return s


def frac_to_float(s):
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m and int(m.group(3)) != 0:
        v = int(m.group(2)) / int(m.group(3))
        return -v if m.group(1) else v
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    if m and int(m.group(2)) != 0:
        return int(m.group(1)) / int(m.group(2))
    return to_float(s)


def math_equiv(a, b):
    if a is None:
        return False
    na, nb = norm(a), norm(b)
    if na == nb:
        return True
    fa, fb = frac_to_float(na), frac_to_float(nb)
    return fa is not None and fb is not None and math.isclose(fa, fb, rel_tol=1e-6, abs_tol=1e-9)


def looped(text):
    """Metnin sonunda kisa bir parca tekrar tekrar geliyorsa True (>= 4 kez, 40+ karakter)."""
    t = text[-1500:]
    if len(t) < 200:
        return False
    for k in range(0, max(1, len(t) - 40), 25):
        sub = t[k:k + 40]
        if sub.strip() and t.count(sub) >= 4:
            return True
    return False


def run_set(name, items, make_prompt, score, max_tok, tag):
    t0 = time.time()

    def one(it):
        txt, fin, ntok, rsn = chat(make_prompt(it), max_tok)
        ok, has_final = score(it, txt)
        return {"ok": ok, "final": has_final, "finish": fin, "ntok": ntok, "loop": looped(rsn + "\n" + txt),
                "text": (rsn + "\n=====\n" + txt) if rsn else txt, "gold": it["gold"], "think": bool(rsn)}

    part = open(os.path.join(ROOT, "results", f"eval_{tag}_{name}.partial.jsonl"), "w", encoding="utf-8")   # cokmeye karsi: her sonuc hemen yazilir
    res = [None] * len(items)
    with cf.ThreadPoolExecutor(SLOTS) as ex:
        futs = {ex.submit(one, it): i for i, it in enumerate(items)}
        for fu in cf.as_completed(futs):
            res[futs[fu]] = fu.result()
            part.write(json.dumps({"i": futs[fu], **{k: v for k, v in res[futs[fu]].items() if k != "text"}}, ensure_ascii=False) + "\n")
            part.flush()
    part.close()
    wall = time.time() - t0
    n = len(res)
    acc = sum(r["ok"] for r in res) / n
    ci = 1.96 * math.sqrt(acc * (1 - acc) / n)
    s = {"set": name, "n": n, "acc": round(100 * acc, 1), "ci95": round(100 * ci, 1),
         "kesilme_yuzde": round(100 * sum(r["finish"] == "length" for r in res) / n, 1),
         "dongu_yuzde": round(100 * sum(r["loop"] for r in res) / n, 1),
         "son_cevap_yok_yuzde": round(100 * sum(not r["final"] for r in res) / n, 1),
         "ort_token": round(sum(r["ntok"] for r in res) / n), "sn": round(wall, 1),
         "tok_sn": round(sum(r["ntok"] for r in res) / wall)}
    with open(os.path.join(ROOT, "results", f"eval_{tag}_{name}.jsonl"), "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return s


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    model, tag = sys.argv[1], sys.argv[2]
    n_g = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    n_m = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    mt_g = int(sys.argv[5]) if len(sys.argv) > 5 else 768
    mt_m = int(sys.argv[6]) if len(sys.argv) > 6 else 1024
    mp = model if os.path.isabs(model) else os.path.join(ROOT, "models", model)
    gs = [json.loads(x) for x in open(os.path.join(ROOT, "data", "eval", "gsm8k_test.jsonl"), encoding="utf-8")][:n_g]
    for g in gs:
        g["gold"] = g["answer"].split("####")[-1].strip().replace(",", "")
    allm = [json.loads(x) for x in open(os.path.join(ROOT, "data", "eval", "math500_test.jsonl"), encoding="utf-8")]
    sel = [int(i) for i in os.environ["EV_IDX"].split(",")] if os.environ.get("EV_IDX") else range(MOFF, MOFF + n_m)
    ms = [allm[i] for i in sel if i < len(allm)]
    for m in ms:
        m["gold"] = m["answer"]
    p, log = start(mp, tag)
    out = {"model": os.path.basename(mp), "tag": tag, "system": SYSTEM, "extra": EXTRA, "sets": []}
    try:
        if n_g:
            def sg(it, txt):
                pred, has = gsm_pred(txt)
                a, b = to_float(pred) if pred else None, to_float(it["gold"])
                return (a is not None and b is not None and math.isclose(a, b, abs_tol=1e-4)), has
            out["sets"].append(run_set("gsm8k", gs, lambda it: it["question"] + "\n\nSolve this step by step. End with the sentence 'The answer is N.' where N is just the number.", sg, mt_g, tag))
            print(json.dumps(out["sets"][-1], ensure_ascii=False), flush=True)
        if n_m:
            def sm(it, txt):
                b = boxed(txt)
                return math_equiv(b, it["gold"]), b is not None
            out["sets"].append(run_set("math500", ms, lambda it: it["problem"] + "\n\nSolve this step by step and put the final answer in \\boxed{}.", sm, mt_m, tag))
            print(json.dumps(out["sets"][-1], ensure_ascii=False), flush=True)
    finally:
        p.terminate()
        try:
            p.wait(15)
        except subprocess.TimeoutExpired:
            p.kill()
        log.close()
    with open(os.path.join(ROOT, "results", f"eval_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
