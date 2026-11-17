#!/usr/bin/env python3
"""Hiz_Deneyi / kod_degerlendir: HumanEval (164 problem) uzerinde yerel modeli olcer.
  - pass@1 (greedy), pass@k (k ornek, sicak ornekleme: en az biri dogru), ve
  - 'ajan dongusu': basarisiz cozumde test hatasini modele geri verip en fazla R tur duzeltme (kendi kendini onarma).
Testler ayri bir python surecinde, zaman asimli calisir (stdlib; tehlikeli cagrilar AST ile elenir).

  python -I kod_degerlendir.py <model_gguf_adi> <etiket> [n=164] [k=8] [R=3]
Ortam: EV_KWARGS (or. {"enable_thinking": false}), EV_CTX, EV_EXTRA (sunucuya ek bayrak), EV_TEMP (varsayilan 0.8)
"""
import ast
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

ROOT = os.environ.get("HIZ_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(ROOT, "bin", "llama-b12473", "llama-server.exe")
PORT = 8092
SLOTS = int(os.environ.get("EV_SLOTS", "16"))
EXTRA = os.environ.get("EV_EXTRA", "").split()
KWARGS = json.loads(os.environ.get("EV_KWARGS", "null"))
TEMP = float(os.environ.get("EV_TEMP", "0.8"))
NOFLAGS = 0x08000000
DENY_IMPORT = ("subprocess", "socket", "ctypes", "shutil", "urllib", "http", "requests", "multiprocessing", "signal",
               "webbrowser", "smtplib", "ftplib", "telnetlib", "asyncio", "sqlite3", "pickle", "marshal", "importlib", "tempfile")
DENY_CALL = {"eval", "exec", "input", "__import__", "compile", "breakpoint", "exit", "quit", "open"}
DENY_ATTR = {("os", "system"), ("os", "popen"), ("os", "remove"), ("os", "unlink"), ("os", "rmdir"), ("os", "removedirs"),
             ("os", "rename"), ("os", "kill"), ("os", "_exit"), ("sys", "exit")}


def start(model_path, tag):
    log = open(os.path.join(ROOT, "logs", f"kodeval_server_{tag}.log"), "w", encoding="utf-8")
    cmd = [SERVER, "-m", model_path, "-ngl", "99", "-fa", "on", "-c", os.environ.get("EV_CTX", "65536"), "-np", str(SLOTS), "-kvu", "-bs",
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


def chat(messages, max_tokens, temperature, seed=1):
    body = {"model": "local", "messages": messages, "max_tokens": max_tokens, "stream": False, "temperature": temperature,
            "seed": seed}
    if temperature > 0:
        body["top_p"] = 0.95
    if KWARGS:
        body["chat_template_kwargs"] = KWARGS
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    m = d["choices"][0]["message"]
    return (m.get("content") or ""), d.get("usage", {}).get("completion_tokens", 0)


FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)(?:```|\Z)", re.S)


def extract(text, prompt, entry):
    blocks = [b for lang, b in FENCE.findall(text) if lang.lower() in ("", "python", "py", "python3") and b.strip()]
    code = blocks[0] if blocks else text
    if f"def {entry}" not in code:                      # yalniz govde geldiyse imzayi basa ekle
        code = prompt + code
    head = "\n".join(ln for ln in prompt.splitlines() if re.match(r"\s*(import|from)\s", ln))
    return head + "\n" + code


def unsafe(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in DENY_IMPORT:
                    return f"yasak import {a.name}"
        elif isinstance(n, ast.ImportFrom):
            if (n.module or "").split(".")[0] in DENY_IMPORT:
                return f"yasak import {n.module}"
        elif isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id in DENY_CALL:
                return f"yasak cagri {f.id}()"
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and (f.value.id, f.attr) in DENY_ATTR:
                return f"yasak cagri {f.value.id}.{f.attr}()"
    return None


def run_tests(code, prob, workdir, uid):
    bad = unsafe(code)
    if bad:
        return False, "REJECTED: " + bad
    prog = code + "\n\n" + prob["test"] + f"\n\ncheck({prob['entry_point']})\n"
    path = os.path.join(workdir, f"p_{uid}.py")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(prog)
    try:
        p = subprocess.run([sys.executable, "-I", path], cwd=workdir, capture_output=True, text=True, timeout=12,
                           creationflags=NOFLAGS, encoding="utf-8", errors="replace")
        ok = p.returncode == 0
        err = (p.stderr or "").strip().splitlines()[-6:]
        return ok, "\n".join(err)[:600]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT (12 s): infinite loop or too slow"
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def ask(prob, temperature, seed):
    user = ("Complete the following Python function. Answer with the complete function in a single ```python code "
            "block and nothing else.\n\n```python\n" + prob["prompt"] + "\n```")
    txt, nt = chat([{"role": "user", "content": user}], 1500, temperature, seed)
    return user, txt, nt


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    model, tag = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 164
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    R = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    mp = model if os.path.isabs(model) else os.path.join(ROOT, "models", model)
    probs = [json.loads(x) for x in open(os.path.join(ROOT, "data", "eval", "humaneval.jsonl"), encoding="utf-8")][:n]
    work = os.path.join(ROOT, "kod_uretimi", f"humaneval_{tag}")
    os.makedirs(work, exist_ok=True)
    proc, log = start(mp, tag)
    res = {"model": os.path.basename(mp), "tag": tag, "n": len(probs), "k": k, "R": R, "kwargs": KWARGS}
    toks = [0]
    try:
        t0 = time.time()

        def greedy(i):
            prob = probs[i]
            user, txt, nt = ask(prob, 0.0, 1)
            toks[0] += nt
            code = extract(txt, prob["prompt"], prob["entry_point"])
            ok, err = run_tests(code, prob, work, f"g{i}")
            return {"i": i, "ok": ok, "err": err, "user": user, "txt": txt, "code": code}

        with cf.ThreadPoolExecutor(SLOTS) as ex:
            g = list(ex.map(greedy, range(len(probs))))
        res["pass1_greedy"] = round(100 * sum(r["ok"] for r in g) / len(g), 1)
        print(f"[greedy] pass@1 = %{res['pass1_greedy']}  ({time.time() - t0:.0f} sn)", flush=True)

        def sample(job):
            i, s = job
            prob = probs[i]
            _, txt, nt = ask(prob, TEMP, 100 + s)
            toks[0] += nt
            code = extract(txt, prob["prompt"], prob["entry_point"])
            ok, _ = run_tests(code, prob, work, f"s{i}_{s}")
            return i, ok

        t1 = time.time()
        with cf.ThreadPoolExecutor(SLOTS) as ex:
            sres = list(ex.map(sample, [(i, s) for i in range(len(probs)) for s in range(k)]))
        per = {}
        for i, ok in sres:
            per.setdefault(i, []).append(ok)
        res["pass1_sampled"] = round(100 * sum(sum(v) / len(v) for v in per.values()) / len(per), 1)
        res["passk"] = round(100 * sum(1 for v in per.values() if any(v)) / len(per), 1)
        print(f"[ornekleme T={TEMP}] pass@1 = %{res['pass1_sampled']}   pass@{k} = %{res['passk']}  ({time.time() - t1:.0f} sn)", flush=True)

        # ajan dongusu: basarisiz greedy cozumleri test hatasiyla duzeltt
        state = {r["i"]: r for r in g}
        solved_at = {r["i"]: 0 for r in g if r["ok"]}
        for rnd in range(1, R + 1):
            todo = [i for i, r in state.items() if i not in solved_at]
            if not todo:
                break

            def repair(i):
                r = state[i]
                msgs = [{"role": "user", "content": r["user"]}, {"role": "assistant", "content": r["txt"]},
                        {"role": "user", "content": "Your solution fails the tests with this error:\n```\n" + r["err"] +
                         "\n```\nFix the function. Answer with the complete corrected function in a single ```python code block."}]
                txt, nt = chat(msgs, 1500, 0.2, 7 + rnd)
                toks[0] += nt
                code = extract(txt, probs[i]["prompt"], probs[i]["entry_point"])
                ok, err = run_tests(code, probs[i], work, f"r{i}_{rnd}")
                return i, {"i": i, "ok": ok, "err": err, "user": r["user"], "txt": txt, "code": code}

            with cf.ThreadPoolExecutor(SLOTS) as ex:
                for i, r2 in ex.map(repair, todo):
                    state[i] = r2
                    if r2["ok"]:
                        solved_at[i] = rnd
            res[f"ajan_tur{rnd}"] = round(100 * len(solved_at) / len(probs), 1)
            print(f"[ajan dongusu] tur {rnd}: toplam gecen = %{res[f'ajan_tur{rnd}']}", flush=True)
        res["toplam_sn"] = round(time.time() - t0, 1)
        res["uretilen_token"] = toks[0]
    finally:
        proc.terminate()
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    with open(os.path.join(ROOT, "results", f"kod_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
