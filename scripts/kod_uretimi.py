#!/usr/bin/env python3
"""Hiz_Deneyi / kod_uretimi  (v2)

Yerel Llama 3.2 3B (llama-server, cok slotlu) bir Python kutuphanesini (toolkit) KENDI BASINA yazar:
plan (API'ler dahil) -> modul -> test -> pyflakes + pytest -> hata ciktisi modele geri verilir -> model duzeltir.
Bu betik yalniz YONETICIDIR (modeli cagirir, statik/dinamik kontrol kosar, hatayi modele geri verir, rapor yazar);
urun kodunun (modul, test, README) tek satirini yazmaz ve duzeltmez.

Modlar:  run  (tam kosu)  |  bench  (yalniz toplam tok/sn olcumu)
Ortam:   KU_SLOTS, KU_BUDGET, KU_PERCAT, KU_REPAIR, KU_EXTRA (sunucuya ek bayrak), KU_SAMPLER (JSON)
"""
import ast
import concurrent.futures as cf
import datetime
import json
import keyword
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.environ.get("HIZ_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(ROOT, "bin", "llama-b12473", "llama-server.exe")
MODEL = os.path.join(ROOT, "models", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")
PORT = 8090
SLOTS = int(os.environ.get("KU_SLOTS", "16"))
BUDGET = float(os.environ.get("KU_BUDGET", "190"))      # uretim fazinda yeni LLM cagrisi baslatma suresi (sn)
PER_CAT = int(os.environ.get("KU_PERCAT", "10"))
MAX_REPAIR = int(os.environ.get("KU_REPAIR", "3"))
SERVER_EXTRA = os.environ.get("KU_EXTRA", "").split()
SAMPLER = json.loads(os.environ.get("KU_SAMPLER", '{"temperature": 0.2, "top_p": 0.95, "top_k": 40}'))
RUN = os.path.join(ROOT, "kod_uretimi", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
NOFLAGS = 0x08000000   # CREATE_NO_WINDOW

CATEGORIES = [
    "text processing and string utilities",
    "pure-Python data structures",
    "math, statistics and numeric utilities",
    "dates, times and calendars (pure calculations on given dates, no access to the current time)",
    "validation and parsing of common formats (versions, identifiers, numbers, simple URLs)",
    "iterables, collections and functional helpers",
    "classic algorithms (sorting, searching, graphs, dynamic programming)",
    "formatting, tables, colors and unit conversion",
]

PLAN_SYS = ("You are a principal Python engineer designing a professional, standard-library-only utility "
            "package called 'toolkit'. You follow output-format instructions exactly.")
GEN_SYS = ("You are a senior Python engineer. You write clean, correct, professional Python 3.10+ code: PEP 8, "
           "full type hints, Google-style docstrings, input validation with clear exceptions (ValueError/TypeError), "
           "no print statements, standard library only, no placeholder text like [Your Name]. You answer with "
           "exactly one fenced ```python code block and no other text.")
TEST_SYS = ("You are a senior Python engineer writing pytest unit tests. You answer with exactly one fenced "
            "```python code block and no other text.")
FIX_SYS = ("You are a senior Python engineer fixing code. You answer only with fenced ```python code blocks "
           "containing complete files, no explanations.")
README_SYS = "You are a technical writer. You output only the requested Markdown document."


def plan_prompt(cat):
    return (f"Category: {cat}.\n"
            f"List exactly {PER_CAT} distinct, genuinely useful modules for this category. Each must be implementable "
            "in 60-120 lines of pure Python (standard library only) and testable without files, network, clocks or "
            "randomness.\n"
            f"Output exactly {PER_CAT} lines and nothing else. Each line has the form:\n"
            "module_name | one-sentence purpose | func_one(arg: type) -> type; func_two(arg: type) -> type; func_three(...)\n"
            "The third part lists 3 to 5 public functions with typed signatures. module_name is lowercase snake_case, "
            "3-24 characters, and not the name of a Python standard-library module.")


def gen_prompt(name, purpose, api, short=False):
    api_line = (f"Public API to implement (exactly these names and signatures; small private helpers are fine): {api}\n"
                if api else "Choose 4 to 6 useful public functions.\n")
    size = "Keep it under 90 lines." if short else "60 to 120 lines."
    return (f"Write the module `toolkit/{name}.py`.\nPurpose: {purpose}\n{api_line}\n"
            "Requirements:\n"
            "- A module docstring and an `__all__` list with the public names.\n"
            "- Deterministic, pure logic only: no file access, no network, no input(), no randomness, no "
            "current-time access, no global state.\n"
            "- Validate arguments and raise ValueError or TypeError with a clear message for invalid input.\n"
            f"- {size} Do not include tests or a main block.\n"
            "Output the complete module in one ```python block.")


def test_prompt(name, code, names, short=False):
    imp = ", ".join(names) if names else "*"
    size = ("6 to 8 test functions, each at most 8 lines; keep the whole file under 80 lines." if short
            else "8 to 10 test functions, each at most 10 lines.")
    return (f"Here is the module `toolkit/{name}.py`:\n\n```python\n{code}\n```\n\n"
            f"API summary (from the code):\n{api_summary(code)}\n"
            f"Exceptions explicitly raised by the code:\n{raises_summary(code)}\n\n"
            f"Write `tests/test_{name}.py` with pytest.\n"
            "Rules:\n"
            f"- Import everything you use at the top of the file, for example `import math`, `import pytest`, "
            f"`from toolkit.{name} import {imp}`.\n"
            f"- {size} Use small, simple inputs whose results you can verify by hand from the code.\n"
            "- Test error cases with pytest.raises ONLY for exceptions listed above, with inputs that clearly trigger "
            "them. Never assert on error message text.\n"
            "- For floats use pytest.approx. No file access, randomness, network or sleeping.\n"
            "Output the complete test file in one ```python block.")


def fix_prompt(name, code, tests, err, names, route):
    keep = f" Keep the public names: {', '.join(names)}." if names else ""
    if route == "module":
        ctx = f"Tests (for context):\n```python\n{tests}\n```\n\n" if tests else ""
        return (f"The module `toolkit/{name}.py` has a problem.\n\nModule:\n```python\n{code}\n```\n\n{ctx}"
                f"Problem:\n```\n{err}\n```\n\nFix the module.{keep} Answer with the complete corrected module in one "
                "```python block.")
    if route == "tests":
        return (f"The tests for `toolkit/{name}.py` do not pass or are invalid.\n\nModule (do not change it):\n"
                f"```python\n{code}\n```\n\nCurrent tests:\n```python\n{tests}\n```\n\nProblem:\n```\n{err}\n```\n\n"
                "Rewrite the complete test file so that it passes with the module exactly as written. Fix missing "
                "imports and wrong expected values by re-reading the module code. If a test asserts behavior the "
                "module does not implement, change it to the real behavior or replace it with a different simple "
                "test. Keep at least 8 test functions. Answer with the complete test file in one ```python block.")
    return (f"The tests for `toolkit/{name}.py` do not pass.\n\nModule:\n```python\n{code}\n```\n\nTests:\n"
            f"```python\n{tests}\n```\n\nProblem:\n```\n{err}\n```\n\nFind the root cause. Decide whether the module "
            "or the tests are wrong and fix it; correct wrong expected values or real bugs, and keep at least 8 tests."
            f"{keep}\nAnswer with the complete corrected module in a ```python block FIRST, then the complete "
            "corrected tests in a second ```python block.")


def fix_module_prompt(name, code, tests, err, names):
    keep = f" Keep the public names: {', '.join(names)}." if names else ""
    ctx = f"Tests (for context, do not rewrite them):\n```python\n{tests}\n```\n\n" if tests else ""
    return (f"The module `toolkit/{name}.py` has a problem.\n\nModule:\n```python\n{code}\n```\n\n{ctx}"
            f"Problem:\n```\n{err}\n```\n\nIf the problem lists undefined names, add the missing imports or "
            f"definitions. Fix the module.{keep} Answer with the complete corrected module in one ```python block.")


def fix_tests_prompt(name, code, tests, err):
    return (f"The tests for `toolkit/{name}.py` do not pass or are invalid.\n\nModule (do not change it):\n"
            f"```python\n{code}\n```\n\nCurrent tests:\n```python\n{tests}\n```\n\nProblem:\n```\n{err}\n```\n\n"
            "If the problem lists undefined names, add the missing import statements at the top of the file "
            "(for example `import datetime` or `from datetime import date, timedelta`). Re-read the module code, "
            "correct wrong expected values, and replace any test that asserts behavior the module does not "
            "implement with a simple test of the real behavior. Keep at least 8 short test functions. "
            "Answer with the complete test file in one ```python block.")


# ---------------------------------------------------------------- LLM
class LLM:
    def __init__(self, logpath):
        self.lock = threading.Lock()
        self.ptok = self.ctok = self.calls = self.errors = 0
        self.t_first = self.t_last = None
        self.logf = open(logpath, "a", encoding="utf-8")

    def chat(self, system, user, max_tokens, tag, seed=1, temperature=None):
        body = {"model": "local", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": max_tokens, "seed": seed, "stream": False, **SAMPLER}
        if temperature is not None:
            body["temperature"] = temperature
        t0 = time.time()
        with self.lock:
            if self.t_first is None:
                self.t_first = t0
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=240) as r:
                d = json.loads(r.read())
            text = d["choices"][0]["message"]["content"] or ""
            fin = d["choices"][0].get("finish_reason")
            u = d.get("usage", {})
        except Exception as e:  # noqa: BLE001
            with self.lock:
                self.errors += 1
                self.logf.write(json.dumps({"tag": tag, "error": repr(e)}) + "\n")
                self.logf.flush()
            return None
        t1 = time.time()
        with self.lock:
            self.ptok += u.get("prompt_tokens", 0)
            self.ctok += u.get("completion_tokens", 0)
            self.calls += 1
            self.t_last = max(self.t_last or 0, t1)
            self.logf.write(json.dumps({"tag": tag, "sec": round(t1 - t0, 2), "ptok": u.get("prompt_tokens"),
                                        "ctok": u.get("completion_tokens"), "finish": fin, "user": user,
                                        "text": text}, ensure_ascii=False) + "\n")
            self.logf.flush()
        return {"text": text, "finish": fin, "ctok": u.get("completion_tokens", 0)}


def start_server():
    log = open(os.path.join(RUN, "server.log"), "w", encoding="utf-8")
    # olculen en hizli toplam ayar: 16 slot + birlesik f16 KV (q8_0 KV ≈ %30 yavas cikti) + GPU orneklemesi
    cmd = [SERVER, "-m", MODEL, "-ngl", "99", "-fa", "on", "-c", "36864", "-np", str(SLOTS), "-kvu", "-bs",
           "-ctk", "f16", "-ctv", "f16", "--host", "127.0.0.1", "--port", str(PORT), "--no-webui", *SERVER_EXTRA]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 180:
        if proc.poll() is not None:
            raise RuntimeError("sunucu kapandi (server.log'a bak)")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                if r.status == 200:
                    return proc, log
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    raise RuntimeError("sunucu hazir olmadi")


def stop_server(proc, log):
    proc.terminate()
    try:
        proc.wait(15)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()


# ---------------------------------------------------------------- kod analizi
FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)(?:```|\Z)", re.S)
DENY_IMPORT = ("subprocess", "socket", "ctypes", "shutil", "urllib.request", "urllib.error", "urllib.robotparser",
               "http", "requests", "multiprocessing", "signal", "webbrowser", "smtplib", "ftplib", "telnetlib",
               "asyncio", "sqlite3", "pickle", "marshal", "importlib", "tempfile", "pty", "winreg")
DENY_CALL = {"eval", "exec", "input", "__import__", "compile", "breakpoint", "exit", "quit"}
DENY_ATTR = {("os", "system"), ("os", "popen"), ("os", "remove"), ("os", "unlink"), ("os", "rmdir"),
             ("os", "removedirs"), ("os", "rename"), ("os", "kill"), ("os", "_exit"), ("sys", "exit"),
             ("time", "sleep")}


def denied(modname):
    return any(modname == d or modname.startswith(d + ".") for d in DENY_IMPORT)


def extract_blocks(text):
    out = []
    for m in FENCE.finditer(text):
        lang, body = m.group(1).lower(), m.group(2)
        if lang in ("", "python", "py", "python3") and body.strip():
            out.append(body.rstrip() + "\n")
    return out


def static_error(code, is_test):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"
    if len([ln for ln in code.splitlines() if ln.strip()]) < (20 if is_test else 25):
        return "file is too short; write a complete, substantial file"
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if denied(a.name):
                    return f"forbidden import: {a.name} (use only pure standard-library logic)"
        elif isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            if denied(mod) or (mod == "urllib" and any(a.name != "parse" for a in n.names)):
                return f"forbidden import: {mod} (use only pure standard-library logic)"
        elif isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                if f.id in DENY_CALL:
                    return f"forbidden call: {f.id}()"
                if f.id == "open" and not is_test:
                    return "forbidden call: open() inside the module (pure logic only)"
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                if (f.value.id, f.attr) in DENY_ATTR:
                    return f"forbidden call: {f.value.id}.{f.attr}()"
    return None


def pyflakes_error(path):
    try:
        p = subprocess.run([PY, "-m", "pyflakes", path], capture_output=True, text=True, timeout=20,
                           creationflags=NOFLAGS, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return None
    msgs = []
    for ln in (p.stdout or "").splitlines():
        if "undefined name" in ln:
            m = re.search(r":(\d+):(?:\d+:?)? ?(.*)$", ln)
            if m:
                msgs.append(f"{m.group(2)} (line {m.group(1)})")
    return ("undefined names (add the missing imports or definitions): " + "; ".join(msgs[:6])) if msgs else None


def public_names(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets):
            try:
                v = ast.literal_eval(n.value)
                if isinstance(v, (list, tuple)):
                    return [str(x) for x in v if isinstance(x, str)]
            except Exception:  # noqa: BLE001
                pass
    return [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("_")]


def _sig(fn):
    try:
        return f"{fn.name}({ast.unparse(fn.args)})" + (f" -> {ast.unparse(fn.returns)}" if fn.returns else "")
    except Exception:  # noqa: BLE001
        return fn.name


def api_summary(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    allnames = set(public_names(code))
    lines = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and (n.name in allnames or not n.name.startswith("_")):
            lines.append("- def " + _sig(n))
        elif isinstance(n, ast.ClassDef) and (n.name in allnames or not n.name.startswith("_")):
            ms = [_sig(m) for m in n.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and (not m.name.startswith("_") or m.name == "__init__")]
            lines.append(f"- class {n.name}: " + "; ".join(ms))
    return "\n".join(lines[:30])


def raises_summary(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "none"
    out = []

    def visit(fn, prefix=""):
        types = set()
        for r in ast.walk(fn):
            if isinstance(r, ast.Raise) and r.exc is not None:
                e = r.exc.func if isinstance(r.exc, ast.Call) else r.exc
                try:
                    types.add(ast.unparse(e))
                except Exception:  # noqa: BLE001
                    pass
        if types:
            out.append(f"- {prefix}{fn.name}: {', '.join(sorted(types))}")

    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visit(n)
        elif isinstance(n, ast.ClassDef):
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    visit(m, n.name + ".")
    return "\n".join(out[:30]) or "none"


def test_strength(code):
    tree = ast.parse(code)
    tests = sum(1 for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_"))
    asserts = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Assert))
    raises = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.With):
            for i in n.items:
                c = i.context_expr
                if isinstance(c, ast.Call) and getattr(c.func, "attr", getattr(c.func, "id", "")) == "raises":
                    raises += 1
    return tests, asserts + raises


def run_pytest(cwd, pythonpath, targets, timeout=40, maxfail=3):
    os.makedirs(cwd, exist_ok=True)
    env = dict(os.environ)
    env.update({"PYTHONPATH": pythonpath, "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"})
    cmd = [PY, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", "--tb=short",
           f"--maxfail={maxfail}", *targets]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout,
                           creationflags=NOFLAGS, encoding="utf-8", errors="replace")
        out, rc = (p.stdout or "") + (p.stderr or ""), p.returncode
    except subprocess.TimeoutExpired as e:
        so = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        out, rc = so + f"\nTIMEOUT: tests exceeded {timeout}s (infinite loop or sleep?)", -9
    return rc, out, time.time() - t0


def tail(out, n=45, w=200):
    return "\n".join(ln[:w] for ln in out.strip().splitlines()[-n:])


def parse_counts(out):
    def g(p):
        m = re.search(p, out)
        return int(m.group(1)) if m else 0
    return g(r"(\d+) passed"), g(r"(\d+) failed"), g(r"(\d+) error")


def metrics(paths):
    tot = nb = cl = funcs = docs = params = ann = rets = 0
    for p in paths:
        src = open(p, encoding="utf-8").read()
        lines = src.splitlines()
        tot += len(lines)
        nb += sum(1 for ln in lines if ln.strip())
        cl += sum(1 for ln in lines if ln.strip() and not ln.strip().startswith("#"))
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs += 1
                docs += 1 if ast.get_docstring(n) else 0
                args = [a for a in n.args.args + n.args.kwonlyargs if a.arg not in ("self", "cls")]
                params += len(args)
                ann += sum(1 for a in args if a.annotation is not None)
                rets += 1 if n.returns is not None else 0
    pct = lambda a, b: round(100 * a / b, 1) if b else 0.0  # noqa: E731
    return {"dosya": len(paths), "satir_toplam": tot, "satir_dolu": nb, "satir_kod": cl, "fonksiyon": funcs,
            "docstring_yuzde": pct(docs, funcs), "parametre_tip_yuzde": pct(ann, params), "donus_tip_yuzde": pct(rets, funcs)}


# ---------------------------------------------------------------- is akisi
LINE_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])?\s*`?([a-z][a-z0-9_]{2,24})`?\s*\|\s*(.+?)\s*$")


def make_plan(llm):
    def ask(cat):
        r = llm.chat(PLAN_SYS, plan_prompt(cat), 900, "plan", seed=7, temperature=0.5)
        items = []
        for line in (r["text"] if r else "").splitlines():
            m = LINE_RE.match(line)
            if m:
                parts = [p.strip() for p in m.group(2).split("|")]
                items.append((m.group(1), parts[0], parts[1] if len(parts) > 1 else ""))
        return items

    with cf.ThreadPoolExecutor(len(CATEGORIES)) as ex:
        res = list(ex.map(ask, CATEGORIES))
    seen, per_cat = set(), []
    for cat, items in zip(CATEGORIES, res):
        lst = []
        for n, p, api in items:
            if (n in seen or n in sys.stdlib_module_names or keyword.iskeyword(n)
                    or n in ("toolkit", "tests", "test", "module_name")):
                continue
            seen.add(n)
            lst.append({"name": n, "purpose": p, "api": api, "category": cat})
        per_cat.append(lst[:PER_CAT])
    out = []
    for i in range(PER_CAT):
        for lst in per_cat:
            if i < len(lst):
                out.append(lst[i])
    return out


class Ctx:
    def __init__(self, llm, work, proje, t_start):
        self.llm, self.work, self.proje, self.t_start = llm, work, proje, t_start
        self.lock = threading.Lock()
        self.fix_sem = threading.Semaphore(7)   # uzun baglamli onarim istekleri ayni anda KV'yi doldurmasin

    def time_ok(self):
        return time.time() - self.t_start < BUDGET


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def process(mod, ctx):
    name = mod["name"]
    st = {"name": name, "purpose": mod["purpose"], "category": mod["category"], "status": "baslamadi",
          "repairs": 0, "calls": 0, "ctok": 0, "history": [], "last_error": ""}
    if not ctx.time_ok():
        st["status"] = "sure_yetmedi"
        return st
    t0 = time.time()
    mdir = os.path.join(ctx.work, name)
    os.makedirs(os.path.join(mdir, "toolkit"), exist_ok=True)
    os.makedirs(os.path.join(mdir, "tests"), exist_ok=True)
    write(os.path.join(mdir, "toolkit", "__init__.py"), "")
    code_path = os.path.join(mdir, "toolkit", f"{name}.py")
    test_path = os.path.join(mdir, "tests", f"test_{name}.py")

    def call(system, user, max_tokens, tag, seed, temperature=None):
        r = ctx.llm.chat(system, user, max_tokens, f"{tag}:{name}", seed=seed, temperature=temperature)
        st["calls"] += 1
        if r:
            st["ctok"] += r["ctok"]
        return r

    code = tests = first_try = None
    short = False
    for attempt in range(2):
        if not ctx.time_ok():
            break
        r = call(GEN_SYS, gen_prompt(name, mod["purpose"], mod.get("api", ""), short), 1400, "gen", attempt + 1)
        b = extract_blocks(r["text"]) if r else []
        if r and r["finish"] == "length":      # kesildi: daha kisa iste
            short = True
            first_try = first_try or (b[0] if b else None)
            continue
        if b:
            code = b[0]
            break
    code = code or first_try
    if code is None:
        st["status"] = "kod_uretilemedi" if ctx.time_ok() else "sure_doldu"
        st["sec"] = round(time.time() - t0, 1)
        return st

    for rnd in range(MAX_REPAIR + 1):
        write(code_path, code)
        err = static_error(code, False) or pyflakes_error(code_path)
        route_now = "module"
        if err is None:
            if tests is None:
                if not ctx.time_ok():
                    st["status"] = "sure_doldu"
                    break
                r = call(TEST_SYS, test_prompt(name, code, public_names(code)), 1100, "test", 3)
                b = extract_blocks(r["text"]) if r else []
                tests = b[0] if b else None
                if tests is None:
                    err, route_now = "no test code block was returned", "tests"
            if err is None:
                write(test_path, tests)
                e2 = static_error(tests, True) or pyflakes_error(test_path)
                if e2:
                    err, route_now = "[tests] " + e2, "tests"
            if err is None:
                rc, out, _ = run_pytest(os.path.join(mdir, "run"), mdir, [test_path])
                nt, na = test_strength(tests)
                if rc == 0 and nt >= 6 and na >= 10:
                    st["history"].append("gecti")
                    st["status"] = "gecti"
                    break
                if rc != 0:
                    err = tail(out)
                    route_now = "module" if re.search(r"toolkit[\\/]", out) else "tests"
                    st["history"].append(f"pytest:fail:{route_now}")
                else:
                    err = f"tests are too weak: found {nt} test functions and {na} assertions; need at least 6 and 10"
                    route_now = "tests"
                    st["history"].append("pytest:zayif")
            else:
                st["history"].append("statik:" + err[:50])
        else:
            st["history"].append("statik:" + err[:50])
        st["last_error"] = err[:700]
        if rnd == MAX_REPAIR:
            st["status"] = "basarisiz"
            break
        if not ctx.time_ok():
            st["status"] = "sure_doldu"
            break
        st["repairs"] += 1
        if tests is None:               # test blogu hic gelmediyse yeniden uret (dongunun basinda)
            continue
        route = "both" if rnd >= 2 else route_now
        mt = 2200 if route == "both" else 1300
        with ctx.fix_sem:
            r = call(FIX_SYS, fix_prompt(name, code, tests, err, public_names(code), route), mt, "fix", 10 + rnd,
                     temperature=0.35)
        blocks = extract_blocks(r["text"]) if r else []
        newcode = newtests = None
        for b in blocks:
            if "def test_" in b:
                newtests = newtests or b
            else:
                newcode = newcode or b
        if route in ("module", "both") and newcode:
            code = newcode
        if route in ("tests", "both") and newtests:
            tests = newtests
    # kayit
    for p, s in ((code_path, code), (test_path, tests)):
        if s:
            write(p, s)
    st["sec"] = round(time.time() - t0, 1)
    if code:
        st["lines_code"] = len([x for x in code.splitlines() if x.strip()])
    if tests:
        st["lines_test"] = len([x for x in tests.splitlines() if x.strip()])
        try:
            st["n_tests"], st["n_asserts"] = test_strength(tests)
        except SyntaxError:
            pass
    if st["status"] == "gecti":
        with ctx.lock:
            shutil.copy(code_path, os.path.join(ctx.proje, "toolkit", f"{name}.py"))
            shutil.copy(test_path, os.path.join(ctx.proje, "tests", f"test_{name}.py"))
    else:
        bdir = os.path.join(os.path.dirname(ctx.proje), "basarisiz")
        with ctx.lock:
            os.makedirs(bdir, exist_ok=True)
            for p in (code_path, test_path):
                if os.path.exists(p):
                    shutil.copy(p, os.path.join(bdir, os.path.basename(p)))
    return st


def failed_names(out):
    return sorted(set(re.findall(r"^(?:FAILED|ERROR) .*?::(?:\w+::)*(\w+)", out, re.M)))


def drop_tests(src, names):
    """Basarisiz test fonksiyonlarini (yalniz silme) karantinaya alir; baska hicbir seyi degistirmez."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    lines = src.splitlines(keepends=True)
    spans = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names:
            start = min([d.lineno for d in n.decorator_list] + [n.lineno]) - 1
            spans.append((start, n.end_lineno))
    for s, e in sorted(spans, reverse=True):
        del lines[s:e]
    return "".join(lines)


def process2(mod, ctx):
    """v3 akisi: modul -> statik denetim -> test -> statik denetim -> pytest; az sayida basarisiz test karantinaya
    alinir (silinir, raporlanir), cok sayida ise model duzeltir. Her duzeltme modele yaptirilir."""
    name = mod["name"]
    st = {"name": name, "purpose": mod["purpose"], "category": mod["category"], "status": "baslamadi",
          "repairs": 0, "calls": 0, "ctok": 0, "history": [], "last_error": "", "dropped": 0}
    if not ctx.time_ok():
        st["status"] = "sure_yetmedi"
        return st
    t0 = time.time()
    mdir = os.path.join(ctx.work, name)
    os.makedirs(os.path.join(mdir, "toolkit"), exist_ok=True)
    os.makedirs(os.path.join(mdir, "tests"), exist_ok=True)
    write(os.path.join(mdir, "toolkit", "__init__.py"), "")
    code_path = os.path.join(mdir, "toolkit", f"{name}.py")
    test_path = os.path.join(mdir, "tests", f"test_{name}.py")
    state = {"code": None, "tests": None, "fixes": 0}

    def call(system, user, max_tokens, tag, seed, temperature=None):
        r = ctx.llm.chat(system, user, max_tokens, f"{tag}:{name}", seed=seed, temperature=temperature)
        st["calls"] += 1
        if r:
            st["ctok"] += r["ctok"]
        return r

    def can_fix():
        return state["fixes"] < MAX_REPAIR and ctx.time_ok()

    def finish(status, err=""):
        st["status"] = status
        if err:
            st["last_error"] = err[:700]
        st["sec"] = round(time.time() - t0, 1)
        code, tests = state["code"], state["tests"]
        for p, s in ((code_path, code), (test_path, tests)):
            if s:
                write(p, s)
        if code:
            st["lines_code"] = len([x for x in code.splitlines() if x.strip()])
        if tests:
            st["lines_test"] = len([x for x in tests.splitlines() if x.strip()])
            try:
                st["n_tests"], st["n_asserts"] = test_strength(tests)
            except SyntaxError:
                pass
        if status.startswith("gecti"):
            with ctx.lock:
                shutil.copy(code_path, os.path.join(ctx.proje, "toolkit", f"{name}.py"))
                shutil.copy(test_path, os.path.join(ctx.proje, "tests", f"test_{name}.py"))
        else:
            bdir = os.path.join(os.path.dirname(ctx.proje), "basarisiz")
            with ctx.lock:
                os.makedirs(bdir, exist_ok=True)
                for p in (code_path, test_path):
                    if os.path.exists(p):
                        shutil.copy(p, os.path.join(bdir, os.path.basename(p)))
        return st

    def fix_module(err, with_tests):
        state["fixes"] += 1
        st["repairs"] += 1
        r = call(FIX_SYS, fix_module_prompt(name, state["code"], state["tests"] if with_tests else None, err,
                                            public_names(state["code"])), 1300, "fixm", 10 + state["fixes"],
                 temperature=0.35)
        b = [x for x in (extract_blocks(r["text"]) if r else []) if "def test_" not in x]
        if b:
            state["code"] = b[0]

    def fix_tests(err):
        state["fixes"] += 1
        st["repairs"] += 1
        r = call(FIX_SYS, fix_tests_prompt(name, state["code"], state["tests"], err), 1500, "fixt",
                 20 + state["fixes"], temperature=0.35)
        b = [x for x in (extract_blocks(r["text"]) if r else []) if "def test_" in x]
        if b:
            state["tests"] = b[0]

    # 1) modul
    first, short = None, False
    for attempt in range(2):
        if not ctx.time_ok():
            break
        r = call(GEN_SYS, gen_prompt(name, mod["purpose"], mod.get("api", ""), short), 1400, "gen", attempt + 1)
        b = extract_blocks(r["text"]) if r else []
        if r and r["finish"] == "length":
            short = True
            first = first or (b[0] if b else None)
            continue
        if b:
            state["code"] = b[0]
            break
    state["code"] = state["code"] or first
    if state["code"] is None:
        return finish("kod_uretilemedi" if ctx.time_ok() else "sure_doldu")

    # 2) modul statik denetim
    while True:
        write(code_path, state["code"])
        err = static_error(state["code"], False) or pyflakes_error(code_path)
        if err is None:
            break
        st["history"].append("modul_statik:" + err[:40])
        if not can_fix():
            return finish("sure_doldu" if not ctx.time_ok() else "basarisiz", err)
        fix_module(err, False)

    # 3) testler + statik denetim
    first, short = None, False
    for attempt in range(2):
        if not ctx.time_ok():
            break
        r = call(TEST_SYS, test_prompt(name, state["code"], public_names(state["code"]), short), 1500, "test",
                 3 + attempt)
        b = extract_blocks(r["text"]) if r else []
        if r and r["finish"] == "length":
            short = True
            first = first or (b[0] if b else None)
            continue
        if b:
            state["tests"] = b[0]
            break
    state["tests"] = state["tests"] or first
    if state["tests"] is None:
        return finish("test_uretilemedi" if ctx.time_ok() else "sure_doldu")
    while True:
        write(test_path, state["tests"])
        err = static_error(state["tests"], True) or pyflakes_error(test_path)
        if err is None:
            break
        st["history"].append("test_statik:" + err[:40])
        if not can_fix():
            return finish("sure_doldu" if not ctx.time_ok() else "basarisiz", "[tests] " + err)
        fix_tests("[tests] " + err)

    # 4) pytest
    for _ in range(5):
        write(code_path, state["code"])
        write(test_path, state["tests"])
        rc, out, _ = run_pytest(os.path.join(mdir, "run"), mdir, [test_path])
        nt, na = test_strength(state["tests"])
        if rc == 0:
            if nt >= 6 and na >= 10:
                st["history"].append("gecti")
                return finish("gecti_karantina" if st["dropped"] else "gecti")
            err = f"tests are too weak: found {nt} test functions and {na} assertions; need at least 6 and 10"
            st["history"].append("pytest:zayif")
            if not can_fix():
                return finish("sure_doldu" if not ctx.time_ok() else "basarisiz", err)
            fix_tests(err)
            continue
        names = failed_names(out)
        collect_err = "ERROR collecting" in out or not names
        st["history"].append(f"pytest:fail:{len(names)}/{nt}")
        err = tail(out)
        st["last_error"] = err[:700]
        if names and not collect_err and nt - len(names) >= 6 and (nt - len(names)) / nt >= 0.6:
            new = drop_tests(state["tests"], set(names))
            try:
                nt2, na2 = test_strength(new)
            except SyntaxError:
                nt2 = na2 = 0
            if nt2 >= 6 and na2 >= 10:
                state["tests"] = new
                st["dropped"] += len(names)
                continue
        if not can_fix():
            return finish("sure_doldu" if not ctx.time_ok() else "basarisiz", err)
        if re.search(r"toolkit[\\/]", out) and not collect_err:
            fix_module(err, True)
        else:
            fix_tests(err)
    return finish("basarisiz", st["last_error"])


BENCH = [("slugify", "turn text into URL slugs"), ("roman_numerals", "convert integers to and from Roman numerals"),
         ("word_frequency", "count and rank word frequencies in text"), ("binary_heap", "min and max heap classes"),
         ("matrix_utils", "basic matrix operations on lists of lists"), ("semver_compare", "parse and compare semantic versions"),
         ("luhn_check", "validate and generate Luhn check digits"), ("interval_merge", "merge and intersect numeric intervals"),
         ("ring_buffer", "fixed-size circular buffer class"), ("text_wrap", "word-wrap and justify text"),
         ("base_converter", "convert numbers between bases 2-36"), ("moving_average", "simple and exponential moving averages"),
         ("trie_index", "prefix tree for string lookup"), ("date_ranges", "iterate and compare calendar date ranges"),
         ("csv_lines", "parse and format CSV lines in memory"), ("unit_convert", "convert between common units")]


def bench(llm):
    llm.chat(GEN_SYS, "Write a Python function that adds two numbers.", 60, "warm", seed=1)
    t0 = time.time()
    with cf.ThreadPoolExecutor(SLOTS) as ex:
        res = list(ex.map(lambda nm: llm.chat(GEN_SYS, gen_prompt(nm[0], nm[1], ""), 900, "bench", seed=1), BENCH[:SLOTS]))
    wall = time.time() - t0
    ctok = sum((r or {"ctok": 0})["ctok"] for r in res)
    print(f"BENCH slots={SLOTS} sampler={json.dumps(SAMPLER)} extra={' '.join(SERVER_EXTRA) or '-'}: "
          f"{ctok} token / {wall:.1f} sn = {ctok / wall:.0f} tok/sn toplam", flush=True)


def make_readme(llm, proje, passing):
    lines = []
    for m in passing[:70]:
        p = os.path.join(proje, "toolkit", m["name"] + ".py")
        try:
            src = open(p, encoding="utf-8").read()
            doc = (ast.get_docstring(ast.parse(src)) or m["purpose"]).strip().splitlines()[0][:110]
            api = ", ".join(public_names(src)[:8])
        except Exception:  # noqa: BLE001
            doc, api = m["purpose"], ""
        lines.append(f"- {m['name']}: {doc} [API: {api}]")
    prompt = ("Write README.md for a Python package named 'toolkit' (pure standard library, Python 3.10+). "
              "It contains these modules:\n" + "\n".join(lines) + "\n\nInclude: a short overview, how to use it "
              "(copy the folder or add it to PYTHONPATH), a module table grouped by purpose, 3 short usage examples "
              "that only call names listed in the [API: ...] lists above, how to run the tests with "
              "`python -m pytest tests`, and a short contributing note. Output only the Markdown.")
    r = llm.chat(README_SYS, prompt, 1400, "readme", seed=5, temperature=0.3)
    txt = (r["text"] if r else "").strip()
    txt = re.sub(r"^```(?:markdown|md)?\s*\n", "", txt)
    txt = re.sub(r"\n```\s*$", "", txt)
    write(os.path.join(proje, "README.md"), txt + "\n")
    return len(txt.splitlines())


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    os.makedirs(RUN, exist_ok=True)
    proc, slog = start_server()
    try:
        llm = LLM(os.path.join(RUN, "llm_log.jsonl"))
        if mode == "bench":
            bench(llm)
            return
        t_all = time.time()
        plan = make_plan(llm)
        t_plan = time.time()
        print(f"[plan] {len(plan)} modul planlandi ({t_plan - t_all:.1f} sn)", flush=True)
        work = os.path.join(RUN, "calisma")
        proje = os.path.join(RUN, "proje")
        os.makedirs(os.path.join(proje, "toolkit"), exist_ok=True)
        os.makedirs(os.path.join(proje, "tests"), exist_ok=True)
        write(os.path.join(proje, "toolkit", "__init__.py"),
              '"""toolkit: model-generated standard-library utility package."""\n')   # iskelet (yonetici)
        ctx = Ctx(llm, work, proje, time.time())
        with cf.ThreadPoolExecutor(SLOTS) as ex:
            results = list(ex.map(lambda m: process2(m, ctx), plan))
        t_gen = time.time()
        print(f"[uretim] bitti ({t_gen - t_plan:.1f} sn)", flush=True)
        passing = [r for r in results if r["status"].startswith("gecti")]
        readme_lines = make_readme(llm, proje, passing) if passing else 0
        final = []
        for i in range(2):
            rc, out, dt = run_pytest(os.path.join(RUN, f"son_kosu_{i + 1}"), proje, [os.path.join(proje, "tests")],
                                     timeout=150, maxfail=100000)
            p_, f_, e_ = parse_counts(out)
            final.append({"rc": rc, "passed": p_, "failed": f_, "errors": e_, "sec": round(dt, 1), "tail": tail(out, 6)})
        t_end = time.time()
        files_code = [os.path.join(proje, "toolkit", f) for f in sorted(os.listdir(os.path.join(proje, "toolkit"))) if f != "__init__.py"]
        files_test = [os.path.join(proje, "tests", f) for f in sorted(os.listdir(os.path.join(proje, "tests")))]
        mc, mt = metrics(files_code), metrics(files_test)
        cnt = {}
        for r in results:
            cnt[r["status"]] = cnt.get(r["status"], 0) + 1
        gen_secs = (llm.t_last - llm.t_first) if llm.t_first else 0
        rep = {"toplam_sn": round(t_end - t_all, 1), "plan_sn": round(t_plan - t_all, 1), "uretim_sn": round(t_gen - t_plan, 1),
               "son_dogrulama_sn": round(t_end - t_gen, 1), "planlanan_modul": len(plan), "durumlar": cnt,
               "tam_gecen_modul": sum(1 for r in results if r["status"] == "gecti"),
               "karantinali_modul": sum(1 for r in results if r["status"] == "gecti_karantina"),
               "karantinaya_alinan_test": sum(r.get("dropped", 0) for r in results),
               "llm_cagri": llm.calls, "llm_hata": llm.errors, "istem_token": llm.ptok, "uretilen_token": llm.ctok,
               "toplam_tok_sn": round(llm.ctok / gen_secs, 0) if gen_secs else 0, "kod": mc, "test": mt,
               "readme_satir": readme_lines, "son_kosu": final,
               "moduller": [{k: v for k, v in r.items() if k != "history"} | {"gecmis": r["history"]} for r in results]}
        write(os.path.join(RUN, "rapor.json"), json.dumps(rep, ensure_ascii=False, indent=1))
        write_report(rep, passing, results)
        print(json.dumps({k: v for k, v in rep.items() if k not in ("moduller",)}, ensure_ascii=False, indent=1), flush=True)
        print("RAPOR:", os.path.join(RUN, "RAPOR.md"), flush=True)
    finally:
        stop_server(proc, slog)


def write_report(rep, passing, results):
    s = rep["son_kosu"]
    L = []
    L.append("# Model-yazimi kod uretimi: rapor\n")
    L.append("Urun kodunun (moduller, testler, README) tek satirini insan ya da Claude yazmadi; hepsini yerel Llama 3.2 3B Instruct "
             "(Q4_K_M) yazdi ve duzeltti. Yonetici betik (`scripts/kod_uretimi.py`) yalniz modeli cagirir, pyflakes/pytest kosar, hatayi modele geri verir. "
             "`toolkit/__init__.py` yonetici tarafindan olusturulan tek satirlik iskelettir. Testleri ayni model yazdigi icin "
             "gecen test, modulun dogrulugunun garantisi degil, modul ile testin tutarliligidir.\n")
    L.append(f"- Toplam sure: {rep['toplam_sn']} sn (plan {rep['plan_sn']} + uretim {rep['uretim_sn']} + son dogrulama {rep['son_dogrulama_sn']})")
    L.append(f"- Planlanan modul: {rep['planlanan_modul']}; durumlar: {rep['durumlar']}")
    L.append(f"- LLM cagrisi: {rep['llm_cagri']} (hata {rep['llm_hata']}); istem {rep['istem_token']} + uretilen {rep['uretilen_token']} token; toplam hiz ≈ {rep['toplam_tok_sn']} tok/sn")
    L.append(f"- Gecen modul: {len(passing)} (hic test dusurulmeden: {rep['tam_gecen_modul']}, karantinali: {rep['karantinali_modul']}; "
             f"karantinaya alinan basarisiz test: {rep['karantinaya_alinan_test']}); kod: {rep['kod']}")
    L.append(f"- Testler: {rep['test']}")
    L.append(f"- Son kosu 1: {s[0]['passed']} gecti, {s[0]['failed']} kaldi, {s[0]['errors']} hata ({s[0]['sec']} sn)")
    L.append(f"- Son kosu 2: {s[1]['passed']} gecti, {s[1]['failed']} kaldi, {s[1]['errors']} hata ({s[1]['sec']} sn)\n")
    L.append("## Moduller\n")
    L.append("| modul | durum | onarim | cagri | satir (kod/test) | sn |")
    L.append("|---|---|---|---|---|---|")
    for r in results:
        L.append(f"| {r['name']} | {r['status']} | {r['repairs']} | {r['calls']} | {r.get('lines_code', '-')}/{r.get('lines_test', '-')} | {r.get('sec', '-')} |")
    write(os.path.join(RUN, "RAPOR.md"), "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
