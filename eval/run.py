#!/usr/bin/env python3
"""Coding eval (HumanEval / MBPP-sanitized) against any OpenAI-compatible server. pass@1, 1 sample per problem."""
import argparse, json, re, subprocess, sys, tempfile, time, os
from concurrent.futures import ThreadPoolExecutor
from urllib import request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--model", default="x")
ap.add_argument("--name", required=True)
ap.add_argument("--bench", choices=["humaneval", "mbpp"], required=True)
ap.add_argument("--n", type=int, default=0, help="first N problems (0 = all)")
ap.add_argument("--max-tokens", type=int, default=8192)
ap.add_argument("--conc", type=int, default=1)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top-p", type=float, default=1.0)
ap.add_argument("--top-k", type=int, default=None)
ap.add_argument("--extra", default="{}", help="extra JSON merged into the request body")
a = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(here, a.bench + ".jsonl"))]
if a.n: rows = rows[:a.n]


def make_prompt(r):
    if a.bench == "humaneval":
        return ("Complete the following Python function. Reply with the complete function (including any needed imports) "
                "in a single ```python code block and nothing else after it.\n\n```python\n%s\n```" % r["prompt"])
    return ("%s\nYour code should pass this test:\n%s\nReply with the complete solution (including any needed imports) "
            "in a single ```python code block and nothing else after it." % (r["prompt"], r["test_list"][0]))


def chat(prompt):
    body = dict(model=a.model, messages=[{"role": "user", "content": prompt}], max_tokens=a.max_tokens,
                temperature=a.temperature, top_p=a.top_p)
    if a.top_k is not None: body["top_k"] = a.top_k
    body.update(json.loads(a.extra))
    req = request.Request(a.url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                          {"Content-Type": "application/json", "Authorization": "Bearer x"})
    t = time.time()
    with request.urlopen(req, timeout=3600) as f:
        j = json.load(f)
    dt = time.time() - t
    c = j["choices"][0]
    m = c["message"]
    return dict(content=m.get("content") or "", reasoning=m.get("reasoning_content") or m.get("reasoning") or "",
                finish=c.get("finish_reason"), tokens=j.get("usage", {}).get("completion_tokens", 0), secs=dt)


def extract(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else ""


def run_py(src, timeout=20):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); p = f.name
    try:
        r = subprocess.run([sys.executable, "-I", p], capture_output=True, timeout=timeout, text=True)
        return r.returncode == 0, r.stderr[-300:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(p)


def one(r):
    try:
        g = chat(make_prompt(r))
    except Exception as e:
        return dict(task_id=r["task_id"], ok=False, err="request: %s" % e, tokens=0, secs=0, finish="error")
    code = extract(g["content"])
    if a.bench == "humaneval":
        imports = "\n".join(l for l in r["prompt"].splitlines() if re.match(r"\s*(import|from)\s", l))
        if ("def %s" % r["entry_point"]) not in code:
            code = r["prompt"] + code
        src = "%s\n%s\n\n%s\n\ncheck(%s)\n" % (imports, code, r["test"], r["entry_point"])
    else:
        src = "%s\n%s\n\n%s\n" % ("\n".join(r.get("test_imports", [])), code, "\n".join(r["test_list"]))
    ok, err = run_py(src) if code.strip() else (False, "no code block")
    return dict(task_id=r["task_id"], ok=ok, err="" if ok else err, tokens=g["tokens"], secs=g["secs"], finish=g["finish"],
                reasoning_chars=len(g["reasoning"]), content=g["content"][-1500:])


t0 = time.time()
with ThreadPoolExecutor(a.conc) as ex:
    res = []
    for i, x in enumerate(ex.map(one, rows)):
        res.append(x)
        print("%3d/%d %-22s %s %5d tok %6.1fs %s" % (i + 1, len(rows), x["task_id"], "PASS" if x["ok"] else "FAIL",
              x["tokens"], x["secs"], x["finish"]), flush=True)
n = len(res); p = sum(x["ok"] for x in res)
tok = sum(x["tokens"] for x in res)
summ = dict(name=a.name, bench=a.bench, n=n, passed=p, pass_at_1=round(100 * p / n, 2), wall_s=round(time.time() - t0, 1),
            total_tokens=tok, mean_tokens=round(tok / n), truncated=sum(x["finish"] == "length" for x in res),
            errors=sum(x["finish"] == "error" for x in res), conc=a.conc, temperature=a.temperature, max_tokens=a.max_tokens)
print(json.dumps(summ))
json.dump(dict(summary=summ, results=res), open(os.path.join(here, "out_%s_%s.json" % (a.name, a.bench)), "w"), indent=1)
