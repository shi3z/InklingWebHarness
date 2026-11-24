#!/usr/bin/env python3
"""Web console for yharness (standard library only): dashboard + give tasks to the harness + start/stop engines + start eval runs.

SECURITY - read this: the console can make this computer run Python code (model-written code, and tests you type). The sandbox in
yharness.py is a soft guard (temp dir, rlimits, scrubbed env, socket guard); user namespaces are not available here, so it is NOT a
security boundary: code can read your files. Therefore:
  * a random token is required for everything (data/web_token, mode 600); it is checked on every request,
  * it binds to this machine's tailnet IPv4 only, never 0.0.0.0,
  * POST needs the token cookie (SameSite=Strict) and a same-origin Origin header,
  * engines can only be the fixed presets below; no arbitrary commands or URLs are accepted,
  * one job at a time; --no-exec turns off everything that runs code (the dashboard and chat-without-verification stay).
"""
import argparse, glob, hmac, html, json, os, re, secrets, signal, socket, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request as urlreq
from urllib.parse import urlparse, parse_qs, unquote, quote

import yharness as Y

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DATA = Y.DATA; EVAL = Y.EVAL
JOBS_DIR = os.path.join(DATA, "jobs"); os.makedirs(JOBS_DIR, exist_ok=True)
SAFE = re.compile(r"^[\w.\-]+$"); e = html.escape
NO_EXEC = False
DEFAULT_CTX = 64000      # total context limit per worker (the paper tested 16K-48K and found it enough; 64K leaves room for a verbose, thinking model)
SANDBOX = False        # the console runs code WITHOUT the sandbox unless started with --sandbox

PRESETS = {
    "inkling": dict(label="Inkling-2-Mini Q4_K_M (llama.cpp)", port=8767, model="inkling",
                    cmd=[ROOT + "/llama.cpp-inkling/build/bin/llama-server", "-m", ROOT + "/gguf/Inkling-2-Mini-UD-Q4_K_M.gguf", "--jinja", "-ngl", "99",
                         "-c", "262144", "-np", "4", "--port", "8767", "--temp", "1.0", "--top-p", "1.0", "--top-k", "0", "--min-p", "0.0"],
                    env={"LD_LIBRARY_PATH": ROOT + "/llama.cpp-inkling/build/bin"},
                    sampling=dict(temperature=1.0, top_p=1.0, extra={"top_k": 0})),
}
PROCS = {}             # preset -> Popen started by this console
JOBS = {}              # id -> dict
LOCK = threading.Lock()
MAX_PARALLEL = 4                        # = llama-server slots (-np); more jobs wait in a queue
SEM = threading.BoundedSemaphore(MAX_PARALLEL)


def token():
    p = os.path.join(DATA, "web_token")
    if not os.path.exists(p):
        fd = os.open(p, os.O_WRONLY | os.O_CREAT, 0o600); os.write(fd, secrets.token_hex(16).encode()); os.close(fd)
    return open(p).read().strip()


def healthy(name):
    try:
        with urlreq.urlopen("http://127.0.0.1:%d/health" % PRESETS[name]["port"], timeout=2) as f: return f.status == 200
    except Exception: return False


def engine_state():
    return {n: dict(label=p["label"], up=healthy(n), mine=n in PROCS and PROCS[n].poll() is None) for n, p in PRESETS.items()}


def start_engine(name):
    if name not in PRESETS: return "unknown engine"
    for other in PRESETS:                                     # one GPU: never two engines at once
        if other != name and healthy(other): return "stop '%s' first (one GPU)" % other
    if healthy(name): return "already running"
    p = PRESETS[name]
    log = open(os.path.join(DATA, "engine_%s.log" % name), "w")
    PROCS[name] = subprocess.Popen(p["cmd"], env=dict(os.environ, **p["env"]), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return "starting (model load takes a while; watch the status)"


def stop_engine(name):
    pr = PROCS.get(name)
    if pr and pr.poll() is None: os.killpg(pr.pid, signal.SIGTERM); return "stopping"
    # the console was restarted and lost its record: stop whatever listens on the preset's port, but only if it is our engine binary
    out = subprocess.run(["ss", "-ltnp", "sport = :%d" % PRESETS[name]["port"]], capture_output=True, text=True).stdout
    m = re.search(r"pid=(\d+)", out)
    if not m: return "not running"
    cmd = open("/proc/%s/cmdline" % m.group(1), "rb").read().decode(errors="replace")
    if PRESETS[name]["cmd"][1] not in cmd and PRESETS[name]["cmd"][0] not in cmd: return "port %d is used by something that is not this engine; not touching it" % PRESETS[name]["port"]
    os.kill(int(m.group(1)), signal.SIGTERM); return "stopping (found by port)"


def engine_for(name, max_tokens=8192, effort=None):
    p = PRESETS[name]; extra = dict(p["sampling"]["extra"])
    if effort in ("low", "medium", "high"): extra["chat_template_kwargs"] = {"reasoning_effort": effort}     # how long the model thinks (the template's "Thinking effort level")
    return Y.Engine("http://127.0.0.1:%d/v1" % p["port"], p["model"], p["sampling"]["temperature"], p["sampling"]["top_p"], max_tokens, extra)


def engine_ctx():
    """(context per slot, whether the slots SHARE one KV cache). With `-np 4 -c 262144` every slot owns 65536 tokens (not shared); an engine started
    with the automatic slot count shares one cache among all slots. (0, False) if unknown."""
    try:
        with urlreq.urlopen("http://127.0.0.1:%d/props" % PRESETS["inkling"]["port"], timeout=3) as f:
            n = int(json.load(f).get("default_generation_settings", {}).get("n_ctx", 0))
        head = open(os.path.join(DATA, "engine_inkling.log"), errors="replace").read(20000)
        return n, "kv_unified = 'true'" in head
    except Exception: return 0, False


# ------------------------------------------------------------------------------------------------ jobs
MEDIA_KIND = {".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image", ".webp": "image", ".bmp": "image",
              ".mp4": "video", ".webm": "video", ".mov": "video", ".m4v": "video", ".mkv": "video"}
MEDIA_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
              ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime", ".m4v": "video/mp4", ".mkv": "video/x-matroska"}   # no .svg on purpose
SKIP_DIRS = {"venv", ".venv", "site-packages", "node_modules", ".git", "__pycache__", ".cache"}


def scan_media(base, limit=5000):
    """Every image/video under `base` (not inside venvs and the like): {relative path: (size, mtime, kind)}."""
    out = {}
    for dp, dns, fns in os.walk(base):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            k = MEDIA_KIND.get(os.path.splitext(fn)[1].lower())
            if not k: continue
            p = os.path.join(dp, fn)
            try: st = os.stat(p)
            except OSError: continue
            out[os.path.relpath(p, base)] = (st.st_size, st.st_mtime, k)
            if len(out) >= limit: return out
    return out


PATH_TOKEN = re.compile(r"(?:~|/)[^\s'\"|;&<>()$`=,]+")
HOME = os.path.expanduser("~")


def register(job, real):
    """Only files the harness has registered can be fetched through /m/<job>/<n>/<name> (never an arbitrary path)."""
    idx = job.setdefault("mindex", {}); fl = job.setdefault("mfiles", [])
    if real not in idx: idx[real] = len(fl); fl.append(real)
    return idx[real]


def shown(job, real, st, kind):
    seen = job.setdefault("media_seen", {})
    if seen.get(real) == (st.st_size, st.st_mtime): return None
    seen[real] = (st.st_size, st.st_mtime)
    base = os.path.realpath(workdir(job)); disp = os.path.relpath(real, base) if real.startswith(base + os.sep) else ("~/" + os.path.relpath(real, HOME) if real.startswith(HOME + os.sep) else real)
    return dict(path=disp, kind=kind, size=st.st_size, url="/m/%s/%d/%s" % (job["id"], register(job, real), quote(os.path.basename(real))), t=st.st_mtime)


def media_check(job):
    """After a command: announce the images/videos that appeared or that the command explicitly used - in the task folder AND wherever the
    command's own paths point (absolute or ~ paths, or the folder an output file template lives in)."""
    found = []; t0 = job.get("cmd_t0", job["t"]) - 1.0; base = workdir(job)
    if os.path.isdir(base):
        for r, (sz, mt, k) in scan_media(base).items():
            real = os.path.realpath(os.path.join(base, r)); it = shown(job, real, os.stat(real), k)
            if it: found.append(it)
    for tok in PATH_TOKEN.findall(job.get("last_cmd", "")):
        p = os.path.realpath(os.path.expanduser(tok.rstrip(".,:)")))
        if not p.startswith(HOME + os.sep) or (p.count(os.sep) < 3 and os.path.isdir(p)): continue          # never walk $HOME itself, "/" or the like
        if not os.path.exists(p): p = os.path.dirname(p)
        if os.path.isfile(p):
            k = MEDIA_KIND.get(os.path.splitext(p)[1].lower())
            if k: it = shown(job, p, os.stat(p), k); found += [it] if it else []
        elif os.path.isdir(p) and not any(part in SKIP_DIRS for part in p.split(os.sep)):
            for r, (sz, mt, k) in scan_media(p, 400).items():
                real = os.path.realpath(os.path.join(p, r))
                if mt >= t0:
                    it = shown(job, real, os.stat(real), k); found += [it] if it else []
    if found:
        found.sort(key=lambda i: i["t"]); job["nmedia"] = len(job.get("mfiles", []))
        job["emit"](dict(node="MEDIA", total=len(found), all=len(job.get("mfiles", [])), items=[{k: v for k, v in i.items() if k != "t"} for i in found[:48]]))


PHASE = {"reasoning": "Thinking", "content": "Writing answer", "toolcall": "Building command"}


def summarize(job, ev):
    """Keep the one-line state of the job (shown in the task list) up to date."""
    n = ev["node"]
    if n == "REQUEST": job["phase"] = "Sent to model, waiting for reply"; job["step"] = ev.get("attempt")
    elif n == "DELTA": job["phase"] = PHASE.get(ev.get("kind"), "Generating")
    elif n == "MODEL": job["phase"] = "Generation finished"
    elif n == "CMD": job["cmd_active"] = True; job["phase"] = "Running command"; job["last_cmd"] = ev["text"]; job["cmd_t0"] = time.time(); job["cmd"] = ev["text"][:300]; job["ncmd"] = job.get("ncmd", 0) + 1; job["last_out"] = ""
    elif n == "OUT": job["last_out"] = ev["text"][:200]
    elif n == "TOOLEND": job["cmd_active"] = False; job["phase"] = "Result returned to model" if ev.get("ok") else "Command failed, returned to model"
    elif n == "VERIFY": job["phase"] = "Tests passed" if ev.get("ok") else "Tests failed"
    elif n == "FIX": job["phase"] = "Requesting fix"
    elif n == "DONE": job["phase"] = "Done"
    elif n == "ESCALATE": job["phase"] = "Needs review (limit reached)"
    elif n == "ERROR": job["phase"] = "Error / aborted"
    elif n == "QUEUED": job["phase"] = "Queued (other tasks running)"
    elif n == "SUP_REQ": job["phase"] = "Supervisor is reviewing"
    elif n == "SUP": job["phase"] = "Supervisor: " + {"ok": "All good", "nudge": "Advice sent", "redirect": "Change of course requested", "stop": "Stop requested", "done": "Judged complete"}.get(ev.get("verdict"), "")
    elif n == "COMPACT": job["phase"] = "Compacting context and handing over (#%d)" % ev.get("n", 0)
    elif n == "APP": job["phase"] = "Server detected: port %s" % ev.get("port")
    elif n == "MEDIA": job["phase"] = "Media generated: %d" % ev.get("total", 0)


def make_emit(job):
    path = os.path.join(JOBS_DIR, job["id"] + ".jsonl"); f = open(path, "a")
    def emit(ev):
        if job.get("cancel") and ev.get("node") in ("DELTA", "SUP_DELTA"): raise RuntimeError("cancelled by user")   # closes the stream: the server stops generating
        ev = dict(ev, t=round(time.time() - job["t"], 2)); job["ev"].append(ev); summarize(job, ev); f.write(json.dumps(ev, ensure_ascii=False) + "\n"); f.flush()
        if ev["node"] in ("TOOLEND", "VERIFY", "RESULT"):
            try: media_check(job)
            except Exception: pass
    return emit


def workdir(job): return os.path.join(ROOT, "workspace", job["id"])      # one folder per task: parallel tasks do not trample each other


def launch(job, fn, *args):
    """Start the job as soon as one of the MAX_PARALLEL slots is free; until then it is 'queued' (and can be cancelled)."""
    def go():
        emit = job["emit"]
        if not SEM.acquire(blocking=False):
            emit(dict(node="QUEUED", text="%d other tasks are running. Waiting for a free slot." % MAX_PARALLEL))
            while not SEM.acquire(timeout=0.5):
                if job.get("cancel"): emit(dict(node="ERROR", text="Cancelled while queued")); job["status"] = "done"; return
        try:
            job["status"] = "running"; job["t_start"] = time.time(); fn(job, *args)
        except Exception as ex:
            emit(dict(node="ERROR", text=repr(ex))); job["status"] = "done"
        finally: SEM.release()
    threading.Thread(target=go, daemon=True).start()


def run_task(job, prompt, tests, retries, smoke, strategy, max_tokens=8192, wall=900):
    emit = job["emit"]
    try:
        sess = Y.Session("web", job["id"])
        r = Y.solve_free(prompt, tests, job.setdefault("eng", engine_for("inkling", max_tokens)), job.setdefault("sb", Y.Sandbox(isolated=SANDBOX, wall=wall, workdir=workdir(job))), sess, retries, strategy, smoke, emit=emit)
        job["result"] = {k: v for k, v in r.items() if k != "trace"}
        emit(dict(node="RESULT", code=r.get("code") or r.get("text", ""), verified=bool(r.get("verified")), final=r["node"], attempts=r.get("attempts")))
        if r["node"] == "ESCALATE":
            open(os.path.join(DATA, "human_review.jsonl"), "a").write(json.dumps(dict(task_id=job["id"], node="ESCALATE", final=False, attempts=r["attempts"], reason="from web console")) + "\n")
    except Exception as ex:
        emit(dict(node="ERROR", text=repr(ex)))
    job["status"] = "done"; job["t_done"] = time.time()


def run_agent(job, prompt, max_steps, max_tokens, wall, sup_mode="key", ctx_limit=64000, effort="medium"):
    emit = job["emit"]
    try:
        sess = Y.Session("web", job["id"])
        sb = job.setdefault("sb", Y.Sandbox(isolated=False, wall=wall, workdir=workdir(job), env_extra={"HARNESS_JOB": job["id"]}))
        job["media_seen"] = {}
        sup = None
        if sup_mode in ("key", "every"):                    # a separate agent in its own fresh context (own request, own prompt, own notes)
            p = PRESETS["inkling"]
            seng = Y.Engine("http://127.0.0.1:%d/v1" % p["port"], p["model"], 0.7, 1.0, 3000, {"top_k": 0, "chat_template_kwargs": {"reasoning_effort": "low"}})
            job["seng"] = seng; sup = Y.Supervisor(seng, sup_mode)
        n_ctx, shared = engine_ctx()                          # never let the context outgrow what the engine has
        active = max(1, sum(1 for j in JOBS.values() if j.get("status") in ("running", "queued") and j.get("kind") == "agent"))
        share = (n_ctx // active if shared else n_ctx) if n_ctx else 14000
        total = ctx_limit if ctx_limit >= 8000 else DEFAULT_CTX
        if total > share:
            emit(dict(node="NOTICE", text="context total limit lowered from %d to %d tokens: the engine has %d per slot%s." % (total, share, n_ctx, (" and the slots share it (%d running tasks)" % active) if shared else ""))); total = share
        ctx_limit = int(total * 0.75)                         # compaction threshold; the last quarter is the room for the recap turn (as in the paper)
        emit(dict(node="NOTICE", text="context: total limit %d tokens, compaction threshold %d tokens. The full log stays on disk (recall tool)." % (total, ctx_limit)))
        r = Y.solve_agent(prompt, job.setdefault("eng", engine_for("inkling", max_tokens, effort)), sb, sess, max_steps, emit=emit, sup=sup, ctx_limit=ctx_limit,
                         notice_hook=lambda: [job["notices"].pop(0) for _ in range(len(job.get("notices", [])))])
        job["result"] = {k: v for k, v in r.items() if k != "trace"}
        try: media_check(job)
        except Exception: pass
        emit(dict(node="RESULT", code=r.get("text") or r.get("code", ""), verified=False, final=r["node"], attempts=r.get("attempts"), agent=True))
        if r["node"] == "ESCALATE":
            open(os.path.join(DATA, "human_review.jsonl"), "a").write(json.dumps(dict(task_id=job["id"], node="ESCALATE", final=False, attempts=r["attempts"], reason="agent from web console")) + "\n")
    except Exception as ex:
        emit(dict(node="ERROR", text=repr(ex)))
    job["status"] = "done"


def run_eval(job, n, retries, strategy, judge):
    emit = job["emit"]; p = PRESETS["inkling"]; log = os.path.join(JOBS_DIR, job["id"] + ".log")
    cmd = [sys.executable, os.path.join(HERE, "yharness.py"), "--url", "http://127.0.0.1:%d/v1" % p["port"], "--model", p["model"], "--name", "web_" + job["id"],
           "--n", str(n), "--retries", str(retries), "--strategy", strategy, "--temperature", str(p["sampling"]["temperature"]), "--top-p", str(p["sampling"]["top_p"]),
           "--extra", json.dumps(p["sampling"]["extra"]), "--replay-failures"] + (["--judge"] if judge else [])
    emit(dict(node="CMD", text="$ " + " ".join(cmd[1:6]) + " ..."))
    job["proc"] = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    pos = 0
    while True:                                           # stream the log file of the eval process line by line
        alive = job["proc"].poll() is None
        with open(log, errors="replace") as lf:
            lf.seek(pos); chunk = lf.read(); pos = lf.tell()
        for line in chunk.splitlines(): emit(dict(node="LOG", text=line))
        if not alive: break
        time.sleep(0.5)
    emit(dict(node="DONE" if job["proc"].returncode == 0 and not job.get("cancel") else "ERROR", text="eval finished (exit %s)%s" % (job["proc"].returncode, ", cancelled" if job.get("cancel") else "")))
    job["status"] = "done"


# ------------------------------------------------------------------------------------------------ servers the agent starts
TAILNET_IP = None
APPS = {}                 # port -> dict(job, pid, bind, direct, url, alive, fw, t, error)
LOOPBACK = {"127.0.0.1", "::1", "localhost"}
WILDCARD = {"*", "0.0.0.0", "::"}


def listeners():
    """(bind address, port, pid) of every TCP listener this user can see."""
    out = subprocess.run(["ss", "-H", "-ltnp"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6: continue
        host, _, port = parts[3].rpartition(":"); m = re.search(r"pid=(\d+)", parts[5])
        if m and port.isdigit(): yield host.strip("[]"), int(port), int(m.group(1))


def job_of_pid(pid):
    try:
        for kv in open("/proc/%d/environ" % pid, "rb").read().split(b"\0"):
            if kv.startswith(b"HARNESS_JOB="): return kv[12:].decode()
    except OSError: pass
    return None


def kill_group(pid):
    try: os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        try: os.kill(pid, signal.SIGTERM)
        except OSError: return False
    return True


def scan_once():
    """A listener whose process carries HARNESS_JOB belongs to that task.
      reachable (0.0.0.0 / a real address)  -> an 'app': shown in the console.
      localhost only AND still there after the command that started it has ended (= a server left running) -> FORBIDDEN: stopped, and the worker is told.
      localhost only while a command is still running -> allowed (a helper such as a screenshot tool's temporary server lives and dies inside one command)."""
    cur = {}; bad = {}
    for host, port, pid in listeners():
        jid = job_of_pid(pid)
        if not jid or jid not in JOBS: continue
        if host in LOOPBACK:
            if port not in cur: bad[port] = (host, pid, jid)
        else: cur[port] = (host, pid, jid); bad.pop(port, None)
    for port, (host, pid, jid) in cur.items():
        a = APPS.get(port)
        if a and a["alive"] and a["pid"] == pid: continue
        APPS[port] = dict(job=jid, pid=pid, bind=host, direct=True, url="http://%s:%d/" % (TAILNET_IP, port), alive=True, t=time.time(), error="")
        try: JOBS[jid]["emit"](dict(node="APP", port=port, url=APPS[port]["url"], direct=True, error=""))
        except Exception: pass
    for port, a in list(APPS.items()):
        if a["alive"] and port not in cur: a["alive"] = False
    for port, (host, pid, jid) in bad.items():
        job = JOBS[jid]
        if job.get("cmd_active"): continue                                                  # still inside the command that started it
        if port in job.setdefault("killed_ports", set()): continue
        job["killed_ports"].add(port)
        if kill_group(pid):
            msg = ("The server on port %d listened on localhost (%s) only and was left running after its command ended: that is FORBIDDEN, so the harness stopped it. " % (port, host)) + Y.BIND_ADVICE + " Start it again bound to 0.0.0.0."
            job.setdefault("notices", []).append(msg)
            try: job["emit"](dict(node="NOTICE", text="[harness] localhost-only server on port %d stopped (forbidden): %s" % (port, host)))
            except Exception: pass


def watch_apps():
    while True:
        try: scan_once()
        except Exception: pass
        time.sleep(1.5)


def stop_app(port):
    a = APPS.get(port)
    if not a or not a["alive"]: return "not running"
    return "stopping" if kill_group(a["pid"]) else "could not stop it"


def new_job(kind, title=""):
    jid = "%s_%d_%s" % (kind, int(time.time()), secrets.token_hex(2))
    job = JOBS[jid] = dict(id=jid, kind=kind, status="queued", ev=[], t=time.time(), title=(title.strip().splitlines() or [kind])[0][:80], phase="Queued", step=0, ncmd=0, cmd="", last_out="")
    job["emit"] = make_emit(job); return job


def load_old_jobs():
    """After a restart the in-memory jobs are gone, but every event was written to data/jobs/<id>.jsonl: read them back (read-only history)."""
    for p in sorted(glob.glob(os.path.join(JOBS_DIR, "*.jsonl"))):
        jid = os.path.basename(p)[:-6]
        if jid in JOBS or not SAFE.match(jid): continue
        ev = jl(p)
        if not ev: continue
        first = next((x for x in ev if x.get("node") == "GENERATE"), {})
        job = dict(id=jid, kind=jid.split("_")[0], status="done", ev=ev, t=os.path.getmtime(p) - ev[-1].get("t", 0), title=((first.get("text") or jid).strip().splitlines() or [jid])[0][:80], step=0,
                   ncmd=sum(1 for x in ev if x.get("node") == "CMD"), cmd="", last_out="", restored=True)
        last = ev[-1].get("node"); job["phase"] = {"DONE": "Done", "RESULT": "Done", "ESCALATE": "Needs review (limit reached)", "ERROR": "Error / aborted"}.get(last, "Aborted (console restarted)")
        if last == "RESULT": job["phase"] = "Done" if not any(x.get("node") in ("ESCALATE", "ERROR") for x in ev[-3:]) else "Needs review (limit reached)"
        for x in ev:                                            # the /m/<job>/<n>/<name> registry lives in memory: rebuild it from the MEDIA events
            if x.get("node") != "MEDIA": continue
            for it in x.get("items", []):
                m = re.match(r"^/m/[\w.\-]+/(\d+)/", it.get("url", ""))
                if not m: continue
                n = int(m.group(1)); d = it["path"]
                real = os.path.realpath(os.path.join(HOME, d[2:]) if d.startswith("~/") else os.path.join(workdir(job), d))
                fl = job.setdefault("mfiles", []); fl.extend([""] * (n + 1 - len(fl))); fl[n] = real; job.setdefault("mindex", {})[real] = n
        job["emit"] = lambda e: None; JOBS[jid] = job


def job_view(job):
    d = {k: v for k, v in job.items() if k not in ("proc", "ev", "sb", "eng", "seng", "emit", "media_seen", "mfiles", "mindex", "last_cmd", "notices", "killed_ports", "cmd_active")}
    d["events"] = len(job["ev"])
    d["apps"] = [dict(port=p, url=a["url"], alive=a["alive"], direct=a["direct"], error=a["error"]) for p, a in APPS.items() if a["job"] == job["id"]]
    d["elapsed"] = (job["ev"][-1]["t"] if job["ev"] else 0) if job["status"] == "done" else round(time.time() - job["t"], 1); return d


# ------------------------------------------------------------------------------------------------ read-only dashboard (as before)
def jl(path):
    if not os.path.exists(path): return []
    out = []
    for l in open(path, errors="replace"):
        try: out.append(json.loads(l))
        except Exception: pass
    return out


def load_runs():
    rows = []
    for p in sorted(glob.glob(os.path.join(DATA, "run_*.json"))):
        try: s = json.load(open(p))["summary"]; rows.append(s)
        except Exception: pass
    for p in sorted(glob.glob(os.path.join(EVAL, "out_*.json"))):
        try:
            s = json.load(open(p))["summary"]
            rows.append(dict(name=s["name"], bench=s["bench"], n=s["n"], final_pct=s["pass_at_1"], first_try_pct=s["pass_at_1"], total_tokens=s["total_tokens"], wall_s=s["wall_s"]))
        except Exception: pass
    return rows


def table(head, rows):
    return "<table><tr>%s</tr>%s</table>" % ("".join("<th>%s</th>" % e(c) for c in head), "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % c for c in r) for r in rows))


CSS = ("body{font:14px system-ui,sans-serif;margin:20px auto;max-width:1100px;padding:0 12px;color:#222}h1{font-size:20px}h2{font-size:16px;margin-top:26px}"
       "table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:4px 8px;text-align:left;vertical-align:top}th{background:#f4f4f4}"
       "pre{background:#f7f7f7;padding:10px;overflow:auto;white-space:pre-wrap;margin:4px 0}textarea,input,select{font:inherit;width:100%;box-sizing:border-box}"
       "textarea{min-height:90px}button{font:inherit;padding:6px 14px;margin:4px 4px 4px 0;cursor:pointer}.row{display:flex;gap:12px}.row>*{flex:1}"
       ".note{color:#666}.ev{border-left:4px solid #ccc;padding-left:8px;margin:8px 0}.GENERATE,.MODEL{border-color:#36c}.VERIFY{border-color:#c80}.DONE{border-color:#0a5}"
       ".ESCALATE,.ERROR{border-color:#c00}.FIX{border-color:#a5a}.warn{background:#fff3cd;padding:8px;border-radius:4px}.ok{color:#064}.bad{color:#a00}")

APP_JS = r"""
const _errs={};
function showErr(msg){let b=document.getElementById('jserr'); if(!b){b=document.createElement('div'); b.id='jserr'; b.style='position:fixed;bottom:0;left:0;right:0;background:#b3261e;color:#fff;padding:6px 10px;font:12px monospace;z-index:99;white-space:pre-wrap;max-height:30vh;overflow:auto'; document.body.append(b)}
  _errs[msg]=(_errs[msg]||0)+1; b.textContent='UI JS error (a console bug, please report it):\n'+Object.entries(_errs).map(([m,n])=>m+(n>1?'  ×'+n:'')).join('\n')}
addEventListener('error',e=>showErr(e.message+' ('+(e.lineno||'?')+':'+(e.colno||'?')+')'));
addEventListener('unhandledrejection',e=>showErr('Async error: '+(e.reason&&e.reason.message||e.reason)));
const $=s=>document.querySelector(s);
function el(tag,cls,text){const x=document.createElement(tag); if(cls)x.className=cls; if(text!==undefined)x.textContent=text; return x}
async function api(path,body){const r=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});
  if(!r.ok){throw new Error(await r.text())} return r.json()}
const S={job:null,es:null,busy:false,synced:false,t0:0,curT:0,phaseEvT:0,finalT:0,att:0,maxAtt:0,chars:{think:0,ans:0},think:null,ans:null,tool:null,cmdout:null,thinkCard:null,
  pend:{think:'',ans:''},engineUp:false,lastEv:0,step:'send'};
const fmt=s=>{s=Math.max(0,Math.floor(s));return Math.floor(s/60)+':'+String(s%60).padStart(2,'0')};
function banner(step,title,sub,cls){S.step=step;S.phaseEvT=S.curT;$('#now').className='now '+(cls||'');
  for(const k of ['send','think','write','run','result','fix']){$('#step-'+k).className='step'+(k===step?' on':'')}
  $('#nowtitle').textContent=title; $('#nowsub').textContent=sub||''}
setInterval(()=>{ if(!S.job){$('#clock').textContent='';return}
  if(!S.synced){$('#clock').textContent='Loading...';return}
  const total=S.busy?(Date.now()-S.t0)/1000:S.finalT;
  $('#clock').textContent='Total '+fmt(total)+(S.busy?' / this phase '+fmt(total-S.phaseEvT):' (finished)')+(S.att?' / step '+S.att+(S.maxAtt?' / '+S.maxAtt:''):'');
  const idle=(Date.now()-S.lastEv)/1000; $('#stall').textContent=(S.busy&&idle>8&&['send','think','write'].includes(S.step))?'⚠ '+Math.floor(idle)+' s with no output from the model (possibly long thinking; you can also check the engine log)':'';
  const s=Math.max(total-S.phaseEvT,1); $('#rate').textContent=!S.busy?'':S.step==='think'?'Thinking '+S.chars.think+' chars ('+(S.chars.think/s).toFixed(0)+' chars/s)':S.step==='write'?'Generated '+S.chars.ans+' chars':''},300);
function stick(p){return p.scrollTop+p.clientHeight>=p.scrollHeight-30}
function addTo(p,t){const k=stick(p);p.textContent+=t;if(k)p.scrollTop=p.scrollHeight}
function flush(){ if(S.pend.think&&S.think){addTo(S.think,S.pend.think);S.pend.think=''} if(S.pend.ans&&S.ans){addTo(S.ans,S.pend.ans);S.pend.ans=''} }
setInterval(flush,60);
function card(cls,title,body){const c=el('div','card '+cls); c.append(el('div','ct',title)); if(body)c.append(body); $('#timeline').append(c); if(S.busy&&S.synced)c.scrollIntoView({block:'nearest'}); return c}
function pre(cls,t){return el('pre',cls||'',t||'')}
function onEv(ev){
 if(ev.node!=='DELTA'&&ev.node!=='SUP_DELTA')flush();
 S.lastEv=Date.now(); S.curT=ev.t||0; S.finalT=S.curT; if(S.synced)S.t0=Date.now()-S.curT*1000;
 switch(ev.node){
 case 'GENERATE': {card('c-task','Your request (as sent to the model)',pre('',ev.text)); break}
 case 'QUEUED': {banner('send','Queued',ev.text); card('c-info','Queued',pre('small',ev.text)); break}
 case 'REQUEST': {S.att=ev.attempt; banner('send','Sent to model','Waiting for the first reply (model is reading the input)...'); card('c-req','Step '+ev.attempt+': request sent',pre('small',ev.text));
   S.think=null;S.ans=null;S.tool=null;S.thinkCard=null;S.chars={think:0,ans:0};S.pend={think:'',ans:''}; break}
 case 'DELTA': {
   if(ev.kind==='reasoning'){ if(!S.think){S.think=pre('think');S.thinkCard=card('c-think','Step '+S.att+': model thinking (live)',S.think); banner('think','Model is thinking','This is the reasoning before the answer. It can take several minutes.')}
     S.chars.think+=ev.text.length; S.pend.think+=ev.text }
   else if(ev.kind==='toolcall'){ if(!S.tool){ if(S.thinkCard)S.thinkCard.classList.add('collapsed'); S.tool=pre('toolbuild'); card('c-tool','Step '+S.att+': model is building a command (live)',S.tool); banner('write','Building command','The model is writing the command to pass to the bash tool.')} addTo(S.tool,ev.text) }
   else { if(!S.ans){ if(S.thinkCard)S.thinkCard.classList.add('collapsed'); S.ans=pre('ans');card('c-ans','Step '+S.att+': model answer (live)',S.ans); banner('write','Writing answer','')}
     S.chars.ans+=ev.text.length; S.pend.ans+=ev.text }
   break}
 case 'MODEL': {banner('write',ev.finish==='length'?'Generation stopped at the token limit':'Generation finished',ev.finish==='length'?'The limit may have been used up by thinking alone. '+ev.text:ev.text); card('c-info','Generation result',pre('small',ev.text)); break}
 case 'CMD': {banner('run','Running command','Live output streams below.'); S.cmdout=pre('cmdout',''); const b=el('div'); b.append(pre('cmd',ev.text),S.cmdout); card('c-cmd','Executed command (live output)',b); break}
 case 'OUT': {if(S.cmdout)addTo(S.cmdout,ev.text+'\n'); break}
 case 'TOOLEND': {banner('fix',ev.ok?'Command succeeded':'Command failed','Result returned to the model. Waiting for its next decision...'); card(ev.ok?'c-ok':'c-bad','Command result: '+ev.text+(ev.secs!==undefined?' ('+ev.secs+' s)':''),null); break}
 case 'VERIFY': {banner('result',ev.ok?'Tests passed':'Tests failed',ev.ok?'Verification passed.':'Returning the failure to the model to fix.',ev.ok?'good':'bad'); card(ev.ok?'c-ok':'c-bad',(ev.ok?'Result: passed':'Result: failed')+(ev.secs!==undefined?' ('+ev.secs+' s)':''),pre('',ev.text)); break}
 case 'FIX': {banner('fix','Failure returned to the model',ev.text+'. Waiting for the next attempt...'); card('c-fix','Requesting fix',pre('small',ev.text)); break}
 case 'LOG': {addTo($('#evlog'),ev.text+'\n'); break}
 case 'RESULT': {if(ev.code){const body=el('div'); body.append(pre('final',ev.code)); const b=el('button','','Copy'); b.onclick=()=>navigator.clipboard.writeText(ev.code); body.append(b);
     card(ev.verified?'c-ok':'c-info',ev.agent?('Final agent answer ('+ev.final+'/'+ev.attempts+' steps)'):('Final code - '+(ev.verified?'verified':'unverified')+' ('+ev.final+'/'+ev.attempts+' attempts)'),body)} break}
 case 'APP': {const b=el('div'); const l=el('a','',ev.url); l.href=ev.url; l.target='_blank'; b.append(el('div','small','Listening on 0.0.0.0, so it is embedded in the panel above.'),l); card('c-app','Server detected: port '+ev.port,b); break}
 case 'NOTICE': {card('c-info','Notice from the harness',pre('small',ev.text)); break}
 case 'SUP_REQ': {S.supPre=pre('think'); S.supOut=pre('small'); const b=el('div'); b.append(S.supPre,S.supOut); S.supCard=card('c-sup','Supervisor agent (separate context) is reviewing',b); banner('fix','Supervisor is checking',ev.text); break}
 case 'SUP_DELTA': {const t=ev.kind==='reasoning'?S.supPre:S.supOut; if(t)addTo(t,ev.text); break}
 case 'SUP': {const c={ok:['All good','c-ok'],nudge:['Advice sent','c-sup'],redirect:['Change of course requested','c-fix'],stop:['Stop requested','c-bad'],done:['Judged complete','c-ok']}[ev.verdict]||['?','c-info'];
   if(S.supCard){S.supCard.classList.add('collapsed')} const b=el('div'); if(ev.message)b.append(el('div','sup-msg','-> To worker: '+ev.message)); b.append(el('div','small','Reason: '+(ev.reason||'')),el('div','small','Supervisor notes: '+(ev.state||'')));
   card(c[1],'Supervisor verdict: '+c[0]+(ev.secs!==undefined?' ('+ev.secs+' s)':''),b); banner('fix','Supervisor verdict: '+c[0],ev.message||ev.reason||''); S.supCard=null; break}
 case 'COMPACT': {const b=el('div'); b.append(el('div','small','Previous context: about '+ev.before+' tokens -> handed over to a new conversation. The full log is searchable with the recall tool.'),pre('ans',ev.text)); card('c-comp','Context compacted and handed over (#'+ev.n+')',b); banner('fix','Context compacted','Continuing in a short new conversation.'); break}
 case 'MEDIA': {const b=el('div'); const g=el('div','mgrid'); const url=p=>'/media/'+S.job+'/'+p.split('/').map(encodeURIComponent).join('/'); const U=it=>it.url||url(it.path);
   for(const it of ev.items){ if(it.kind==='video'){const w=el('div','mv'); const v=document.createElement('video'); v.controls=true; v.preload='metadata'; v.src=U(it); w.append(v,el('div','small',it.path+' ('+(it.size/1e6).toFixed(1)+' MB)')); g.append(w)}
     else{const a=document.createElement('a'); a.href=U(it); a.target='_blank'; const i=document.createElement('img'); i.loading='lazy'; i.src=U(it); i.title=it.path; a.append(i); g.append(a)} }
   b.append(g); if(ev.total>ev.items.length)b.append(el('div','small','and '+(ev.total-ev.items.length)+' more ('+ev.all+' in this task)'));
   const l=el('a','','View all in gallery'); l.href='/gallery#'+S.job; l.target='_blank'; b.append(l); card('c-media','Generated videos and images ('+ev.total+')',b); break}
 case 'DONE': {banner('result','Done',ev.text,'good'); break}
 case 'ESCALATE': {banner('result','Human review needed','Limit reached. Added to the review queue.','bad'); card('c-bad','Human review needed',pre('',ev.text)); break}
 case 'ERROR': {banner('result','Error / aborted',ev.text,'bad'); card('c-bad','Error',pre('',ev.text)); break}
 }}
function attach(id){ if(S.es)S.es.close();
 Object.assign(S,{supCard:null,supPre:null,supOut:null,job:id,busy:true,synced:false,att:0,think:null,ans:null,tool:null,cmdout:null,thinkCard:null,chars:{think:0,ans:0},pend:{think:'',ans:''},lastEv:Date.now(),curT:0,phaseEvT:0,finalT:0});
 $('#timeline').replaceChildren(); $('#apps').replaceChildren(); S.appEls={}; $('#evlog').textContent=''; $('#stop').disabled=false; banner('send','Loading...','');
 const es=S.es=new EventSource('/api/stream/'+id);
 es.onmessage=m=>{ if(S.es===es)onEv(JSON.parse(m.data)) };
 es.addEventListener('sync',m=>{ if(S.es!==es)return; const d=JSON.parse(m.data); S.synced=true; S.t0=Date.now()-d.elapsed*1000; S.lastEv=Date.now() });
 es.addEventListener('end',()=>{ if(S.es!==es)return; es.close(); S.busy=false; $('#stop').disabled=true; if(S.step!=='result')banner('result','Finished','');pollJobs()});
 es.onerror=()=>{ if(S.es===es&&es.readyState===2){S.busy=false;$('#stop').disabled=true} };
 document.querySelectorAll('.job').forEach(j=>j.classList.toggle('sel',j.dataset.id===id)) }
async function submit(kind){try{
 S.maxAtt=$('#mode').value==='agent'?+$('#steps').value:+$('#retries').value+1;
 const body=kind==='task'?{mode:$('#mode').value,steps:+$('#steps').value,supervisor:$('#sup').value,ctx_total:+$('#ctxlim').value,effort:$('#effort').value,prompt:$('#prompt').value,tests:$('#tests').value,retries:+$('#retries').value,max_tokens:+$('#maxtok').value,wall:+$('#wall').value,verify:$('#verify').value,strategy:$('#strategy').value}
  :{n:+$('#evn').value,retries:+$('#retries').value,strategy:$('#strategy').value,judge:$('#judge').checked};
 const j=await api('/api/'+kind,body); attach(j.id); if(kind==='task'){$('#prompt').value=''} pollJobs()}catch(x){alert(x.message)}}
$('#go').onclick=()=>submit('task'); $('#goeval').onclick=()=>submit('eval');
$('#stop').onclick=async()=>{try{await api('/api/job/cancel',{id:S.job});$('#nowsub').textContent='Cancel requested...'}catch(x){alert(x.message)}};
$('#estart').onclick=async()=>{alert((await api('/api/engine/start',{name:'inkling'})).msg);refreshEngine()};
$('#estop').onclick=async()=>{alert((await api('/api/engine/stop',{name:'inkling'})).msg);refreshEngine()};
async function refreshEngine(){try{const s=(await api('/api/engines')).inkling;S.engineUp=s.up;
 $('#estate').textContent=s.up?'Running':(s.mine?'Starting (loading model)':'Stopped'); $('#estate').className=s.up?'ok':'bad';
 $('#go').disabled=!s.up; $('#goeval').disabled=!s.up;
 $('#hint').textContent=s.up?'':'The engine is stopped. Press Start and wait until it shows Running.'}catch(x){}}
// ---- task list (all jobs, live)
function badge(j){ if(j.status==='queued')return['Queued','b-q']; if(j.status==='running')return['Running','b-r'];
  if((j.phase||'').includes('Done'))return['Done','b-ok']; if((j.phase||'').includes('Aborted')||(j.phase||'').includes('Error'))return['Aborted','b-bad']; return['Needs review','b-bad']}
S.appEls={};
function renderApps(apps){ const box=$('#apps'); const want=new Set();
  for(const a of apps||[]){ const k=S.job+':'+a.port; want.add(k); let d=S.appEls[k];
    if(!d){ d=el('div','app'); const h=el('div','ah'); const t=el('a','',a.url); t.href=a.url; t.target='_blank'; t.rel='noopener'; const st=el('span','small'); st.className='small ast';
      const b1=el('button','','Open (new tab)'); b1.onclick=()=>window.open(a.url,'_blank','noopener');
      const b3=el('button','','Stop'); b3.onclick=async()=>{ if(confirm('Stop the server on port '+a.port+'?')){try{alert((await api('/api/app/stop',{port:a.port})).msg)}catch(e){alert(e.message)}} };
      h.append(el('b','','▶ Launched app'),t,st,b1,b3); d.append(h); box.append(d); S.appEls[k]=d }
    d.classList.toggle('dead',!a.alive); d.querySelector('.ast').textContent=a.alive?' (listening on 0.0.0.0)':' (stopped)'; }
  for(const k of Object.keys(S.appEls)) if(!want.has(k)){S.appEls[k].remove(); delete S.appEls[k]} }
async function pollJobs(){ try{ const js=await api('/api/jobs'); const box=$('#jobs'); const cur=js.find(j=>j.id===S.job); if(cur)renderApps(cur.apps); const keep=box.scrollTop; box.querySelectorAll('.job').forEach(n=>n.remove());
  const run=js.filter(j=>j.status==='running').length, q=js.filter(j=>j.status==='queued').length; $('#jobsum').textContent='Running '+run+' / Queued '+q+' / Total '+js.length;
  for(const j of js){ const [b,bc]=badge(j); const d=el('div','job '+(j.id===S.job?'sel ':'')+(j.status==='done'?'fin':'')); d.dataset.id=j.id;
    const t=el('div','jt',j.title||j.id); const m=el('div','jm'); m.append(el('span','badge '+bc,b),el('span','',' '+fmt(j.elapsed)+(j.step?' · step '+j.step:'')+(j.ncmd?' · commands '+j.ncmd:'')));
    d.append(t,m,el('div','jp',j.phase||'')); if(j.status==='running'&&j.cmd)d.append(el('div','jc',j.cmd)); if(j.status==='running'&&j.last_out)d.append(el('div','jo',j.last_out));
    if(j.status!=='done'){const x=el('button','jx','Cancel'); x.onclick=async e=>{e.stopPropagation(); try{await api('/api/job/cancel',{id:j.id})}catch(err){alert(err.message)}}; d.append(x)}
    d.onclick=()=>{ if(S.job!==j.id){location.hash='job='+j.id; attach(j.id)} }; box.append(d)} box.scrollTop=keep }catch(x){} }
(function(){const pre_=$('#englog'),line=$('#englast');const e2=new EventSource('/api/enginelog');let lines=[];
 e2.onmessage=m=>{const k=stick(pre_);const t=JSON.parse(m.data);lines.push(t);if(lines.length>400)lines=lines.slice(-400);pre_.textContent=lines.join('\n');if(k)pre_.scrollTop=pre_.scrollHeight; if(t.trim())line.textContent=t.slice(0,160)}})();
refreshEngine(); setInterval(refreshEngine,4000); pollJobs(); setInterval(pollJobs,1200);
banner('send','Idle','Enter a request and press Run. You can submit several tasks in a row.');
fetch('/api/jobs').then(r=>r.json()).then(j=>{const m=location.hash.match(/^#job=([\w.\-]+)$/); if(m){attach(m[1]);return} const r=j.find(x=>x.status==='running'); if(r)attach(r.id)}).catch(()=>{});
addEventListener('hashchange',()=>{const m=location.hash.match(/^#job=([\w.\-]+)$/); if(m&&S.job!==m[1])attach(m[1])});
"""

CSS += (".layout{display:grid;grid-template-columns:330px minmax(0,1fr);gap:18px;align-items:start}#side{position:sticky;top:8px;max-height:calc(100vh - 16px);overflow:auto}"
        "@media(max-width:950px){.layout{grid-template-columns:1fr}#side{position:static;max-height:none}}"
        ".job{border:1px solid #d5d8de;border-left:5px solid #9aa;border-radius:4px;padding:6px 8px;margin:6px 0;cursor:pointer;background:#fff;position:relative}.job:hover{background:#f6f8fc}.job.sel{outline:2px solid #2b7bd6;background:#eef5fd}.job.fin{opacity:.85}"
        ".jt{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding-right:44px}.jm{font-size:12px;color:#555;margin-top:2px}.jp{font-size:13px;margin-top:2px}"
        ".jc{font:12px monospace;background:#111;color:#7CFC9A;padding:2px 5px;border-radius:3px;margin-top:3px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.jo{font:11px monospace;color:#666;margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}"
        ".badge{font-size:11px;padding:1px 6px;border-radius:9px;color:#fff}.b-q{background:#8a8f98}.b-r{background:#2b7bd6;animation:pulse 1.4s infinite}.b-ok{background:#14663a}.b-bad{background:#b3261e}@keyframes pulse{50%{opacity:.55}}"
        ".jx{position:absolute;right:6px;top:6px;font-size:11px;padding:1px 6px;margin:0}"
        ".now{position:sticky;top:0;z-index:5;background:#1d2b4a;color:#fff;padding:12px 16px;border-radius:6px;margin:10px 0}.now.good{background:#14663a}.now.bad{background:#8a1c1c}"
        "#nowtitle{font-size:22px;font-weight:700}#nowsub{opacity:.9;margin-top:2px;min-height:18px}#clock,#rate{font-size:13px;opacity:.9;margin-top:6px;display:inline-block;margin-right:16px}"
        ".steps{display:flex;gap:4px;margin-top:8px}.step{flex:1;text-align:center;font-size:12px;padding:4px 2px;border-radius:3px;background:rgba(255,255,255,.12);color:rgba(255,255,255,.65)}.step.on{background:#ffd54a;color:#000;font-weight:700}"
        ".card{border-left:5px solid #999;background:#fff;margin:10px 0;padding:6px 12px;box-shadow:0 1px 3px rgba(0,0,0,.12);border-radius:3px}.ct{font-weight:700;margin-bottom:4px}"
        ".c-task{border-color:#789}.c-req{border-color:#aab}.c-think{border-color:#7a5fc4;background:#faf8ff}.c-ans{border-color:#2b7bd6}.c-cmd{border-color:#222}.c-ok{border-color:#0a8f4d}.c-bad{border-color:#c62828}.c-fix{border-color:#b05bc4}.c-info{border-color:#aaa}.c-tool{border-color:#f0a500}.c-app{border-color:#0b7285;background:#f1fbfc}.app{border:2px solid #0b7285;border-radius:6px;margin:10px 0;background:#fff}.app .ah{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:6px 10px;background:#e3fafc}.app iframe{width:100%;height:65vh;border:0;display:block;background:#fff;resize:vertical;overflow:auto}.app.dead{opacity:.6}.c-sup{border-color:#8e24aa;background:#fbf5fd}.c-comp{border-color:#00838f;background:#f3fbfc}.c-media{border-color:#e65100}.sup-msg{background:#f3e5f5;padding:6px 8px;border-radius:3px;margin:4px 0;font-weight:600}.mgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:6px}.mgrid img{width:100%;height:105px;object-fit:cover;border-radius:3px;background:#eee}.mgrid .mv{grid-column:span 3}.mgrid video{width:100%;max-height:50vh;background:#000}"
        ".card pre{margin:4px 0}.small{font-size:12px;color:#555}.think{background:#f1ecfb;color:#3a2d63;max-height:38vh;overflow:auto}.ans{background:#eef5fd;max-height:38vh;overflow:auto}.toolbuild{background:#fff8e1;color:#5d4a00;max-height:20vh;overflow:auto;font-size:12px}"
        ".cmd{background:#111;color:#7CFC9A}.cmdout{background:#0b0b0b;color:#ddd;max-height:40vh;overflow:auto;font-size:12px}.final{background:#fff;border:1px solid #ccc;max-height:50vh;overflow:auto}.collapsed pre.think{max-height:5em}"
        "#englog,#evlog{background:#111;color:#ccc;max-height:28vh;overflow:auto;font-size:12px}#englast{font-size:12px;color:#666}")


def page(title, body, script=""):
    return "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'><title>%s</title><style>%s</style>%s<script>%s</script>" % (e(title), CSS.replace("max-width:1100px", "max-width:1500px"), body, script)


def console():
    if NO_EXEC: warn = "<p class=warn>Code execution: OFF (--no-exec). Only unverified answers are available.</p>"
    elif SANDBOX: warn = "<p class=warn><b>Code execution: ON (soft sandbox)</b> Temp directory and CPU/memory limits apply. This is not isolation.</p>"
    else: warn = ("<p class=warn><b>Code execution: ON / no sandbox</b> Commands and code written by the model are <b>run as-is on this machine with your privileges</b>"
                  "(real environment, network, all files). Each task works in <code>~/rrr/workspace/&lt;task-id&gt;</code>. Anyone who knows the token can do the same.</p>")
    steps = "".join("<div class=step id=step-%s>%s</div>" % (k, l) for k, l in [("send", "1. Send"), ("think", "2. Think"), ("write", "3. Generate"), ("run", "4. Run command"), ("result", "5. Result"), ("fix", "6. Return result")])
    side = "<aside id=side><h2 style='margin-top:0'>Tasks</h2><div id=jobsum class=note></div><div id=jobs></div></aside>"
    main = ("<main><p>Engine: <b id=estate class=bad>?</b> <button id=estart>Start</button><button id=estop>Stop</button> <span id=hint class=note></span><br><span id=englast></span></p>"
        "<details><summary>Engine log (live)</summary><pre id=englog></pre></details>"
        "<h2>New task <span class=note>(you can submit several; they run in parallel)</span></h2><label>Request<textarea id=prompt placeholder='e.g. Set up https://github.com/yt-dlp/yt-dlp'></textarea></label>"
        "<details><summary>Advanced settings</summary><label>Tests (code mode only; Python; optional)<textarea id=tests placeholder='assert is_prime(7)'></textarea></label>"
        "<div class=row><div><label>Mode<select id=mode><option value=agent>Agent (runs bash, reads results, decides next step)</option><option value=code>Code generation + verification (pass/fail by tests)</option></select></label></div>"
        "<div><label>Max steps (agent)<input id=steps type=number value=25 min=1 max=80></label></div></div>"
        "<div class=row><div><label>Supervisor agent (monitors and advises in a separate context)<select id=sup><option value=key>Key points only (errors, repeats, final answer)</option><option value=every>Every step</option><option value=off>Off</option></select></label></div>"
        "<div><label>Worker context limit (tokens). Above the threshold (75%%) it is summarized and handed to a new conversation; past context is available via the recall tool. There is no unlimited option<input id=ctxlim type=number value=64000 min=8000 max=250000 step=8000></label></div>"
        "<div><label>Thinking depth (shallower is faster)<select id=effort><option value=medium>Medium</option><option value=low>Shallow</option><option value=high>Deep (slow)</option></select></label></div></div>"
        "<div class=row><div><label>Verification (code mode)<select id=verify><option value=tests>Run my tests</option><option value=smoke>OK if it runs without errors</option><option value=none>No verification</option></select></label></div>"
        "<div><label>Retry limit (code mode)<input id=retries type=number value=3 min=0 max=6></label></div>"
        "<div><label>Max tokens per call<input id=maxtok type=number value=16000 min=512 max=30000 step=1000></label></div>"
        "<div><label>Per-command timeout (s)<input id=wall type=number value=900 min=5 max=7200></label></div>"
        "<div><label>Prompt strategy<select id=strategy><option>direct</option><option>plan</option></select></label></div></div></details>"
        "<button id=go disabled>Run</button> <details style='display:inline-block'><summary>MBPP evaluation</summary><div class=row><div><label>Problems<input id=evn type=number value=10 min=1 max=200></label></div>"
        "<div><label><input id=judge type=checkbox style='width:auto'> LLM judge</label></div></div><button id=goeval disabled>Run evaluation</button><pre id=evlog></pre></details>"
        "<div class=now id=now><div id=nowtitle></div><div id=nowsub></div><div><span id=clock></span><span id=rate></span><button id=stop disabled style='float:right'>Cancel this task</button></div>"
        "<div id=stall style='color:#ffd54a;font-size:13px'></div><div class=steps>%s</div></div>"
        "<div id=apps></div><div id=timeline></div><p><a href='/gallery' target=_blank>Media gallery (videos and images)</a> · <a href='/dashboard'>Dashboard (results, review queue, failures)</a></p></main>") % steps
    return page("yharness console", "<h1>yharness console <span class=note>Inkling-2-Mini · up to %d parallel</span></h1>%s<div class=layout>%s%s</div>" % (MAX_PARALLEL, warn, side, main), APP_JS)


def dashboard():
    runs = load_runs()
    r1 = table(["name", "bench", "n", "first try %", "final %", "escalated", "mean attempts", "tokens", "wall s"],
               [[e(str(r.get(k, "-"))) for k in ("name", "bench", "n", "first_try_pct", "final_pct", "escalated", "mean_attempts", "total_tokens", "wall_s")] for r in runs])
    hr = jl(os.path.join(DATA, "human_review.jsonl"))
    r2 = table(["task", "node", "final", "attempts", "reason"], [[e(str(x.get(k))) for k in ("task_id", "node", "final", "attempts", "reason")] for x in hr[-30:]])
    fl = jl(os.path.join(DATA, "failures.jsonl"))
    r3 = table(["task", "run", "error tail"], [[e(str(x.get("task_id"))), e(str(x.get("name"))), e(str(x.get("err"))[-120:])] for x in fl[-30:]])
    ls = "<ul>%s</ul>" % "".join("<li>%s</li>" % e(x.get("text", "")) for x in jl(os.path.join(DATA, "lessons.jsonl"))[-15:])
    sess = []
    for d in sorted(glob.glob(os.path.join(DATA, "sessions", "*"))):
        n = os.path.basename(d); files = sorted(os.listdir(d), key=lambda f: -os.path.getmtime(os.path.join(d, f)))
        sess.append([e(n), str(len(files)), " ".join("<a href='/session/%s/%s'>%s</a>" % (e(n), e(f), e(f[:-6])) for f in files[:40])])
    return page("yharness dashboard", "<p><a href='/'>&larr; console</a></p><h1>Dashboard</h1><h2>Eval runs</h2>%s<h2>Human review (latest 30 of %d)</h2>%s"
                "<h2>Failure set (latest 30 of %d)</h2>%s<h2>Lessons</h2>%s<h2>Sessions</h2>%s" % (r1, len(hr), r2, len(fl), r3, ls, table(["run", "tasks", "transcripts"], sess)))


def session_page(run, task):
    p = os.path.join(DATA, "sessions", run, task)
    if not (SAFE.match(run) and SAFE.match(task) and os.path.isfile(p)): return None
    out = "".join("<h3>%s %s</h3><pre>%s</pre>" % (e(str(m.get("role"))), e(str(m.get("tool", ""))), e(str(m.get("content")))) for m in jl(p))
    return page("%s/%s" % (run, task), "<p><a href='/dashboard'>&larr; back</a></p><h1>%s / %s</h1>%s" % (e(run), e(task), out))


# ------------------------------------------------------------------------------------------------ http
def media_base(jid):
    """Folder of a task's files. '_shared' = the old common folder of the first version (before every task had its own)."""
    if jid == "_shared": return os.path.join(ROOT, "workspace")
    if not SAFE.match(jid): return None
    b = os.path.join(ROOT, "workspace", jid)
    return b if os.path.isdir(b) else None


def gallery():
    """All images and videos of all task folders, newest folder first. Plain HTML: videos play in place, images are lazy thumbnails."""
    bases = [("_shared", "Shared folder (legacy)", os.path.join(ROOT, "workspace"))]
    ws = os.path.join(ROOT, "workspace")
    for d in sorted(os.listdir(ws), key=lambda x: -os.path.getmtime(os.path.join(ws, x))) if os.path.isdir(ws) else []:
        p = os.path.join(ws, d)
        if os.path.isdir(p) and d not in ("venv", "frames", "yt-dlp") and SAFE.match(d): bases.append((d, (JOBS[d]["title"] if d in JOBS else d), p))
    out = []
    for jid, title, p in bases:
        items = scan_media(p)
        if jid == "_shared": items = {r: v for r, v in items.items() if not re.match(r"^(agent|task|eval)_\d+_\w+", r)}
        if not items: continue
        vids = sorted([r for r, v in items.items() if v[2] == "video"], key=lambda r: -items[r][1]); imgs = sorted([r for r, v in items.items() if v[2] == "image"])
        h = "<h2 id='%s'>%s <span class=note>Videos %d · Images %d</span></h2>" % (e(jid), e(title), len(vids), len(imgs))
        for r in vids[:12]:
            h += "<div class=gv><video controls preload=metadata src='/media/%s/%s'></video><div class=note>%s(%.1f MB)</div></div>" % (e(jid), quote(r), e(r), items[r][0] / 1e6)
        h += "<div class=grid>" + "".join("<a href='/media/%s/%s' target=_blank><img src='/media/%s/%s' title='%s'></a>" % (e(jid), quote(r), e(jid), quote(r), e(r)) for r in imgs[:120]) + "</div>"
        if len(imgs) > 120: h += "<p class=note>%d more images (showing only the first 120)</p>" % (len(imgs) - 120)
        out.append(h)
    css = ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:6px}.grid img{width:100%;height:110px;object-fit:cover;border-radius:3px;background:#eee}.gv video{max-width:640px;width:100%;background:#000}.gv{margin:8px 0}"
    return page("Media", "<style>" + css + "</style><p><a href='/'>&larr; Console</a></p><h1>Media gallery</h1>" + ("".join(out) or "<p>No videos or images yet.</p>"))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, code, body, ctype="text/html; charset=utf-8", extra=()):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff"); self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; media-src 'self'; frame-src http://%s:*" % (TAILNET_IP or "127.0.0.1"))
        for k, v in extra: self.send_header(k, v)
        self.end_headers(); self.wfile.write(b)

    def authed(self, q):
        cands = [q.get("token", [""])[0]]
        for c in self.headers.get("Cookie", "").split(";"):
            if c.strip().startswith("yh_token="): cands.append(c.strip()[9:])
        return any(hmac.compare_digest(c, TOKEN) for c in cands if c)

    def serve_file(self, base, rel):
        """Send an image/video from a task folder. Confined to the folder, whitelisted extensions, HTTP Range supported (needed to seek in videos)."""
        real = os.path.realpath(os.path.join(base, rel)); b = os.path.realpath(base); ext = os.path.splitext(real)[1].lower()
        if not (real.startswith(b + os.sep) and ext in MEDIA_MIME and os.path.isfile(real)): return self.send(404, "not found")
        size = os.path.getsize(real); a, z = 0, size - 1; code = 200
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m and (m.group(1) or m.group(2)):
            if m.group(1): a = int(m.group(1)); z = int(m.group(2)) if m.group(2) else size - 1
            else: a = max(0, size - int(m.group(2)))
            if a > z or a >= size: self.send_response(416); self.send_header("Content-Range", "bytes */%d" % size); self.end_headers(); return
            z = min(z, size - 1); code = 206
        self.send_response(code); self.send_header("Content-Type", MEDIA_MIME[ext]); self.send_header("Accept-Ranges", "bytes"); self.send_header("Content-Length", str(z - a + 1))
        self.send_header("Cache-Control", "private, max-age=60"); self.send_header("X-Content-Type-Options", "nosniff")
        if code == 206: self.send_header("Content-Range", "bytes %d-%d/%d" % (a, z, size))
        self.end_headers()
        try:
            with open(real, "rb") as f:
                f.seek(a); left = z - a + 1
                while left > 0:
                    chunk = f.read(min(262144, left))
                    if not chunk: break
                    self.wfile.write(chunk); left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, OSError): pass

    def sse_start(self):
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-store"); self.send_header("X-Accel-Buffering", "no")
        self.send_header("Content-Security-Policy", "default-src 'none'"); self.end_headers(); self.close_connection = True

    def sse_write(self, data, event=None):
        self.wfile.write(((("event: %s\n" % event) if event else "") + "data: %s\n\n" % data).encode()); self.wfile.flush()

    def sse_job(self, job):
        """Server-sent events: replays every event of the job from the start, then follows it live until the job is done."""
        self.sse_start(); sent = 0; idle = 0.0; synced = False
        try:
            while True:
                evs = job["ev"]; n = len(evs)
                while sent < n: self.sse_write(json.dumps(evs[sent], ensure_ascii=False)); sent += 1; idle = 0
                if not synced:                                  # the replay has caught up: tell the page how old the job is, so its clocks are right
                    synced = True; self.sse_write(json.dumps(dict(elapsed=job_view(job)["elapsed"], status=job["status"], queued=job["status"] == "queued")), "sync")
                if job["status"] == "done" and sent >= len(job["ev"]): return self.sse_write("{}", "end")
                time.sleep(0.05); idle += 0.05
                if idle >= 10: self.wfile.write(b": keepalive\n\n"); self.wfile.flush(); idle = 0
        except (BrokenPipeError, ConnectionResetError, OSError): pass

    def sse_file(self, path):
        """Tail -f of a log file: the last 40 lines first, then every new line."""
        self.sse_start(); idle = 0.0
        try:
            while not os.path.exists(path):
                self.wfile.write(b": waiting for log\n\n"); self.wfile.flush(); time.sleep(1)
            with open(path, errors="replace") as f:
                for line in f.read().splitlines()[-40:]: self.sse_write(json.dumps(line))
                while True:
                    line = f.readline()
                    if line: self.sse_write(json.dumps(line.rstrip("\n"))); idle = 0
                    else:
                        time.sleep(0.3); idle += 0.3
                        if idle >= 10: self.wfile.write(b": keepalive\n\n"); self.wfile.flush(); idle = 0
        except (BrokenPipeError, ConnectionResetError, OSError): pass

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if not self.authed(q): return self.send(401, "token required: open the URL with ?token=...")
        extra = (("Set-Cookie", "yh_token=%s; HttpOnly; SameSite=Strict; Path=/" % TOKEN),)
        if u.path == "/": return self.send(200, console().replace("<h1>", ("<img src='/api/sleep?s=%d' hidden><h1>" % min(10, int(q.get("hold", ["0"])[0] or 0))) if q.get("hold") else "<h1>", 1), extra=extra)
        if u.path == "/api/sleep": time.sleep(min(10, int(q.get("s", ["1"])[0] or 1))); return self.send(200, "ok", "text/plain")
        if u.path == "/dashboard": return self.send(200, dashboard(), extra=extra)
        if u.path == "/gallery": return self.send(200, gallery(), extra=extra)
        m = re.match(r"^/media/([\w.\-]+)/(.+)$", u.path)
        if m:
            base = media_base(m.group(1)); return self.serve_file(base, unquote(m.group(2))) if base else self.send(404, "not found")
        m = re.match(r"^/m/([\w.\-]+)/(\d+)/([^/]+)$", u.path)
        if m:
            j = JOBS.get(m.group(1)); fl = (j or {}).get("mfiles", []); i = int(m.group(2))
            if i >= len(fl) or not fl[i]: return self.send(404, "not found")
            return self.serve_file(os.path.dirname(fl[i]), os.path.basename(fl[i]))
        m = re.match(r"^/api/media/([\w.\-]+)$", u.path)
        if m:
            base = media_base(m.group(1)); items = scan_media(base) if base else {}
            return self.send(200, json.dumps([dict(path=r, kind=v[2], size=v[0]) for r, v in sorted(items.items(), key=lambda kv: kv[1][1])][:500]), "application/json")
        if u.path == "/api/engines": return self.send(200, json.dumps(engine_state()), "application/json")
        if u.path == "/api/runs": return self.send(200, json.dumps(load_runs()), "application/json")
        if u.path == "/api/jobs": return self.send(200, json.dumps([job_view(j) for j in sorted(JOBS.values(), key=lambda j: -j["t"])[:30]]), "application/json")
        if u.path == "/api/enginelog": return self.sse_file(os.path.join(DATA, "engine_inkling.log"))
        m = re.match(r"^/api/stream/([\w.\-]+)$", u.path)
        if m and m.group(1) in JOBS: return self.sse_job(JOBS[m.group(1)])
        m = re.match(r"^/api/job/([\w.\-]+)$", u.path)
        if m and m.group(1) in JOBS: return self.send(200, json.dumps(job_view(JOBS[m.group(1)])), "application/json")
        m = re.match(r"^/session/([\w.\-]+)/([\w.\-]+)$", u.path)
        if m:
            s = session_page(*m.groups()); return self.send(200, s) if s else self.send(404, "not found")
        self.send(404, "not found")

    def do_POST(self):
        u = urlparse(self.path)
        if not self.authed({}): return self.send(401, "token required")
        origin = self.headers.get("Origin", "")
        if origin and urlparse(origin).netloc != self.headers.get("Host", ""): return self.send(403, "cross-origin request refused")
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > 200_000: return self.send(413, "too large")
            b = json.loads(self.rfile.read(n) or b"{}")
        except Exception: return self.send(400, "bad json")
        J = lambda o: self.send(200, json.dumps(o), "application/json")
        if u.path == "/api/engine/start": return J(dict(msg=start_engine(str(b.get("name")))))
        if u.path == "/api/engine/stop": return J(dict(msg=stop_engine(str(b.get("name")))))
        if u.path == "/api/app/stop":
            return J(dict(msg=stop_app(int(b.get("port", 0)))))
        if u.path == "/api/job/cancel":
            j = JOBS.get(str(b.get("id")))
            if not j or j["status"] == "done": return self.send(404, "no such running job")
            j["cancel"] = True
            if j.get("proc") and j["proc"].poll() is None: os.killpg(j["proc"].pid, signal.SIGTERM)
            if j.get("eng"): j["eng"].abort()               # a generation that is silent right now
            if j.get("seng"): j["seng"].abort()
            if j.get("sb"): j["sb"].kill()                  # a command that is running right now
            return J(dict(msg="cancelling"))
        if u.path in ("/api/task", "/api/eval"):
            if not healthy("inkling"): return self.send(409, "the engine is not running: press Start and wait until it says running")
            retries = max(0, min(6, int(b.get("retries", 3)))); strat = b.get("strategy") if b.get("strategy") in Y.STRATEGIES else "direct"
            if u.path == "/api/task":
                prompt = str(b.get("prompt", ""))[:20000]; tests = str(b.get("tests", ""))[:20000]; verify = b.get("verify", "tests")
                if not prompt.strip(): return self.send(400, "empty task")
                mode = b.get("mode", "agent"); mt = max(512, min(30000, int(b.get("max_tokens", 16000)))); wall = max(5, min(7200, int(b.get("wall", 900))))
                if mode == "agent":
                    if NO_EXEC or SANDBOX: return self.send(403, "agent mode runs shell commands without a sandbox: not available with --no-exec / --sandbox")
                    job = new_job("agent", prompt); launch(job, run_agent, prompt, max(1, min(80, int(b.get("steps", 25)))), mt, wall, b.get("supervisor", "key") if b.get("supervisor", "key") in ("off", "key", "every") else "key", max(0, min(250000, int(b.get("ctx_total", b.get("ctx_limit", 64000))))), b.get("effort", "medium") if b.get("effort", "medium") in ("low", "medium", "high") else "medium")
                    return J(dict(id=job["id"]))
                if NO_EXEC and verify != "none": return self.send(403, "code execution is disabled (--no-exec): choose verification 'none'")
                job = new_job("task", prompt)
                launch(job, run_task, prompt, tests if verify == "tests" else "", retries, verify == "smoke" or (verify == "tests" and not tests.strip()), strat, mt, wall)
                return J(dict(id=job["id"]))
            if NO_EXEC: return self.send(403, "code execution is disabled (--no-exec)")
            job = new_job("eval", "MBPP eval n=%s" % b.get("n", 10)); launch(job, run_eval, max(1, min(200, int(b.get("n", 10)))), retries, strat, bool(b.get("judge")))
            return J(dict(id=job["id"]))
        self.send(404, "not found")


def main():
    global TOKEN, NO_EXEC, SANDBOX
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None, help="default: this machine's tailnet IPv4 (tailscale ip -4)")
    ap.add_argument("--port", type=int, default=8770); ap.add_argument("--no-exec", action="store_true"); ap.add_argument("--sandbox", action="store_true", help="run code in the soft sandbox (temp dir, rlimits, scrubbed env)")
    a = ap.parse_args(); NO_EXEC = a.no_exec; SANDBOX = a.sandbox; TOKEN = token()
    host = a.host or subprocess.check_output(["tailscale", "ip", "-4"], text=True).split()[0]
    global TAILNET_IP
    TAILNET_IP = host
    load_old_jobs()
    threading.Thread(target=watch_apps, daemon=True).start()
    ThreadingHTTPServer.daemon_threads = True
    srv = ThreadingHTTPServer((host, a.port), H)
    print("yharness console: http://%s:%d/?token=%s   (code execution %s, sandbox %s)" % (host, a.port, TOKEN, "OFF" if NO_EXEC else "ON", "ON" if SANDBOX else "OFF"), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
