"""Silent-death monitor for bots.

The most expensive bot failure is often not a bad trade but a bot that silently stopped: the
scheduler says "OK", the process is alive, yet nothing has been written for hours.

A :class:`PulseMonitor` runs a list of *checks*. Each check returns ``{key: error_or_None}``.
A problem must be seen on ``confirm`` passes in a row before ``notify`` is called (no flapping),
and recovery is reported once. Notification and publishing are plain callbacks, so you can plug in
Telegram, Slack, e-mail, a log file or nothing - no endpoints are hard-coded here.

Built-in checks:
* :func:`file_fresh` - a file or folder (newest file inside) was modified recently.
* :func:`windows_tasks_check` - Windows Task Scheduler (via the COM API through PowerShell):
  repeating tasks must have run within 2 x interval + 5 min with an OK result code; logon/boot
  ("loop") tasks must be in the Running state. Windows only; optional.

Example::

    mon = PulseMonitor([file_fresh("bot_a/trades.log", 30)], state_path="pulse_state.json",
                       notify=print)
    mon.loop(period_s=300)
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import time
from typing import Callable, Dict, Iterable, List, Optional

Check = Callable[[], Dict[str, Optional[str]]]
SKIP_EXT = {".py", ".pyc", ".lock", ".bak", ".md"}


# ----------------------------------------------------------------------------- generic checks
def newest_mtime(path: str, max_depth: int = 2, limit: int = 4000) -> float:
    """mtime of a file, or of the newest non-code file in a folder (0.0 if none)."""
    if os.path.isfile(path):
        return os.path.getmtime(path)
    best, n, base = 0.0, 0, path.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "node_modules", "venv", ".venv")]
        if root.count(os.sep) - base >= max_depth:
            dirs[:] = []
        for f in files:
            if os.path.splitext(f)[1].lower() in SKIP_EXT:
                continue
            try:
                best = max(best, os.path.getmtime(os.path.join(root, f)))
            except OSError:
                pass
            n += 1
            if n > limit:
                return best
    return best


def file_fresh(path: str, max_age_min: float, name: Optional[str] = None,
               now: Optional[Callable[[], float]] = None) -> Check:
    """Check that ``path`` (file or folder) produced output in the last ``max_age_min`` minutes."""
    key = f"fresh:{name or path}"
    clock = now or time.time

    def _check():
        if not os.path.exists(path):
            return {key: f"{name or path}: path does not exist"}
        m = newest_mtime(path)
        age = (clock() - m) / 60 if m else None
        if age is None or age > max_age_min:
            return {key: f"{name or path}: no new output for "
                         f"{'ever' if age is None else f'{age:.0f} min'} (limit {max_age_min:g}) - silent death?"}
        return {key: None}
    return _check


# ----------------------------------------------------------------------------- Windows Task Scheduler
OK_CODES = {0, 267009, 267011, 267014, 267008}   # success, running, ...
_TASKS_PS = r"""
$s = New-Object -ComObject Schedule.Service; $s.Connect()
$tm = @{0='Event';1='Time';2='Daily';3='Weekly';4='Monthly';8='Boot';9='Logon'}
$sm = @{0='Unknown';1='Disabled';2='Queued';3='Ready';4='Running'}
$out = foreach ($t in $s.GetFolder('__FOLDER__').GetTasks(1)) {
  $d = $t.Definition; $lr = [datetime]$t.LastRunTime
  if ($lr.Year -lt 2000) { $ts = '' } else { $ts = ([datetimeoffset]$lr).ToString('o') }
  [pscustomobject]@{n=$t.Name; s=$(if ($t.Enabled) { $sm[[int]$t.State] } else { 'Disabled' });
    r=[int64]([uint32]$t.LastTaskResult); t=$ts;
    trig=@(foreach($g in $d.Triggers){ "$($tm[[int]$g.Type])|$($g.Repetition.Interval)|$(try{$g.DaysInterval}catch{''})|$(try{$g.WeeksInterval}catch{''})" }) }
}
$out | ConvertTo-Json -Depth 4 -Compress
"""


def iso_duration_minutes(s: str) -> Optional[float]:
    """'PT5M' -> 5, 'PT1H30M' -> 90, 'P1D' -> 1440, '' -> None."""
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s or "")
    if not m or not any(m.groups()):
        return None
    d, h, mi, se = (int(x or 0) for x in m.groups())
    return d * 1440 + h * 60 + mi + se / 60


def task_interval(triggers: Iterable[str]):
    """Shortest interval in minutes from trigger strings 'Kind|PTxM|days|weeks', 'loop' for logon/boot
    tasks with no repetition, or None (event-only)."""
    best, loop = None, False
    for tr in triggers:
        kind, rep, days, weeks = (tr.split("|") + ["", "", "", ""])[:4]
        v = iso_duration_minutes(rep)
        if v is None:
            if kind == "Daily":
                v = 1440 * int(days or 1)
            elif kind == "Weekly":
                v = 10080 * int(weeks or 1)
            elif kind in ("Logon", "Boot"):
                loop = True
        if v is not None:
            best = v if best is None else min(best, v)
    if best is None and loop:
        return "loop"
    return best


def judge_task(t: dict, now: dt.datetime) -> Optional[str]:
    """Error string for one task dict {n, s, r, t, trig} or None if healthy."""
    iv = task_interval(t.get("trig") or [])
    if iv is None or t.get("s") == "Disabled":
        return None
    if iv == "loop":
        return None if t.get("s") == "Running" else f"{t['n']}: loop bot not running (state {t.get('s')})"
    errs = []
    if t.get("t"):
        age = (now - dt.datetime.fromisoformat(t["t"])).total_seconds() / 60
        if age > 2 * iv + 5:
            errs.append(f"last run {age:.0f} min ago (runs every {iv:.0f} min)")
        if t.get("r") not in OK_CODES:
            errs.append(f"last result code {t.get('r')}")
    return f"{t['n']}: " + "; ".join(errs) if errs else None


def read_windows_tasks(folder: str = "\\") -> List[dict]:
    """All tasks in a Task Scheduler folder (Windows only)."""
    ps = _TASKS_PS.replace("__FOLDER__", folder.replace("'", "''"))
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                         timeout=90, encoding="utf-8", errors="replace",
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    j = json.loads(out or "[]")
    return j if isinstance(j, list) else [j]


def windows_tasks_check(name_pattern: str = ".*", folder: str = "\\",
                        reader: Callable[[], List[dict]] = None) -> Check:
    """Check every enabled task whose name matches ``name_pattern``."""
    rx = re.compile(name_pattern)
    read = reader or (lambda: read_windows_tasks(folder))

    def _check():
        now = dt.datetime.now().astimezone()
        try:
            tasks = read()
        except Exception as e:
            return {"tasks:read": f"cannot read Task Scheduler: {e}"}
        return {f"task:{t['n']}": judge_task(t, now) for t in tasks if rx.search(t.get("n", ""))}
    return _check


# ----------------------------------------------------------------------------- the monitor
class PulseMonitor:
    def __init__(self, checks: List[Check], state_path: str = "pulse_state.json", confirm: int = 2,
                 notify: Optional[Callable[[str], None]] = None,
                 publish: Optional[Callable[[dict], None]] = None, label: str = "bots"):
        self.checks, self.state_path, self.confirm = checks, state_path, max(1, int(confirm))
        self.notify, self.publish, self.label = notify, publish, label

    def _load(self) -> dict:
        try:
            with open(self.state_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def run_once(self) -> dict:
        res: Dict[str, Optional[str]] = {}
        for c in self.checks:
            try:
                res.update(c())
            except Exception as e:  # a broken check must not kill the monitor
                res[f"check:{getattr(c, '__name__', 'check')}"] = f"check crashed: {e!r}"
        st = self._load()
        bad, good = [], []
        for k, err in res.items():
            s = st.setdefault(k, {"fails": 0, "alerted": False})
            if err:
                s["fails"] += 1
                s["last"] = err
                if s["fails"] >= self.confirm and not s["alerted"]:
                    bad.append(err)
                    s["alerted"] = True
            else:
                if s["alerted"]:
                    good.append(f"{k.split(':', 1)[-1]} is OK again")
                s["fails"], s["alerted"] = 0, False
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(st, f, indent=1)
        summary = {"at": dt.datetime.now().isoformat(timespec="seconds"), "label": self.label,
                   "problems": sum(1 for e in res.values() if e), "errors": [e for e in res.values() if e],
                   "new_alerts": bad, "recovered": good}
        if self.notify and bad:
            self.notify(f"[{self.label}] problems\n" + "\n".join(f"- {b}" for b in bad))
        if self.notify and good:
            self.notify(f"[{self.label}] back to normal\n" + "\n".join(f"- {g}" for g in good))
        if self.publish:
            try:
                self.publish(summary)
            except Exception:
                pass
        return summary

    def loop(self, period_s: float = 300, max_passes: Optional[int] = None) -> None:
        k = 0
        while max_passes is None or k < max_passes:
            t0 = time.time()
            self.run_once()
            k += 1
            if max_passes is not None and k >= max_passes:
                break
            time.sleep(max(1.0, period_s - (time.time() - t0)))
