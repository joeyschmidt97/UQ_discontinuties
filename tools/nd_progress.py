"""Progress view for the high-dimensional runs: one line per worker, live.

Reads the worker logs written by `python -m benchmarknd`, so it never touches
the results files and can be started, stopped and restarted at any time.
"""
import argparse
import pathlib
import re
import time

DONE = re.compile(r"\[(\d\d:\d\d:\d\d)\] done\s+(\S+) seed=(\d+) (\S+) final ([\d.]+) in ([\d.]+) min")
START = re.compile(r"\[(\d\d:\d\d:\d\d)\] start (\S+) seed=(\d+) (\S+) budget=(\d+)")
FAIL = re.compile(r"\[(\d\d:\d\d:\d\d)\] FAILED (\S+) seed=(\d+) (\S+): (.+)")
CHECK = re.compile(r"\s+N=\s*(\d+) error ([\d.]+)")


def read_log(path):
    """Workers launched from PowerShell redirect as UTF-16; plain runs as UTF-8."""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or raw[1:2] == b"\x00":
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def finished_from_results(results_dir):
    """Completed trajectories per worker, read from the saved results files.

    The logs are truncated whenever a worker restarts, so they undercount after
    a resume. The results files are the authoritative record.
    """
    import json
    counts = {}
    for path in sorted(pathlib.Path(results_dir).glob("*/results.json")):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
        except (OSError, ValueError):
            continue
        counts[path.parent.name] = len({r["arm"] for r in rows
                                        if r.get("status") == "ok" and r.get("n") == r.get("budget")})
    return counts


def snapshot(log_dir, expected_arms, results_dir=None):
    workers, done, failed = {}, 0, []
    saved = finished_from_results(results_dir) if results_dir else {}
    for path in sorted(pathlib.Path(log_dir).glob("*.log")):
        text = read_log(path)
        finished = DONE.findall(text)
        failures = FAIL.findall(text)
        started = START.findall(text)
        last_check = CHECK.findall(text)
        done += max(saved.get(path.stem, 0), len(finished))
        failed.extend(failures)
        current = started[-1] if started else None
        workers[path.stem] = dict(
            completed=max(saved.get(path.stem, 0), len(finished)+len(failures)), expected=expected_arms,
            arm=current[3] if current else "-",
            at=f"N={last_check[-1][0]} err {last_check[-1][1]}" if last_check else "-",
            last=finished[-1] if finished else None)
    return workers, done, failed


def render(log_dir, expected_arms, results_dir=None):
    workers, done, failed = snapshot(log_dir, expected_arms, results_dir)
    total = len(workers)*expected_arms
    width = max((len(name) for name in workers), default=10)
    lines = [f"{'worker':{width}}  {'progress':>9}  {'running':16}  latest checkpoint"]
    for name, state in workers.items():
        lines.append(f"{name:{width}}  {state['completed']:3}/{state['expected']:<5}  "
                     f"{state['arm']:16}  {state['at']}")
    lines.append("")
    lines.append(f"trajectories finished {done}/{total}   workers {len(workers)}   failures {len(failed)}")
    for failure in failed[:5]:
        lines.append(f"  FAILED {failure[1]} seed={failure[2]} {failure[3]}: {failure[4][:90]}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=pathlib.Path, required=True)
    parser.add_argument("--results", type=pathlib.Path, help="run directory holding <worker>/results.json")
    parser.add_argument("--arms", type=int, default=7, help="trajectories expected per worker")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=20.)
    args = parser.parse_args()
    while True:
        print("\033[2J\033[H" + time.strftime("%H:%M:%S") + "\n"
              + render(args.logs, args.arms, args.results), flush=True)
        if not args.watch:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
