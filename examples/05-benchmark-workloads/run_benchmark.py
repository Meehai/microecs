"""Run every workload x library x N, verify every result, print tables + the winner map.

Layout: one subfolder per workload (wN_*), one file per library inside it, each exposing
`build(n)`, `step(state)`, `collect(state)`. This driver imports `<workload>.<lib>`, runs it
through common.run_workload (which verifies against a float64 reference), and reports timings.
A missing file means "this library can't express this workload" -- recorded as N/A with the
reason from NA below (a real capability gap, never faked).

Usage:
    python run_benchmark.py                 # full matrix + columnar tail -> results.json + tables
    python run_benchmark.py 200 1000 5000   # custom N list (main matrix only)
"""
import importlib
import json
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # find common + workload packages
import numpy as np
import common as C

LIBS      = ["microecs", "xecs", "esper", "snecs", "ecs_pattern"]
WORKLOADS = ["w1_physics", "w2_bounce", "w3_ai", "w4_random", "w5_churn", "w6_mixed", "w7_migrate"]
MAIN_NS   = [200, 1000, 5000, 20000, 100000]
TAIL_NS   = [200000, 500000, 1000000]          # columnar-only, microecs vs xecs

# capability gaps -- a missing <workload>/<lib>.py is reported N/A with these reasons.
NA = {
    ("w5_churn",   "xecs"):        "no despawn primitive (fixed-capacity, spawn-only)",
    ("w7_migrate", "xecs"):        "no component migration (fixed per-component pools)",
    ("w7_migrate", "ecs_pattern"): "entities are fixed inheritance classes (no runtime add/remove)",
}


def _load(workload, lib):
    """Import <workload>.<lib> and return (build, step, collect), or None if absent/NA."""
    try:
        mod = importlib.import_module(f"{workload}.{lib}")
    except ModuleNotFoundError:
        return None
    return mod.build, mod.step, mod.collect


def run_matrix(ns):
    rows = []
    for n in ns:
        refs = C.references(n)
        print(f"\n### N = {n:,}  (k={C.k_for(n)} touched/frame, b={C.b_for(n)} churned/frame)", flush=True)
        for wl in WORKLOADS:
            for lib in LIBS:
                if (wl, lib) in NA:
                    rows.append({"workload": wl, "library": lib, "n": n, "na": True, "reason": NA[(wl, lib)]})
                    print(f"  {lib:12} {wl:11} N/A  {NA[(wl, lib)]}", flush=True)
                    continue
                fns = _load(wl, lib)
                if fns is None:
                    rows.append({"workload": wl, "library": lib, "n": n, "na": True, "reason": "no adapter"})
                    print(f"  {lib:12} {wl:11} N/A  no adapter", flush=True)
                    continue
                try:
                    r = C.run_workload(wl, *fns, refs[wl], n)
                except Exception as e:      # a crash on one cell must not kill the whole sweep
                    r = {"workload": wl, "n": n, "error": f"{type(e).__name__}: {e}"}
                r["library"] = lib
                tag = "ok" if r.get("ok") else ("ERR" if r.get("error") else "FAIL")
                extra = (r.get("error") or (f"{r['step_ms']:.4f}ms {r['ns_per_entity']:.1f}ns/e"
                         if tag == "ok" else f"d={r.get('max_diff')}"))
                print(f"  {lib:12} {wl:11} {tag:4} {extra}", flush=True)
                rows.append(r)
    return rows


def run_tail():
    """microecs vs xecs, w1/w2 only, at large N -- locate the columnar crossover asymptote."""
    rows = []
    for n in TAIL_NS:
        refs = C.references(n)
        print(f"\n### TAIL N = {n:,} (microecs vs xecs, columnar)", flush=True)
        for lib in ("microecs", "xecs"):
            for wl in ("w1_physics", "w2_bounce"):
                fns = _load(wl, lib)
                if fns is None:
                    continue
                try:
                    r = C.run_workload(wl, *fns, refs[wl], n)
                except Exception as e:
                    r = {"workload": wl, "n": n, "error": f"{type(e).__name__}: {e}"}
                r["library"] = lib
                r["tail"] = True
                if r.get("ok"):
                    print(f"  {lib:12} {wl:11} {r['step_ms']:.4f}ms {r['ns_per_entity']:.2f}ns/e", flush=True)
                rows.append(r)
    return rows


def _cell(rows, lib, wl, n):
    for r in rows:
        if r.get("library") == lib and r["workload"] == wl and r["n"] == n and not r.get("tail"):
            if r.get("na"):     return "  N/A"
            if r.get("error"):  return "  ERR"
            if not r.get("ok"): return " FAIL"
            return f"{r['step_ms']:7.4f}"
    return "    -"


def print_tables(rows, ns):
    for wl in WORKLOADS:
        print(f"\n=== {wl}  (step ms/frame, lower is better) ===")
        print(f"{'library':<13}" + "".join(f"{'N=' + str(n):>10}" for n in ns))
        for lib in LIBS:
            print(f"{lib:<13}" + "".join(f"{_cell(rows, lib, wl, n):>10}" for n in ns))


def print_winner_map(rows, ns):
    """Fastest library per (workload, N) -- the headline: the ranking flips by workload AND N."""
    print("\n=== winner map (fastest library per cell) ===")
    print(f"{'workload':<13}" + "".join(f"{'N=' + str(n):>13}" for n in ns))
    for wl in WORKLOADS:
        cells = []
        for n in ns:
            best, who = float("inf"), "-"
            for lib in LIBS:
                for r in rows:
                    if (r.get("library") == lib and r["workload"] == wl and r["n"] == n
                            and not r.get("tail") and r.get("ok") and r["step_ms"] < best):
                        best, who = r["step_ms"], lib
            cells.append(who)
        print(f"{wl:<13}" + "".join(f"{c:>13}" for c in cells))


def main():
    ns = [int(x) for x in sys.argv[1:]] or MAIN_NS
    rows = run_matrix(ns)
    tail = run_tail() if ns == MAIN_NS else []
    meta = {
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "warmup": C.WARMUP, "measure": C.MEASURE,
        "k_rule": "max(16, n//50)", "b_rule": "max(16, n//100)", "k_mig_rule": "max(4, n//200)",
        "libs": LIBS, "workloads": WORKLOADS,
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w") as f:
        json.dump({"meta": meta, "results": rows + tail}, f, indent=1)
    print_tables(rows, ns)
    print_winner_map(rows, ns)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
