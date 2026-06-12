"""Shared workload + harness for the cross-library ECS benchmark.

One physics frame = two batched systems, applied in this order (semi-implicit Euler):
    1. integrate velocity:  vel += acc * dt   (only the entities that have Acceleration)
    2. integrate position:  pos += vel * dt   (every entity)

Every library script simulates the SAME scene: same entity count, same initial data
(same SEED), same number of steps. Each result is checked against a float64 numpy
ground truth via an order-independent fingerprint -- so a library cannot look fast by
silently skipping work.

Fairness note (representation):
    Each library uses its OWN idiomatic, best-case layout.
    - The pure-python object ECSs (esper, snecs, ecs-pattern) store position/velocity as
      plain python-float scalars -- their fastest path. (Per-entity numpy (2,) arrays are
      ~13x slower for them; see the microecs README microbenchmark.)
    - microecs stores (2,) float32 arrays -- its natural vectorized layout.
    We compare the *computation*, not the storage. Results are verified equal within a
    tolerance loose enough for microecs' float32 accumulation vs the float64 reference.
"""
import sys
import time
from importlib.metadata import version, PackageNotFoundError
import numpy as np

# ---- workload config (shared by every script) ----
N_DEFAULT = 100_000          # number of entities in the scene
WARMUP    = 3                # untimed steps, run before measuring
MEASURE   = 30               # timed steps; the reported step time is the per-step mean
SEED      = 0
DT        = 0.016            # python float (float64) -- the pure-python libs use this
DT32      = np.float32(DT)   # float32 -- microecs, so writes stay in-dtype (no upcast)

# fingerprint tolerance (loose enough for microecs' float32 path vs the float64 reference)
_RTOL, _ATOL = 1e-3, 1e-2


def make_data(n):
    """Deterministic initial scene. Odd-index entities (half) carry acceleration.

    Returns float64 arrays pos0 (n,2), vel0 (n,2), acc (n,2) and a bool mask has_acc (n,).
    """
    rng = np.random.default_rng(SEED)
    pos0 = rng.uniform(-40, 40, (n, 2))
    vel0 = rng.uniform(-10, 10, (n, 2))
    acc  = rng.uniform(-10, 10, (n, 2))
    has_acc = (np.arange(n) % 2 == 1)
    return pos0, vel0, acc, has_acc


def lib_version(dist_name):
    """Installed version of a distribution, or '?' if it can't be found."""
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return "?"


def _reference_positions(n, steps):
    """float64 ground-truth final positions after `steps` frames (vectorized numpy)."""
    pos, vel, acc, has_acc = make_data(n)
    pos = pos.copy()
    vel = vel.copy()
    a = acc[has_acc]
    for _ in range(steps):
        vel[has_acc] += a * DT
        pos += vel * DT
    return pos


def fingerprint(positions):
    """Order-independent signature of a scene: every x and y coordinate, pooled and sorted.

    Robust to entity ordering differing between libraries, and to float32 vs float64,
    while still catching any library that skips entities or uses the wrong dynamics.
    """
    return np.sort(np.asarray(positions, dtype=np.float64).ravel())


def check(positions, n, steps):
    """Return (ok, max_abs_diff) comparing a library's final positions to ground truth."""
    fp = fingerprint(positions)
    ref = fingerprint(_reference_positions(n, steps))
    if fp.shape != ref.shape:
        return False, float("inf")
    max_diff = float(np.max(np.abs(fp - ref))) if fp.size else 0.0
    return bool(np.allclose(fp, ref, rtol=_RTOL, atol=_ATOL)), max_diff


class Timer:
    """`with Timer() as t: ...` then read `t.seconds`."""
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0


def summarize(name, ver, n, build_s, measured_s, measure, ok, max_diff):
    """Pack one library's run into a result dict (one frame = both systems)."""
    step_s = measured_s / measure
    return {
        "name": name,
        "version": ver,
        "n": n,
        "build_s": build_s,
        "step_s": step_s,
        "ns_per_entity": step_s / n * 1e9,
        "m_updates_per_s": n / step_s / 1e6,
        "ok": ok,
        "max_diff": max_diff,
    }


def cli_n_measure(argv=None):
    """Parse optional `[N] [measure_steps]` argv; fall back to the shared defaults.

    Used by every script's main() and by run_all so `python bench_x.py 20000 15` works
    the same everywhere.
    """
    argv = sys.argv if argv is None else argv
    n = int(argv[1]) if len(argv) > 1 else N_DEFAULT
    measure = int(argv[2]) if len(argv) > 2 else MEASURE
    return n, measure


def run_bench(name, ver, build, step, collect, n=N_DEFAULT, warmup=WARMUP, measure=MEASURE):
    """Shared driver: build scene (timed) -> warmup -> measure steps (timed) -> verify.

    `build(n)` returns an opaque state; `step(state)` runs one frame; `collect(state)`
    returns the final positions for the correctness check. Every library script funnels
    through here, so the methodology is identical across all of them.
    """
    with Timer() as build_timer:
        state = build(n)
    for _ in range(warmup):
        step(state)
    with Timer() as step_timer:
        for _ in range(measure):
            step(state)
    ok, max_diff = check(collect(state), n, warmup + measure)
    return summarize(name, ver, n, build_timer.seconds, step_timer.seconds, measure, ok, max_diff)


def print_result(r):
    """One readable line for a single library's result."""
    flag = "ok" if r["ok"] else f"MISMATCH(maxdiff={r['max_diff']:.2e})"
    print(f"{r['name']:<14} v{r['version']:<8} "
          f"build={r['build_s'] * 1e3:8.1f}ms  "
          f"step={r['step_s'] * 1e3:8.3f}ms  "
          f"{r['ns_per_entity']:9.1f} ns/entity/frame  "
          f"{r['m_updates_per_s']:7.1f} M/s  [{flag}]")
