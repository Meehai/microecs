#!/usr/bin/env python3
"""Focused tracker for microecs task #26 — the low-N columnar overhead.

microecs wins columnar physics at N≥20k but LOSES to xecs in the 1k-10k band (Part 8 crossover ~10k).
The cause is fixed per-op Python overhead in QueryResult/Field that does not scale with N: each
`qr.field` access allocates a fresh Field, each op/write allocates more. This benchmark isolates that.

Single archetype (every entity has Pos+Vel+Acc -> one pool -> the single-pool fast path we want to
optimize). One frame = `vel += acc*dt; pos += vel*dt`. Three contestants at each N:
  microecs     the lib under test (QueryResult write-through)
  xecs         the lib that wins this band today (Rust SoA, in-place +=)
  numpy-floor  the same math in-place on raw numpy arrays -- the theoretical floor microecs should approach

Every result is verified against a float64 reference. A per-phase breakdown at the end shows WHERE
microecs' time goes (query lookup vs one Field access vs the full step) so the fix has a target.

    python run.py                 # default N sweep + breakdown
    python run.py 1000 5000       # custom N list
"""
import gc
import sys
import time
from typing import Any, Callable, TypeVar
import numpy as np
from dataclasses import field
from microecs import World, Component
import xecs as xx

_State = TypeVar("_State")  # per-contestant state passed build -> step -> collect

SEED = 0
DT = 0.016
DT32 = np.float32(DT)
WARMUP, MEASURE = 5, 50
FRAMES = WARMUP + MEASURE
DEFAULT_NS = [200, 500, 1000, 2000, 5000, 10000, 20000]
_RTOL, _ATOL = 2e-3, 2e-2


def scene(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    return (rng.uniform(-40, 40, (n, 2)), rng.uniform(-10, 10, (n, 2)), rng.uniform(-10, 10, (n, 2)))


def reference(n: int) -> np.ndarray:
    """float64 ground truth: sorted pooled positions after FRAMES steps."""
    pos, vel, acc = (a.copy() for a in scene(n))
    for _ in range(FRAMES):
        vel += acc * DT
        pos += vel * DT
    return np.sort(pos.ravel())


def _fp(a: np.ndarray) -> np.ndarray:
    return np.sort(np.asarray(a, np.float64).ravel())


def _ok(fp: np.ndarray, ref: np.ndarray) -> bool:
    return fp.shape == ref.shape and bool(np.allclose(fp, ref, rtol=_RTOL, atol=_ATOL))


# --------------------------------------------------------------------------- microecs
def _f(shape: tuple[int, ...]) -> Any:
    return field(metadata={"shape": shape, "dtype": "float32", "default": None})


class Pos(Component):
    position: np.ndarray = _f((2,))
    velocity: np.ndarray = _f((2,))
class Acc(Component):
    acceleration: np.ndarray = _f((2,))


def mecs_build(n: int) -> World:
    p, v, a = scene(n)
    w = World([Pos, Acc])
    for i in range(n):
        w.add_entity([Pos, Acc], position=np.asarray(p[i], np.float32),
                     velocity=np.asarray(v[i], np.float32), acceleration=np.asarray(a[i], np.float32))
    w.update()
    return w


def mecs_step(w: World) -> None:
    qr = w.query(Pos, Acc)
    qr.velocity[:] = qr.velocity + qr.acceleration * DT32
    qr.position[:] = qr.position + qr.velocity * DT32


def mecs_collect(w: World) -> np.ndarray:
    return _fp(w.query(Pos, Acc).position.numpy())


# --------------------------------------------------------------------------- xecs
class Position(xx.Component):
    x: xx.Float32
    y: xx.Float32
class Velocity(xx.Component):
    x: xx.Float32
    y: xx.Float32
class Acceleration(xx.Component):
    x: xx.Float32
    y: xx.Float32


def _c(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, np.float32)


def xecs_build(n: int) -> xx.World:
    p, v, a = scene(n)
    app = xx.SimulationApp()
    app.add_pool(Position.create_pool(n)); app.add_pool(Velocity.create_pool(n))
    app.add_pool(Acceleration.create_pool(n))
    cmd, world = app._commands, app.world
    cmd.spawn([Position, Velocity, Acceleration], n)
    pv, vv, av = world.get_view(Position), world.get_view(Velocity), world.get_view(Acceleration)
    pv.x.fill(_c(p[:, 0])); pv.y.fill(_c(p[:, 1]))
    vv.x.fill(_c(v[:, 0])); vv.y.fill(_c(v[:, 1]))
    av.x.fill(_c(a[:, 0])); av.y.fill(_c(a[:, 1]))
    return world


def xecs_step(world: xx.World) -> None:
    pos, vel, acc = world.get_view(Position), world.get_view(Velocity), world.get_view(Acceleration)
    vel.x += acc.x * DT
    vel.y += acc.y * DT
    pos.x += vel.x * DT
    pos.y += vel.y * DT


def xecs_collect(world: xx.World) -> np.ndarray:
    pv = world.get_view(Position)
    return _fp(np.column_stack([pv.x.numpy(), pv.y.numpy()]))


# ------------------------------------------------------------------ raw-numpy floor
def np_build(n: int) -> dict[str, np.ndarray]:
    p, v, a = (_c(x) for x in scene(n))
    return {"pos": p, "vel": v, "acc": a}


def np_step(st: dict[str, np.ndarray]) -> None:
    st["vel"] += st["acc"] * DT32
    st["pos"] += st["vel"] * DT32


def np_collect(st: dict[str, np.ndarray]) -> np.ndarray:
    return _fp(st["pos"])


# --------------------------------------------------------------------------- timing
def _reps(n: int) -> int:
    return 8 if n <= 5000 else 4


def bench(build: Callable[[int], _State], step: Callable[[_State], None],
          collect: Callable[[_State], np.ndarray], n: int) -> tuple[float, np.ndarray]:
    best = float("inf")
    fp: np.ndarray | None = None
    for _ in range(_reps(n)):
        gc.collect(); gc.disable()
        try:
            st = build(n)
            for _ in range(WARMUP):
                step(st)
            t0 = time.perf_counter()
            for _ in range(MEASURE):
                step(st)
            best = min(best, (time.perf_counter() - t0) / MEASURE)
            fp = collect(st)
        finally:
            gc.enable()
    assert fp is not None  # >=1 rep always runs -> fp is set
    return best, fp


def _t(fn: Callable[[], Any], iters: int = 200) -> float:
    for _ in range(5):
        fn()
    gc.collect(); gc.disable()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    dt = (time.perf_counter() - t0) / iters
    gc.enable()
    return dt


def breakdown(n: int) -> None:
    """Where does microecs' per-frame time go? Isolate query lookup / one Field access / full step."""
    w = mecs_build(n)
    t_query = _t(lambda: w.query(Pos, Acc))
    qr = w.query(Pos, Acc)
    t_access = _t(lambda: qr.velocity)                       # one __getattr__ -> one Field alloc
    t_step = _t(lambda: mecs_step(w))
    npst = np_build(n)
    t_floor = _t(lambda: np_step(npst))
    print(f"\n=== microecs per-frame breakdown at N={n:,} (us) ===")
    print(f"  query(Pos,Acc) lookup        {t_query*1e6:8.3f}")
    print(f"  one qr.field access (Field)  {t_access*1e6:8.3f}")
    print(f"  full step (2 field-updates)  {t_step*1e6:8.3f}")
    print(f"  raw-numpy floor (same math)  {t_floor*1e6:8.3f}   <- target")
    print(f"  overhead over floor          {(t_step-t_floor)*1e6:8.3f}  ({t_step/t_floor:.1f}x)")


def main() -> None:
    ns = [int(x) for x in sys.argv[1:]] or DEFAULT_NS
    print(f"# columnar physics (single archetype), {WARMUP} warmup + {MEASURE} timed frames, min over reps")
    print(f"\n{'N':>7} {'microecs':>10} {'xecs':>10} {'np-floor':>10} {'mecs/xecs':>11} {'mecs/floor':>11}  ok")
    for n in ns:
        (m, mfp), (x, xfp), (f, ffp) = (bench(mecs_build, mecs_step, mecs_collect, n),
                                        bench(xecs_build, xecs_step, xecs_collect, n),
                                        bench(np_build, np_step, np_collect, n))
        ref = reference(n)
        ok = _ok(mfp, ref) and _ok(xfp, ref) and _ok(ffp, ref)
        band = " <-- target band" if 500 <= n <= 10000 else ""
        print(f"{n:>7} {m*1e3:>9.4f}m {x*1e3:>9.4f}m {f*1e3:>9.4f}m "
              f"{m/x:>10.2f}x {m/f:>10.2f}x  {'ok' if ok else 'FAIL'}{band}")
    for n in (500, 5000):
        breakdown(n)


if __name__ == "__main__":
    main()
