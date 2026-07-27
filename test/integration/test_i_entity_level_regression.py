"""Perf regression guard for the three per-entity access paths, against plain-OOP as the yardstick.

`docs/source/benchmarks.md` prices the same physics step (`pos += vel*dt`) four ways and publishes the
result as a *ratio to `oop-scalar`* (plain python floats, no numpy): **zip-rows 16x, pool-loop 19x,
get_entity 29x**. Those three numbers are a contract. `#45` bought get_entity 40x -> 29x by inlining
`_locate` into `Entity.__getattr__`/`__setattr__`, and the two defects found reviewing it (a duplicated
write, a `def __reduce__` nested inside `__setattr__`) were each worth ~200 ns/write while leaving the
whole 464-test correctness suite green. **A correctness suite cannot see a perf regression.** This can.

Why ratios and not nanoseconds: absolutes wander +/-20% run to run and vary per machine, so an
ns/entity assertion is a flake generator. A ratio measured *in the same interleaved run* divides the
machine out. Two denominators are used, because they fail differently:

  * **`x oop-scalar`** -- what the docs quote. The number a reader checks.
  * **`get-entity / zip-rows`** -- the tight one. Both are microecs on the same pools in the same
    process, so it cancels almost everything: 1.77 measured, 2.48 before `#45`.

Anti-flake design, in order of importance:

  1. **Every path is fingerprint-checked before it is timed** (`test_..._paths_all_compute_the_same_step`).
     A path that silently stops doing the work would post a wonderful number. This is the load-bearing
     test of the two; the timing test calls the same builder.
  2. **`min` of N rounds, interleaved round-robin.** Timing noise is one-sided -- interference only ever
     makes a run slower -- so the minimum is the estimator, and interleaving puts all four paths on the
     same thermal state.
  3. **`zip-rows` and `pool-loop` are controls.** Neither touches `Entity`. If a control is out of its
     own band the *machine* is not measuring cleanly, so the test **skips** instead of blaming
     `entity.py`. A perf test that cries wolf gets deleted; one that abstains survives.
  4. **Best-of-3 attempts.** A single interleaved sweep is ~100 ms, so retrying is cheap insurance.
  5. **Skipped on CI** (`CI=true`), and read sceptically on a loaded laptop -- see below.

**The controls have a blind spot, and it produced a false failure (2026-07-27).** Anti-flake measure 3
assumes interference scales every path alike, so a clean control implies a clean machine. It does not:
interference hits the *slowest* path hardest, and `ent` is the slowest by construction. Under load,
`ent/oop` drifted to **35.46** while `zip/oop` (16, ceiling 22) and `pool/oop` (18.6, ceiling 26) stayed
well in band and declined to trip -- so the test failed and blamed `entity.py` for the machine. Re-measured
idle minutes later: **ent/oop 28.1-29.8, ent/zip 1.76-1.86** over 15 samples. Hence measure 5. Note 35.46 is
uncomfortably near a genuine `#45` revert (37.97), so widening the bound was rejected: it would leave ~2
points of daylight between "fine" and "regressed". The environment is the thing to control, not the ceiling.

Thresholds sit roughly midway between today and the pre-`#45` baseline: they catch a real regression
(a revert lands at 40x / 2.48) without tripping on an *idle* laptop. Retune with
`test/manual/get-entity-perf/ratio_stability.py`, which is where N and these bounds came from -- and only
from an idle-machine reading.

Known hole: this pins the *entity* path. If a change slowed `zip-rows` itself by the same factor, the
control would trip and the test would skip rather than fail. That is the deliberate trade.
"""
import gc
import os
import time
from dataclasses import field
import numpy as np
import pytest

from microecs import World, Component

N = 2_000                  # measured sweet spot: ~6.4 ms/round, tightest spread, reproduces the doc ratios
ROUNDS = 15                # min-of-15 interleaved
ATTEMPTS = 3               # a whole sweep is ~100 ms; retry before failing
SEED = 0
DT = np.float32(0.016)     # array paths stay float32 -> no upcast writing back into the pools
DT_PY = 0.016              # scalar OOP uses a python float: the fastest pure-python path

# ratio ceilings. "now" is the measured median on the reference machine; "pre-#45" is what a revert scores.
MAX_ZIP_OVER_OOP = 22.0    # control  (now ~15.9, docs 16x)
MAX_POOL_OVER_OOP = 26.0   # control  (now ~18.6, docs 19x)
MAX_ENT_OVER_OOP = 34.0    # GUARDED  (now ~27.7, docs 29x; a #45 revert measured 37.97 here)
MAX_ENT_OVER_ZIP = 2.15    # GUARDED  (now ~1.77;           a #45 revert measured 2.46 here)


class HasPos(Component):
    position: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})


class HasVel(Component):
    velocity: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})


class HasTag(Component):    # exists only to force a 2nd archetype, so the query spans 2 pools
    tag: np.ndarray = field(metadata={"shape": (1,), "dtype": "float32", "default": None})


class OOPScalar:
    """The yardstick: plain python floats + native scalar math. The fastest thing an ECS replaces."""
    __slots__ = ("px", "py", "vx", "vy")

    def __init__(self, px, py, vx, vy):
        self.px, self.py, self.vx, self.vy = px, py, vx, vy


def _data(n):
    rng = np.random.default_rng(SEED)
    return (rng.uniform(-40, 40, (n, 2)).astype("float32"),
            rng.uniform(-10, 10, (n, 2)).astype("float32"))


def _build(n):
    """A committed 2-pool world plus the equivalent OOP scene, from identical starting data."""
    pos, vel = _data(n)
    world = World(components=[HasPos, HasVel, HasTag])
    for i in range(n):
        if i % 2:                                        # odd entities also carry HasTag -> 2nd archetype
            world.add_entity([HasPos, HasVel, HasTag], position=pos[i].copy(), velocity=vel[i].copy(),
                             tag=np.zeros(1, "float32"))
        else:
            world.add_entity([HasPos, HasVel], position=pos[i].copy(), velocity=vel[i].copy())
    world.update()
    objs = [OOPScalar(float(pos[i, 0]), float(pos[i, 1]), float(vel[i, 0]), float(vel[i, 1])) for i in range(n)]
    return world, objs


def _paths(world, objs):
    """The four spellings of `pos += vel*dt`, mirroring examples/04-benchmark-ecs-vs-oop.py exactly."""
    qr = world.query(HasPos, HasVel)
    ids = [int(x) for x in qr.entity_ids]

    def oop_scalar():
        for o in objs:
            o.px += o.vx * DT_PY
            o.py += o.vy * DT_PY

    def zip_rows():
        for p, v in zip(qr.position, qr.velocity):
            p += v * DT

    def pool_loop():
        for pool in qr.pool_list:
            posa, vela = pool.position, pool.velocity
            for i in range(len(pool)):
                posa[i] += vela[i] * DT

    def get_entity():
        for eid in ids:
            ent = world.get_entity(eid)
            ent.position += ent.velocity * DT

    return {"oop": oop_scalar, "zip": zip_rows, "pool": pool_loop, "ent": get_entity}


def _fingerprint(positions):
    """Order-independent digest of one step's result, so all four paths can be compared."""
    return np.sort(np.asarray(positions, dtype="float64").ravel())


def _sweep():
    """One interleaved round-robin sweep. Returns {path: best_seconds}."""
    world, objs = _build(N)
    ops = _paths(world, objs)
    for op in ops.values():
        op()                                             # warm every path before any of them is timed
    best = {name: float("inf") for name in ops}
    gc_was_on = gc.isenabled()
    gc.disable()
    try:
        for _ in range(ROUNDS):
            for name, op in ops.items():
                t0 = time.perf_counter()
                op()
                best[name] = min(best[name], time.perf_counter() - t0)
    finally:
        if gc_was_on:
            gc.enable()
    return best


# --- 1. correctness: the timing loops must actually be doing the work ---------------------------------------------

def test_i_entity_access_paths_all_compute_the_same_step():
    """All four spellings of `pos += vel*dt` agree -- otherwise the ratio test prices a no-op.

    Each path gets its own freshly built world from the same seed, runs exactly one step, and is
    compared to the OOP scene. This is what stops a broken path from posting a fast number.
    """
    references = {}
    for name in ("oop", "zip", "pool", "ent"):
        world, objs = _build(N)
        _paths(world, objs)[name]()                      # exactly one step
        if name == "oop":
            references[name] = _fingerprint([[o.px, o.py] for o in objs])
        else:
            references[name] = _fingerprint(world.query(HasPos, HasVel).position.numpy())

    for name in ("zip", "pool", "ent"):
        np.testing.assert_allclose(references[name], references["oop"], rtol=1e-6, atol=1e-5,
                                   err_msg=f"path '{name}' does not compute the same step as plain OOP")


# --- 2. perf: the published ratios still hold ----------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("CI") == "true",
                    reason="timing-based; unreliable on shared CI runners. See the note in this test's "
                           "docstring -- run it locally on an idle machine instead.")
def test_i_entity_access_path_ratios_stay_within_published_bounds():
    """`get_entity` must stay near 29x OOP-scalar and ~1.8x zip-rows -- a revert of #45 scores 40x / 2.48.

    Skips (does not fail) when a control path is out of band: that means the machine is too noisy to
    attribute anything to `Entity`.

    **Not run on CI, and not run trustworthily on a loaded laptop either** (added 2026-07-27). The control
    mechanism below has a blind spot that a false failure exposed: interference does not scale every path
    equally, it hits the SLOWEST path hardest, and `ent` is the slowest by construction (~29x oop). So a
    busy machine inflates `ent/oop` while `zip/oop` and `pool/oop` -- measured at 16 and 18.6 against
    ceilings of 22 and 26 -- sit comfortably in band and decline to trip. The observed case scored
    **ent/oop = 35.46** with both controls clean, i.e. the test asserted "this is the entity path, not the
    machine" and was wrong.

    That is close enough to a real `#45` revert (37.97) that the two are indistinguishable under load, which
    is the whole reason for the skip rather than a looser bound: raising the ceiling to swallow 35.46 would
    leave ~2 points between "fine" and "regressed" and gut the test. The bounds stay where they are.

    Re-measured on an idle machine right after that failure
    (`test/manual/get-entity-perf/ratio_stability.py`, 5 reps x 3 values of N):
    **ent/oop 28.1-29.8, ent/zip 1.76-1.86** -- dead on the published 29x, and the spread across 15 samples
    is under 2 points. The bounds are correctly placed; the environment was the variable.
    """
    failures = None
    for attempt in range(ATTEMPTS):
        b = _sweep()
        r = {"zip/oop": b["zip"] / b["oop"], "pool/oop": b["pool"] / b["oop"],
             "ent/oop": b["ent"] / b["oop"], "ent/zip": b["ent"] / b["zip"]}
        shown = ", ".join(f"{k}={v:.2f}" for k, v in r.items())

        noisy = [f"{k}={r[k]:.1f} > {lim}" for k, lim in
                 (("zip/oop", MAX_ZIP_OVER_OOP), ("pool/oop", MAX_POOL_OVER_OOP)) if r[k] > lim]
        failures = [f"{k}={r[k]:.2f} > {lim}" for k, lim in
                    (("ent/oop", MAX_ENT_OVER_OOP), ("ent/zip", MAX_ENT_OVER_ZIP)) if r[k] > lim]

        if not failures:
            return                                       # within bounds -- done, no retry needed
        if noisy and attempt == ATTEMPTS - 1:
            pytest.skip(f"machine too noisy to measure: control path(s) out of band ({'; '.join(noisy)}). "
                        f"All ratios: {shown}")

    pytest.fail(
        f"Entity access path MAY have regressed: {'; '.join(failures)}. "
        f"Best of {ATTEMPTS} interleaved sweeps at N={N}.\n"
        f"Reference (post-#45): ent/oop ~27.7, ent/zip ~1.77. Pre-#45: 40 and 2.48.\n"
        f"CHECK THE MACHINE FIRST. The controls being in band does NOT rule it out: interference hits the "
        f"slowest path hardest and `ent` is the slowest, so a loaded machine moves ent/oop while zip/oop "
        f"and pool/oop stay clean. A measured false failure scored ent/oop=35.46 that way. Confirm with "
        f"`uptime` and re-run `test/manual/get-entity-perf/ratio_stability.py` on an idle machine -- an "
        f"idle reading is ent/oop 28-30, ent/zip 1.76-1.86, and it is tight.\n"
        f"Only if an IDLE machine still reproduces it: retune here and update the get-entity row in "
        f"docs/source/benchmarks.md to match.")
