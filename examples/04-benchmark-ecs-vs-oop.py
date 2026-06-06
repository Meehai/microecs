"""Standalone benchmark: ECS (microecs) vs OOP, same physics step computed 8 ways.

Intended home: examples/04-benchmark-ecs-vs-oop.py (kept under test/manual/ per project rules
that route benchmarks here; the dev promotes it to examples/ with a single `mv`).

Every mode applies the SAME step -- pos += vel*dt over N entities, the same number of times,
from the same seed -- so they all produce the SAME result (asserted in main via an
order-independent fingerprint). We report the average wall-time per step, in seconds.

Run:    python test/manual/perf/bench_ecs_vs_oop.py
Output: {mode: avg_seconds_per_step}
"""
import time
from dataclasses import field
import numpy as np

from microecs import World, Component

# ---- config (globals) ----
N = 100_000
REPEATS = 7
SEED = 0
DT = np.float32(0.016)     # array variants keep float32 -> no upcast writing back to pools
DT_PY = 0.016              # scalar OOP uses a python float: the fastest pure-python path

MODES = [
    "micro-ecs-pool-vectorized",
    "micro-ecs-vectorized",
    "oop-scalar",
    "oop-numpy",
    "micro-ecs-zip-rows",
    "micro-ecs-pool-loop",
    "micro-ecs-index",
    "micro-ecs-get-entity",
]


# ---- components: (2,) pos/vel; HasTag only exists to force a 2nd pool ----
class HasPos(Component):
    position: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})
class HasVel(Component):
    velocity: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})
class HasTag(Component):
    tag: np.ndarray = field(metadata={"shape": (1,), "dtype": "float32"})


# ---- OOP object-first scene (the thing an ECS replaces) ----
class OOPNumpy:
    """Each entity is an object owning its own (2,) numpy arrays."""
    __slots__ = ("position", "velocity")
    def __init__(self, position, velocity):
        self.position, self.velocity = position, velocity

class OOPScalar:
    """Steel-man OOP: plain python floats + native scalar math."""
    __slots__ = ("px", "py", "vx", "vy")
    def __init__(self, px, py, vx, vy):
        self.px, self.py, self.vx, self.vy = px, py, vx, vy


# ---- shared helpers ----
def _data(n):
    rng = np.random.default_rng(SEED)
    pos = rng.uniform(-40, 40, (n, 2)).astype("float32")
    vel = rng.uniform(-10, 10, (n, 2)).astype("float32")
    return pos, vel

def _world(n):
    pos, vel = _data(n)
    w = World(components=[HasPos, HasVel, HasTag])
    for i in range(n):
        if i % 2:   # odd entities also carry HasTag -> a 2nd archetype, so the query spans 2 pools
            w.add_entity(components=[HasPos, HasVel, HasTag],
                         position=pos[i].copy(), velocity=vel[i].copy(), tag=np.zeros(1, "float32"))
        else:
            w.add_entity(components=[HasPos, HasVel], position=pos[i].copy(), velocity=vel[i].copy())
    w.update()
    return w

def _bench(step):
    """mean seconds per step over REPEATS runs, after one warmup."""
    step()
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        step()
    return (time.perf_counter() - t0) / REPEATS

def _key(positions):
    """order-independent fingerprint of the result, so all variants can be checked equal."""
    return np.sort(np.asarray(positions, dtype="float64").ravel())


# ---- the 8 modes: each builds size-n data, times one step, returns (avg_seconds, result_key) ----

def micro_ecs_vectorized(n=N):
    qr = _world(n).query(HasPos, HasVel)
    def step():
        qr.position[:] = qr.position + qr.velocity * DT          # _Field spans both pools; loop hidden
    return _bench(step), _key(qr.position.numpy())

def micro_ecs_pool_vectorized(n=N):
    qr = _world(n).query(HasPos, HasVel)
    def step():
        for pool in qr.pool_list:                                # you write the per-pool loop; no _Field object
            pool.position[:] = pool.position + pool.velocity * DT
    return _bench(step), _key(qr.position.numpy())

def micro_ecs_zip_rows(n=N):
    qr = _world(n).query(HasPos, HasVel)
    def step():
        for p, v in zip(qr.position, qr.velocity):               # __iter__ yields row-views from the buffers
            p += v * DT
    return _bench(step), _key(qr.position.numpy())

def micro_ecs_pool_loop(n=N):
    qr = _world(n).query(HasPos, HasVel)
    def step():
        for pool in qr.pool_list:
            posa, vela = pool.position, pool.velocity
            for i in range(len(pool)):
                posa[i] += vela[i] * DT
    return _bench(step), _key(qr.position.numpy())

def micro_ecs_index(n=N):
    qr = _world(n).query(HasPos, HasVel)
    def step():
        posf, velf = qr.position, qr.velocity
        for i in range(n):
            row = posf[i]            # qr.f[i] read = one np.searchsorted to find the pool (query_result.py:78)
            row += velf[i] * DT      # in-place on the view writes back; qr.f[i] += ... would raise (setitem)
    return _bench(step), _key(qr.position.numpy())

def micro_ecs_get_entity(n=N):
    w = _world(n)
    qr = w.query(HasPos, HasVel)
    ids = [int(x) for x in qr.entity_ids]
    def step():
        for eid in ids:
            ent, _ = w.get_entity(eid)               # builds a dict + one numpy index per field, every call
            ent["position"] += ent["velocity"] * DT
    return _bench(step), _key(qr.position.numpy())

def oop_numpy(n=N):
    pos, vel = _data(n)
    objs = [OOPNumpy(pos[i].copy(), vel[i].copy()) for i in range(n)]
    def step():
        for o in objs:
            o.position += o.velocity * DT
    return _bench(step), _key([o.position for o in objs])

def oop_scalar(n=N):
    pos, vel = _data(n)
    objs = [OOPScalar(float(pos[i, 0]), float(pos[i, 1]), float(vel[i, 0]), float(vel[i, 1])) for i in range(n)]
    def step():
        for o in objs:
            o.px += o.vx * DT_PY
            o.py += o.vy * DT_PY
    return _bench(step), _key([[o.px, o.py] for o in objs])


def main():
    timings, keys = {}, {}
    for mode in MODES:
        if mode == "micro-ecs-pool-vectorized":
            sec, key = micro_ecs_pool_vectorized(n=N)
        elif mode == "micro-ecs-vectorized":
            sec, key = micro_ecs_vectorized(n=N)
        elif mode == "oop-scalar":
            sec, key = oop_scalar(n=N)
        elif mode == "oop-numpy":
            sec, key = oop_numpy(n=N)
        elif mode == "micro-ecs-zip-rows":
            sec, key = micro_ecs_zip_rows(n=N)
        elif mode == "micro-ecs-pool-loop":
            sec, key = micro_ecs_pool_loop(n=N)
        elif mode == "micro-ecs-index":
            sec, key = micro_ecs_index(n=N)
        elif mode == "micro-ecs-get-entity":
            sec, key = micro_ecs_get_entity(n=N)
        else:
            raise ValueError(f"unknown mode: {mode}")
        timings[mode] = sec
        keys[mode] = key

    ref = keys[MODES[0]]                              # all modes must compute the SAME physics
    for mode, key in keys.items():
        assert np.allclose(key, ref, rtol=1e-3, atol=1e-3), f"{mode} diverged from {MODES[0]}"

    print({mode: round(sec, 8) for mode, sec in timings.items()})


if __name__ == "__main__":
    main()
