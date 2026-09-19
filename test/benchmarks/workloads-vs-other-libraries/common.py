"""Shared scene + deterministic event streams + numpy/python ground-truth references + timing.

Every library simulates the SAME scene from the SAME SEED for the SAME frames, driven by the
SAME precomputed per-frame event streams (damage targets, spawn payloads, migrate windows). A
float64 numpy/python reference computes the ground truth for each workload; each library's final
state is checked against it via an order-independent fingerprint -- so a library cannot look fast
by skipping work, and results are comparable across wildly different data models.

Workloads (one subfolder each -- see README):
  w1 physics   columnar batch integrate                          (vectorizable)
  w2 bounce    integrate + data-parallel wall bounce (np.where)  (vectorizable-with-branch)
  w3 ai        per-entity data-dependent state machine           (branchy row logic)
  w4 random    read-modify-write K entities by external id       (random access)
  w5 churn     spawn B + despawn B (FIFO) per frame + integrate  (structural churn)
  w6 mixed     physics + ai + K targeted damage, fixed set       (steady-state frame)
  w7 migrate   integrate all + rolling component add/remove      (archetype migration)

A workload adapter (one file per library under each wN_* folder) is three callables:
    build(n)       -> opaque state           (timed once per rep)
    step(state)    -> None                    (one frame; timed over MEASURE frames)
    collect(state) -> 1-D float64 fingerprint (checked against a reference below)
"""
import gc
import time
import numpy as np

SEED     = 0
DT       = 0.016
DT32     = np.float32(DT)
BOUND    = 50.0        # w2 wall half-extent
DRAIN    = 40.0        # w3 hp drained per second while alive
RESPAWN  = 0.5         # w3 seconds spent dead before respawn
DMG      = 3.0         # w4/w6 damage per hit

# fingerprint tolerance -- loose enough for float32 accumulation vs the float64 reference
_RTOL, _ATOL = 2e-3, 2e-2


# ----------------------------------------------------------------------------- scene

def make_scene(n):
    """Deterministic initial scene as float64 arrays. Odd-index entities carry acceleration."""
    rng = np.random.default_rng(SEED)
    pos     = rng.uniform(-40, 40, (n, 2))
    vel     = rng.uniform(-10, 10, (n, 2))
    acc     = rng.uniform(-10, 10, (n, 2))
    has_acc = (np.arange(n) % 2 == 1)
    hp      = rng.uniform(30, 100, n)
    state   = np.zeros(n, dtype=np.int64)   # 0 = alive, 1 = dead
    timer   = np.zeros(n)
    return dict(pos=pos, vel=vel, acc=acc, has_acc=has_acc, hp=hp, state=state, timer=timer)


def damage_targets(n, frames, k):
    """(frames, k) int array of DISTINCT logical entity indices hit each frame (same for every library).

    Distinct within a frame so a column store (xecs) can scatter the K hits correctly without
    needing add.at accumulation -- models "K distinct entities take damage this frame".
    """
    rng = np.random.default_rng(SEED + 1)
    k = min(k, n)
    return np.stack([rng.choice(n, size=k, replace=False) for _ in range(frames)])


def migrate_windows(n, frames, k):
    """(frames, k) int array of DISTINCT, non-overlapping id-blocks -- window(f) = [f*k, f*k+k).

    Used by w7: each frame adds a component to window(f) and removes it from window(f-1), forcing
    2k archetype migrations/frame. Blocks are disjoint (frames*k < n) so every add is to an entity
    that lacks the component and every remove is from one that has it -- always valid, no per-entity
    state needed to drive it. Caller must size k so frames*k <= n.
    """
    base = np.arange(k)
    return np.stack([f * k + base for f in range(frames)])


def spawn_payloads(frames, b):
    """(frames, b) float payloads for entities spawned each frame; payload = global birth counter.

    Payload is an immutable per-entity tag: it lets the churn fingerprint check *which*
    entities survived, independent of id/ordering differences between libraries.
    """
    total = frames * b
    return np.arange(total, dtype=np.float64).reshape(frames, b)


# ----------------------------------------------------------------- ground-truth references
# Each returns a 1-D float64 fingerprint: an order-independent signature of the final state.

def _fp(*arrays):
    """Pool every value, sort -> order-independent signature robust to entity reordering."""
    return np.sort(np.concatenate([np.asarray(a, np.float64).ravel() for a in arrays]))


def ref_physics(n, frames):
    s = make_scene(n)
    pos, vel, acc, m = s["pos"].copy(), s["vel"].copy(), s["acc"], s["has_acc"]
    a = acc[m]
    for _ in range(frames):
        vel[m] += a * DT
        pos    += vel * DT
    return _fp(pos)


def ref_bounce(n, frames):
    s = make_scene(n)
    pos, vel, acc, m = s["pos"].copy(), s["vel"].copy(), s["acc"], s["has_acc"]
    a = acc[m]
    for _ in range(frames):
        vel[m] += a * DT
        pos    += vel * DT
        flip = np.abs(pos) > BOUND          # data-parallel branch: reflect at the walls
        vel = np.where(flip, -vel, vel)
    return _fp(pos, vel)


def ref_ai(n, frames):
    """Per-entity health state machine. Same logic every library runs (masked or looped)."""
    s = make_scene(n)
    hp, state, timer = s["hp"].copy(), s["state"].copy(), s["timer"].copy()
    for _ in range(frames):
        alive = state == 0
        dead  = ~alive
        hp[alive] -= DRAIN * DT
        just_died = alive & (hp <= 0)
        state[just_died] = 1
        timer[just_died] = RESPAWN
        timer[dead] -= DT
        respawn = dead & (timer <= 0)
        state[respawn] = 0
        hp[respawn]    = 100.0
    return _fp(hp, state, timer)


def ref_random(n, frames, k):
    s = make_scene(n)
    hp = s["hp"].copy()
    tg = damage_targets(n, frames, k)
    for f in range(frames):
        np.add.at(hp, tg[f], -DMG)          # additive, so duplicate targets in a frame stack (== per-entity loop)
    return _fp(hp)


def ref_churn(n, frames, b):
    """Start with n entities (payload -1), each frame despawn B oldest + spawn B, integrate all.

    Fingerprint is the multiset of surviving payloads -> checks the right entities are alive,
    independent of how each library stores/orders them.
    """
    from collections import deque
    payloads = deque([-1.0] * n)            # initial cohort tagged -1
    sp = spawn_payloads(frames, b)
    for f in range(frames):
        for _ in range(min(b, len(payloads))):
            payloads.popleft()              # FIFO despawn oldest
        payloads.extend(sp[f].tolist())     # spawn B new at the tail
    return np.sort(np.array(payloads, np.float64))


def ref_mixed(n, frames, k):
    """Fixed entity set: physics + ai + K targeted damage every frame (steady-state frame)."""
    s = make_scene(n)
    pos, vel, acc, m = s["pos"].copy(), s["vel"].copy(), s["acc"], s["has_acc"]
    hp, state, timer = s["hp"].copy(), s["state"].copy(), s["timer"].copy()
    a = acc[m]
    tg = damage_targets(n, frames, k)
    for f in range(frames):
        vel[m] += a * DT
        pos    += vel * DT
        alive = state == 0
        dead  = ~alive
        hp[alive] -= DRAIN * DT
        just_died = alive & (hp <= 0)
        state[just_died] = 1
        timer[just_died] = RESPAWN
        timer[dead] -= DT
        respawn = dead & (timer <= 0)
        state[respawn] = 0
        hp[respawn]    = 100.0
        np.add.at(hp, tg[f], -DMG)
    return _fp(pos, hp, state, timer)


def ref_migrate(n, frames, k):
    """Integrate all + rolling component add/remove. Fingerprint: positions + buffed-entity tags.

    Buffed set at the end = window(frames-1); each buffed entity is tagged with amount=float(id),
    so the fingerprint checks the RIGHT entities ended up migrated, order-independently.
    """
    s = make_scene(n)
    pos, vel = s["pos"].copy(), s["vel"]
    win = migrate_windows(n, frames, k)
    buffed = {}
    for f in range(frames):
        pos += vel * DT
        if f > 0:
            for t in win[f - 1]:
                buffed.pop(int(t), None)
        for t in win[f]:
            buffed[int(t)] = float(t)
    return _fp(pos, np.array(sorted(buffed.values()), np.float64))


def verify(fp, ref):
    """(ok, max_abs_diff) comparing a library's fingerprint to the reference."""
    fp  = np.asarray(fp, np.float64)
    ref = np.asarray(ref, np.float64)
    if fp.shape != ref.shape:
        return False, float("inf")
    if fp.size == 0:
        return True, 0.0
    d = float(np.max(np.abs(fp - ref)))
    return bool(np.allclose(fp, ref, rtol=_RTOL, atol=_ATOL)), d


# --------------------------------------------------------------------------- timing harness

WARMUP  = 3
MEASURE = 30
FRAMES  = WARMUP + MEASURE


def k_for(n):  # random-access / mixed: ~2% of entities touched per frame (min 16)
    return max(16, n // 50)


def b_for(n):  # churn: ~1% spawned+despawned per frame (min 16)
    return max(16, n // 100)


def k_mig(n):  # component-migration: 2*k migrations/frame; kept small so FRAMES*k < n (disjoint blocks)
    return max(4, n // 200)


def references(n):
    """Ground-truth fingerprint for every workload at this N (computed once, float64)."""
    k, b = k_for(n), b_for(n)
    return {
        "w1_physics": ref_physics(n, FRAMES),
        "w2_bounce":  ref_bounce(n, FRAMES),
        "w3_ai":      ref_ai(n, FRAMES),
        "w4_random":  ref_random(n, FRAMES, k),
        "w5_churn":   ref_churn(n, FRAMES, b),
        "w6_mixed":   ref_mixed(n, FRAMES, k),
        "w7_migrate": ref_migrate(n, FRAMES, k_mig(n)),
    }


def _reps_for(n):
    return 6 if n <= 5_000 else (4 if n <= 30_000 else 2)


def run_workload(name, build, step, collect, ref, n):
    """Run one (library, workload) pair. Returns a result dict (build/step ms, ns/entity, ok)."""
    reps = _reps_for(n)
    best_build = float("inf")
    best_step  = float("inf")
    fp = None
    for _ in range(reps):
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter()
            state = build(n)
            best_build = min(best_build, time.perf_counter() - t0)
            for _ in range(WARMUP):
                step(state)
            t0 = time.perf_counter()
            for _ in range(MEASURE):
                step(state)
            best_step = min(best_step, (time.perf_counter() - t0) / MEASURE)
            fp = collect(state)
        finally:
            gc.enable()
    ok, max_diff = verify(fp, ref)
    return {
        "workload": name,
        "n": n,
        "build_ms": best_build * 1e3,
        "step_ms":  best_step * 1e3,
        "ns_per_entity": best_step / n * 1e9,
        "m_updates_per_s": n / best_step / 1e6,
        "ok": ok,
        "max_diff": max_diff,
    }
