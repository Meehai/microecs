"""w6 mixed -- microecs: physics (batched) + ai (masked) + K targeted hits (per-pool columnar scatter).

Health spans 2 archetypes (with/without Acc), so each target id is mapped once to (pool, local row)
and the per-frame per-pool rows are precomputed -- the damage stays a columnar O(k) scatter.
"""
from dataclasses import field
import numpy as np
from microecs import World, Component
import common as C

DT32    = C.DT32
DMG32   = np.float32(C.DMG)
DRAINDT = np.float32(C.DRAIN * C.DT)
RESPAWN = np.float32(C.RESPAWN)


def _f(shape, dtype="float32"):
    return field(metadata={"shape": shape, "dtype": dtype, "default": None})


class Pos(Component):
    position: np.ndarray = _f((2,))
    velocity: np.ndarray = _f((2,))
class Acc(Component):
    acceleration: np.ndarray = _f((2,))
class Health(Component):
    hp:    np.ndarray = _f((1,))
    state: np.ndarray = _f((1,), "int32")
    timer: np.ndarray = _f((1,))


def _v2(a): return np.asarray(a, np.float32)
def _s(x):  return np.array([x], np.float32)
def _si(x): return np.array([x], np.int32)


def _ai_tick(q):
    hp    = q.hp.numpy()
    state = q.state.numpy()
    timer = q.timer.numpy()
    alive = state == 0
    dead  = ~alive
    new_hp    = np.where(alive, hp - DRAINDT, hp)
    just_died = alive & (new_hp <= 0)
    new_state = np.where(just_died, 1, state)
    new_timer = np.where(just_died, RESPAWN, timer)
    new_timer = np.where(dead, new_timer - DT32, new_timer)
    respawn   = dead & (new_timer <= 0)
    new_state = np.where(respawn, 0, new_state)
    new_hp    = np.where(respawn, np.float32(100.0), new_hp)
    q.hp[:]    = new_hp.astype(np.float32)
    q.state[:] = new_state.astype(np.int32)
    q.timer[:] = new_timer.astype(np.float32)


def build(n):
    s = C.make_scene(n)
    w = World([Pos, Acc, Health])
    for i in range(n):
        comps = [Pos, Acc, Health] if s["has_acc"][i] else [Pos, Health]
        kw = dict(position=_v2(s["pos"][i]), velocity=_v2(s["vel"][i]),
                  hp=_s(s["hp"][i]), state=_si(s["state"][i]), timer=_s(s["timer"][i]))
        if s["has_acc"][i]:
            kw["acceleration"] = _v2(s["acc"][i])
        w.add_entity(comps, **kw)
    w.update()
    # map each target id -> (pool, local row) once (fixed set); precompute per-frame per-pool rows.
    tg = C.damage_targets(n, C.FRAMES, C.k_for(n))
    qh = w.query(Health)
    pools = qh.pool_list
    ids_arr = qh.entity_ids
    sizes = [len(p) for p in pools]
    bounds = np.cumsum([0, *sizes])
    maxid = int(ids_arr.max())
    id_pool = np.full(maxid + 1, -1, np.int64)
    id_local = np.full(maxid + 1, -1, np.int64)
    for pi in range(len(pools)):
        seg = ids_arr[bounds[pi]:bounds[pi + 1]]
        id_pool[seg] = pi
        id_local[seg] = np.arange(len(seg))
    frame_pool_rows = []
    for f in range(C.FRAMES):
        pp, pl = id_pool[tg[f]], id_local[tg[f]]
        frame_pool_rows.append([pl[pp == pi] for pi in range(len(pools))])
    return {"w": w, "hp_cols": [p.hp for p in pools], "rows": frame_pool_rows, "fc": [0]}


def step(st):
    w = st["w"]
    f = st["fc"][0]; st["fc"][0] += 1
    qv = w.query(Pos, Acc)
    qv.velocity[:] = qv.velocity + qv.acceleration * DT32
    qp = w.query(Pos)
    qp.position[:] = qp.position + qp.velocity * DT32
    _ai_tick(w.query(Health))
    for col, rows in zip(st["hp_cols"], st["rows"][f]):   # per-pool columnar scatter (O(k))
        col[rows, 0] -= DMG32


def collect(st):
    w = st["w"]
    qp, qh = w.query(Pos), w.query(Health)
    return C._fp(qp.position.numpy(), qh.hp.numpy(), qh.state.numpy(), qh.timer.numpy())
