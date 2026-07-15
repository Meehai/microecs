"""w4 random -- microecs: FAIR columnar scatter (microecs is a columnar SoA lib, like xecs).

The K distinct hits/frame are scattered into the pool column with a precomputed id->row map:
`col[rows] -= DMG`, O(k), zero per-entity python. (The naive `get_entity(id)` loop is ~400x
slower -- quantified in probes/microecs_random.py; never use it in a hot loop.)
"""
from dataclasses import field
import numpy as np
from microecs import World, Component
import common as C

DMG32 = np.float32(C.DMG)


def _f(shape, dtype="float32"):
    return field(metadata={"shape": shape, "dtype": dtype, "default": None})


class Dmg(Component):
    hp: np.ndarray = _f((1,))


def _s(x): return np.array([x], np.float32)


def build(n):
    s = C.make_scene(n)
    w = World([Dmg])
    for i in range(n):
        w.add_entity([Dmg], hp=_s(s["hp"][i]))
    w.update()
    # single archetype -> qr.hp.numpy() is a live, zero-copy view into pool memory.
    qr = w.query(Dmg)
    ids_arr = qr.entity_ids
    row_of = np.empty(int(ids_arr.max()) + 1, np.int64)
    row_of[ids_arr] = np.arange(len(ids_arr))
    tg = C.damage_targets(n, C.FRAMES, C.k_for(n))
    return {"w": w, "col": qr.hp.numpy(), "rows": row_of[tg], "fc": [0]}


def step(st):
    f = st["fc"][0]; st["fc"][0] += 1
    st["col"][st["rows"][f], 0] -= DMG32   # columnar scatter, O(k) (the fair SoA idiom, == xecs)


def collect(st):
    return C._fp(st["col"])
