"""w7 migrate -- microecs: get_entity(id).add/remove_component + update() moves entities between
archetypes ([Pos] <-> [Pos,Buff]); integrate all each frame. Migration pays an archetype copy."""
from dataclasses import field
import numpy as np
from microecs import World, Component
import common as C

DT32 = C.DT32


def _f(shape, dtype="float32"):
    return field(metadata={"shape": shape, "dtype": dtype, "default": None})


class Pos(Component):
    position: np.ndarray = _f((2,))
    velocity: np.ndarray = _f((2,))
class Buff(Component):
    amount: np.ndarray = _f((1,))


def _v2(a): return np.asarray(a, np.float32)
def _s(x):  return np.array([x], np.float32)


def build(n):
    s = C.make_scene(n)
    w = World([Pos, Buff])
    ids = [w.add_entity([Pos], position=_v2(s["pos"][i]), velocity=_v2(s["vel"][i])) for i in range(n)]
    w.update()
    return {"w": w, "ids": ids, "win": C.migrate_windows(n, C.FRAMES, C.k_mig(n)), "fc": [0]}


def step(st):
    w, ids, win = st["w"], st["ids"], st["win"]
    f = st["fc"][0]; st["fc"][0] += 1
    if f > 0:
        for t in win[f - 1]:
            w.get_entity(ids[t]).remove_component(Buff)      # [Pos,Buff] -> [Pos]
    for t in win[f]:
        w.get_entity(ids[t]).add_component(Buff, amount=_s(float(t)))  # [Pos] -> [Pos,Buff]
    w.update()
    qp = w.query(Pos)
    qp.position[:] = qp.position + qp.velocity * DT32


def collect(st):
    w = st["w"]
    buffs = w.query(Buff).amount.numpy().ravel() if len(w.query(Buff)) else np.empty(0, np.float32)
    return C._fp(w.query(Pos).position.numpy(), buffs)
