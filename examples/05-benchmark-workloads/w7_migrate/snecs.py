"""w7 migrate -- snecs: add_component / remove_component per entity id (sparse-set, the migration champ)."""
import numpy as np
import snecs
from snecs import Component, register_component, Query
import common as C

DT = C.DT


@register_component
class Position(Component):
    __slots__ = ("x", "y", "vx", "vy")
    def __init__(self, x, y, vx, vy): self.x, self.y, self.vx, self.vy = x, y, vx, vy
@register_component
class Buff(Component):
    __slots__ = ("amount",)
    def __init__(self, amount): self.amount = amount


def build(n):
    w = snecs.World()
    s = C.make_scene(n)
    ids = [snecs.new_entity([Position(float(s["pos"][i, 0]), float(s["pos"][i, 1]),
                                      float(s["vel"][i, 0]), float(s["vel"][i, 1]))], world=w)
           for i in range(n)]
    return {"w": w, "ids": ids, "win": C.migrate_windows(n, C.FRAMES, C.k_mig(n)), "fc": [0],
            "pos_q": Query((Position,), world=w).compile()}


def step(st):
    w, ids, win = st["w"], st["ids"], st["win"]
    f = st["fc"][0]; st["fc"][0] += 1
    if f > 0:
        for t in win[f - 1]:
            snecs.remove_component(ids[t], Buff, world=w)
    for t in win[f]:
        snecs.add_component(ids[t], Buff(float(t)), world=w)
    for _e, (p,) in st["pos_q"]:
        p.x += p.vx * DT
        p.y += p.vy * DT


def collect(st):
    pos = C._fp([(p.x, p.y) for _e, (p,) in st["pos_q"]])
    buffs = np.sort(np.array([b.amount for _e, (b,) in Query((Buff,), world=st["w"]).compile()], float))
    return C._fp(pos, buffs)
