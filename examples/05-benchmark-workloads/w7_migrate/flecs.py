"""w7 migrate -- flecs: set/remove a component per frame (archetype/table migration, like microecs)."""
import numpy as np
import flecs
import common as C

DT = C.DT


class Position:
    __slots__ = ("x", "y", "vx", "vy")
    def __init__(self, x, y, vx, vy): self.x, self.y, self.vx, self.vy = x, y, vx, vy
class Buff:
    __slots__ = ("amount",)
    def __init__(self, amount): self.amount = amount


def build(n):
    w = flecs.World()
    s = C.make_scene(n)
    ids = []
    for i in range(n):
        e = w.entity()
        e.set(Position(float(s["pos"][i, 0]), float(s["pos"][i, 1]),
                       float(s["vel"][i, 0]), float(s["vel"][i, 1])))
        ids.append(e)
    return {"w": w, "qp": w.query(Position), "ids": ids,
            "win": C.migrate_windows(n, C.FRAMES, C.k_mig(n)), "fc": [0]}


def step(st):
    ids, win = st["ids"], st["win"]
    f = st["fc"][0]; st["fc"][0] += 1
    if f > 0:
        for t in win[f - 1]:
            ids[t].remove(Buff)
    for t in win[f]:
        ids[t].set(Buff(float(t)))
    st["qp"].reset()
    for _e, p in st["qp"]:
        p.x += p.vx * DT
        p.y += p.vy * DT


def collect(st):
    w = st["w"]
    pos = C._fp([(p.x, p.y) for _e, p in w.query(Position)])
    buffs = np.sort(np.array([b.amount for _e, b in w.query(Buff)], float))
    return C._fp(pos, buffs)
