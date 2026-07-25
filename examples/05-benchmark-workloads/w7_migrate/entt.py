"""w7 migrate -- PyEnTT: emplace/remove a component per frame (sparse-set migration, no data move)."""
import numpy as np
import entt
import common as C

DT = C.DT


class Position:
    __slots__ = ("x", "y", "vx", "vy")
    def __init__(self, x, y, vx, vy): self.x, self.y, self.vx, self.vy = x, y, vx, vy
class Buff:
    __slots__ = ("amount",)
    def __init__(self, amount): self.amount = amount


def build(n):
    r = entt.Registry()
    s = C.make_scene(n)
    ids = []
    for i in range(n):
        e = r.create()
        r.emplace(e, Position, float(s["pos"][i, 0]), float(s["pos"][i, 1]),
                  float(s["vel"][i, 0]), float(s["vel"][i, 1]))
        ids.append(e)
    return {"r": r, "ids": ids, "win": C.migrate_windows(n, C.FRAMES, C.k_mig(n)), "fc": [0]}


def step(st):
    r, ids, win = st["r"], st["ids"], st["win"]
    f = st["fc"][0]; st["fc"][0] += 1
    if f > 0:
        for t in win[f - 1]:
            r.remove(ids[t], Buff)
    for t in win[f]:
        r.emplace(ids[t], Buff, float(t))
    for _e, p in r.view(Position):
        p.x += p.vx * DT
        p.y += p.vy * DT


def collect(st):
    r = st["r"]
    pos = C._fp([(p.x, p.y) for _e, p in r.view(Position)])
    buffs = np.sort(np.array([b.amount for _e, b in r.view(Buff)], float))
    return C._fp(pos, buffs)
