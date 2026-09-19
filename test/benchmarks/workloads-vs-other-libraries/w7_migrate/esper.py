"""w7 migrate -- esper: add_component / remove_component per entity id; integrate all each frame."""
import numpy as np
import esper
import common as C

DT = C.DT


class Position:
    __slots__ = ("x", "y", "vx", "vy")
    def __init__(self, x, y, vx, vy): self.x, self.y, self.vx, self.vy = x, y, vx, vy
class Buff:
    __slots__ = ("amount",)
    def __init__(self, amount): self.amount = amount


def build(n):
    esper.clear_database()
    s = C.make_scene(n)
    ids = [esper.create_entity(Position(float(s["pos"][i, 0]), float(s["pos"][i, 1]),
                                        float(s["vel"][i, 0]), float(s["vel"][i, 1]))) for i in range(n)]
    return {"ids": ids, "win": C.migrate_windows(n, C.FRAMES, C.k_mig(n)), "fc": [0]}


def step(st):
    ids, win = st["ids"], st["win"]
    f = st["fc"][0]; st["fc"][0] += 1
    if f > 0:
        for t in win[f - 1]:
            esper.remove_component(ids[t], Buff)
    for t in win[f]:
        esper.add_component(ids[t], Buff(float(t)))
    for _e, (p,) in esper.get_components(Position):
        p.x += p.vx * DT
        p.y += p.vy * DT


def collect(_st):
    pos = C._fp([(p.x, p.y) for _e, (p,) in esper.get_components(Position)])
    buffs = np.sort(np.array([b.amount for _e, (b,) in esper.get_components(Buff)], float))
    return C._fp(pos, buffs)
