"""w5 churn -- esper: create_entity / delete_entity(immediate=True) with a FIFO id deque."""
from collections import deque
import numpy as np
import esper
import common as C

DT = C.DT


class Position:
    __slots__ = ("x", "y", "vx", "vy")
    def __init__(self, x, y, vx, vy): self.x, self.y, self.vx, self.vy = x, y, vx, vy
class Payload:
    __slots__ = ("payload",)
    def __init__(self, payload): self.payload = payload


def build(n):
    esper.clear_database()
    s = C.make_scene(n)
    order = deque()
    for i in range(n):
        eid = esper.create_entity(Position(float(s["pos"][i, 0]), float(s["pos"][i, 1]),
                                            float(s["vel"][i, 0]), float(s["vel"][i, 1])),
                                  Payload(-1.0))
        order.append(eid)
    sp = C.spawn_payloads(C.FRAMES, C.b_for(n))
    return {"order": order, "sp": sp, "fc": [0]}


def step(st):
    order, sp = st["order"], st["sp"]
    f = st["fc"][0]; st["fc"][0] += 1
    b = sp.shape[1]
    for _ in range(min(b, len(order))):
        esper.delete_entity(order.popleft(), immediate=True)
    for p in sp[f]:
        order.append(esper.create_entity(Position(0.0, 0.0, 1.0, 1.0), Payload(float(p))))
    for _e, (p,) in esper.get_components(Position):
        p.x += p.vx * DT
        p.y += p.vy * DT


def collect(_st):
    return np.sort(np.array([pl.payload for _e, (pl,) in esper.get_components(Payload)], float))
