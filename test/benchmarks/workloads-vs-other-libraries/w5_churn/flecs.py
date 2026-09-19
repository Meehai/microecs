"""w5 churn -- flecs: entity()/destroy() each frame + integrate all."""
from collections import deque
import numpy as np
import flecs
import common as C

DT = C.DT


class Position:
    __slots__ = ("x", "y", "vx", "vy")
    def __init__(self, x, y, vx, vy): self.x, self.y, self.vx, self.vy = x, y, vx, vy
class Payload:
    __slots__ = ("payload",)
    def __init__(self, payload): self.payload = payload


def build(n):
    w = flecs.World()
    s = C.make_scene(n)
    order = deque()
    for i in range(n):
        e = w.entity()
        e.set(Position(float(s["pos"][i, 0]), float(s["pos"][i, 1]),
                       float(s["vel"][i, 0]), float(s["vel"][i, 1])))
        e.set(Payload(-1.0))
        order.append(e)
    return {"w": w, "qp": w.query(Position), "order": order,
            "sp": C.spawn_payloads(C.FRAMES, C.b_for(n)), "fc": [0]}


def step(st):
    w, order, sp = st["w"], st["order"], st["sp"]
    f = st["fc"][0]; st["fc"][0] += 1
    for _ in range(min(sp.shape[1], len(order))):
        order.popleft().destroy()
    for p in sp[f]:
        e = w.entity()
        e.set(Position(0.0, 0.0, 1.0, 1.0))
        e.set(Payload(float(p)))
        order.append(e)
    st["qp"].reset()
    for _e, p in st["qp"]:
        p.x += p.vx * DT
        p.y += p.vy * DT


def collect(st):
    return np.sort(np.array([pl.payload for _e, pl in st["w"].query(Payload)], float))
