"""w5 churn -- PyEnTT: create/destroy each frame (EnTT recycles ids) + integrate all."""
from collections import deque
import numpy as np
import entt
import common as C

DT = C.DT


class Position:
    __slots__ = ("x", "y", "vx", "vy")
    def __init__(self, x, y, vx, vy): self.x, self.y, self.vx, self.vy = x, y, vx, vy
class Payload:
    __slots__ = ("payload",)
    def __init__(self, payload): self.payload = payload


def build(n):
    r = entt.Registry()
    s = C.make_scene(n)
    order = deque()
    for i in range(n):
        e = r.create()
        r.emplace(e, Position, float(s["pos"][i, 0]), float(s["pos"][i, 1]),
                  float(s["vel"][i, 0]), float(s["vel"][i, 1]))
        r.emplace(e, Payload, -1.0)
        order.append(e)
    return {"r": r, "order": order, "sp": C.spawn_payloads(C.FRAMES, C.b_for(n)), "fc": [0]}


def step(st):
    r, order, sp = st["r"], st["order"], st["sp"]
    f = st["fc"][0]; st["fc"][0] += 1
    for _ in range(min(sp.shape[1], len(order))):
        r.destroy(order.popleft())
    for p in sp[f]:
        e = r.create()
        r.emplace(e, Position, 0.0, 0.0, 1.0, 1.0)
        r.emplace(e, Payload, float(p))
        order.append(e)
    for _e, p in r.view(Position):
        p.x += p.vx * DT
        p.y += p.vy * DT


def collect(st):
    return np.sort(np.array([pl.payload for _e, pl in st["r"].view(Payload)], float))
