"""w5 churn -- snecs: new_entity / delete_entity_immediately with a FIFO id deque (sparse-set)."""
from collections import deque
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
class Payload(Component):
    __slots__ = ("payload",)
    def __init__(self, payload): self.payload = payload


def build(n):
    w = snecs.World()
    s = C.make_scene(n)
    order = deque()
    for i in range(n):
        eid = snecs.new_entity([Position(float(s["pos"][i, 0]), float(s["pos"][i, 1]),
                                          float(s["vel"][i, 0]), float(s["vel"][i, 1])),
                                Payload(-1.0)], world=w)
        order.append(eid)
    sp = C.spawn_payloads(C.FRAMES, C.b_for(n))
    return {"w": w, "order": order, "sp": sp, "fc": [0],
            "pos_q": Query((Position,), world=w).compile(),
            "pay_q": Query((Payload,), world=w).compile()}


def step(st):
    w, order, sp = st["w"], st["order"], st["sp"]
    f = st["fc"][0]; st["fc"][0] += 1
    b = sp.shape[1]
    for _ in range(min(b, len(order))):
        snecs.delete_entity_immediately(order.popleft(), world=w)
    for p in sp[f]:
        order.append(snecs.new_entity([Position(0.0, 0.0, 1.0, 1.0), Payload(float(p))], world=w))
    for _e, (p,) in st["pos_q"]:
        p.x += p.vx * DT
        p.y += p.vy * DT


def collect(st):
    return np.sort(np.array([pl.payload for _e, (pl,) in st["pay_q"]], float))
