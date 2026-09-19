"""w1 physics -- snecs: pure-python sparse-set; queries compiled outside the hot loop (its fast path)."""
import snecs
from snecs import Component, register_component, Query
import common as C

DT = C.DT


@register_component
class Position(Component):
    __slots__ = ("x", "y", "vx", "vy")
    def __init__(self, x, y, vx, vy): self.x, self.y, self.vx, self.vy = x, y, vx, vy
@register_component
class Acceleration(Component):
    __slots__ = ("ax", "ay")
    def __init__(self, ax, ay): self.ax, self.ay = ax, ay


def build(n):
    w = snecs.World()
    s = C.make_scene(n)
    for i in range(n):
        p = Position(float(s["pos"][i, 0]), float(s["pos"][i, 1]),
                     float(s["vel"][i, 0]), float(s["vel"][i, 1]))
        if s["has_acc"][i]:
            snecs.new_entity([p, Acceleration(float(s["acc"][i, 0]), float(s["acc"][i, 1]))], world=w)
        else:
            snecs.new_entity([p], world=w)
    return {"w": w,
            "acc_q": Query((Position, Acceleration), world=w).compile(),
            "pos_q": Query((Position,), world=w).compile()}


def step(st):
    for _e, (p, a) in st["acc_q"]:
        p.vx += a.ax * DT
        p.vy += a.ay * DT
    for _e, (p,) in st["pos_q"]:
        p.x += p.vx * DT
        p.y += p.vy * DT


def collect(st):
    return C._fp([(p.x, p.y) for _e, (p,) in st["pos_q"]])
