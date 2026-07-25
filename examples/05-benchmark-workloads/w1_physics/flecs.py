"""w1 physics -- flecs (C core, pybind11): entity/query over python-object components."""
import flecs
import common as C

DT = C.DT


class Position:
    __slots__ = ("x", "y", "vx", "vy")
    def __init__(self, x, y, vx, vy): self.x, self.y, self.vx, self.vy = x, y, vx, vy
class Acceleration:
    __slots__ = ("ax", "ay")
    def __init__(self, ax, ay): self.ax, self.ay = ax, ay


def build(n):
    w = flecs.World()
    s = C.make_scene(n)
    for i in range(n):
        e = w.entity()
        e.set(Position(float(s["pos"][i, 0]), float(s["pos"][i, 1]),
                       float(s["vel"][i, 0]), float(s["vel"][i, 1])))
        if s["has_acc"][i]:
            e.set(Acceleration(float(s["acc"][i, 0]), float(s["acc"][i, 1])))
    return {"w": w, "qpa": w.query(Position, Acceleration), "qp": w.query(Position)}


def step(st):
    st["qpa"].reset()
    for _e, p, a in st["qpa"]:
        p.vx += a.ax * DT
        p.vy += a.ay * DT
    st["qp"].reset()
    for _e, p in st["qp"]:
        p.x += p.vx * DT
        p.y += p.vy * DT


def collect(st):
    return C._fp([(p.x, p.y) for _e, p in st["w"].query(Position)])
