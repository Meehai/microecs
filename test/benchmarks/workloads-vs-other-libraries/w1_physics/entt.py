"""w1 physics -- PyEnTT: the C++ EnTT sparse-set core via nanobind; components are python objects."""
import entt
import common as C

DT = C.DT


class Position:
    __slots__ = ("x", "y", "vx", "vy")
    def __init__(self, x, y, vx, vy): self.x, self.y, self.vx, self.vy = x, y, vx, vy
class Acceleration:
    __slots__ = ("ax", "ay")
    def __init__(self, ax, ay): self.ax, self.ay = ax, ay


def build(n):
    r = entt.Registry()
    s = C.make_scene(n)
    for i in range(n):
        e = r.create()
        r.emplace(e, Position, float(s["pos"][i, 0]), float(s["pos"][i, 1]),
                  float(s["vel"][i, 0]), float(s["vel"][i, 1]))
        if s["has_acc"][i]:
            r.emplace(e, Acceleration, float(s["acc"][i, 0]), float(s["acc"][i, 1]))
    return r


def step(r):
    for _e, p, a in r.view(Position, Acceleration):
        p.vx += a.ax * DT
        p.vy += a.ay * DT
    for _e, p in r.view(Position):
        p.x += p.vx * DT
        p.y += p.vy * DT


def collect(r):
    return C._fp([(p.x, p.y) for _e, p in r.view(Position)])
