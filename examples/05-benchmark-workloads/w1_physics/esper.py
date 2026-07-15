"""w1 physics -- esper: the most popular pure-python ECS (per-entity objects, plain floats)."""
import esper
import common as C

DT = C.DT


class Position:
    __slots__ = ("x", "y", "vx", "vy")
    def __init__(self, x, y, vx, vy): self.x, self.y, self.vx, self.vy = x, y, vx, vy
class Acceleration:
    __slots__ = ("ax", "ay")
    def __init__(self, ax, ay): self.ax, self.ay = ax, ay


def build(n):
    esper.clear_database()
    s = C.make_scene(n)
    for i in range(n):
        p = Position(float(s["pos"][i, 0]), float(s["pos"][i, 1]),
                     float(s["vel"][i, 0]), float(s["vel"][i, 1]))
        if s["has_acc"][i]:
            esper.create_entity(p, Acceleration(float(s["acc"][i, 0]), float(s["acc"][i, 1])))
        else:
            esper.create_entity(p)
    return None


def step(_):
    for _e, (p, a) in esper.get_components(Position, Acceleration):
        p.vx += a.ax * DT
        p.vy += a.ay * DT
    for _e, (p,) in esper.get_components(Position):
        p.x += p.vx * DT
        p.y += p.vy * DT


def collect(_):
    return C._fp([(p.x, p.y) for _e, (p,) in esper.get_components(Position)])
