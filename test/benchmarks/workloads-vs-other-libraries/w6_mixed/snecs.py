"""w6 mixed -- snecs: physics loop + ai if-branch (compiled queries) + K hits via entity_component(id)."""
import snecs
from snecs import Component, register_component, Query
import common as C

DT, DMG = C.DT, C.DMG
DRAINDT, RESPAWN = C.DRAIN * C.DT, C.RESPAWN


@register_component
class Position(Component):
    __slots__ = ("x", "y", "vx", "vy")
    def __init__(self, x, y, vx, vy): self.x, self.y, self.vx, self.vy = x, y, vx, vy
@register_component
class Acceleration(Component):
    __slots__ = ("ax", "ay")
    def __init__(self, ax, ay): self.ax, self.ay = ax, ay
@register_component
class Health(Component):
    __slots__ = ("hp", "state", "timer")
    def __init__(self, hp, state, timer): self.hp, self.state, self.timer = hp, state, timer


def _ai_tick(q):
    for _e, (h,) in q:
        if h.state == 0:
            h.hp -= DRAINDT
            if h.hp <= 0:
                h.state = 1
                h.timer = RESPAWN
        else:
            h.timer -= DT
            if h.timer <= 0:
                h.state = 0
                h.hp = 100.0


def build(n):
    w = snecs.World()
    s = C.make_scene(n)
    ids = []
    for i in range(n):
        p = Position(float(s["pos"][i, 0]), float(s["pos"][i, 1]),
                     float(s["vel"][i, 0]), float(s["vel"][i, 1]))
        h = Health(float(s["hp"][i]), int(s["state"][i]), float(s["timer"][i]))
        if s["has_acc"][i]:
            ids.append(snecs.new_entity([p, Acceleration(float(s["acc"][i, 0]), float(s["acc"][i, 1])), h], world=w))
        else:
            ids.append(snecs.new_entity([p, h], world=w))
    tg = C.damage_targets(n, C.FRAMES, C.k_for(n))
    return {"w": w, "ids": ids, "tg": tg, "fc": [0],
            "acc_q": Query((Position, Acceleration), world=w).compile(),
            "pos_q": Query((Position,), world=w).compile(),
            "hp_q":  Query((Health,), world=w).compile()}


def step(st):
    w, ids, tg = st["w"], st["ids"], st["tg"]
    f = st["fc"][0]; st["fc"][0] += 1
    for _e, (p, a) in st["acc_q"]:
        p.vx += a.ax * DT
        p.vy += a.ay * DT
    for _e, (p,) in st["pos_q"]:
        p.x += p.vx * DT
        p.y += p.vy * DT
    _ai_tick(st["hp_q"])
    for t in tg[f]:
        snecs.entity_component(ids[t], Health, world=w).hp -= DMG


def collect(st):
    pos = [(p.x, p.y) for _e, (p,) in st["pos_q"]]
    heal = [(h.hp, h.state, h.timer) for _e, (h,) in st["hp_q"]]
    return C._fp([r[0] for r in pos], [r[1] for r in pos],
                 [r[0] for r in heal], [r[1] for r in heal], [r[2] for r in heal])
