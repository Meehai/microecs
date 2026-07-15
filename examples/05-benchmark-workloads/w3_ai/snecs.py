"""w3 ai -- snecs: per-entity if-branch state machine over a compiled Health query."""
import snecs
from snecs import Component, register_component, Query
import common as C

DT = C.DT
DRAINDT, RESPAWN = C.DRAIN * C.DT, C.RESPAWN


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
    for i in range(n):
        snecs.new_entity([Health(float(s["hp"][i]), int(s["state"][i]), float(s["timer"][i]))], world=w)
    return {"w": w, "q": Query((Health,), world=w).compile()}


def step(st):
    _ai_tick(st["q"])


def collect(st):
    xs = [(h.hp, h.state, h.timer) for _e, (h,) in st["q"]]
    return C._fp([r[0] for r in xs], [r[1] for r in xs], [r[2] for r in xs])
