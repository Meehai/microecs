"""w3 ai -- PyEnTT: per-entity if-branch state machine over the C++ sparse-set view."""
import entt
import common as C

DT = C.DT
DRAINDT, RESPAWN = C.DRAIN * C.DT, C.RESPAWN


class Health:
    __slots__ = ("hp", "state", "timer")
    def __init__(self, hp, state, timer): self.hp, self.state, self.timer = hp, state, timer


def build(n):
    r = entt.Registry()
    s = C.make_scene(n)
    for i in range(n):
        e = r.create()
        r.emplace(e, Health, float(s["hp"][i]), int(s["state"][i]), float(s["timer"][i]))
    return r


def step(r):
    for _e, h in r.view(Health):
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


def collect(r):
    xs = [(h.hp, h.state, h.timer) for _e, h in r.view(Health)]
    return C._fp([x[0] for x in xs], [x[1] for x in xs], [x[2] for x in xs])
