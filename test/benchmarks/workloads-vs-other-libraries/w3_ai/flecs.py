"""w3 ai -- flecs: per-entity if-branch state machine over a cached query."""
import flecs
import common as C

DT = C.DT
DRAINDT, RESPAWN = C.DRAIN * C.DT, C.RESPAWN


class Health:
    __slots__ = ("hp", "state", "timer")
    def __init__(self, hp, state, timer): self.hp, self.state, self.timer = hp, state, timer


def build(n):
    w = flecs.World()
    s = C.make_scene(n)
    for i in range(n):
        w.entity().set(Health(float(s["hp"][i]), int(s["state"][i]), float(s["timer"][i])))
    return {"w": w, "q": w.query(Health)}


def step(st):
    st["q"].reset()
    for _e, h in st["q"]:
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


def collect(st):
    xs = [(h.hp, h.state, h.timer) for _e, h in st["w"].query(Health)]
    return C._fp([x[0] for x in xs], [x[1] for x in xs], [x[2] for x in xs])
