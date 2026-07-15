"""w3 ai -- esper: per-entity if-branch state machine (its natural, fast path for branchy logic)."""
import esper
import common as C

DT = C.DT
DRAINDT, RESPAWN = C.DRAIN * C.DT, C.RESPAWN


class Health:
    __slots__ = ("hp", "state", "timer")
    def __init__(self, hp, state, timer): self.hp, self.state, self.timer = hp, state, timer


def _ai_tick():
    for _e, (h,) in esper.get_components(Health):
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
    esper.clear_database()
    s = C.make_scene(n)
    for i in range(n):
        esper.create_entity(Health(float(s["hp"][i]), int(s["state"][i]), float(s["timer"][i])))
    return None


def step(_):
    _ai_tick()


def collect(_):
    xs = [(h.hp, h.state, h.timer) for _e, (h,) in esper.get_components(Health)]
    return C._fp([r[0] for r in xs], [r[1] for r in xs], [r[2] for r in xs])
