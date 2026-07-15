"""w6 mixed -- esper: physics loop + ai if-branch + K targeted hits via component_for_entity(id)."""
import esper
import common as C

DT, DMG = C.DT, C.DMG
DRAINDT, RESPAWN = C.DRAIN * C.DT, C.RESPAWN


class Position:
    __slots__ = ("x", "y", "vx", "vy")
    def __init__(self, x, y, vx, vy): self.x, self.y, self.vx, self.vy = x, y, vx, vy
class Acceleration:
    __slots__ = ("ax", "ay")
    def __init__(self, ax, ay): self.ax, self.ay = ax, ay
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
    ids = []
    for i in range(n):
        p = Position(float(s["pos"][i, 0]), float(s["pos"][i, 1]),
                     float(s["vel"][i, 0]), float(s["vel"][i, 1]))
        h = Health(float(s["hp"][i]), int(s["state"][i]), float(s["timer"][i]))
        if s["has_acc"][i]:
            ids.append(esper.create_entity(p, Acceleration(float(s["acc"][i, 0]), float(s["acc"][i, 1])), h))
        else:
            ids.append(esper.create_entity(p, h))
    tg = C.damage_targets(n, C.FRAMES, C.k_for(n))
    return {"ids": ids, "tg": tg, "fc": [0]}


def step(st):
    ids, tg = st["ids"], st["tg"]
    f = st["fc"][0]; st["fc"][0] += 1
    for _e, (p, a) in esper.get_components(Position, Acceleration):
        p.vx += a.ax * DT
        p.vy += a.ay * DT
    for _e, (p,) in esper.get_components(Position):
        p.x += p.vx * DT
        p.y += p.vy * DT
    _ai_tick()
    for t in tg[f]:
        esper.component_for_entity(ids[t], Health).hp -= DMG


def collect(_st):
    pos = [(p.x, p.y) for _e, (p,) in esper.get_components(Position)]
    heal = [(h.hp, h.state, h.timer) for _e, (h,) in esper.get_components(Health)]
    return C._fp([r[0] for r in pos], [r[1] for r in pos],
                 [r[0] for r in heal], [r[1] for r in heal], [r[2] for r in heal])
