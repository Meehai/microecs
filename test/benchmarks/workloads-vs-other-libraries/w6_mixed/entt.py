"""w6 mixed -- PyEnTT: physics view + ai view + K targeted hits by handle (a realistic steady frame)."""
import entt
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


def build(n):
    r = entt.Registry()
    s = C.make_scene(n)
    ids = []
    for i in range(n):
        e = r.create()
        r.emplace(e, Position, float(s["pos"][i, 0]), float(s["pos"][i, 1]),
                  float(s["vel"][i, 0]), float(s["vel"][i, 1]))
        if s["has_acc"][i]:
            r.emplace(e, Acceleration, float(s["acc"][i, 0]), float(s["acc"][i, 1]))
        r.emplace(e, Health, float(s["hp"][i]), int(s["state"][i]), float(s["timer"][i]))
        ids.append(e)
    return {"r": r, "ids": ids, "tg": C.damage_targets(n, C.FRAMES, C.k_for(n)), "fc": [0]}


def step(st):
    r, ids, tg = st["r"], st["ids"], st["tg"]
    f = st["fc"][0]; st["fc"][0] += 1
    for _e, p, a in r.view(Position, Acceleration):
        p.vx += a.ax * DT
        p.vy += a.ay * DT
    for _e, p in r.view(Position):
        p.x += p.vx * DT
        p.y += p.vy * DT
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
    for t in tg[f]:
        r.get(ids[t], Health).hp -= DMG


def collect(st):
    r = st["r"]
    pos = [(p.x, p.y) for _e, p in r.view(Position)]
    heal = [(h.hp, h.state, h.timer) for _e, h in r.view(Health)]
    return C._fp([x[0] for x in pos], [x[1] for x in pos],
                 [x[0] for x in heal], [x[1] for x in heal], [x[2] for x in heal])
