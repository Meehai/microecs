"""w6 mixed -- flecs: physics + ai + K targeted hits by handle (a realistic steady frame)."""
import flecs
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
    w = flecs.World()
    s = C.make_scene(n)
    ids = []
    for i in range(n):
        e = w.entity()
        e.set(Position(float(s["pos"][i, 0]), float(s["pos"][i, 1]),
                       float(s["vel"][i, 0]), float(s["vel"][i, 1])))
        if s["has_acc"][i]:
            e.set(Acceleration(float(s["acc"][i, 0]), float(s["acc"][i, 1])))
        e.set(Health(float(s["hp"][i]), int(s["state"][i]), float(s["timer"][i])))
        ids.append(e)
    return {"w": w, "qpa": w.query(Position, Acceleration), "qp": w.query(Position),
            "qh": w.query(Health), "ids": ids,
            "tg": C.damage_targets(n, C.FRAMES, C.k_for(n)), "fc": [0]}


def step(st):
    ids, tg = st["ids"], st["tg"]
    f = st["fc"][0]; st["fc"][0] += 1
    st["qpa"].reset()
    for _e, p, a in st["qpa"]:
        p.vx += a.ax * DT
        p.vy += a.ay * DT
    st["qp"].reset()
    for _e, p in st["qp"]:
        p.x += p.vx * DT
        p.y += p.vy * DT
    st["qh"].reset()
    for _e, h in st["qh"]:
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
        ids[t].get(Health).hp -= DMG


def collect(st):
    w = st["w"]
    pos = [(p.x, p.y) for _e, p in w.query(Position)]
    heal = [(h.hp, h.state, h.timer) for _e, h in w.query(Health)]
    return C._fp([x[0] for x in pos], [x[1] for x in pos],
                 [x[0] for x in heal], [x[1] for x in heal], [x[2] for x in heal])
