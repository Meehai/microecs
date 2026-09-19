"""w6 mixed -- ecs-pattern: physics loop + ai if-branch + K targeted hits via direct object refs."""
from ecs_pattern import component, entity, EntityManager
import common as C

DT, DMG = C.DT, C.DMG
DRAINDT, RESPAWN = C.DRAIN * C.DT, C.RESPAWN


@component
class Position:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
@component
class Acceleration:
    ax: float = 0.0
    ay: float = 0.0
@component
class Health:
    hp: float = 0.0
    state: int = 0
    timer: float = 0.0


@entity
class MixAcc(Position, Acceleration, Health): pass
@entity
class MixNoAcc(Position, Health): pass


def _ai_tick(em):
    for e in em.get_with_component(Health):
        if e.state == 0:
            e.hp -= DRAINDT
            if e.hp <= 0:
                e.state = 1
                e.timer = RESPAWN
        else:
            e.timer -= DT
            if e.timer <= 0:
                e.state = 0
                e.hp = 100.0


def build(n):
    em = EntityManager()
    s = C.make_scene(n)
    ents = []
    for i in range(n):
        px, py = float(s["pos"][i, 0]), float(s["pos"][i, 1])
        vx, vy = float(s["vel"][i, 0]), float(s["vel"][i, 1])
        hp, stt, tm = float(s["hp"][i]), int(s["state"][i]), float(s["timer"][i])
        if s["has_acc"][i]:
            ents.append(MixAcc(x=px, y=py, vx=vx, vy=vy, ax=float(s["acc"][i, 0]), ay=float(s["acc"][i, 1]),
                               hp=hp, state=stt, timer=tm))
        else:
            ents.append(MixNoAcc(x=px, y=py, vx=vx, vy=vy, hp=hp, state=stt, timer=tm))
    em.add(*ents)
    tg = C.damage_targets(n, C.FRAMES, C.k_for(n))
    return {"em": em, "ents": ents, "tg": tg, "fc": [0]}


def step(st):
    em, ents, tg = st["em"], st["ents"], st["tg"]
    f = st["fc"][0]; st["fc"][0] += 1
    for e in em.get_with_component(Position, Acceleration):
        e.vx += e.ax * DT
        e.vy += e.ay * DT
    for e in em.get_with_component(Position):
        e.x += e.vx * DT
        e.y += e.vy * DT
    _ai_tick(em)
    for t in tg[f]:
        ents[t].hp -= DMG


def collect(st):
    em = st["em"]
    pos = [(e.x, e.y) for e in em.get_with_component(Position)]
    heal = [(e.hp, e.state, e.timer) for e in em.get_with_component(Health)]
    return C._fp([r[0] for r in pos], [r[1] for r in pos],
                 [r[0] for r in heal], [r[1] for r in heal], [r[2] for r in heal])
