"""w3 ai -- ecs-pattern: per-entity if-branch state machine over entity objects."""
from ecs_pattern import component, entity, EntityManager
import common as C

DT = C.DT
DRAINDT, RESPAWN = C.DRAIN * C.DT, C.RESPAWN


@component
class Health:
    hp: float = 0.0
    state: int = 0
    timer: float = 0.0


@entity
class Mob(Health): pass


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
    em.add(*[Mob(hp=float(s["hp"][i]), state=int(s["state"][i]), timer=float(s["timer"][i]))
             for i in range(n)])
    return em


def step(em):
    _ai_tick(em)


def collect(em):
    xs = [(e.hp, e.state, e.timer) for e in em.get_with_component(Health)]
    return C._fp([r[0] for r in xs], [r[1] for r in xs], [r[2] for r in xs])
