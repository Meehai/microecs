"""w2 bounce -- ecs-pattern: physics + per-entity wall reflection over entity objects."""
from ecs_pattern import component, entity, EntityManager
import common as C

DT, BOUND = C.DT, C.BOUND


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


@entity
class PhysAcc(Position, Acceleration): pass
@entity
class Phys(Position): pass


def build(n):
    em = EntityManager()
    s = C.make_scene(n)
    ents = []
    for i in range(n):
        px, py = float(s["pos"][i, 0]), float(s["pos"][i, 1])
        vx, vy = float(s["vel"][i, 0]), float(s["vel"][i, 1])
        if s["has_acc"][i]:
            ents.append(PhysAcc(x=px, y=py, vx=vx, vy=vy,
                                ax=float(s["acc"][i, 0]), ay=float(s["acc"][i, 1])))
        else:
            ents.append(Phys(x=px, y=py, vx=vx, vy=vy))
    em.add(*ents)
    return em


def step(em):
    for e in em.get_with_component(Position, Acceleration):
        e.vx += e.ax * DT
        e.vy += e.ay * DT
    for e in em.get_with_component(Position):
        e.x += e.vx * DT
        e.y += e.vy * DT
    for e in em.get_with_component(Position):
        if e.x > BOUND or e.x < -BOUND: e.vx = -e.vx
        if e.y > BOUND or e.y < -BOUND: e.vy = -e.vy


def collect(em):
    return C._fp([(e.x, e.y, e.vx, e.vy) for e in em.get_with_component(Position)])
