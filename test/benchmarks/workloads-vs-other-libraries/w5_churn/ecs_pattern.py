"""w5 churn -- ecs-pattern: em.add / em.delete with a FIFO deque of entity objects."""
from collections import deque
import numpy as np
from ecs_pattern import component, entity, EntityManager
import common as C

DT = C.DT


@component
class Position:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
@component
class PayloadC:
    payload: float = 0.0


@entity
class Bullet(Position, PayloadC): pass


def build(n):
    em = EntityManager()
    s = C.make_scene(n)
    order = deque()
    ents = []
    for i in range(n):
        e = Bullet(x=float(s["pos"][i, 0]), y=float(s["pos"][i, 1]),
                   vx=float(s["vel"][i, 0]), vy=float(s["vel"][i, 1]), payload=-1.0)
        ents.append(e)
        order.append(e)
    em.add(*ents)
    sp = C.spawn_payloads(C.FRAMES, C.b_for(n))
    return {"em": em, "order": order, "sp": sp, "fc": [0]}


def step(st):
    em, order, sp = st["em"], st["order"], st["sp"]
    f = st["fc"][0]; st["fc"][0] += 1
    b = sp.shape[1]
    for _ in range(min(b, len(order))):
        em.delete(order.popleft())
    for p in sp[f]:
        e = Bullet(x=0.0, y=0.0, vx=1.0, vy=1.0, payload=float(p))
        em.add(e)
        order.append(e)
    for e in em.get_with_component(Position):
        e.x += e.vx * DT
        e.y += e.vy * DT


def collect(st):
    return np.sort(np.array([e.payload for e in st["em"].get_with_component(PayloadC)], float))
