"""w4 random -- ecs-pattern: no entity ids, so random access holds direct object references (its
fastest possible path -- a plain list of the entity objects)."""
from ecs_pattern import component, entity, EntityManager
import common as C

DMG = C.DMG


@component
class DmgC:
    dhp: float = 0.0


@entity
class Target(DmgC): pass


def build(n):
    em = EntityManager()
    s = C.make_scene(n)
    ents = [Target(dhp=float(s["hp"][i])) for i in range(n)]
    em.add(*ents)
    tg = C.damage_targets(n, C.FRAMES, C.k_for(n))
    return {"em": em, "ents": ents, "tg": tg, "fc": [0]}


def step(st):
    ents, tg = st["ents"], st["tg"]
    f = st["fc"][0]; st["fc"][0] += 1
    for t in tg[f]:
        ents[t].dhp -= DMG


def collect(st):
    return C._fp([e.dhp for e in st["em"].get_with_component(DmgC)])
