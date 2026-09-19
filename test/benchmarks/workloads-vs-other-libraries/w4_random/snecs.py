"""w4 random -- snecs: per-entity access via entity_component(id, T, world)."""
import snecs
from snecs import Component, register_component, Query
import common as C

DMG = C.DMG


@register_component
class Dmg(Component):
    __slots__ = ("hp",)
    def __init__(self, hp): self.hp = hp


def build(n):
    w = snecs.World()
    s = C.make_scene(n)
    ids = [snecs.new_entity([Dmg(float(s["hp"][i]))], world=w) for i in range(n)]
    tg = C.damage_targets(n, C.FRAMES, C.k_for(n))
    return {"w": w, "ids": ids, "tg": tg, "fc": [0], "q": Query((Dmg,), world=w).compile()}


def step(st):
    w, ids, tg = st["w"], st["ids"], st["tg"]
    f = st["fc"][0]; st["fc"][0] += 1
    for t in tg[f]:
        snecs.entity_component(ids[t], Dmg, world=w).hp -= DMG


def collect(st):
    return C._fp([d.hp for _e, (d,) in st["q"]])
