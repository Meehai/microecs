"""w4 random -- esper: per-entity access via component_for_entity(id) (an O(1) dict lookup)."""
import esper
import common as C

DMG = C.DMG


class Dmg:
    __slots__ = ("hp",)
    def __init__(self, hp): self.hp = hp


def build(n):
    esper.clear_database()
    s = C.make_scene(n)
    ids = [esper.create_entity(Dmg(float(s["hp"][i]))) for i in range(n)]
    tg = C.damage_targets(n, C.FRAMES, C.k_for(n))
    return {"ids": ids, "tg": tg, "fc": [0]}


def step(st):
    ids, tg = st["ids"], st["tg"]
    f = st["fc"][0]; st["fc"][0] += 1
    for t in tg[f]:
        esper.component_for_entity(ids[t], Dmg).hp -= DMG


def collect(_st):
    return C._fp([d.hp for _e, (d,) in esper.get_components(Dmg)])
