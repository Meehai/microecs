"""w4 random -- flecs: read-modify-write by entity handle via entity.get(T)."""
import flecs
import common as C

DMG = C.DMG


class Dmg:
    __slots__ = ("hp",)
    def __init__(self, hp): self.hp = hp


def build(n):
    w = flecs.World()
    s = C.make_scene(n)
    ids = []
    for i in range(n):
        e = w.entity()
        e.set(Dmg(float(s["hp"][i])))
        ids.append(e)
    return {"w": w, "ids": ids, "tg": C.damage_targets(n, C.FRAMES, C.k_for(n)), "fc": [0]}


def step(st):
    ids, tg = st["ids"], st["tg"]
    f = st["fc"][0]; st["fc"][0] += 1
    for t in tg[f]:
        ids[t].get(Dmg).hp -= DMG


def collect(st):
    return C._fp([d.hp for _e, d in st["w"].query(Dmg)])
