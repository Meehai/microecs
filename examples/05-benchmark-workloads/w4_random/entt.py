"""w4 random -- PyEnTT: read-modify-write by entity handle via registry.get(e, T) (C++ sparse-set O(1))."""
import entt
import common as C

DMG = C.DMG


class Dmg:
    __slots__ = ("hp",)
    def __init__(self, hp): self.hp = hp


def build(n):
    r = entt.Registry()
    s = C.make_scene(n)
    ids = []
    for i in range(n):
        e = r.create()
        r.emplace(e, Dmg, float(s["hp"][i]))
        ids.append(e)
    return {"r": r, "ids": ids, "tg": C.damage_targets(n, C.FRAMES, C.k_for(n)), "fc": [0]}


def step(st):
    r, ids, tg = st["r"], st["ids"], st["tg"]
    f = st["fc"][0]; st["fc"][0] += 1
    for t in tg[f]:
        r.get(ids[t], Dmg).hp -= DMG


def collect(st):
    return C._fp([d.hp for _e, d in st["r"].view(Dmg)])
