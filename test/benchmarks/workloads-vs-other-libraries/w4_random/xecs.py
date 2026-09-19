"""w4 random -- xecs: column store; scatter the K hits by round-tripping the column .numpy()->.fill()."""
# ruff: noqa
import numpy as np
import xecs as xx
import common as C

DMG = C.DMG


class Dmg(xx.Component):
    hp: xx.Float32


def _f32(a): return np.ascontiguousarray(a, dtype=np.float32)


def build(n):
    s = C.make_scene(n)
    app = xx.SimulationApp()
    app.add_pool(Dmg.create_pool(n))
    cmd, world = app._commands, app.world
    cmd.spawn([Dmg], n)
    world.get_view(Dmg).hp.fill(_f32(s["hp"]))
    tg = C.damage_targets(n, C.FRAMES, C.k_for(n))
    return {"world": world, "tg": tg, "fc": [0]}


def step(st):
    f = st["fc"][0]; st["fc"][0] += 1
    h = st["world"].get_view(Dmg)
    hp = h.hp.numpy()                    # column store: scatter the K hits, round-trip the column
    hp[st["tg"][f]] -= DMG
    h.hp.fill(hp)


def collect(st):
    return C._fp(st["world"].get_view(Dmg).hp.numpy())
