"""Is microecs' random-access cliff (w4) intrinsic, or just the naive get_entity() API?

A savvy user with a static entity set builds an id->row map once, then scatters the K hits into
the pool column with numpy fancy-indexing (`col[rows] -= DMG`). This probe compares that against
the naive `world.get_entity(id).hp = ...` per-hit path. Same result, verified. Shows how much of
microecs' random-access cost is API overhead (get_entity view + __getattr__) vs intrinsic.
"""
import gc
import os
import sys
import time
import numpy as np
from dataclasses import field
from microecs import World, Component

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C


class Dmg(Component):
    hp: np.ndarray = field(metadata={"shape": (1,), "dtype": "float32", "default": None})


def _time(fn, iters=30):
    for _ in range(3):
        fn()
    gc.collect(); gc.disable()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    dt = (time.perf_counter() - t0) / iters
    gc.enable()
    return dt


def main():
    print(f"{'N':>8} {'K':>6} {'get_entity(ms)':>16} {'batched(ms)':>13} "
          f"{'ns/hit get':>12} {'ns/hit batch':>14} {'speedup':>9}")
    for n in [1000, 20000, 100000]:
        k = C.k_for(n)
        tg = C.damage_targets(n, C.FRAMES, k)
        s = C.make_scene(n)

        # --- naive per-entity get_entity path ---
        w = World([Dmg])
        ids = [w.add_entity([Dmg], hp=np.array([s["hp"][i]], np.float32)) for i in range(n)]
        w.update()
        fc = [0]
        def step_get():
            f = fc[0] % C.FRAMES; fc[0] += 1
            for t in tg[f]:
                e = w.get_entity(ids[t])
                e.hp = e.hp - np.float32(C.DMG)
        t_get = _time(step_get)

        # --- batched scatter path (id->row map built once; static set) ---
        w2 = World([Dmg])
        ids2 = [w2.add_entity([Dmg], hp=np.array([s["hp"][i]], np.float32)) for i in range(n)]
        w2.update()
        qr = w2.query(Dmg)
        col = qr.hp.numpy()                     # single pool -> live view into pool memory
        row_of = {eid: r for r, eid in enumerate(qr.entity_ids.tolist())}
        rows_by_frame = [np.array([row_of[ids2[t]] for t in tg[f]], np.int64) for f in range(C.FRAMES)]
        fc2 = [0]
        def step_batch():
            f = fc2[0] % C.FRAMES; fc2[0] += 1
            col[rows_by_frame[f], 0] -= np.float32(C.DMG)
        t_batch = _time(step_batch)

        print(f"{n:>8} {k:>6} {t_get*1e3:>16.4f} {t_batch*1e3:>13.4f} "
              f"{t_get/k*1e9:>12.0f} {t_batch/k*1e9:>14.0f} {t_get/t_batch:>8.1f}x")


if __name__ == "__main__":
    main()
