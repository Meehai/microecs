"""w5 churn -- microecs: add/remove_entity + update() with a FIFO id deque, then integrate all."""
from collections import deque
from dataclasses import field
import numpy as np
from microecs import World, Component
import common as C

DT32 = C.DT32


def _f(shape, dtype="float32"):
    return field(metadata={"shape": shape, "dtype": dtype, "default": None})


class Pos(Component):
    position: np.ndarray = _f((2,))
    velocity: np.ndarray = _f((2,))
class Pay(Component):
    payload: np.ndarray = _f((1,))


def _v2(a): return np.asarray(a, np.float32)
def _s(x):  return np.array([x], np.float32)


def build(n):
    s = C.make_scene(n)
    w = World([Pos, Pay])
    order = deque()
    for i in range(n):
        eid = w.add_entity([Pos, Pay], position=_v2(s["pos"][i]), velocity=_v2(s["vel"][i]),
                           payload=_s(-1.0))
        order.append(eid)
    w.update()
    sp = C.spawn_payloads(C.FRAMES, C.b_for(n))
    return {"w": w, "order": order, "sp": sp, "fc": [0],
            "spos": _v2([0.0, 0.0]), "svel": _v2([1.0, 1.0])}


def step(st):
    w, order, sp = st["w"], st["order"], st["sp"]
    f = st["fc"][0]; st["fc"][0] += 1
    b = sp.shape[1]
    for _ in range(min(b, len(order))):
        w.remove_entity(order.popleft())
    for p in sp[f]:
        order.append(w.add_entity([Pos, Pay], position=st["spos"], velocity=st["svel"], payload=_s(p)))
    w.update()
    qp = w.query(Pos)
    qp.position[:] = qp.position + qp.velocity * DT32


def collect(st):
    return np.sort(st["w"].query(Pay).payload.numpy().ravel().astype(np.float64))
