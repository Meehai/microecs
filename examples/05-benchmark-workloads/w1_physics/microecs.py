"""w1 physics -- microecs: numpy struct-of-arrays, batched QueryResult write-through (the fast path)."""
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
class Acc(Component):
    acceleration: np.ndarray = _f((2,))


def _v2(a): return np.asarray(a, np.float32)


def build(n):
    s = C.make_scene(n)
    w = World([Pos, Acc])
    for i in range(n):
        if s["has_acc"][i]:
            w.add_entity([Pos, Acc], position=_v2(s["pos"][i]), velocity=_v2(s["vel"][i]),
                         acceleration=_v2(s["acc"][i]))
        else:
            w.add_entity([Pos], position=_v2(s["pos"][i]), velocity=_v2(s["vel"][i]))
    w.update()
    return w


def step(w):
    qv = w.query(Pos, Acc)
    qv.velocity[:] = qv.velocity + qv.acceleration * DT32
    qp = w.query(Pos)
    qp.position[:] = qp.position + qp.velocity * DT32


def collect(w):
    return C._fp(w.query(Pos).position.numpy())
