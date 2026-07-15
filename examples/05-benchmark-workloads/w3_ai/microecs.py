"""w3 ai -- microecs: branchy per-entity state machine pushed into np.where masks (the microecs idiom)."""
from dataclasses import field
import numpy as np
from microecs import World, Component
import common as C

DT32    = C.DT32
DRAINDT = np.float32(C.DRAIN * C.DT)
RESPAWN = np.float32(C.RESPAWN)


def _f(shape, dtype="float32"):
    return field(metadata={"shape": shape, "dtype": dtype, "default": None})


class Health(Component):
    hp:    np.ndarray = _f((1,))
    state: np.ndarray = _f((1,), "int32")
    timer: np.ndarray = _f((1,))


def _s(x):  return np.array([x], np.float32)
def _si(x): return np.array([x], np.int32)


def _ai_tick(q):
    """One masked state-machine tick over a Health query (works across any number of pools)."""
    hp    = q.hp.numpy()
    state = q.state.numpy()
    timer = q.timer.numpy()
    alive = state == 0
    dead  = ~alive
    new_hp    = np.where(alive, hp - DRAINDT, hp)
    just_died = alive & (new_hp <= 0)
    new_state = np.where(just_died, 1, state)
    new_timer = np.where(just_died, RESPAWN, timer)
    new_timer = np.where(dead, new_timer - DT32, new_timer)
    respawn   = dead & (new_timer <= 0)
    new_state = np.where(respawn, 0, new_state)
    new_hp    = np.where(respawn, np.float32(100.0), new_hp)
    q.hp[:]    = new_hp.astype(np.float32)
    q.state[:] = new_state.astype(np.int32)
    q.timer[:] = new_timer.astype(np.float32)


def build(n):
    s = C.make_scene(n)
    w = World([Health])
    for i in range(n):
        w.add_entity([Health], hp=_s(s["hp"][i]), state=_si(s["state"][i]), timer=_s(s["timer"][i]))
    w.update()
    return w


def step(w):
    _ai_tick(w.query(Health))


def collect(w):
    q = w.query(Health)
    return C._fp(q.hp.numpy(), q.state.numpy(), q.timer.numpy())
