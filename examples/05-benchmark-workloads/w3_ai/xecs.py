"""w3 ai -- xecs: masked state machine; every masked write round-trips .numpy()->.fill() through Rust."""
# ruff: noqa
import numpy as np
import xecs as xx
import common as C

DT = C.DT
DRAINDT, RESPAWN = np.float32(C.DRAIN * C.DT), np.float32(C.RESPAWN)


class Health(xx.Component):
    hp: xx.Float32
    state: xx.Int32
    timer: xx.Float32


def _f32(a): return np.ascontiguousarray(a, dtype=np.float32)


def _ai_fill(h):
    hp, state, timer = h.hp.numpy(), h.state.numpy(), h.timer.numpy()
    alive = state == 0
    dead  = ~alive
    new_hp    = np.where(alive, hp - DRAINDT, hp)
    just_died = alive & (new_hp <= 0)
    new_state = np.where(just_died, 1, state)
    new_timer = np.where(just_died, RESPAWN, timer)
    new_timer = np.where(dead, new_timer - np.float32(DT), new_timer)
    respawn   = dead & (new_timer <= 0)
    new_state = np.where(respawn, 0, new_state)
    new_hp    = np.where(respawn, np.float32(100.0), new_hp)
    h.hp.fill(new_hp.astype(np.float32))
    h.state.fill(new_state.astype(np.int32))
    h.timer.fill(new_timer.astype(np.float32))


def build(n):
    s = C.make_scene(n)
    app = xx.SimulationApp()
    app.add_pool(Health.create_pool(n))
    cmd, world = app._commands, app.world
    cmd.spawn([Health], n)
    h = world.get_view(Health)
    h.hp.fill(_f32(s["hp"]))
    h.state.fill(np.ascontiguousarray(s["state"], np.int32))
    h.timer.fill(_f32(s["timer"]))
    return {"world": world}


def step(st):
    _ai_fill(st["world"].get_view(Health))


def collect(st):
    h = st["world"].get_view(Health)
    return C._fp(h.hp.numpy(), h.state.numpy(), h.timer.numpy())
