"""w6 mixed -- xecs: physics (in-place) + ai (round-trip masks) + K targeted hits (column scatter)."""
# ruff: noqa
import numpy as np
import xecs as xx
import common as C

DT, DMG = C.DT, C.DMG
DRAINDT, RESPAWN = np.float32(C.DRAIN * C.DT), np.float32(C.RESPAWN)


class Position(xx.Component):
    x: xx.Float32
    y: xx.Float32
class Velocity(xx.Component):
    x: xx.Float32
    y: xx.Float32
class Acceleration(xx.Component):
    x: xx.Float32
    y: xx.Float32
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
    m = s["has_acc"]
    acc_idx = np.flatnonzero(m)
    order = np.concatenate([acc_idx, np.flatnonzero(~m)])
    n_acc = int(acc_idx.size)
    inv = np.empty(n, np.int64); inv[order] = np.arange(n)   # scene index -> xecs row
    pos_ord, vel_ord = s["pos"][order], s["vel"][order]
    acc_ord = s["acc"][acc_idx]
    hp_ord, st_ord, tm_ord = s["hp"][order], s["state"][order], s["timer"][order]

    app = xx.SimulationApp()
    app.add_pool(Position.create_pool(n))
    app.add_pool(Velocity.create_pool(n))
    app.add_pool(Acceleration.create_pool(n_acc))
    app.add_pool(Health.create_pool(n))
    cmd, world = app._commands, app.world
    _p, a_vel, _a, _h = cmd.spawn([Position, Velocity, Acceleration, Health], n_acc)
    if n - n_acc:
        cmd.spawn([Position, Velocity, Health], n - n_acc)

    pv, vv, av, hv = (world.get_view(Position), world.get_view(Velocity),
                      world.get_view(Acceleration), world.get_view(Health))
    pv.x.fill(_f32(pos_ord[:, 0])); pv.y.fill(_f32(pos_ord[:, 1]))
    vv.x.fill(_f32(vel_ord[:, 0])); vv.y.fill(_f32(vel_ord[:, 1]))
    av.x.fill(_f32(acc_ord[:, 0])); av.y.fill(_f32(acc_ord[:, 1]))
    hv.hp.fill(_f32(hp_ord)); hv.state.fill(np.ascontiguousarray(st_ord, np.int32)); hv.timer.fill(_f32(tm_ord))
    tg = C.damage_targets(n, C.FRAMES, C.k_for(n))
    return {"world": world, "a_vel": a_vel, "tg": inv[tg], "fc": [0]}


def step(st):
    world, a_vel = st["world"], st["a_vel"]
    f = st["fc"][0]; st["fc"][0] += 1
    vel_a = world.get_view(Velocity, a_vel)
    acc = world.get_view(Acceleration)
    vel_a.x += acc.x * DT
    vel_a.y += acc.y * DT
    pos, vel = world.get_view(Position), world.get_view(Velocity)
    pos.x += vel.x * DT
    pos.y += vel.y * DT
    h = world.get_view(Health)
    _ai_fill(h)
    hp = h.hp.numpy()
    hp[st["tg"][f]] -= DMG
    h.hp.fill(hp)


def collect(st):
    world = st["world"]
    pv, hv = world.get_view(Position), world.get_view(Health)
    return C._fp(np.column_stack([pv.x.numpy(), pv.y.numpy()]),
                 hv.hp.numpy(), hv.state.numpy(), hv.timer.numpy())
