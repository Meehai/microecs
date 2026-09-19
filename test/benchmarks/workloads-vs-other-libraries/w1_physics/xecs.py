"""w1 physics -- xecs: Rust struct-of-arrays, in-place `+=` fast path (the one other vectorized ECS).

Entities with acceleration are spawned first so the velocity integrate is a contiguous sub-view
(xecs' natural layout); the fingerprint is order-independent so this still verifies.
"""
# ruff: noqa
import numpy as np
import xecs as xx
import common as C

DT = C.DT


class Position(xx.Component):
    x: xx.Float32
    y: xx.Float32
class Velocity(xx.Component):
    x: xx.Float32
    y: xx.Float32
class Acceleration(xx.Component):
    x: xx.Float32
    y: xx.Float32


def _f32(a): return np.ascontiguousarray(a, dtype=np.float32)


def build(n):
    s = C.make_scene(n)
    m = s["has_acc"]
    acc_idx = np.flatnonzero(m)
    order = np.concatenate([acc_idx, np.flatnonzero(~m)])
    n_acc = int(acc_idx.size)
    pos_ord, vel_ord, acc_ord = s["pos"][order], s["vel"][order], s["acc"][acc_idx]

    app = xx.SimulationApp()
    app.add_pool(Position.create_pool(n))
    app.add_pool(Velocity.create_pool(n))
    app.add_pool(Acceleration.create_pool(n_acc))
    cmd, world = app._commands, app.world
    _p, a_vel, _a = cmd.spawn([Position, Velocity, Acceleration], n_acc)
    if n - n_acc:
        cmd.spawn([Position, Velocity], n - n_acc)

    pv, vv, av = world.get_view(Position), world.get_view(Velocity), world.get_view(Acceleration)
    pv.x.fill(_f32(pos_ord[:, 0])); pv.y.fill(_f32(pos_ord[:, 1]))
    vv.x.fill(_f32(vel_ord[:, 0])); vv.y.fill(_f32(vel_ord[:, 1]))
    av.x.fill(_f32(acc_ord[:, 0])); av.y.fill(_f32(acc_ord[:, 1]))
    return {"world": world, "a_vel": a_vel}


def step(st):
    world, a_vel = st["world"], st["a_vel"]
    vel_a = world.get_view(Velocity, a_vel)
    acc = world.get_view(Acceleration)
    vel_a.x += acc.x * DT
    vel_a.y += acc.y * DT
    pos, vel = world.get_view(Position), world.get_view(Velocity)
    pos.x += vel.x * DT
    pos.y += vel.y * DT


def collect(st):
    pv = st["world"].get_view(Position)
    return C._fp(np.column_stack([pv.x.numpy(), pv.y.numpy()]))
