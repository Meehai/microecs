"""w2 bounce -- xecs: physics + wall reflection. The branch must round-trip through .numpy()+.fill()
(xecs' masked array assignment is unreliable in 0.9.0 and .numpy() writes don't persist)."""
# ruff: noqa
import numpy as np
import xecs as xx
import common as C

DT, BOUND = C.DT, C.BOUND


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
    px, py = pos.x.numpy(), pos.y.numpy()
    vx, vy = vel.x.numpy(), vel.y.numpy()
    vel.x.fill(np.where(np.abs(px) > BOUND, -vx, vx).astype(np.float32))
    vel.y.fill(np.where(np.abs(py) > BOUND, -vy, vy).astype(np.float32))


def collect(st):
    world = st["world"]
    pv, vv = world.get_view(Position), world.get_view(Velocity)
    return C._fp(np.column_stack([pv.x.numpy(), pv.y.numpy()]),
                 np.column_stack([vv.x.numpy(), vv.y.numpy()]))
