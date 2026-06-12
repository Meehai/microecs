"""xecs benchmark -- a Rust-backed, struct-of-arrays Python ECS (tweng/xecs).

xecs is the one competitor here that is *also* vectorized: components are columns of
Float32 arrays (backed by Rust), and ops like `pos.x += vel.x*dt` run batched, not
per entity. So this is the real fight against microecs' numpy SoA.

Like every other script we drive the data directly (component views), bypassing the
app scheduler -- the same level at which esper/snecs/ecs-pattern/microecs are driven --
to isolate the batch-update cost. Entities are grouped acceleration-first (xecs' natural
contiguous pool layout); the fingerprint is order-independent, so it still verifies.
"""
import numpy as np
import xecs as xx
import _common as C


class Position(xx.Component):
    x: xx.Float32
    y: xx.Float32
class Velocity(xx.Component):
    x: xx.Float32
    y: xx.Float32
class Acceleration(xx.Component):
    x: xx.Float32
    y: xx.Float32


def _f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def build(n):
    pos0, vel0, acc, has_acc = C.make_data(n)
    acc_idx = np.flatnonzero(has_acc)        # entities with acceleration -> spawned first
    order = np.concatenate([acc_idx, np.flatnonzero(~has_acc)])
    n_acc = int(acc_idx.size)
    pos_ord, vel_ord = pos0[order], vel0[order]
    acc_ord = acc[acc_idx]

    app = xx.SimulationApp()
    app.add_pool(Position.create_pool(n))
    app.add_pool(Velocity.create_pool(n))
    app.add_pool(Acceleration.create_pool(n_acc))
    cmd, world = app._commands, app.world   # _commands is what the app injects into systems

    # group A (acceleration) occupies pool slots [0, n_acc); group B fills the rest
    _a_pos, a_vel, _a_acc = cmd.spawn([Position, Velocity, Acceleration], n_acc)
    if n - n_acc:
        cmd.spawn([Position, Velocity], n - n_acc)

    pos_v, vel_v, acc_v = world.get_view(Position), world.get_view(Velocity), world.get_view(Acceleration)
    pos_v.x.fill(_f32(pos_ord[:, 0])); pos_v.y.fill(_f32(pos_ord[:, 1]))
    vel_v.x.fill(_f32(vel_ord[:, 0])); vel_v.y.fill(_f32(vel_ord[:, 1]))
    acc_v.x.fill(_f32(acc_ord[:, 0])); acc_v.y.fill(_f32(acc_ord[:, 1]))
    return world, a_vel


def step(state):
    world, a_vel = state
    dt = C.DT
    # integrate velocity over the acceleration group (a sub-view of the Velocity pool)
    vel_a = world.get_view(Velocity, a_vel)
    acc_v = world.get_view(Acceleration)
    vel_a.x += acc_v.x * dt
    vel_a.y += acc_v.y * dt
    # integrate position over all entities
    pos_v = world.get_view(Position)
    vel_v = world.get_view(Velocity)
    pos_v.x += vel_v.x * dt
    pos_v.y += vel_v.y * dt


def collect_positions(state):
    world, _a_vel = state
    pos_v = world.get_view(Position)
    return np.column_stack([pos_v.x.numpy(), pos_v.y.numpy()])


def run(n=C.N_DEFAULT, warmup=C.WARMUP, measure=C.MEASURE):
    return C.run_bench("xecs", C.lib_version("xecs"),
                       build, step, collect_positions, n, warmup, measure)


def main():
    n, measure = C.cli_n_measure()
    C.print_result(run(n=n, measure=measure))


if __name__ == "__main__":
    main()
