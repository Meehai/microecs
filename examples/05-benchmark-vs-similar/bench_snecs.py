"""snecs benchmark -- a sparse-set, type-hinted Python ECS (slavfox/snecs).

snecs components subclass `Component` and are registered once. Queries are *compiled*
outside the hot loop (snecs' documented fast path) and iterated each frame, mutating the
component objects in place. Fastest layout: __slots__ components with python-float x/y.
"""
import snecs
from snecs import Component, register_component, Query
import _common as C


@register_component
class Position(Component):
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x, self.y = x, y
@register_component
class Velocity(Component):
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x, self.y = x, y
@register_component
class Acceleration(Component):
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x, self.y = x, y


def build(n):
    world = snecs.World()
    pos0, vel0, acc, has_acc = C.make_data(n)
    for i in range(n):
        if has_acc[i]:
            snecs.new_entity([Position(float(pos0[i, 0]), float(pos0[i, 1])),
                              Velocity(float(vel0[i, 0]), float(vel0[i, 1])),
                              Acceleration(float(acc[i, 0]), float(acc[i, 1]))], world=world)
        else:
            snecs.new_entity([Position(float(pos0[i, 0]), float(pos0[i, 1])),
                              Velocity(float(vel0[i, 0]), float(vel0[i, 1]))], world=world)
    # compile queries once, outside the hot loop (the snecs-recommended pattern)
    vel_q = Query((Velocity, Acceleration), world=world).compile()
    pos_q = Query((Position, Velocity), world=world).compile()
    only_pos_q = Query((Position,), world=world).compile()
    return world, vel_q, pos_q, only_pos_q


def step(state):
    _world, vel_q, pos_q, _only_pos_q = state
    dt = C.DT
    for _eid, (vel, acc) in vel_q:
        vel.x += acc.x * dt
        vel.y += acc.y * dt
    for _eid, (pos, vel) in pos_q:
        pos.x += vel.x * dt
        pos.y += vel.y * dt


def collect_positions(state):
    _world, _vel_q, _pos_q, only_pos_q = state
    return [(pos.x, pos.y) for _eid, (pos,) in only_pos_q]


def run(n=C.N_DEFAULT, warmup=C.WARMUP, measure=C.MEASURE):
    return C.run_bench("snecs", C.lib_version("snecs"),
                       build, step, collect_positions, n, warmup, measure)


def main():
    n, measure = C.cli_n_measure()
    C.print_result(run(n=n, measure=measure))


if __name__ == "__main__":
    main()
