"""esper benchmark -- the most popular Python ECS (benmoran56/esper).

esper stores each component as a plain python object per entity; a system iterates
`get_components(...)` and mutates the objects in place. We give it its fastest layout:
__slots__ components holding plain python-float x/y (no numpy per entity).
"""
import esper
import _common as C


class Position:
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x, self.y = x, y
class Velocity:
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x, self.y = x, y
class Acceleration:
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x, self.y = x, y


def build(n):
    esper.clear_database()   # esper uses a module-global world; start clean
    pos0, vel0, acc, has_acc = C.make_data(n)
    for i in range(n):
        if has_acc[i]:
            esper.create_entity(Position(float(pos0[i, 0]), float(pos0[i, 1])),
                                Velocity(float(vel0[i, 0]), float(vel0[i, 1])),
                                Acceleration(float(acc[i, 0]), float(acc[i, 1])))
        else:
            esper.create_entity(Position(float(pos0[i, 0]), float(pos0[i, 1])),
                                Velocity(float(vel0[i, 0]), float(vel0[i, 1])))
    return None   # state lives in esper's global world


def step(_):
    dt = C.DT
    for _ent, (vel, acc) in esper.get_components(Velocity, Acceleration):
        vel.x += acc.x * dt
        vel.y += acc.y * dt
    for _ent, (pos, vel) in esper.get_components(Position, Velocity):
        pos.x += vel.x * dt
        pos.y += vel.y * dt


def collect_positions(_):
    return [(p.x, p.y) for _ent, (p,) in esper.get_components(Position)]


def run(n=C.N_DEFAULT, warmup=C.WARMUP, measure=C.MEASURE):
    return C.run_bench("esper", C.lib_version("esper"),
                       build, step, collect_positions, n, warmup, measure)


def main():
    n, measure = C.cli_n_measure()
    C.print_result(run(n=n, measure=measure))


if __name__ == "__main__":
    main()
