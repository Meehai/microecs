"""ecs-pattern benchmark -- a dataclass-based Python ECS (ikvk/ecs-pattern).

ecs-pattern merges component fields onto an entity via inheritance, so field names must
be unique across the components an entity combines (Position->x,y; Velocity->vx,vy;
Acceleration->ax,ay). A system iterates `get_with_component(...)`, which yields entity
objects, and mutates their fields in place.
"""
from ecs_pattern import component, entity, EntityManager
import _common as C


@component
class Position:
    x: float = 0.0
    y: float = 0.0
@component
class Velocity:
    vx: float = 0.0
    vy: float = 0.0
@component
class Acceleration:
    ax: float = 0.0
    ay: float = 0.0


@entity
class Ball(Position, Velocity):
    pass
@entity
class AccBall(Position, Velocity, Acceleration):
    pass


def build(n):
    em = EntityManager()
    pos0, vel0, acc, has_acc = C.make_data(n)
    ents = []
    for i in range(n):
        if has_acc[i]:
            ents.append(AccBall(x=float(pos0[i, 0]), y=float(pos0[i, 1]),
                                vx=float(vel0[i, 0]), vy=float(vel0[i, 1]),
                                ax=float(acc[i, 0]), ay=float(acc[i, 1])))
        else:
            ents.append(Ball(x=float(pos0[i, 0]), y=float(pos0[i, 1]),
                             vx=float(vel0[i, 0]), vy=float(vel0[i, 1])))
    em.add(*ents)
    return em


def step(em):
    dt = C.DT
    for e in em.get_with_component(Velocity, Acceleration):
        e.vx += e.ax * dt
        e.vy += e.ay * dt
    for e in em.get_with_component(Position, Velocity):
        e.x += e.vx * dt
        e.y += e.vy * dt


def collect_positions(em):
    return [(e.x, e.y) for e in em.get_with_component(Position)]


def run(n=C.N_DEFAULT, warmup=C.WARMUP, measure=C.MEASURE):
    return C.run_bench("ecs-pattern", C.lib_version("ecs-pattern"),
                       build, step, collect_positions, n, warmup, measure)


def main():
    n, measure = C.cli_n_measure()
    C.print_result(run(n=n, measure=measure))


if __name__ == "__main__":
    main()
