"""microecs benchmark: the vectorized numpy SoA path (this project's whole point).

A frame is two batched numpy ops over a QueryResult that spans both archetypes:
    vel += acc*dt   (the HasAcc pool)
    pos += vel*dt   (both pools, stitched by the Field)
No per-entity python in the hot loop -- that is the design.
"""
from dataclasses import field
import numpy as np
from microecs import World, Component
import _common as C


class HasPos(Component):
    position: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})
class HasVel(Component):
    velocity: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})
class HasAcc(Component):
    acceleration: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})


def build(n):
    pos0, vel0, acc, has_acc = C.make_data(n)
    w = World(components=[HasPos, HasVel, HasAcc])
    for i in range(n):
        if has_acc[i]:
            w.add_entity(components=[HasPos, HasVel, HasAcc],
                         position=pos0[i].astype("float32"),
                         velocity=vel0[i].astype("float32"),
                         acceleration=acc[i].astype("float32"))
        else:
            w.add_entity(components=[HasPos, HasVel],
                         position=pos0[i].astype("float32"),
                         velocity=vel0[i].astype("float32"))
    w.update()   # commit the buffered spawns
    return w


def step(w):
    # query per frame is the idiomatic microecs system pattern (see README main loop)
    qv = w.query(HasVel, HasAcc)
    qv.velocity[:] = qv.velocity + qv.acceleration * C.DT32
    qp = w.query(HasPos, HasVel)
    qp.position[:] = qp.position + qp.velocity * C.DT32


def collect_positions(w):
    return w.query(HasPos, HasVel).position.numpy()


def run(n=C.N_DEFAULT, warmup=C.WARMUP, measure=C.MEASURE):
    return C.run_bench("microecs", C.lib_version("microecs"),
                       build, step, collect_positions, n, warmup, measure)


def main():
    n, measure = C.cli_n_measure()
    C.print_result(run(n=n, measure=measure))


if __name__ == "__main__":
    main()
