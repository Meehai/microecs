"""Why does pure-python+numpy microecs OVERTAKE Rust xecs on columnar updates at large N?

Hypothesis (REFUTED by this probe): the crossover is a dtype artifact. xecs' idiomatic step uses a
python-float dt (`view.x += acc.x * dt`); if that upcast the float32 column to float64 and back, the
extra memory traffic would dominate at large N. This times xecs' step with dt = python-float vs
np.float32 and inspects the intermediate dtype. Result: no speedup (numpy 2.x NEP 50 keeps float32),
so the crossover is NOT a dtype artifact -- see boundary.py for the real cause (the Rust<->numpy copy).
"""
import gc
import os
import sys
import time
import numpy as np
import xecs as xx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C


class Position(xx.Component):
    x: xx.Float32
    y: xx.Float32
class Velocity(xx.Component):
    x: xx.Float32
    y: xx.Float32


def _time(fn, state, iters=50):
    for _ in range(3):
        fn(state)
    gc.collect(); gc.disable()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(state)
    dt = (time.perf_counter() - t0) / iters
    gc.enable()
    return dt


def build(n):
    s = C.make_scene(n)
    app = xx.SimulationApp()
    app.add_pool(Position.create_pool(n)); app.add_pool(Velocity.create_pool(n))
    cmd, world = app._commands, app.world
    cmd.spawn([Position, Velocity], n)
    pv, vv = world.get_view(Position), world.get_view(Velocity)
    pv.x.fill(np.ascontiguousarray(s["pos"][:, 0], np.float32))
    pv.y.fill(np.ascontiguousarray(s["pos"][:, 1], np.float32))
    vv.x.fill(np.ascontiguousarray(s["vel"][:, 0], np.float32))
    vv.y.fill(np.ascontiguousarray(s["vel"][:, 1], np.float32))
    return world


def step_f64(world):
    dt = 0.016                       # python float -> float64 intermediate?
    pos, vel = world.get_view(Position), world.get_view(Velocity)
    pos.x += vel.x * dt
    pos.y += vel.y * dt


def step_f32(world):
    dt = np.float32(0.016)           # stay in float32
    pos, vel = world.get_view(Position), world.get_view(Velocity)
    pos.x += vel.x * dt
    pos.y += vel.y * dt


def main():
    w = build(1000)
    prod = w.get_view(Velocity).x * 0.016
    prod32 = w.get_view(Velocity).x * np.float32(0.016)
    print(f"type of (view.x * dt):             {type(prod).__module__}.{type(prod).__name__}")
    print(f"dtype of (Float32 * python-float): {np.asarray(prod).dtype}")
    print(f"dtype of (Float32 * np.float32):   {np.asarray(prod32).dtype}")
    print()
    print(f"{'N':>10} {'xecs f64-dt (ms)':>18} {'xecs f32-dt (ms)':>18} {'speedup':>9}")
    for n in [1000, 10000, 100000, 500000, 1000000]:
        w = build(n)
        t64 = _time(step_f64, w)
        w = build(n)
        t32 = _time(step_f32, w)
        print(f"{n:>10} {t64 * 1e3:>18.4f} {t32 * 1e3:>18.4f} {t64 / t32:>8.2f}x")


if __name__ == "__main__":
    main()
