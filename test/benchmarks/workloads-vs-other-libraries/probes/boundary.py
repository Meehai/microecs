"""Locate the real source of xecs' large-N columnar slowdown vs pure-numpy microecs.

Refuted: float64 upcast (numpy 2.x NEP 50 keeps float32; float32 dt gave 0 speedup -- xecs_dtype.py).
Confirmed here: the Rust<->numpy boundary. `.numpy()` is a COPY. `view.x * dt` returns a plain numpy
array -- xecs does its arithmetic in numpy, not Rust; the Rust core is storage. So every xecs
vectorized op pays copy-out + copy-back that pure-numpy microecs (operating in-place on the very
arrays it stores) never pays.

Compares, per position-integrate step at each N:
  xecs      : pos.x += vel.x*dt ; pos.y += vel.y*dt          (Rust buffers via numpy)
  numpy-2col: ax += vx*dt ; ay += vy*dt                       (raw numpy, in-place, x/y split)
  numpy-fused: p += v*dt                                      (raw numpy, in-place, (N,2))
and times a bare `vel.x.numpy()` to expose the copy cost directly.
"""
import gc
import time
import numpy as np
import xecs as xx


class Position(xx.Component):
    x: xx.Float32
    y: xx.Float32
class Velocity(xx.Component):
    x: xx.Float32
    y: xx.Float32


def _time(fn, iters=50):
    for _ in range(3):
        fn()
    gc.collect(); gc.disable()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    dt = (time.perf_counter() - t0) / iters
    gc.enable()
    return dt


def build_xecs(n):
    app = xx.SimulationApp()
    app.add_pool(Position.create_pool(n)); app.add_pool(Velocity.create_pool(n))
    cmd, world = app._commands, app.world
    cmd.spawn([Position, Velocity], n)
    v = world.get_view(Velocity)
    v.x.fill(np.ones(n, np.float32)); v.y.fill(np.ones(n, np.float32))
    return world


def main():
    dt = np.float32(0.016)
    print(f"{'N':>9} {'xecs(ms)':>10} {'np-2col(ms)':>12} {'np-fused(ms)':>13} "
          f"{'.numpy() copy(ms)':>18} {'xecs/np-fused':>14}")
    for n in [10_000, 100_000, 500_000, 1_000_000]:
        world = build_xecs(n)

        def step_xecs():
            pos, vel = world.get_view(Position), world.get_view(Velocity)
            pos.x += vel.x * dt
            pos.y += vel.y * dt

        px = np.zeros(n, np.float32); py = np.zeros(n, np.float32)
        vx = np.ones(n, np.float32); vy = np.ones(n, np.float32)

        def step_np2():
            px[:] += vx * dt
            py[:] += vy * dt

        p2 = np.zeros((n, 2), np.float32); v2 = np.ones((n, 2), np.float32)

        def step_fused():
            p2[:] += v2 * dt

        velview = world.get_view(Velocity).x
        t_copy = _time(lambda: velview.numpy())

        tx, t2, tf = _time(step_xecs), _time(step_np2), _time(step_fused)
        print(f"{n:>9} {tx*1e3:>10.4f} {t2*1e3:>12.4f} {tf*1e3:>13.4f} "
              f"{t_copy*1e3:>18.4f} {tx/tf:>13.2f}x")


if __name__ == "__main__":
    main()
