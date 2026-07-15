# Moving & Colliding Balls

[`examples/02-moving-colliding-balls.py`](examples/02-moving-colliding-balls.py) picks up where [Hello World](example-1-hello-world.md) left off: the balls now **move**, **bounce off the walls**, and turn red on contact — all driven by a fixed-timestep physics loop. Run it:

```bash
python examples/02-moving-colliding-balls.py --n_objects 10
```

Two new components carry the extra state (`HasColor` from Hello World is dropped — the renderer just picks red/black from `is_colliding`):

```python
class HasMotion2D(Component):
    velocity: np.ndarray = field(metadata={"shape": (2, ), "dtype": "float32", "default": None})
class HasCollision(Component):
    is_colliding: np.ndarray = field(metadata={"shape": (1, ), "dtype": "bool", "default": None})
```

## Everything is a vectorized system

Three update systems, each one batched over the whole query — no per-entity loop anywhere:

```python
class MotionSystem:                # integrate: pos += vel*dt, all entities at once
    def __call__(self, world):
        qr = world.query(HasMotion2D, HasPosition2D)
        qr.position[:] = qr.position + qr.velocity * DT

class WallBounceSystem:            # flip velocity where a ball crossed a wall (data-parallel branch)
    def __call__(self, world):
        qr = world.query(HasPosition2D, HasMotion2D, HasRadius)
        (w, h), r = self.scene_size, qr.radius[:, 0]
        mask = np.zeros((len(qr), 2), bool)
        mask[:, 0] = (qr.position[:, 0] - r < 0) | (qr.position[:, 0] + r > w)
        mask[:, 1] = (qr.position[:, 1] - r < 0) | (qr.position[:, 1] + r > h)
        qr.velocity[:] = np.where(mask, -qr.velocity, qr.velocity)

class CollisionDetectionSystem:    # pairwise overlap, one broadcast -> (N, N) distances
    def __call__(self, world):
        qr = world.query(HasPosition2D, HasMotion2D, HasRadius, HasCollision)
        qr.is_colliding[:] = self._get_collisions(qr.position.numpy(), qr.radius.numpy())
```

`WallBounceSystem` is the textbook case for pushing an `if` into `np.where`; `CollisionDetectionSystem` does the whole O(N²) overlap test as a single broadcast (`(N,1,2) - (1,N,2) → (N,N)` distances). `RenderSystem` is the same `zip` loop as [Hello World](example-1-hello-world.md), just coloured red when `is_colliding`. See [Systems](systems.md) for why the batched form wins.

## Fixed-timestep loop (the new idea)

Physics must step at a constant `dt` regardless of frame rate, or fast and slow machines simulate differently. This is the accumulator pattern from the canonical [*Fix Your Timestep!*](https://gafferongames.com/post/fix_your_timestep/): a small `Clock` decouples the two — it banks real elapsed time and hands out fixed-`dt` subticks.

```python
class Clock:
    def tick(self):                       # bank the real time since last frame
        now = rl.GetTime(); self.accumulator += now - self.prev_time; self.prev_time = now
    def drain(self):                      # yield one fixed-dt step per banked dt (capped at max_ticks)
        n = 0
        while self.accumulator >= self.dt and n < self.max_ticks:
            yield; self.accumulator -= self.dt; n += 1
```

The main loop flushes structure once per render tick, then runs the physics systems once **per subtick**:

```python
clock = Clock(dt=DT, max_ticks=MAX_SUBTICKS_PER_RENDER_TICK)
while not rl.WindowShouldClose():
    world.update()                        # once per render tick: commit spawns/despawns
    clock.wait_and_tick()
    if rl.IsMouseButtonPressed(rl.MOUSE_BUTTON_LEFT):
        _spawn_circle(world, ...)         # lazy -> appears at next world.update()
    for _ in clock.drain():               # 0..max_ticks fixed-dt steps this frame
        for system in update_systems:     # MotionSystem, WallBounceSystem, CollisionDetectionSystem
            system(world=world)
    # ... BeginDrawing / RenderSystem / EndDrawing ...
```

`world.update()` runs **once per render tick** (structure is committed at render granularity), while the vectorized field writes inside the systems are eager and run **per subtick** — the [eager-vs-deferred split](primitives.md) in action.

## See also

- [Systems & Per-Entity Iteration](systems.md) — the vectorized / `np.where` patterns these systems use.
- [Serialization (save & load)](example-3-serialization.md) — the next example.
