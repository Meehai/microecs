# 2. Moving & Colliding Balls

[`examples/02-motion-and-collision.py`](examples/02-motion-and-collision.py) picks up where [Hello World](example-1-hello-world.md) left off: the balls now **move**, **bounce off the walls**, and turn red on contact — all driven by a fixed-timestep physics loop. Run it:

```bash
python examples/02-motion-and-collision.py --n_objects 10
```

Two new components carry the extra state (`HasColor` from Hello World is dropped — the renderer just picks red/black from `is_colliding`):

```python
class HasMotion2D(Component):
    velocity: np.ndarray = f32((2, ))
class HasCollision(Component):
    is_colliding: np.ndarray = boolf((1, ))
```

## Everything is a vectorized system

Three update systems, each one batched over the whole query — no per-entity loop anywhere:

```python
class MotionSystem:                # integrate: pos += vel*dt, all entities at once
    def __call__(self, world):
        qr = world.query(HasMotion2D, HasPosition2D)
        qr.position = qr.position + qr.velocity * DT

class WallBounceSystem:            # flip velocity where a ball crossed a wall (data-parallel branch)
    def __call__(self, world):
        qr = world.query(HasPosition2D, HasMotion2D, HasRadius)
        (w, h), r = self.scene_size, qr.radius[:, 0]
        mask = np.zeros((len(qr), 2), bool)
        mask[:, 0] = (qr.position[:, 0] - r < 0) | (qr.position[:, 0] + r > w)
        mask[:, 1] = (qr.position[:, 1] - r < 0) | (qr.position[:, 1] + r > h)
        qr.velocity = np.where(mask, -qr.velocity, qr.velocity)

class CollisionDetectionSystem:    # pairwise overlap, one broadcast -> (N, N) distances
    def __call__(self, world):
        qr = world.query(HasPosition2D, HasMotion2D, HasRadius, HasCollision)
        qr.is_colliding = self._get_collisions(qr.position.numpy(), qr.radius.numpy())
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

## Full code
```python
#!/usr/bin/env python3
from typing import Callable
from argparse import ArgumentParser, Namespace
import random
import numpy as np
import raylib as rl
from loggez import loggez_logger as logger

from microecs import World, Component, f32, boolf

Point2D = tuple[float, float]
DT = 1 / 100
MAX_SUBTICKS_PER_RENDER_TICK = 3

# utils

class Clock:
    """clock used for physics with fixed DT in main loops"""

    def __init__(self, dt: float, max_ticks: int):
        self.dt = dt
        self.max_ticks = max_ticks
        self.prev_time = rl.GetTime()
        self.accumulator = 0

    def tick(self):
        """tick once by adding the delta between prev frame and now"""
        now = rl.GetTime()
        frame_time = now - self.prev_time
        self.prev_time = now
        self.accumulator += frame_time

    def drain(self):
        """drain the accumulator. in main loop: for _ in clock.drain(): ..."""
        n_ticks = 0
        while self.accumulator >= self.dt and n_ticks < self.max_ticks:
            yield
            self.accumulator -= self.dt
            n_ticks += 1
        self.accumulator = min(self.accumulator, self.dt) # Drop residual debt instead of it piling up across frames

    def wait(self):
        """waits the leftover time in case the previous tick ran too fast to maintain consistent FPS"""
        rl.WaitTime(max(self.dt - (rl.GetTime() - self.prev_time), 0))

    def wait_and_tick(self):
        """calls wait() then tick(). Put this at the beginning of the main loop :)"""
        self.wait()
        self.tick()

# components

class HasRadius(Component):
    radius: np.ndarray = f32((1, ))

class HasPosition2D(Component):
    position: np.ndarray = f32((2, ))

class HasMotion2D(Component):
    velocity: np.ndarray = f32((2, ))

class HasCollision(Component):
    is_colliding: np.ndarray = boolf((1, ))

# systems

class RenderSystem:
    def __call__(self, world: World):
        qr = world.query(HasRadius, HasPosition2D, HasCollision)
        for position, radius, is_colliding in zip(qr.position, qr.radius, qr.is_colliding):
            color = rl.RED if is_colliding else rl.BLACK
            rl.DrawCircle(int(position[0].item()), int(position[1].item()), int(radius.item()), color)

class MotionSystem:
    def __call__(self, world: World):
        qr = world.query(HasMotion2D, HasPosition2D)
        qr.position = qr.position + qr.velocity * DT # (N, 2)

class WallBounceSystem:
    def __init__(self, scene_size: tuple[int, int]):
        self.scene_size = scene_size

    def __call__(self, world: World):
        qr = world.query(HasPosition2D, HasMotion2D, HasRadius)
        mask_velocity = np.zeros((len(qr.position), 2), bool)
        mask_velocity[:, 0] = np.logical_or(qr.position[:, 0] - qr.radius[:, 0] < 0,
                                            qr.position[:, 0] + qr.radius[:, 0] > self.scene_size[0])
        mask_velocity[:, 1] = np.logical_or(qr.position[:, 1] - qr.radius[:, 0] < 0,
                                            qr.position[:, 1] + qr.radius[:, 0] > self.scene_size[1])
        qr.velocity = np.where(mask_velocity, -qr.velocity, qr.velocity)

class CollisionDetectionSystem:
    def __call__(self, world: World):
        qr = world.query(HasPosition2D, HasMotion2D, HasRadius, HasCollision)
        collisions = self._get_collisions(qr.position.numpy(), qr.radius.numpy())
        qr.is_colliding = np.where(collisions, True, False)

    def _get_collisions(self, positions: np.ndarray, radii: np.ndarray) -> np.ndarray:
        dists = np.sqrt(((positions[:, None] - positions[None])**2).sum(-1))  # (N, 1, 2) - (1, N, 2) -> ... -> (N, N)
        radii_sum =  (radii[None] + radii[:, None])[..., 0] # (N, N)
        collisions_nn = (dists < radii_sum) - np.eye(len(positions)) # (N, N)
        res = (collisions_nn.sum(axis=1) > 0)[..., None] # (N, 1)
        return res

def _spawn_circle(world: World, position: Point2D, radius: float, velocity: Point2D):
    world.add_entity(components=(HasRadius, HasPosition2D, HasMotion2D, HasCollision),
                     position=np.array(position, "float32"), velocity=np.array(velocity, "float32"),
                     radius=np.array([radius], "float32"), is_colliding=np.zeros((1, ), "bool"))

def main(args: Namespace):
    rl.InitWindow(800, 800, b"Entity Component Style + SoA (batched)")
    scene_size = (600, 600)
    mouse_radius = 10

    render_systems: list[Callable] = [RenderSystem()]
    update_systems: list[Callable] = [MotionSystem(), WallBounceSystem(scene_size), CollisionDetectionSystem()]

    world = World(components=[HasRadius, HasPosition2D, HasMotion2D, HasCollision])
    for _ in range(args.n_objects):
        radius = random.randint(5, 15)
        position = random.randint(radius, scene_size[0] - radius), random.randint(radius, scene_size[1] - radius)
        velocity = (200 * random.random() * 2 - 1, 200 * random.random() * 2 - 1)
        _spawn_circle(world, position, radius, velocity)

    clock = Clock(dt=DT, max_ticks=MAX_SUBTICKS_PER_RENDER_TICK)
    while not rl.WindowShouldClose():
        world.update()
        clock.wait_and_tick()

        mouse_pos = rl.GetMousePosition()

        if rl.IsMouseButtonPressed(rl.MOUSE_BUTTON_LEFT):
                if (mouse_pos.x - mouse_radius > 0 and mouse_pos.x + mouse_radius < scene_size[0] and
                    mouse_pos.y - mouse_radius > 0 and mouse_pos.y + mouse_radius < scene_size[1]):
                    velocity = (200 * random.random() * 2 - 1, 200 * random.random() * 2 - 1)
                    _spawn_circle(world, (mouse_pos.x, mouse_pos.y), mouse_radius, velocity)

        for _ in clock.drain():
            logger.log_every_s(f"Applying {clock.accumulator // clock.dt} update ticks per render tick", "DEBUG", True)
            _ = [system(world=world) for system in update_systems]

        rl.BeginDrawing()
        rl.ClearBackground(rl.RAYWHITE)
        rl.DrawFPS(rl.GetScreenWidth() - 100, 0)

        rl.DrawRectangleLinesEx((0, 0, *scene_size), 2, rl.BLACK)
        _ = [system(world=world) for system in render_systems]

        rl.EndDrawing()

        logger.log_every_s(f"FPS: {rl.GetFPS()}", "DEBUG")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_objects", type=int, default=10)
    main(parser.parse_args())
```

## Everything is a vectorized system

Three update systems, each one batched over the whole query — no per-entity loop anywhere:

```python
class MotionSystem:                # integrate: pos += vel*dt, all entities at once
    def __call__(self, world):
        qr = world.query(HasMotion2D, HasPosition2D)
        qr.position = qr.position + qr.velocity * DT

class WallBounceSystem:            # flip velocity where a ball crossed a wall (data-parallel branch)
    def __call__(self, world):
        qr = world.query(HasPosition2D, HasMotion2D, HasRadius)
        (w, h), r = self.scene_size, qr.radius[:, 0]
        mask = np.zeros((len(qr), 2), bool)
        mask[:, 0] = (qr.position[:, 0] - r < 0) | (qr.position[:, 0] + r > w)
        mask[:, 1] = (qr.position[:, 1] - r < 0) | (qr.position[:, 1] + r > h)
        qr.velocity = np.where(mask, -qr.velocity, qr.velocity)

class CollisionDetectionSystem:    # pairwise overlap, one broadcast -> (N, N) distances
    def __call__(self, world):
        qr = world.query(HasPosition2D, HasMotion2D, HasRadius, HasCollision)
        qr.is_colliding = self._get_collisions(qr.position.numpy(), qr.radius.numpy())
```

`WallBounceSystem` is the textbook case for pushing an `if` into `np.where`; `CollisionDetectionSystem` does the whole O(N²) overlap test as a single broadcast (`(N,1,2) - (1,N,2) → (N,N)` distances). `RenderSystem` is the same `zip` loop as [Hello World](example-1-hello-world.md), just coloured red when `is_colliding`. See [Systems](<docs/source/systems.md>) for why the batched form wins.

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

`world.update()` runs **once per render tick** (structure is committed at render granularity), while the vectorized field writes inside the systems are eager and run **per subtick** — the [eager-vs-deferred split](<docs/source/primitives.md>) in action.

## Full code

Everything above, in one file — including the `Clock` in full. Copy it, run it, click to add a ball.

```python
#!/usr/bin/env python3
from dataclasses import field
from typing import Callable
from argparse import ArgumentParser, Namespace
import random
import numpy as np
import raylib as rl
from loggez import loggez_logger as logger

from microecs import World, Component

Point2D = tuple[float, float]
DT = 1 / 100
MAX_SUBTICKS_PER_RENDER_TICK = 3

# utils

class Clock:
    """clock used for physics with fixed DT in main loops"""

    def __init__(self, dt: float, max_ticks: int):
        self.dt = dt
        self.max_ticks = max_ticks
        self.prev_time = rl.GetTime()
        self.accumulator = 0

    def tick(self):
        """tick once by adding the delta between prev frame and now"""
        now = rl.GetTime()
        frame_time = now - self.prev_time
        self.prev_time = now
        self.accumulator += frame_time

    def drain(self):
        """drain the accumulator. in main loop: for _ in clock.drain(): ..."""
        n_ticks = 0
        while self.accumulator >= self.dt and n_ticks < self.max_ticks:
            yield
            self.accumulator -= self.dt
            n_ticks += 1
        self.accumulator = min(self.accumulator, self.dt) # Drop residual debt instead of it piling up across frames

    def wait(self):
        """waits the leftover time in case the previous tick ran too fast to maintain consistent FPS"""
        rl.WaitTime(max(self.dt - (rl.GetTime() - self.prev_time), 0))

    def wait_and_tick(self):
        """calls wait() then tick(). Put this at the beginning of the main loop :)"""
        self.wait()
        self.tick()

# components

class HasRadius(Component):
    radius: np.ndarray = field(metadata={"shape": (1, ), "dtype": "float32", "default": None})

class HasPosition2D(Component):
    position: np.ndarray = field(metadata={"shape": (2, ), "dtype": "float32", "default": None})

class HasMotion2D(Component):
    velocity: np.ndarray = field(metadata={"shape": (2, ), "dtype": "float32", "default": None})

class HasCollision(Component):
    is_colliding: np.ndarray = field(metadata={"shape": (1, ), "dtype": "bool", "default": None})

# systems

class RenderSystem:
    def __call__(self, world: World):
        qr = world.query(HasRadius, HasPosition2D, HasCollision)
        for position, radius, is_colliding in zip(qr.position, qr.radius, qr.is_colliding):
            color = rl.RED if is_colliding else rl.BLACK
            rl.DrawCircle(int(position[0].item()), int(position[1].item()), int(radius.item()), color)

class MotionSystem:
    def __call__(self, world: World):
        qr = world.query(HasMotion2D, HasPosition2D)
        qr.position = qr.position + qr.velocity * DT # (N, 2)

class WallBounceSystem:
    def __init__(self, scene_size: tuple[int, int]):
        self.scene_size = scene_size

    def __call__(self, world: World):
        qr = world.query(HasPosition2D, HasMotion2D, HasRadius)
        mask_velocity = np.zeros((len(qr.position), 2), bool)
        mask_velocity[:, 0] = np.logical_or(qr.position[:, 0] - qr.radius[:, 0] < 0,
                                            qr.position[:, 0] + qr.radius[:, 0] > self.scene_size[0])
        mask_velocity[:, 1] = np.logical_or(qr.position[:, 1] - qr.radius[:, 0] < 0,
                                            qr.position[:, 1] + qr.radius[:, 0] > self.scene_size[1])
        qr.velocity = np.where(mask_velocity, -qr.velocity, qr.velocity)

class CollisionDetectionSystem:
    def __call__(self, world: World):
        qr = world.query(HasPosition2D, HasMotion2D, HasRadius, HasCollision)
        collisions = self._get_collisions(qr.position.numpy(), qr.radius.numpy())
        qr.is_colliding = np.where(collisions, True, False)

    def _get_collisions(self, positions: np.ndarray, radii: np.ndarray) -> np.ndarray:
        dists = np.sqrt(((positions[:, None] - positions[None])**2).sum(-1))  # (N, 1, 2) - (1, N, 2) -> ... -> (N, N)
        radii_sum =  (radii[None] + radii[:, None])[..., 0] # (N, N)
        collisions_nn = (dists < radii_sum) - np.eye(len(positions)) # (N, N)
        res = (collisions_nn.sum(axis=1) > 0)[..., None] # (N, 1)
        return res

def _spawn_circle(world: World, position: Point2D, radius: float, velocity: Point2D):
    world.add_entity(components=(HasRadius, HasPosition2D, HasMotion2D, HasCollision),
                     position=np.array(position, "float32"), velocity=np.array(velocity, "float32"),
                     radius=np.array([radius], "float32"), is_colliding=np.zeros((1, ), "bool"))

def main(args: Namespace):
    rl.InitWindow(800, 800, b"Entity Component Style + SoA (batched)")
    scene_size = (600, 600)
    mouse_radius = 10

    render_systems: list[Callable] = [RenderSystem()]
    update_systems: list[Callable] = [MotionSystem(), WallBounceSystem(scene_size), CollisionDetectionSystem()]

    world = World(components=[HasRadius, HasPosition2D, HasMotion2D, HasCollision])
    for _ in range(args.n_objects):
        radius = random.randint(5, 15)
        position = random.randint(radius, scene_size[0] - radius), random.randint(radius, scene_size[1] - radius)
        velocity = (200 * random.random() * 2 - 1, 200 * random.random() * 2 - 1)
        _spawn_circle(world, position, radius, velocity)

    clock = Clock(dt=DT, max_ticks=MAX_SUBTICKS_PER_RENDER_TICK)
    while not rl.WindowShouldClose():
        world.update()
        clock.wait_and_tick()

        mouse_pos = rl.GetMousePosition()

        if rl.IsMouseButtonPressed(rl.MOUSE_BUTTON_LEFT):
                if (mouse_pos.x - mouse_radius > 0 and mouse_pos.x + mouse_radius < scene_size[0] and
                    mouse_pos.y - mouse_radius > 0 and mouse_pos.y + mouse_radius < scene_size[1]):
                    velocity = (200 * random.random() * 2 - 1, 200 * random.random() * 2 - 1)
                    _spawn_circle(world, (mouse_pos.x, mouse_pos.y), mouse_radius, velocity)

        for _ in clock.drain():
            logger.log_every_s(f"Applying {clock.accumulator // clock.dt} update ticks per render tick", "DEBUG", True)
            _ = [system(world=world) for system in update_systems]

        rl.BeginDrawing()
        rl.ClearBackground(rl.RAYWHITE)
        rl.DrawFPS(rl.GetScreenWidth() - 100, 0)

        rl.DrawRectangleLinesEx((0, 0, *scene_size), 2, rl.BLACK)
        _ = [system(world=world) for system in render_systems]

        rl.EndDrawing()

        logger.log_every_s(f"FPS: {rl.GetFPS()}", "DEBUG")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_objects", type=int, default=10)
    main(parser.parse_args())
```

## See also

- [Systems & Per-Entity Iteration](<docs/source/systems.md>) — the vectorized / `np.where` patterns these systems use.
- [Serialization (save & load)](example-3-serialization.md) — the next example.
