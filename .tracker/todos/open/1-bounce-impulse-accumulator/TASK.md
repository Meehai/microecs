# Bounce as impulse accumulator (fix wall leak + collision jitter)

**Created**: 2026-06-02
**Priority**: 1

## Why

`WallBounceSystem` and `CollisionBounceSystem` (`main.py`) both flip velocity on
**overlap** instead of on **approach**. Two bugs fall out of that:

1. **Sticky/jitter at walls.** `velocity = where(overlap, -velocity, velocity)`
   re-flips every tick the ball is still overlapping the wall. The sign toggles
   instead of settling → ball jitters / sticks at the edge.

2. **Balls leak out of the scene.** Update order is
   `Motion → WallBounce → CollisionBounce`. For a ball touching a wall *and*
   overlapping a neighbor, WallBounce corrects it inward, then CollisionBounce
   runs last and flips it back outward. Next subtick Motion walks it further out.
   Collision always gets the last word → ball marches through the wall.

Root cause is one idea: **flipping is non-composable.** `-velocity` is a negation;
two negations cancel. A squeezed ball (wall pushing in, neighbor pushing out) can
only keep one correction, so the other is silently dropped.

## Fix — two parts

### Part A — direction guard (decide *whether* to act)

Only reverse a component when the object is actually moving *into* the obstacle.

- **Wall**: flip `vx` only if `(pos - r < 0 & vx < 0) | (pos + r > size & vx > 0)`
  (same for `vy`). Idempotent — once moving inward it won't re-trigger.
- **Collision**: bounce only when the pair is *approaching*:
  `dot(v_i - v_j, normal) > 0`, where `normal = (pos_j - pos_i)/|...|`.
  Balls already separating are left alone.

### Part B — impulse accumulator (decide *how* corrections combine)

Detection systems stop writing velocity. They **read** pos/vel and **add** a
velocity delta into a shared buffer. One system applies the sum at the end of the
subtick. All detectors then read the *same* pre-tick snapshot → order no longer
matters, and impulses **superpose** (the squeeze case cancels correctly instead of
dropping a correction).

New trait:
```
HasImpulse: impulse (2,) float32   # Δvelocity, zeroed each subtick
```

Subtick order:
```
ZeroImpulseSystem        # impulse[:] = 0
WallBounceSystem         # READ pos/vel  -> ADD reflect impulse   (no vel write)
CollisionBounceSystem    # READ pos/vel  -> ADD normal impulse    (no vel write)
ApplyImpulseSystem       # vel += impulse
MotionSystem             # pos += vel*dt
```

Stays fully vectorized SoA: `impulse` is a fixed `(N,2)` array, zero allocation,
no per-tick component churn, no event queue.

## Done when

- A ball driven into a corner while overlapping a neighbor stays inside
  `scene_size` for N subticks.
- Two head-on equal balls separate after contact (no clump, no re-stick).
- A ball resting against a wall does not jitter (velocity sign stable).
- System order is swappable for the two detectors without changing the result.

## Tests (tester writes, under `test/`)

- `test_wall_bounce_keeps_inside` — single ball at each wall + corner, fast vel,
  assert `pos ± r` stays within `[0, size]` over many subticks.
- `test_squeeze_into_corner_no_leak` — ball + neighbor pinned in a corner, assert
  no escape (this is the current failing case).
- `test_collision_separates` — two approaching balls end up moving apart.
- `test_no_jitter_at_rest` — ball touching wall with outward vel already removed
  keeps a stable velocity sign across ticks.
- `test_detector_order_independent` — run with detectors in both orders, assert
  identical post-`ApplyImpulse` velocities.

## Out of scope

- Real elastic physics with mass/restitution. Equal-mass normal exchange is enough
  for now; restitution coefficient is a later knob.
- De-penetration / positional correction (pushing overlapping balls apart). Velocity
  impulse only; a leftover frame of overlap is acceptable.
- Spatial hashing / broad-phase. `_get_collisions` stays O(N²) for now.
- Promoting impulses to a full event bus. Accumulator buffer is the minimal shape;
  revisit only if more producers (forces, drag) appear.

## Related

- `main.py`: `WallBounceSystem`, `CollisionBounceSystem`, `MotionSystem`.
- Mirrors robosim's "collisions as events" direction
  (robosim task 125 / plan 9) but kept SoA-batched and queue-free.
