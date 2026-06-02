# Deferred command buffer (safe structural change during iteration)

**Created**: 2026-06-02
**Priority**: 3

## Why

A system iterates pools from `query_and(...)` and operates on them. The moment a
system spawns, despawns, or migrates an entity **during** that loop, the iteration
it is standing on gets mutated underneath it. Three concrete corruptions, all from
the current code:

1. **Swap-remove shifts a row you haven't visited yet.** `Pool.remove_entity`
   (`pool.py:47`) copies the last row into the freed slot and shrinks `size`. If
   you despawn entity at index `i` while looping `0..N`, the entity that *was* last
   is now at `i` — already-passed → it gets **skipped** this tick.

2. **`add_component` moves the entity to a different pool** (`world.py:43`). If that
   richer pool is later in your `query_and` list, the same entity gets processed a
   **second time** this tick.

3. **Reclamation deletes the pool mid-tick** (`world.py:85`). When the last entity
   leaves a pool, `del self.pools[key]`. A system holding that pool (or iterating
   `self.pools`) now walks a dropped/`del`'d archetype.

Root cause is one idea: **structural changes and iteration don't mix.** Reads over
SoA arrays assume the arrays don't move while you read them. Every spawn/despawn/
migrate moves them.

The standard fix: systems don't mutate the world directly. They **queue** a command
(`spawn` / `despawn` / `add_component` / `remove_component`) into a buffer, keep
iterating the *stable* snapshot, and the world **applies the whole buffer at the
tick boundary** — after all systems have read, before the next tick.

## Flush point — DECIDED: `world.update()`

The buffer must be applied at a moment when no system is mid-iteration. **Decision:
an explicit `world.update()` method** that applies every deferred call, called by the
main loop at the safe points it chooses.

Why explicit (not a `world.run(systems)` runner): **there is no single "run all
systems" moment.** Systems run in groups at different phases — before render, after
render, per physics subtick. So the developer owns where flushing happens. The
boundary *between* phase groups is a safe point (nobody iterating), and that's where
`world.update()` goes:

```python
for s in pre_render_systems: s.on_tick(world)
world.update()                 # apply deferred buffer
render(world)
for s in post_render_systems: s.on_tick(world)
world.update()
# subticks fit the same shape:
for _ in clock.drain():
    for s in physics_systems: s.on_tick(world)
    world.update()
```

Accepted trade-off: `update()` is callable-by-hand, so it can be forgotten. A
forgotten `update()` means structural changes silently never apply (entities never
spawn/despawn) — loud enough to notice fast. Keeping it explicit beats hiding flush
inside a loop-owning scheduler the phased loop can't use.

Sub-decision (orthogonal, developer's call): call `update()` **per phase** (a spawn
in pre-render is real before render) vs **once at frame end** (all structural changes
batched; spawns visible next frame). Per-phase makes each phase read a settled world;
frame-end batches everything. Either works — `update()` is just called at the chosen
boundaries.

## API sketch (developer's call; not binding)

```python
# inside a system's on_tick — queue, don't mutate
world.spawn(components=(HasPosition,), position=...)   # deferred add_entity
world.despawn(eid)                                     # deferred remove_entity
world.defer_add_component(eid, HasVelocity, velocity=...)
world.defer_remove_component(eid, HasColor)

world.update()   # applies the whole queue, in order
```

Immediate (`add_entity` / `remove_entity` / `add_component` / `remove_component`)
stays — it's correct **outside** a system loop (setup, tests). Deferred is the
in-loop path. They share the same apply code; deferred just records args and replays
them on `update()`.

## Done when

- A system that despawns the entity it is currently visiting does **not** skip or
  double-process any *other* entity that tick.
- A system that adds a component (migrating an entity to a pool later in its own
  query) does not process that entity twice in the same tick.
- A system that empties a pool does not crash the loop via reclamation; the pool is
  gone only *after* `update()`.
- Deferred commands take effect only on `world.update()`, never mid-iteration.
- Commands apply in the order they were queued (deterministic).
- Immediate (non-deferred) API still works for setup/teardown outside a tick.

## Tests (tester writes, under `test/`)

- `test_deferred_despawn_takes_effect_only_on_update` — queue despawn, assert entity
  still present before `update()`, gone after.
- `test_despawn_during_iteration_visits_each_other_entity_once` — loop a pool, queue
  despawn of the current entity, `update()`, assert every survivor was visited
  exactly once (no skip from swap-remove).
- `test_deferred_add_component_no_double_process` — entity migrates to a later pool
  only on `update()`; assert it's touched once during the iterating tick.
- `test_deferred_spawn_absent_until_update` — spawn queued mid-tick is absent from
  the query until `update()`, present after.
- `test_commands_apply_in_queue_order` — interleave spawn/despawn/add/remove,
  assert final world state matches in-order application on `update()`.
- `test_emptying_pool_via_deferred_despawn_no_crash` — despawn the last entity of a
  pool from inside a loop; `update()`; pool reclaimed, no error.
- `test_update_on_empty_buffer_is_noop` — `world.update()` with nothing queued does
  nothing and doesn't error (safe to call every phase).
- `test_immediate_api_unchanged_outside_tick` — direct add/remove still works (the
  existing task-02 tests must keep passing).

## Out of scope

- A system registry / ordered scheduler. The flush point is decided
  (`world.update()`, called by the main loop); a scheduler is a separate later
  concern and not required by this task.
- A general event bus / message passing between systems. Command buffer is
  structural-change-only (spawn/despawn/migrate), not arbitrary events.
- Command coalescing / dedup (e.g. despawn-then-add-component on the same id in one
  tick). Apply in order, let last-write-win fall out naturally; optimize only if a
  real case needs it.
- Parallel system execution / threading. Single-threaded flush for now.
- Generational ids to catch despawn-then-reuse within a flush. Still plain monotonic
  int (see task 02 out-of-scope).

## Related

- `ecs/world.py`: `add_entity`, `remove_entity`, `add_component`, `remove_component`,
  `_pop_from_pool` (swap-remove + reclamation), `query_and`.
- `ecs/pool.py`: `remove_entity` (swap-remove, `pool.py:47`).
- Builds directly on [task 02](../../done/2-id-based-world-and-trait-migration/TASK.md)
  (stable ids are what make a *deferred* `despawn(eid)` safe — the id must still
  resolve at flush time even after siblings moved).
- Unblocks [task 1](../1-bounce-impulse-accumulator/TASK.md): spawning/despawning
  balls from inside the physics systems without corrupting the SoA iteration.
- Flush is owned by the main loop via `world.update()` — no scheduler dependency.
