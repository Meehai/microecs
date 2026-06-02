# Deferred structural changes — one mode, committed by `world.update()`

**Created**: 2026-06-02
**Priority**: 3

## Why

A system iterates pools from `query_and(...)` and operates on them. The moment a
system spawns, despawns, or migrates an entity **during** that loop, the iteration it
is standing on gets mutated underneath it. Three concrete corruptions, all from the
current code:

1. **Swap-remove shifts a row you haven't visited yet.** `Pool.remove_entity`
   (`pool.py:47`) copies the last row into the freed slot and shrinks `size`. Despawn
   the entity at index `i` while looping `0..N` and the row that *was* last is now at
   `i` — already passed → **skipped** this tick.

2. **`add_component` moves the entity to a different pool** (`world.py:46`). If that
   richer pool is later in the same `query_and` list, the entity is processed a
   **second time** this tick.

3. **Reclamation deletes the pool mid-tick** (`world.py:89`). When the last entity
   leaves a pool, `del self.pools[key]`. A system iterating that registry walks a
   `del`'d archetype.

Root cause is one idea: **structural changes and iteration don't mix.** Reads over
SoA arrays assume the arrays don't move while you read them. Every spawn/despawn/
migrate moves them.

## Decision — ONE mode: every structural change is deferred

No immediate-vs-deferred split, no parallel `defer_*` family. The **existing**
methods become deferred: they record a command and return; nothing happens to the
pools until `world.update()` commits the queue at the tick boundary.

```python
eid = world.add_entity(components=(HasPosition,), position=...)  # queued; id minted & returned NOW
world.add_component(eid, HasVelocity, velocity=...)              # queued
world.remove_component(eid, HasColor)                           # queued
world.remove_entity(eid)                                        # queued
...
world.update()   # commits the whole queue, in enqueue order
```

Two properties make the one mode coherent:

- **Id is minted eagerly, row materializes lazily.** `add_entity` returns a real id
  immediately (just `_last_id += 1`), so the caller has a usable handle and can queue
  follow-up commands against it in the same tick. The entity's pool row only appears
  on `update()`. Commands referencing it (add_component, remove_entity) sit later in
  the same queue and apply after the spawn — enqueue order = causal order.
- **Only STRUCTURAL ops are deferred.** Spawn / despawn / add-component /
  remove-component change archetypes and move arrays → deferred. In-place field
  writes (`pool.position[:] += v`) do **not** move arrays or change archetypes →
  they stay direct and immediate. "One mode" is about structural change, not data.

### Commit point: `world.update()`, called by the main loop

The stub already exists (`world.py:30`). There is no single "run all systems"
moment — systems run in groups across the loop — so the developer owns where commit
happens. The boundary *between* phase groups is safe (nobody iterating):

```python
for s in pre_render_systems: s.on_tick(world)
world.update()                 # commit
render(world)
for s in post_render_systems: s.on_tick(world)
world.update()
for _ in clock.drain():        # subticks fit the same shape
    for s in physics_systems: s.on_tick(world)
    world.update()
```

Accepted trade-off: `update()` is callable by hand, so it can be forgotten — a
forgotten commit means structural changes silently never apply, loud enough to notice
fast. Sub-decision (orthogonal): commit **per phase** vs **once at frame end**. Either
works; per-phase makes each phase read a settled world.

## Consequence — this is a breaking semantic change ⚠️

Today `add_entity` takes effect immediately; after this, it doesn't. **Every call
site that adds/removes entities or components and then expects to see the result must
call `world.update()` first.** That includes:

- **The whole existing unit suite** (`test/unit/test_world.py`, 46 tests). They do
  `add_entity(...)` then immediately assert on `world.pools` / `_eid_to_pool_ix`.
  Each structural op needs a following `update()`, or the assertions must move after a
  single `update()`. This is a deliberate, suite-wide migration — the tester (me)
  does it in lockstep with the implementation.
- **`examples/01-hello-world.py`** — the setup loop that seeds entities, then every
  subtick after running systems.

This cost is the price of one mode. It is expected, not a regression.

## Done when

- `add_entity` returns an id immediately, but the entity is invisible to `query_and`
  / `_eid_to_pool_ix` until the next `world.update()`; visible after.
- `remove_entity` / `add_component` / `remove_component` likewise take effect only on
  `update()`.
- A system that despawns or migrates an entity mid-iteration does **not** skip or
  double-process any entity that tick (the corruptions above are gone).
- A system that empties a pool does not crash the loop; reclamation happens during
  `update()`, not mid-iteration.
- Commands commit in enqueue order (deterministic); a spawn and a follow-up
  add_component on its id in the same tick both land correctly.
- `world.update()` on an empty queue is a safe no-op (already true of the stub).
- The full suite (migrated to call `update()`) and the example still pass.

## Tests (tester writes, under `test/`)

Unit (`test/unit/test_world.py`), matching the one-mode API:

- `test_add_entity_returns_id_before_update` — id minted eagerly, distinct/monotonic,
  even though the row isn't materialized yet.
- `test_add_entity_is_invisible_until_update` — queued spawn absent from `query_and`,
  present after `update()`.
- `test_remove_entity_is_deferred_until_update` — entity still present before
  `update()`, gone after; sibling intact.
- `test_add_component_is_deferred_until_update` — archetype change visible only after
  `update()`.
- `test_spawn_then_add_component_same_tick_commit_in_order` — eager id used to queue a
  follow-up; after one `update()` the entity is in the richer pool with both fields.
- `test_add_then_remove_component_same_tick_nets_to_original` — enqueue-order proof.
- `test_update_on_empty_buffer_is_noop` — already passing; keep.

Integration (`test/integration/test_i_world.py`, fake 2-tick loop):

- `test_structural_change_during_iteration_updates_every_entity` — a system migrates
  some entities mid-tick; because the change is deferred and committed by `update()`,
  after 2 ticks **every** counter == 2 (nobody skipped, nobody double-counted). This
  is the headline hazard, fixed.

## Out of scope

- A system registry / ordered scheduler. Commit point is `world.update()`, owned by
  the main loop; a scheduler is a separate later concern.
- A general event bus / arbitrary inter-system messages. Structural ops only.
- Command coalescing / dedup (e.g. despawn-then-add-component on one id in a tick).
  Apply in order; optimize only if a real case needs it.
- Parallel system execution / threading. Single-threaded commit.
- Generational ids to catch despawn-then-reuse within a commit. Plain monotonic int
  (see task 02 out-of-scope).
- In-place field writes — they stay immediate and are not part of this task.

## Related

- `ecs/world.py`: `update` (stub, `world.py:30`), `add_entity`, `remove_entity`,
  `add_component`, `remove_component`, `_pop_from_pool` (swap-remove + reclamation),
  `query_and`.
- `ecs/pool.py`: `remove_entity` (swap-remove, `pool.py:47`).
- Builds on [task 02](../../done/2-id-based-world-and-trait-migration/TASK.md): stable
  eager ids are what make a deferred `remove_entity(eid)` / follow-up commands resolve
  correctly at commit time even after siblings moved.
- Unblocks [task 1](../1-bounce-impulse-accumulator/TASK.md): spawning/despawning
  balls from inside the physics systems without corrupting SoA iteration.
