# Id-based World + trait-migration API

**Created**: 2026-06-02
**Priority**: 2
**Status**: ✅ Done (2026-06-02)

## Outcome

Implemented on `World` using **component** terminology (the codebase was renamed `trait`→`component`
mid-task, so the methods are `add_component` / `remove_component`, not `add_trait` / `remove_trait`).
`add_entity` returns a stable id; `add_component` / `remove_component` migrate an entity between
archetype pools keeping the **same** id; `remove_entity` drops it. Swap-remove bookkeeping verified
for empty / last-index / middle rows.

Bonus beyond scope: emptied pools are now **reclaimed** from both `pools` and `pool_to_components`,
so archetypes don't leak.

All "Done when" criteria covered in `test/test_world.py` (46 passing):
- ids unique & stable: `test_add_entity_returns_unique_ids`, `test_add_component_keeps_entity_id`,
  `test_id_resolves_after_sibling_removed`
- migrate + preserve fields: `test_add_component_moves_entity_and_preserves_fields`,
  `test_add_component_only_needs_new_fields`, `test_remove_component_narrows_archetype`,
  `test_add_then_remove_component_round_trips`
- removal + sibling resolves: `test_remove_entity_by_id`, `test_remove_last_index_drops_only_that_entity`,
  `test_remove_middle_entity_repoints_swapped_id`
- clear error on misuse: `test_add_duplicate_component_raises`, `test_remove_absent_component_raises`,
  `test_add_unknown_component_raises`
- reclamation: `test_empty_pool_is_reclaimed`

Open follow-up (minor, undecided): removing an entity's **last** component errors via
`_get_entity_pool([])` ("Entity has no components") — delete the entity vs reject cleanly is a design call.

## Why

Today an entity has no stable name. The only handle is `(archetype, index)`, and
`Pool.remove_entity`/`pop_entity` are **swap-remove** — they shift the tail into the
freed slot, so any index you hold for another entity goes stale on the next removal.
Index handles don't survive mutation.

`World`'s own docstring already promises *"Entities are id-based … assigned a unique
id"* — but the code never implements it. The migration pattern from
[task 1](../1-bounce-impulse-accumulator/TASK.md) / the world tests
(pop from pool A → add a trait → re-add to pool B) is currently done by reaching into
`pool.pop_entity(index)` directly. Make ids real, then the operation becomes a clean
`World` method that doesn't leak pool/index details.

Migration = *change which components an entity has* → it moves to the pool for the new
archetype. Three inputs always: **which entity**, **what changes**, **data for any
added fields**. A stable id is what makes "which entity" survive across removals.

## API (id-based)

```python
eid = world.add_entity(traits, **fields)        # CHANGE: now returns a stable int id
world.add_trait(eid, HasVelocity, velocity=...) # widen archetype -> moves entity to richer pool
world.remove_trait(eid, HasColor)               # narrow archetype -> moves entity to smaller pool
world.remove_entity(eid)                         # drop entirely
```

Inputs:

| method          | inputs                              | notes                                  |
|-----------------|-------------------------------------|----------------------------------------|
| `add_entity`    | `traits`, `**fields`                | returns `eid`                          |
| `add_trait`     | `eid`, `Trait`, `**fields_of_trait` | existing fields carry over automatically |
| `remove_trait`  | `eid`, `Trait`                      | no field data needed                   |
| `remove_entity` | `eid`                               | —                                      |

Key property: the caller never passes `from_traits`. World looks up the entity's
current archetype itself. The id is the only handle the caller holds, and it stays
valid across every add/remove.

## Internals (developer's call; sketch only)

- `World` assigns a monotonic id (`self._next_id`) and keeps
  `self._locations: dict[eid, (PoolKey, index)]`.
- **Swap-remove bookkeeping is the one gotcha.** When the tail row fills a freed slot,
  the moved entity's index changed and `_locations` must be patched. Clean trick:
  carry the id as a column in each pool (an `_id` field). After a swap, World reads
  `pool._id[index]` to learn which entity now sits at `index` and fixes its entry.
  Pool stays index-dumb; World owns identity.
- `add_trait` / `remove_trait` = pop the entity's current fields, add/drop the trait's
  fields, compute the new archetype key, add to that pool, update `_locations`.
- Reuse the existing field/shape/dtype validation from `_get_entity_pool`.

## Done when

- `add_entity` returns an id; the id is stable across any number of unrelated
  add/remove/migrate operations.
- `add_trait` moves the entity to the richer pool, preserves all prior field values,
  and requires only the new trait's fields.
- `remove_trait` moves the entity to the smaller pool, dropping that trait's fields.
- `remove_entity(eid)` removes it; a sibling entity's id still resolves correctly
  afterwards (no stale-handle corruption from swap-remove).
- Adding a trait the entity already has, or removing one it lacks, raises a clear error.

## Tests (tester writes first, under `test/` — red spec to implement against)

- `test_add_entity_returns_unique_ids` — ids distinct, monotonic.
- `test_id_resolves_after_sibling_removed` — remove entity A, B's id still points at B
  (the swap-remove bookkeeping check).
- `test_add_trait_moves_entity_and_preserves_fields` — entity leaves pool A, lands in
  pool B, old field values intact, new field set.
- `test_remove_trait_narrows_archetype` — entity moves to the smaller pool, dropped
  field gone.
- `test_add_trait_only_needs_new_fields` — caller passes just the new trait's fields,
  not the carried-over ones.
- `test_add_duplicate_trait_raises` / `test_remove_absent_trait_raises`.
- `test_remove_entity_by_id` — gone from its pool; counts conserved.

## Out of scope

- Generation/versioned ids (detecting use-after-free of a recycled id). Plain
  monotonic int is enough for now; revisit only if id reuse becomes a thing.
- Batched / bulk migration (move many entities at once). One entity per call.
- A general `set_traits(eid, add=(...), remove=(...))` combo form — add only if a
  caller actually needs add+remove in one step.
- Querying *by id* across pools beyond the `_locations` lookup.

## Related

- Supersedes the direct-`pop_entity` migration exercised in
  `test/test_world.py::test_pop_then_migrate_entity_to_a_richer_archetype`.
- `ecs/world.py` (`add_entity`, `_get_entity_pool`, `_make_key`), `ecs/pool.py`
  (`remove_entity`, `pop_entity`, swap-remove).
- Unblocks cleaner entity lifecycle for [task 1](../1-bounce-impulse-accumulator/TASK.md)
  (spawning/despawning balls without index juggling).
