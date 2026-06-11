# Lazy `Entity` allocation + `Pool.fields_set` (don't tax the vectorized path)

**Created**: 2026-06-11
**Priority**: 3

## Why

Two small costs that the `Entity` view (task `14-entity-object`) pushes onto users who may never touch it.
Both cut against the library's whole pitch — *don't pay per-entity python costs unless you ask for them*.

1. **An `Entity` object is allocated per entity at spawn, eagerly.** `add_entity` does
   `self.live_entities[id] = Entity(...)` (`world.py:61`) for **every** entity, even a pure-vectorized sim
   that never calls `get_entity`. 100k entities → 100k `Entity` objects sitting in a dict, unused. The dict
   is eager because it doubles as the liveness registry (the `id in self.live_entities` asserts in
   `remove_entity` / `add_component` / `remove_component` / `get_entity`).

2. **Field membership is a linear scan.** `Entity.__getattr__` / `__setattr__` do
   `if name not in (_fields := pool.fields)` (`entity.py:60`, `:70`) — `pool.fields` is a *list*, so every
   attribute read/write is O(fields). It's on the slowest path the README already calls out
   (`get-entity`, ~1450 ns/entity); a set membership is free speed there.

## Fix

### Part A — lazy, cached `Entity`

Split the two jobs `live_entities` currently does:

- **Liveness registry** — a cheap set/dict of live ids. `add_entity` adds the id (no `Entity` alloc);
  `remove_entity` drops it. This is what the existing asserts check.
- **Entity cache** — a separate `dict[EntityId, Entity]`, populated **on first `get_entity`**, returning
  the cached object thereafter (preserves task-14's stable-identity + usable-as-handle property), evicted
  in `remove_entity`.

Net: an entity that's never fetched costs **zero** `Entity` objects; one that is fetched is built once and
cached (no per-call reconstruction — task 14's decision log measured reconstruction at ~600 ns, so lazy
must still **cache**, not rebuild each call).

Note: archetype migration (`_do_add_component` / `_do_remove_component`) only touches `_eid_to_pool_ix` /
`_pool_ids`, never `live_entities` — so a cached view survives a migration untouched (it re-resolves on
access). Splitting liveness from cache doesn't change that.

### Part B — `Pool.fields_set`

Add `self.fields_set = set(fields)` in `Pool.__init__` (`pool.py`). `Entity.__getattr__` / `__setattr__`
test membership against `pool.fields_set` (O(1)) but keep reporting the ordered `pool.fields` list in the
error message. `pool.fields` stays public/ordered (used elsewhere, e.g. `get_fields`, serialization order).

## Done when

- `add_entity` allocates no `Entity`; an entity never passed to `get_entity` has no `Entity` object.
- `get_entity(id)` builds on first call, returns the **same** object on repeat calls (identity stable).
- `remove_entity` evicts from both the liveness registry and the cache.
- Liveness asserts (`get_entity` / `remove_entity` / `add_component` / `remove_component`) still fire for
  unknown / removed ids.
- `Entity` field read/write membership goes through `Pool.fields_set`; bad name still raises the same
  `AttributeError` naming the field + the ordered field list.
- Behaviour unchanged across swap-remove and archetype migration (task-14 invariants hold).
- Full suite green.

## Tests (tester — to add / extend)

- `test_world.py` — `add_entity` does not create an `Entity` (e.g. cache empty until `get_entity`);
  `get_entity` returns the same object twice (`is`); `remove_entity` clears it from the cache; liveness
  asserts still raise on unknown/removed ids.
- `test_entity.py` — existing read / write-through / `+=` / swap-remove / migration / `to_dict` suite must
  stay green unchanged (these are the invariants Part A must not break).
- Optional micro-check: a fetched-then-removed-then-respawned id doesn't return a stale cached view.

## Out of scope

- Deprecating `set_entity_data` (flagged in task 14 as redundant with `e.f = v`) — separate call.
- Any change to the vectorized `QueryResult` / `_Field` path — already O(1) (dict lookup).
- `__slots__` on `Entity` — task 14 measured GC as a non-issue (~1.2 ns); not worth it.

## Related

- `microecs/world.py:40` `live_entities` (the dict to split), `:61` eager alloc in `add_entity`,
  `:67` `remove_entity`, `:73` `get_entity`.
- `microecs/entity.py:60,70` linear `pool.fields` membership.
- `microecs/pool.py:16` `self.fields` (add `fields_set` alongside).
- Follows task `14-entity-object` (done) — its decision log: eager cache was chosen over reconstruction;
  this keeps the cache but makes it lazy, getting both (no alloc for unused, no rebuild for used).
