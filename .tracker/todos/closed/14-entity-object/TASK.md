# `Entity` object — `world.get_entity(id)` returns a live view, not a tuple

**Created**: 2026-06-11
**Priority**: 2
**Status**: ✅ Done (2026-06-11) — live `Entity` view via eager `live_entities` cache; attribute read/write-through; `to_dict()` serialization; reserved-name guard for `Entity` **and** `QueryResult`; `EntityData` retired. **226 tests green; branch ready to merge.**

## Final shape (as landed)

- `microecs/entity.py` — `Entity` is a live view (re-resolves `(pool, row)` from the id each access). `__getattr__` reads (with `try/except KeyError → AttributeError`, so a read on an uncommitted spawn says "call `world.update()`" instead of raising a raw `KeyError`); `__setattr__` writes through to the pool; `get_components()` / `get_fields()` / `to_dict(serialization_field=None)`. `ENTITY_INTERNAL_ATTRS = {entity_id, _eid_to_pool_ix, _pool_to_components}` is the **attrs-only** set `__setattr__` routes to `super()` — methods are deliberately excluded, so `e.get_components = x` raises instead of shadowing the method.
- `World.live_entities: dict[EntityId, Entity]` replaces `live_ids` — doubles as the eager entity cache. Built in `add_entity`, evicted in `remove_entity` (migrations pop+re-add, so the cached view survives). `get_entity` returns the cached object (stable identity; usable as a handle pre-commit, with a friendly error on field access until `update()`).
- Reserved-name guard: `World._check_components` rejects component fields colliding with either class — `ENTITY_INTERNAL_ATTRS | {vars(Entity) methods} | QUERY_RESULT_INTERNAL_ATTRS | {vars(QueryResult) methods}`. Each class declares its instance-attr set (can't be introspected without an instance); methods are auto-derived from the class dict. One source of truth per class.
- `EntityData` type alias removed from `utils.py` / `__init__.py`; `_pop_from_pool` now annotated `tuple[dict[str, np.ndarray], ...]`. `Entity` exported from `microecs`.
- Decision log: chose the `live_entities` cache over `__slots__` (measured: GC is a non-issue ~1.2 ns; the cache removes the ~600 ns reconstruct on repeat fetches). Fully-programmatic `dir()`-derived reserved names rejected — needs `object.__setattr__` in `__init__` to break the `__setattr__`→constant cycle, not worth it since end-users don't edit the lib.

## Resolved during review

- **`examples/03-serialization.py`** rewritten onto the `Entity` API (`entity.to_dict(serialization_field="serializable")` over `world.live_entities.values()`).
- **Shadowing fixed** — `ENTITY_INTERNAL_ATTRS` (attrs-only) split out from the guard set, so `e.<method> = x` raises.
- **Pre-commit read** now raises `AttributeError` (was raw `KeyError`).
- **`examples/04` + README benchmark** updated (`get-entity` row, lazy Entity ~13% faster).

## Known follow-ups (not blocking merge)

- The pre-commit guard only covers **reads** (`__getattr__`). `__setattr__` / `get_components` / `get_fields` / `to_dict` on an uncommitted spawn still raise raw `KeyError`. A shared `_resolve()` helper would make it symmetric — low value (who writes/serializes before `update()`?).
- `set_entity_data(id, f, v)` is now redundant with `get_entity(id).f = v` (post-commit) — consider deprecating.
- Cosmetic: the `__getattr__` `raise AttributeError(...)` chains the caught `KeyError` ("During handling of the above exception…"); `from None` would clean the traceback.

## Why

`get_entity` used to return a snapshot tuple `(EntityData, list[ComponentType])` — a dict of the
entity's fields copied out at call time. Two problems:

1. **Stale after `update()`.** The dict didn't know the row moved (swap-remove / migration), so a held
   snapshot silently went wrong.
2. **Dict ergonomics.** `data["position"]`, `components` as a separate return value — clunky for the
   "object-like" use case `get_entity` exists for.

The replacement is `microecs/entity.py::Entity`: a **live view** that re-resolves `(pool, row)` from the
id on every access, so it tracks pool changes. `e.position` reads, `e.get_components()` / `e.get_fields()`
report the current archetype.

## What landed (`microecs/entity.py`)

- **Live read** — `__getattr__` resolves `_eid_to_pool_ix[entity_id]` each call → correct after swaps and
  archetype migrations (the old snapshot couldn't do this).
- **Write-through (issue 1)** — `__setattr__` scatters into the pool buffer, eager (like
  `set_entity_data`). `e.field = x` *and* `e.field += x` both persist. The constructor's internal attrs
  (`ENTITY_RESERVED_NAMES = ["entity_id", "_eid_to_pool_ix", "_pool_to_components"]`) branch to
  `super().__setattr__` so init doesn't recurse / isn't mistaken for a field write.
- **Named errors (issue 2)** — bad field name raises `AttributeError(f"... not in entity fields: {fields}
  (entity id: {id})")` on **both** read and write (matches `QueryResult`'s message style).
- `get_fields()` added alongside `get_components()`.

## Issue 3 (reserved-name guard) — landed (historical note; see "Final shape" for what shipped)

`World._check_components` (`world.py:200`) only rejects component fields that collide with
**`QueryResult`** attribute names. It does **not** guard `Entity`'s names — so a component with a field
named `entity_id` / `get_components` / `get_fields` / `_eid_to_pool_ix` / `_pool_to_components` is
silently shadowed: `e.entity_id` returns the id (not the field), `e.get_components` a bound method.

**Fix** — union Entity's names into the reserved set:

```python
from .entity import Entity, ENTITY_RESERVED_NAMES
...
entity_reserved = set(ENTITY_RESERVED_NAMES) | {n for n in vars(Entity) if not n.startswith("__")}
reserved = set(vars(QueryResult([], {}, {}, np.array([], "int64")))) | entity_reserved
```

`vars(Entity)` picks up the methods (`get_components`, `get_fields`, + any added later);
`ENTITY_RESERVED_NAMES` covers the instance attrs (not in the class dict). Keep the methods **out** of
`ENTITY_RESERVED_NAMES` itself — adding them there would make `e.get_components = x` silently shadow the
method via the `super().__setattr__` branch instead of raising.

## Done when

- Building a `World` with a component field named like any Entity attr/method raises at construction.
- The 5 red params in `test_world_rejects_component_field_named_like_an_entity_attribute` go green.
- No regression: full suite green (currently 210 pass + 5 red = the issue-3 set).

## Tests (tester — already in place)

- `test/unit/test_entity.py` (9, green) — read / write-through / `+=` / eager-visible / copy-not-alias /
  named errors on read+write / live-view across swap-remove + archetype migration.
- `test/unit/test_world.py::test_world_rejects_component_field_named_like_an_entity_attribute` (5,
  **red** — flips green when issue 3 lands).
- Migrated off the old tuple pattern: `test_world.py` get_entity/set_entity_data suite + churn +
  entity_ids alignment + tag-migration + zero-dim; `test_field_shape_invariant.py::
  test_field_shape_survives_migration`. All on `e.field` / `e.get_components()` now.

## Docs / examples (dev — flagged, not tester's to edit)

- ~~`examples/03-serialization.py:50` still uses the old `entity, components = ...` pattern → crashes.~~
  **Resolved** — rewritten onto `entity.to_dict()`.
- `examples/04-benchmark-ecs-vs-oop.py:139` comment "builds a dict + one numpy index per field" is stale
  (no dict built now). README benchmark row for `get-entity` updated: `1674 → 1450 ns`, `35× → 30×` (the
  lazy Entity dropped the per-call dict-of-all-fields build → ~13% faster, verified).
- `README.md:106` reproduce path points at `test/manual/perf/bench_ecs_vs_oop.py`, which doesn't exist
  (only `examples/04` does). Pre-existing; left for dev to reconcile.

## Related

- `microecs/entity.py` — the view (`__getattr__`, `__setattr__`, `ENTITY_RESERVED_NAMES`).
- `microecs/world.py:73` `get_entity`, `:78` `set_entity_data` (parallel single-entity write), `:200`
  `_check_components` (the guard to extend).
- Supersedes the snapshot-tuple shape from task `4-get-entity-by-id` (done).
