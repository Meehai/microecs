# World._check_components: complete the reserved-name union, and raise instead of assert

**Created**: 2026-07-25
**Closed**: 2026-07-25
**Priority**: 3

## Why
`World._check_components` (`world.py:263-285`) already rejects a component field named like Entity's or
QueryResult's attributes — the right layer, since such a name is invisible to Pool and *wins* over the data
branch in `QueryResult.__setattr__` (the allow-list is checked first), so it can never be caught downstream.
Two gaps, both measured:

1. **The union is missing `POOL_INTERNAL_ATTRS`** (`size`, `data`, `capacity`, `shapes`, `dtypes`,
   `fields_set`). A field named `size` passes `World(...)` and only dies later, inside `update()`, on Pool's own
   assert when the first entity of that archetype commits — exactly the late, quiet failure a construction-time
   check exists to prevent.

2. ~~**Every check in there is an `assert`**, so `python -O` drops all of them.~~ **DONE.** Was measured: under
   `-O`, `World()` accepted a field named `entity_ids`; the first `qr.entity_ids = v` then replaced the id array
   with the written value, and `len(qr)` / `repr` / id lookups read garbage. Component definitions are user input
   → `raise`, not `assert` (asserts are for our own bugs; same rule as tasks 20 and 22).

## What — both landed
1. ~~`reserved_names |= POOL_INTERNAL_ATTRS`~~ **DONE** (`world.py:264-266`): one construction-time gate unions all
   three layers' names. `Pool.RESERVED_NAMES` became the module constant `POOL_INTERNAL_ATTRS` (`pool.py:6`,
   mirroring `ENTITY_INTERNAL_ATTRS` / `QUERY_RESULT_INTERNAL_ATTRS`), and each class contributes attrs +
   class-dict methods — Pool's methods were collidable too (below).
2. ~~Convert the checks to raises.~~ **DONE** — `_check_components`, `_make_key` (`world.py:255`) and the `Pool`
   ctor (`pool.py:16-19`). Types: `TypeError` (not a dataclass / not an ndarray / bad shape / bad dtype),
   `ValueError` (reserved field name, metadata key mismatch, ctor list-length mismatch).

## Two extra holes found while converting — both fixed
- **`cn` used before assignment**: the non-dataclass `raise` read `cn` one line before `cn = c.__name__`, so a
  non-dataclass component died with `UnboundLocalError` instead of the TypeError naming it.
- **Pool's method names were not reserved**: `Pool.__getattr__` only runs after normal lookup fails, so
  `Pool(fields=["add_entity"], ...)` was accepted and `pool.add_entity` returned the bound method — the column
  reachable only via `pool.data["add_entity"]`. Same attrs/methods split as Entity, now enforced at both layers.

## Known, deliberately left (cosmetic)
The `Pool` ctor message (`pool.py:19`) prints only `POOL_INTERNAL_ATTRS` while the check tests the wider union, so
a field named `add_entity` is reported as "in {size, capacity, ...}" — a set that does not contain it. The union
expression is also duplicated verbatim in `world.py` and `pool.py`; hoisting it to one module-level name in
`pool.py` (after the class body) would fix both. Not worth its own task — a drive-by for whoever next edits
`pool.py` (task 34 does).

## Validation (tester) — in place, all green
`test/unit/test_world.py`: `..._colliding_with_queryresult_internals`, `..._with_pool_internals`,
`..._with_pool_methods` (all pin `ValueError`), `test_world_rejects_non_dataclass_component`,
`test_world_reserved_name_guard_survives_python_dash_o` (subprocess, `-O` is fixed at interpreter start).
`test/unit/test_pool.py`: `test_reserved_pool_method_names_raise`, `test_mismatched_field_shape_dtype_lengths_raise`,
plus the three ctor tests now pinning `ValueError`.

## Relates
- Sibling of task 23 (duplicate field name across components): same layer, same "earliest + loudest" argument.
- Found while testing task 28's `QueryResult.__setattr__` name guard.
