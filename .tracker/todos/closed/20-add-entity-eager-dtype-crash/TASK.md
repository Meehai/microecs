# microecs: add_entity crashes eagerly on wrong dtype/shape

**Created**: 2026-07-01
**Closed**: 2026-07-02
**Priority**: 3

## Resolution

`_check_components_against_pool` (world.py) now validates dtype and shape against component metadata, so a
wrong dtype/shape crashes **at the `add_entity` call**, not deferred to `update()`. Missing field also
crashes eagerly. Implemented with explicit `raise` (`TypeError` dtype, `ValueError` shape, `KeyError`
missing) rather than `assert` — microecs is a library and must validate regardless of `python -O`. Tests
`test_add_entity_wrong_dtype_crashes_eagerly` / `..._wrong_shape_crashes_eagerly` pass; xfail markers
removed. Full microecs suite green (234 passed).

## The question that started this

`world.add_entity(components=(HasRadius,), collider_radii=np.zeros((1,), "int32"))` seems to "work" even
though `collider_radii` is declared `float32`.

**Why it looks like it works:** `add_entity` is **lazy**. The eager check
(`_check_components_against_pool`, world.py) validates field **names only** — presence and no-extras. dtype
and shape are validated **later**, in `pool.add_entity` (`np.issubdtype(new_item.dtype, field_dtype)`),
which only runs at `world.update()`. So a wrong dtype passes the call silently and only crashes at commit.

## Decision: crash eagerly

Bad field **name** already crashes at the `add_entity` call. dtype/shape must too — no asymmetry where one
error is eager and the other deferred to `update()`.

## How

Add the check inside `_check_components_against_pool` (world.py), which already runs eagerly from
`add_entity`. For each provided field, assert against component metadata:
- `np.issubdtype(value.dtype, declared_dtype)`
- `value.shape == declared_shape`

Same checks `pool.add_entity` already does at commit — just moved earlier. Bonus: this method also runs
from `_do_add_component` / `_do_remove_component`, so component migrations get the same eager guard for free.

## Test (added)

`test/unit/test_world.py` — two tests:
- `test_add_entity_wrong_dtype_crashes_eagerly` — float32 field rejects an int32 array **at the call**.
- `test_add_entity_wrong_shape_crashes_eagerly` — shape (1,) field rejects a (2,) array **at the call**.

Both dtype **and** shape must crash eagerly — no asymmetry.

## Files

- `microecs/world.py` (`_check_components_against_pool`)
- Existing commit-time check to mirror: `microecs/pool.py` (`add_entity`)
