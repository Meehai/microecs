# Reserve World's own kwarg names: a field named `component` crashes add_entity with a confusing TypeError

**Created**: 2026-07-25
**Priority**: 3

## Why
Task 31 completed the reserved-name union over the three *classes* (Entity / Pool / QueryResult attrs +
methods). It misses a fourth collidable surface: **the keyword-parameter names of the World methods that take
field data as `**kwargs`**. `add_entity(components, **kwargs)` and `_validate_component(component, strict,
check_extra, **kwargs)` shadow their own parameters when a field happens to share a name.

Four names pass `World(...)` and only blow up at the first `add_entity`, with an internal `TypeError` that
names a *parameter*, not the offending field — measured:

```
component      -> TypeError: World._validate_component() got multiple values for argument 'component'
components     -> TypeError: World.add_entity() got multiple values for argument 'components'
strict         -> TypeError: World._validate_component() got multiple values for argument 'strict'
check_extra    -> TypeError: World._validate_component() got multiple values for argument 'check_extra'
```

`data` and `entity_id` are already rejected at construction, but only by accident — they happen to be Pool /
Entity attrs. Same argument as tasks 23 and 31: reject at construction, the earliest and loudest place, with a
message that names the field.

Sharpened by microecs #29: SET_DATA's command args are now **flat** (`{"component": c, <field>: value}`, like
ADD_COMPONENT), so `component` is a live key in two verbs' arg dicts, not just a parameter name.

## What
Add the internal kwarg names to the union in `World._check_components` (`world.py:264-266`), so
`World([HasComponent])` raises `ValueError` naming the field, exactly like the reserved-attr case.

Derive the set, don't hardcode it — `inspect.signature` over the `**kwargs`-taking methods (`add_entity`,
`_validate_component`, `_validate_components`, `_defaults_for`, `_do_add_component`) collecting every
non-`**kwargs` parameter name, minus `self`. Hardcoding drifts the moment a parameter is renamed (task 31's
`POOL_INTERNAL_ATTRS` is derived for the same reason).

Cheaper alternative worth weighing: make those parameters **positional-only** (`def _validate_component(self,
component, strict, check_extra, /, **kwargs)`), which removes the collision at the source instead of
blacklisting names. Then only `add_entity`'s public `components` needs reserving. Prefer this if it doesn't
churn call sites — a fixed language-level guarantee beats a name list. Whoever implements: measure both.

## Notes
- Pre-existing, not a #29 regression — the `**kwargs`-shadows-parameter pattern predates it.
- Obscure but not hypothetical: `component`/`components` are plausible field names in a user's component
  (e.g. a field holding a component-type tag), and the failure mode is a confusing internal `TypeError`.

## Validation (tester) — in place 2026-07-25, RED as xfail
- `test_world.py::test_world_rejects_field_name_colliding_with_its_own_kwargs` — parametrized over
  `component` / `components` / `strict` / `check_extra`, `xfail(strict=True)`. Flips to XPASS when the guard
  lands (mirrors `test_world_rejects_component_field_named_like_an_entity_attribute`). Drop the marker then.
- `test_world.py::test_world_kwarg_name_clash_is_currently_a_confusing_late_typeerror` — pins the CURRENT bad
  behaviour (World accepts it; `add_entity` dies with `got multiple values for [keyword] argument`) so there is a
  measured before/after. **Delete this test when the guard lands** — construction will never reach `add_entity`.
- Still to add once implemented: the `python -O` subprocess check (like task 31's), and a regression that
  `data` / `entity_id` stay rejected (today they are, but only incidentally — they are Pool / Entity attrs).
- If the **positional-only** route is taken instead, the xfail above still applies (`World()` must reject
  `components`), but the three private-method names stop being reachable as kwargs at all — adjust the
  parametrize list to whatever the derived set becomes rather than hardcoding four names.

## Relates
- Task 31 (reserved-name union + raises) — this is the fourth surface it missed.
- Task 23 (duplicate field name across components) — same layer, same "earliest + loudest" argument.
- Found while reviewing microecs #29's flat SET_DATA args shape.
