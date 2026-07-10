# microecs: "default" field metadata for add_entity

**Created**: 2026-07-01
**Closed**: 2026-07-02
**Priority**: 3

## Resolution

Shipped as a **mandatory** `default` metadata key on every field (not optional as the design below mused —
a single always-present key is a simpler invariant). `default=None` = required (omitting → `KeyError`);
`default=<array>` = filled when omitted. Author-supplied defaults are validated at `World()` construction
(guarded `if default is not None`, raises `TypeError`/`ValueError`). Fill logic in
`_check_components_against_pool` returns the filled defaults (`.copy()` per entity) and reuses #20's eager
dtype/shape checks. Both `add_entity` **and** `entity.add_component` fill omitted defaults
(`_do_add_component` passes the filled defaults to `_add_to_pool`). Builds on #20 (stacked).

Tests in `test/unit/test_world.py`: `test_add_entity_fills_default_when_field_omitted`,
`..._explicit_value_overrides_default`, `test_default_and_explicit_coexist_per_row`,
`test_component_default_wrong_{dtype,shape}_rejected`, `test_add_component_fills_default_when_field_omitted`,
`..._explicit_value_overrides_default`. Every component def across `test/` declares `default`. Suite: 241 passed.

## Why?

`add_entity` requires *every* field of every declared component as a kwarg — walls of `np.zeros((6,), "float32")`
at call sites. Most are pure zero-init; the caller only cares about a couple (e.g. `pose`). Boilerplate hides
the fields that matter.

## What?

Field metadata key `"default"`. Rule:
- Field default is **an array** → may be omitted from `add_entity`; it's filled from the default.
- Field default is **`None`** → still required; omitting it crashes (unchanged behavior).

So `add_entity` works for an omitted field **iff** that field provides a non-None default. Explicit per-field
opt-in — not a blanket "everything defaults to 0". A field the caller must always set (e.g. `pose`) declares
`default=None` and stays mandatory.

## How?

**Where the fill happens.** In `_check_components_against_pool` (world.py): for each expected field not in
kwargs, if its metadata default is not None, fill `default.copy()` (per-entity copy dodges the
shared-mutable-array footgun); else raise. Keep the "extra field" assert intact.

**Metadata plumbing.** `_check_components` requires `{"shape", "dtype", "default", *extra_metadata}` on every
field. Author defaults are shape/dtype-checked at `World()` construction so a bad default fails fast even if
the field is always passed explicitly.

Open question (deferred): `object`-dtype default semantics — document when it comes up.

## Interaction with #20

#20 makes dtype/shape validation eager. The fill path reuses those same checks, so a filled default is
validated exactly like an explicit value.

## Files

- `microecs/world.py` (`__init__`, `add_entity`, `_do_add_component`, `_check_components`,
  `_check_components_against_pool`)
