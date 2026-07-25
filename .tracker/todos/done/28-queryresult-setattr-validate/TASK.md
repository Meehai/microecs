# QueryResult.__setattr__: validate names, never silently create attrs

**Created**: 2026-07-24
**Closed**: 2026-07-25
**Priority**: 3

## Resolution
`QueryResult.__setattr__` (`query_result.py:52-61`) now mirrors Entity: `QUERY_RESULT_INTERNAL_ATTRS` route to
`super()`, a name in `_data` scatters, anything else `raise AttributeError` naming the attr and the field set.
Both dunders reach their own state via `self.__dict__.get(...)` — see the recursion finding below.

The `[:]`-retirement decision that was discussed here moved to **task 32** (docs/messages/examples sweep, no
logic); nothing in it blocks this guard.

## Why
`Entity.__setattr__` (`entity.py:86`) validated; `QueryResult.__setattr__` did not — any name that wasn't a live
data field fell through to `super().__setattr__` and **silently created a new instance attribute**. So
`qr.postion = v` (typo) or `qr.foo = 5` succeeded, did nothing, raised nothing — while the READ half raised.
Entity guarded both halves; QueryResult guarded one. The asymmetry was the bug.

## Found while testing: both dunders recursed on a `__dict__`-less instance
`__getattr__`'s message interpolated `self.fields` and `__setattr__`'s guard read `self._data`. On an instance
whose `__dict__` lacks them those lookups miss, `__getattr__` runs again, and an ordinary `AttributeError` path
becomes `RecursionError`. Not exotic: `copy`, `deepcopy` and `pickle.loads` build their target with
`cls.__new__(cls)` and immediately probe `hasattr(y, "__setstate__")` — all three died. Fixed with
`self.__dict__.get(...)` in both.

The `_data is None` escape hatch that was considered for `__setattr__` was **dropped as dead code**: `__init__`
only ever writes allow-listed names (pinned by `set(vars(qr)) == QUERY_RESULT_INTERNAL_ATTRS`), so the clause
never fires during real construction. Its only reachable effect would be to silently accept a new internal attr
that `__init__` sets and the allow-list forgot — which then becomes clobberable by the first user write.

## Validation (tester) — landed in `test/unit/test_queryresult.py`
- `test_qr_unknown_attribute_read_raises_named_error` — the READ contract the write half mirrors.
- `test_qr_known_field_write_scatters_single_and_multi_pool` — positive control, `_QRArray` and `QRField` paths.
- `test_qr_unknown_attribute_write_raises_named_error[postion|velocity|foo]` — typo, unqueried field, junk.
- `test_qr_write_to_unqueried_field_raises_even_though_the_pool_has_it` — the query's field set is the contract.
- `test_qr_internal_attrs_stay_settable` + `test_qr_internal_attrs_cover_every_init_attribute` — the allow-list
  is exactly `__init__`'s attrs, which is what makes the dropped clause provably dead.
- `test_qr_write_on_uninitialised_instance_is_rejected_not_silently_allowed`,
  `test_qr_read_on_uninitialised_instance_raises_attributeerror_not_recursion`,
  `test_qr_survives_copy_and_pickle[copy|deepcopy|pickle]` — the recursion finding above.
- `test_assigning_a_field_scatters_like_a_recarray` — `qr.f = v` cross-checked against `np.recarray`.
- `test_qr_data_views_match_pool_accessor` — `_data`'s hand-rolled `p.data[f][0:len(p)]` must stay equal to
  `getattr(pool, f)` in shape/memory/values, so Pool's live-rows rule can't drift in two places.

## Spun off
- **Task 31** — `World._check_components` is missing `Pool.RESERVED_NAMES` and validates with `assert` (stripped
  under `-O`), so a field named like a QueryResult internal can win over the data branch.
- **Task 32** — retire `[:]` from docs/messages/examples.

## Relates
- Mirror of the Entity guard (`entity.py:86`, `ENTITY_INTERNAL_ATTRS`).
- `QUERY_RESULT_INTERNAL_ATTRS` (`query_result.py:9`) is the allow-list.
- Source: robosim `183-feedback` item 11.
