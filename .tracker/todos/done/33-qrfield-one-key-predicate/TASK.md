# QRField: one shared entity-axis key predicate

**Created**: 2026-07-25
**Closed**: 2026-07-25
**Priority**: 3

## Resolution
`QRField._selects_axis0` (`qr_field.py:49-65`) is now the single predicate both `__setitem__` (`:86`) and
`__getitem__` (`:104`) call, so the two halves cannot disagree again. `qr.f[:]` reads and `qr.f[:] += 1` work;
`__setitem__` also gained Ellipsis keys (`qr.f[...] = v`, `qr.f[..., None] = v`), which is sound because
`part[...]` is a view. Both error messages reworded to lead with `qr.attr = xxx` and name the offending key.

Verified: microecs 298 passed / 8 xfailed (the 8 are tasks 23 and 31, untouched); robosim
`test/e2e/run_all.sh` green except the four protocol-fuzz payloads that already fail on master.

## Why it happened
`__setitem__` and `__getitem__` each spelled out "is the entity axis untouched?" and drifted: the write half took
a bare `slice(None)`, the read half took `Ellipsis` or a tuple leading with `:` but never a bare `slice(None)`.
Python desugars `x[k] += v` into `__getitem__(k)` → op → `__setitem__(k, ...)`, so the read half rejected the exact
key the write half allowed. Measured before the fix:

```
READ  qr.f[:]            RAISE TypeError      WRITE qr.f[:] = 9        OK
READ  qr.f[...]          OK                   AUG   qr.f[:] += 1       RAISE TypeError
READ  qr.f[:, 0]         OK                   AUG   qr.f[:, 0] += 1    OK
                                              AUG   qr.f += 1          OK
```

Nothing in-tree wrote `[:] +=`, so no bug was masked — but `qr_field.py:49`'s docstring advertised
`qr.position[:] += 1` for as long as it existed. Same duplicate-rule shape as task 28's `_data` finding.

## Two spots that had the old asymmetry written down as if intended
- `test/unit/test_field_numpy_parity.py:187` listed `slice(None)` among the entity-axis indexers that must raise.
  Removed; `slice(None, None, 2)` (`[::2]`) added in its place to guard the near-miss.
- `docs/source/primitives.md:44` listed `qr.f[:]` under "these raise, never lie". Now says selection raises while
  `qr.f[:]` / `qr.f[...]` read the whole field.

Both were mine, and both are the reason the fix looked like a regression at first: 3 red tests, none of them the
library's fault.

## Validation (tester)
`test/unit/test_queryresult.py`:
- `test_qr_field_whole_slice_read_returns_the_whole_field` — pinned against `qr.f[...]`, so the two keys must agree.
- `test_qr_field_whole_slice_augmented_write_scatters` — the `qr.f[:] += 1` symptom.
- `test_qr_field_whole_field_write_keys_are_interchangeable` — the deliberate `__setitem__` widening.
- `test_qr_field_whole_field_keys_that_work_and_selections_that_must_not` — control: `[...]`, `[:, k]`,
  `[:, k] +=`, `qr.f +=` keep working; int / negative / range / mask keys keep raising.

`test_entity_axis_read_indexing_raises[mask]` also covers the `isinstance(key, slice)` guard: without it,
`np.array([True, False]) == slice(None)` returns an *array* (verified, numpy 2.2.6), so a mask key would raise
`ValueError: truth value ... ambiguous` from inside the guard instead of our `TypeError`.

## Relates
- Found while sweeping task 32 (retire `[:]`), which it retroactively justifies: `[:]` was never uniformly
  available.
- Same duplicate-rule shape as task 28's `_data` finding.
