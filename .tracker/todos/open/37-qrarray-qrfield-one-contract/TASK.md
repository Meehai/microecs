# `qr.field` has two contracts: `_QRArray` at 1 pool, `QRField` at 2+

**Created**: 2026-07-25
**Priority**: 1

## Why

`QueryResult.__getattr__` (`query_result.py:43-48`) returns a **different type** depending on how many
archetypes matched: `_QRArray` (a real `np.ndarray` subclass, so the whole numpy API is live) for 0-1 pools,
`QRField` (the shim, which enforces the batch-only contract) for 2+. Same user code, two behaviours:

| expression | 1 pool | 2+ pools |
|---|---|---|
| `qr.f[0]`, `qr.f[::2]`, `qr.f[mask]` | works | `TypeError` |
| `qr.f[0] = x` | **works, eager, unbuffered** | `TypeError` |
| `qr.f.sum()`, `qr.f.dtype` | works | `AttributeError` |

Two distinct problems:

1. **It breaks the buffered-`Entity` invariant.** `qr.f[0] = x` at one pool is a per-entity write that lands
   straight in the pool with an empty command buffer — exactly what #29 removed from the `Entity` surface.
   The guard exists only on the `QRField` path.
2. **It is a latent time bomb.** A call site that works today starts raising the moment an unrelated spawn
   elsewhere creates a second archetype. Nothing in the app changed; the archetype count did.

Verified: `test/manual/structural-audit/probe_edges.py` (sections A-D).

## What

One accessor, one contract, whatever the pool count. `#26`'s perf win must survive — the point of `_QRArray`
is that a single-archetype query is native-C numpy with no per-op object.

## How (dev writes the code)

Two candidate shapes, dev picks:

- **Narrow the subclass.** Keep `_QRArray` but override `__getitem__`/`__setitem__` to run the same
  `QRField._selects_axis0` gate, and `__getattr__` to hide ndarray methods not in the contract. Cheapest, keeps
  the fast path; risk is that `np.ndarray` has a wide surface to close off.
- **Stop returning an ndarray subclass.** Give `QRField` a genuine one-part path (no `np.concatenate`, direct
  delegation) so there is only ever one type. Cleaner contract; must re-measure against the `#26` numbers.

Either way the axis-0 gate has to be shared, not duplicated.

## Validation (tester)

Parametrize the existing QueryResult suite over **1 and 2 matching pools** and assert identical behaviour —
that is the whole bug. Today every rejection test only exercises the 2-pool path.

## Relates

- **#26** introduced `_QRArray` for the low-N win; this is its unintended second contract.
- **#38** is the other half of the numpy-surface problem (functions `QRField` accepts but cannot honour).
- **#29** (one entity write path) — `qr.f[0] = x` is the remaining hole in that invariant.
