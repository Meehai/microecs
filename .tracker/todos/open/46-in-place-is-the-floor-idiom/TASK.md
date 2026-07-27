# The in-place idiom is the floor idiom — stop the self-copy, stop steering people away from it

**Created**: 2026-07-27
**Priority**: 2

One finding at three sites: **an in-place ufunc on a hoisted column is exactly the raw-numpy floor, and
every spelling the library offers or documents costs 1.4–2.4× more.** Measurements and the soundness
argument are in [plan 3](../../../plans/3-access-path-performance.md) Part 1; only what to do is here.

## 1. `QueryResult.__setattr__`: skip the write when the value *is* the column (code, XS)

`query_result.py:61` ends in `getattr(self, name)[:] = value`. For `qr.f += x` the in-place ufunc has
already written into the pool, so this copies the whole column onto itself. **2.3× at 2 pools / N=100k**
(1.07 → 0.47 ns/entity).

```python
if value is getattr(self, name):    # `qr.f += x`: the ufunc returned the cached field object itself
    return
```

Sound because identity is on the **cached field wrapper** (`self._cache[name]`), not on memory — two
fields never share one wrapper. Verified on both surfaces: only `qr.f += x` and `qr.f = qr.f` compare
`True`, and both are genuine no-ops; `qr.f = qr.other_f`, `qr.f + 1`, `.numpy()`, `[:, 0:1]` and scalars
all compare `False` and still copy.

## 2. `Pool.__setattr__`: the message recommends the slow spelling (message, XS)

`pool.py:78` says *"Use `pool.component[:] = ...`"*. That spelling builds two temporaries and copies;
hoisting to a local is **2.1–2.4×** faster and is the floor:

```python
pos = pool.position          # a view -- no __setattr__ involved
pos += pool.velocity * dt    # true in-place ufunc, straight into the pool
```

The guard itself is right (`pool.f = arr` must not rebind a column) — only the advice is wrong. Note
`Pool` **cannot** use item 1's identity trick: its `__getattr__` builds a fresh view per call, so there is
no stable object to compare.

## 3. Docs: name the floor idiom (docs, S)

`benchmarks.md` and `primitives.md` teach `pool.f[:] = pool.f + …` and `qr.f = qr.f + …`. Both are ~2× off
the floor. Add the hoisted-local form for hot loops. Same one level down on the entity path:
`ent.position += x` is **three** touches (read, read, write-back through `__setattr__`), worth 250 ns of
1409 — `p = ent.position; p += x` is two. That one is docs-only; the library cannot fix it (Python always
assigns back on `+=`, and unlike item 1 a row view's identity cannot tell "write yourself" from "write
your neighbour").

## Validation

Item 1 needs a **correctness** test before the line ships — it is the soundness argument, not a perf
tweak: the four "must still copy" assignments above plus `qr.f = qr.f`, run against a 1-pool **and** a
2-pool world (the two surfaces are different objects). Then re-measure with
`test/manual/get-entity-perf/columnar_gap.py`, interleaved.

## Relates

- Plan 3 items 1, 2, 5 — this task is all three; they are one finding.
- [#45](../45-entity-accessor-cost-and-recursion/TASK.md) — item 3's entity-side twin.
- [#37](../37-qrarray-qrfield-one-contract/TASK.md) — item 1 touches `__setattr__`, which is on the
  `_QRArray`/`QRField` boundary. Do item 1 first (it is one line and helps both branches).
