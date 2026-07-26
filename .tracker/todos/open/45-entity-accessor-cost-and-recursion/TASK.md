# Make the Entity accessor path cheap — and stop it recursing

**Created**: 2026-07-26
**Priority**: 2

Two changes to `Entity.__getattr__`/`__setattr__` and one to `World.get_entity`. Both are measured, both
are local, neither touches the API. Supersedes the "live item 1" note left in
[#36](../../done/36-optimize-entity-read-write-path/TASK.md) with a bigger, re-measured win.

Evidence: `test/manual/get-entity-perf/{ladder,per_touch,recursion_fix}.py` on `3c229ae`, numpy 2.2.6,
Python 3.12. `test/manual/` is gitignored, so every number is inlined here.

## Why

`world.get_entity(eid).field` is the ergonomic surface — the one an app reaches for when it addresses
*one* entity. It costs **~2000 ns per entity tick** (read + read + write), and **half of that is
bookkeeping we can delete**: per field touch, `__getattr__` calls `_locate(names=[name])`, which allocates
a one-element list and runs `pool.fields_set.issuperset(...)` to check a field the very next line looks up
in `pool.data` anyway.

#36 priced the `issuperset` → `in` swap alone at ~15%. Inlining the whole thing is worth **~46% of a read
and ~37% of a write**:

| per field touch (N=20k) | read | write |
|---|--:|--:|
| current: `_locate(names=[name])` + `issuperset` | 305 | 392 |
| `_locate1(name)`: str instead of list, `in` instead of `issuperset` (#36's item) | 237 | — |
| inline `_locate`, keep the `in` check | 201 | 249 |
| **inline, `pool.data[name]` KeyError *is* the check** | **164** | **248** |
| + `__slots__` | 161 | 248 |
| raw numpy floor `col[ix]` | ~95 | — |

End to end at N=100k, 2 pools, `ent.position += ent.velocity*dt`, as a ratio to the `zip`-rows row
(absolutes drift ±20% between runs, ratios do not):

| | ns/entity | ×zip | benchmarks.md scale |
|---|--:|--:|--:|
| today | 1929 | 2.53 | ~40× |
| C1 alone (cheap `get_entity`) | 1905 | 2.50 | 39× |
| C2 alone (inline accessors) | 1376 | 1.80 | 28× |
| **C1 + C2** | **1409** | **1.85** | **~29×** |
| C1+C2 and the caller avoids `+=` (`p = ent.position; p += …`) | 1156 | 1.52 | 24× |
| the floor for *any* id-addressed row access | 925 | 1.21 | ~19× |

## What

### C1 — `World.get_entity`: 3 dict lookups → 1

`world.py:98-106` probes `live_entities` three times (`not in`, `is None`, `[...]`). One `try/except
KeyError` does it. ~80 ns/entity, ~5 lines.

### C2 — inline `_locate` into `__getattr__`/`__setattr__`

`_locate` **stays** — `set_data`, `get_fields`, `get_components` still need the multi-name form, and
`set_data(**many)` passes a `dict_keys`, which is the case `issuperset` is actually good at. Only the two
dunders stop calling it, because they pass exactly one name and then look that name up in `pool.data`
regardless. The dict lookup *is* the check.

```python
def __getattr__(self, name):
    try:
        pool, ix = self._eid_to_pool_ix[self.entity_id]
        col = pool.data[name]                 # <- this lookup replaces the issuperset
    except KeyError:
        raise AttributeError(...) from None   # cold path: build the rich message HERE
    return col[ix]
```

Three requirements on the shape:

1. **Only the two lookups go inside the `try`.** `col[ix] = value` stays outside it, so a `KeyError`
   raised *by numpy* (an `object`-dtype field holding a dict-like) can never be mislabelled "no such
   field".
2. **The error messages do not change.** Both of `_locate`'s messages are preserved by building them in
   the `except` branch — free, it is the cold path — and telling the two states apart there with
   `self.entity_id in self._eid_to_pool_ix`. Plan 2's Appendix A warns that this message calls
   `get_components()`, which calls `_locate(names=[])`; that stays safe (an empty name list cannot fail
   the field check) but keep the nesting in mind.
3. **`__setattr__` keeps the `_ENTITY_INTERNAL_ATTRS` guard.** `__slots__` would make it redundant and
   would let `ENTITY_RESERVED_NAMES` be derived from `vars()` instead of a hand-maintained set (verified:
   slot descriptors do show up in `vars()`) — but it is worth **13 ns** and it changes the object model.
   Not part of this task.

### C3 — the copy/pickle self-recursion (a separate defect, same two methods)

```python
e = world.get_entity(eid)
copy.copy(e)        # RecursionError: maximum recursion depth exceeded
copy.deepcopy(e)    # same
pickle.dumps(e)     # same
```

`copy._reconstruct` builds the instance via `cls.__new__(cls)` — no `__init__` — so `self._eid_to_pool_ix`
misses, `__getattr__` fires, and it reads `self._eid_to_pool_ix` again, forever. **`Pool` and
`QueryResult` both guard exactly this** (`self.__dict__.get("data")` at `pool.py:72`, `_data` at
`query_result.py:39` with a comment naming this failure); `Entity` is the odd one out. This is a pattern
instance, not a one-off — worth fixing where the pattern is broken.

The two obvious guards both **tax the hot path**, measured, so do not use them here:

| guard | cost per touch |
|---|--:|
| read internals from `self.__dict__[...]` (what Pool/QueryResult do) | **+45 ns** |
| class-level sentinels `entity_id = None`, `_eid_to_pool_ix = {}` | **+35 ns** |

(Both are real, not noise: a same-named type attribute makes generic getattr classify it before falling
back to the instance dict.)

Fix it with a dunder **on the class** instead — copy/pickle stop probing missing internals, and a field
touch never looks that name up, so it is free (measured 165 → 169 ns, inside noise):

```python
def __reduce__(self):
    raise TypeError("Entity is a live view into the world's pools; it cannot be copied or pickled. "
                    "Use entity.to_dict(), or serialize the world.")
```

That is the honest semantics — an `Entity` is a handle whose meaning is "row `ix` of pool `p` *right
now*". A copy of it cannot be anything sensible.

## Validation

- The whole existing entity suite must stay green, unchanged. Only `test_entity.py:148`
  (`raises(AttributeError, match="velocity")`) asserts on message text, and the preserved message still
  contains the field name.
- **New: both error messages are pinned.** One test per row-less state (never committed / despawned) and
  one for an unknown field, asserting the current sentences — otherwise the `except`-branch rewrite can
  quietly degrade them. Plan 2's `test_entity_without_a_row_raises_attributeerror_everywhere` (7 entry
  points × 2 states) is the existing net; extend it rather than duplicating.
- **New: an `object`-dtype field whose value is a dict-like** does not turn a numpy `KeyError` into
  `AttributeError`. This is what requirement 1 buys and the only way the KeyError-as-check can bite.
- **New: `copy.copy` / `deepcopy` / `pickle.dumps` on an Entity raise `TypeError`, not `RecursionError`.**
- Re-run `examples/04-benchmark-ecs-vs-oop.py` and update the `get-entity` row of
  `docs/source/benchmarks.md` (and the per-operation table under it, which quotes 320/471/2018 ns).

## Non-goals

- **`__slots__`** — 13 ns, changes the object model. Its real argument is deleting a hand-maintained set;
  file that separately if it is wanted.
- **A version-stamped `(pool.data, index)` cache on the Entity.** Prototyped and **rejected on
  measurement**: 1416 ns warm (≈ C1+C2's 1409, i.e. nothing) and **2008 ns cold** — worse than today — on
  any tick with a structural change, because every handle re-locates on first touch. It would also add a
  soundness invariant to defend ("all structural mutation happens inside `update()`") for zero gain.
- **Matching the `zip`-rows row (16×).** Impossible, and not this task's target — see
  [plan 3](../../../plans/3-access-path-performance.md). `zip` uses numpy's array iterator (718 ns/row);
  any indexed access pays `col[i]` getitem (885 ns/row) before microecs does anything at all. The floor
  for an id-addressed path is ~19×, which the `pool-loop` row already sits on.

## Relates

- Supersedes #36's live item 1 (that task stays closed; its verdict — the cost was *accepted* — is
  unchanged, this just makes the cost smaller).
- #36's live item 2 (`set_data(f=v)` slower than `e.f = v`) is untouched and still true: measured 2129 vs
  1929 ns/entity for the same effect.
- Docs: `benchmarks.md` "What one entity operation costs", `primitives.md`.
- Plan 2 Appendix A findings 1 and 4 own the error messages this task must preserve.
