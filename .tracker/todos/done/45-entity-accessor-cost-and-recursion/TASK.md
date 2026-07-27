# Make the Entity accessor path cheap — and stop it recursing

**Created**: 2026-07-26
**Closed**: 2026-07-27
**Priority**: 2

## Verdict (2026-07-27): DONE — shipped, and it landed on the forecast

All three items shipped in `microecs/entity.py` and `microecs/world.py`. The plan predicted −27% end to
end and a 2.53 → 1.85 ratio; the measured result is **−26%, 2.48 → 1.86**. Prediction and outcome agree
to within noise, which is the part worth trusting.

**Per field touch** (`test/manual/get-entity-perf/task45_review.py`, N=20k, min of 30 interleaved rounds;
the baseline is reconstructed as a subclass routing both dunders back through `_locate`, so both variants
run in one process against the same world):

| | read | write | r+w |
|---|--:|--:|--:|
| `3c229ae` | 290 | 430 | 719 |
| shipped | **159** | **267** | **426** |

**End to end** (`examples/04`, N=100k, 2 pools, 3 runs each; compare ratios, not absolutes — absolutes
wander ±20%):

| | ns/entity | ×zip-rows | ×pool-loop | benchmarks.md scale |
|---|--:|--:|--:|--:|
| `3c229ae` | 2124 / 2193 / 2448 | 2.46 / 2.48 / 2.67 | 2.11 | 40× |
| shipped | 1592 / 1638 / 1701 | **1.79 / 1.88 / 1.86** | **1.58** | **29×** |

~19× remains the floor for any id-addressed access (plan 3); `get_entity` is now ~1.5× off it.

### What shipped

- **C1 — `World.get_entity`: 3 dict lookups → 1.** `world.py:95-108`, one `try/except KeyError`.
- **C2 — `_locate` inlined into `__getattr__`/`__setattr__`.** `entity.py:104-124`. The `pool.data[name]`
  lookup *is* the field check; the messages are rebuilt in the `except` branch, which is cold and so free.
  `_locate` **stays** for `set_data` / `get_fields` / `get_components`, where the multi-name `issuperset`
  is the right tool.
- **C3 — `__reduce__` on the class** (`entity.py:126-128`) raising `TypeError`. Measured free: 203/308 with
  it, 205/309 with it shadowed away.

### Two corrections to this task's own analysis

1. **The pickle claim was wrong.** `pickle.dumps(e)` did **not** recurse on Python 3.12 — it **succeeded**,
   1841 bytes, silently serializing `_eid_to_pool_ix`, i.e. every pool in the world, as "one entity". It is
   `pickle.loads` that then hit `RecursionError` on the far side. Dumps-succeeds / loads-explodes is a worse
   failure than the crash, and it is the whole justification for C3 — `copy`/`deepcopy` alone raise
   `RecursionError`, which is loud and self-diagnosing. Traced probe chains: `copy.copy` → `__setstate__` →
   `_eid_to_pool_ix` → … (`object.__setstate__` does not exist on 3.12, so the probe falls through to
   `__getattr__`); `deepcopy` → `__deepcopy__` → `__setstate__` → same.
2. **Requirement 1 (keep the write outside the `try`) was dropped — it is unreachable.** `ndarray[int]` and
   `ndarray[int] = v` never raise `KeyError`: not for `object` dtype (stores fine), not for structured dtype
   (that is a `TypeError`). There is no numpy `KeyError` to mislabel. Write-inside-the-`try` measured 235 vs
   270 ns, so the cheaper shape is also the correct one. The validation case built for this is dropped with it.

Also confirmed: **`__slots__` would not have fixed C3** — an unset slot still falls back to `__getattr__`,
which reads the missing internal and recurses identically. It stays a non-goal for its own reasons (13 ns).

### Two defects caught in review, before the branch landed

Both in `__setattr__`, both costing real time, neither caught by the 459-test suite:

- **`__reduce__` nested one level too deep**, inside `__setattr__` — a throwaway local, never on the class.
  C3 was not fixed *and* every write paid to build the function object.
- **The write was executed twice** (the `try` body, then again after it). Idempotent, so no test failed, but
  it cost ~212 ns — more than everything the inlining had just saved. With both present the write path was
  *slower than before the task* (506 vs 430 ns) and the net end-to-end win was zero.

The second is the lesson: a duplicated idempotent statement is invisible to a correctness suite. Only the
interleaved A/B caught it.

### Validation

- 466 tests pass, 8 xfailed (459 + 5 copy/pickle + 2 perf regression). The existing entity suite is unchanged.
- **New: the ratios are now pinned by a test** — `test/integration/test_i_entity_level_regression.py`. This is
  the direct answer to the two review defects: both were pure perf, both left the whole correctness suite
  green. A correctness suite cannot see a perf regression, so the contract in benchmarks.md (16× / 19× / 29×
  vs `oop-scalar`) needed its own guard.
  - Mirrors `examples/04`'s four loops at N=2000, interleaved round-robin, min of 15, best of 3 sweeps.
    ~0.4 s for both tests.
  - **Every path is fingerprint-checked before it is timed** — a path that stopped doing the work would
    otherwise post a wonderful number. Verified: stubbing the `get_entity` path to a no-op fails that test.
  - Asserts **ratios, never nanoseconds**: `ent/oop ≤ 34` (now ~27.7) and `ent/zip ≤ 2.15` (now ~1.77).
    `zip/oop` and `pool/oop` are **controls** — neither touches `Entity`, so if one is out of band the test
    **skips** rather than blaming `entity.py`. Both branches verified.
  - Verified against a revert to `3c229ae`: fails with `ent/oop=39.19 > 34.0; ent/zip=2.44 > 2.15`.
    Six consecutive runs on the current code pass with no drift.
  - N and the thresholds came from `test/manual/get-entity-perf/ratio_stability.py`, which sweeps N and prints
    the ratio spread. N=2000 was the sweet spot: fastest, tightest spread, and it reproduces the documented
    16/19/29 almost exactly (15.9 / 18.6 / 27.7).
- **New**: `test_entity_cannot_be_copied_or_pickled[copy|deepcopy|pickle]`,
  `test_entity_deepcopy_refusal_propagates_out_of_a_container`, `test_entity_stays_usable_after_a_refused_copy`
  in `test/unit/test_entity.py`. Verified to fail with the guard removed — all 5 hit `RecursionError` at
  `entity.py:106`, so they pin the actual loop, not just the exception type. They assert behaviour rather than
  `hasattr(Entity, "__reduce__")` precisely so the nested-`def` defect above cannot come back.
- Error messages **did** change (the `- Components: [...]` line is gone, wording differs). No test asserts on
  them beyond the field name, and plan 2 Appendix A's message pinning was **not** added — recorded here as a
  known gap rather than pretended away. File it if the messages are worth pinning.
- Docs updated: `benchmarks.md` (`get-entity` row 2163→1560 / 40×→29×, the whole per-operation table, the
  16–40× → 16–29× line, the ~19× floor note, the random-access paragraph), `primitives.md` (610 vs 471 →
  505 vs 247), `examples/05-benchmark-workloads/FINDINGS.md` (P3 re-measured, P5 replaced).
- New reusable harness: `test/manual/get-entity-perf/per_operation.py` regenerates the benchmarks.md
  per-operation table in one interleaved run.

### The one thing that did NOT improve

`get_entity(id).f` in a random-access hot loop (`probes/microecs_random.py`, medians of 3): 1k 3291→2055,
20k 2615→2407, **100k 2913→3050 (unchanged)**. At scale that workload is dominated by cache-missing random
reads into the pool column, not by accessor overhead. **The trap is the access pattern, not the API** — a
1.8× cheaper accessor buys nothing there. Recorded in FINDINGS P3 and benchmarks.md finding 4.

## Relates

- Supersedes [#36](../36-optimize-entity-read-write-path/TASK.md)'s live item 1 (that task's verdict — the
  cost was *accepted* — is unchanged; this just makes the cost smaller). #36's live item 2 is **still open and
  now relatively worse**: `set_data(f=v)` is ~2× `e.f = v` (505 vs 247 ns), since only the attribute path got
  cheaper.
- [Plan 3](../../../plans/3-access-path-performance.md) item 3 — done.
- [#46](../../open/46-in-place-is-the-floor-idiom/TASK.md) — item 3's entity-side twin, still open.
- **Not filed** (dev's call): pinning the two `AttributeError` messages; `__slots__` (13 ns, and it would let
  `ENTITY_RESERVED_NAMES` be derived from `vars()` instead of a hand-maintained set — that is its real
  argument, not perf).
