# w5 churn beyond #44: a cheap pool check, a discarded despawn copy — and batch spawn deferred

**Created**: 2026-07-27
**Priority**: 2

## Status (2026-07-27): items 1 and 2 DONE, item 3 DEFERRED — this task is closed

Items 1 and 2 landed. **Item 3 (batch spawn) is deliberately not being done** — a 92× ceiling on a workload
we wrote ourselves is not enough to justify the API surface; see the "DEFERRED" block under item 3 for the
four reasons and the trigger to revisit. Nothing actionable is left here, so the task closes with the
measurement kept as the record for whoever picks batch spawn up later.

Landed on branch `optimize-49` (`8fd6963` + `3c16d06`), then **merged with master (which had #44)**. The
per-item A/B below was taken pre-merge, on the #44-less base; for what shipped, see "Combined result".
A/B in one process
(`test/manual/churn/task49_ab.py`, N=3000, min-of-7, 3 fields):

| item | before | after | speedup |
|---|--:|--:|--:|
| 2 — pool dtype/shape check | 12432 ns/spawn | 11026 | **1.13×** |
| 1 — despawn bookkeeping | 2668 ns/despawn | 1568 | **1.70×** |
| churn pair | 15100 ns | 12594 | **1.20×** |

The dev also did two things the task did not ask for, both good:

- **`Pool.add_entity(**kwargs)` → `add_entity(entity_data: dict)`**, threaded through `_add_to_pool` /
  `_do_add_component` / `_do_remove_component`. Kills a dict unpack+repack at every call boundary on the
  spawn path. (The A/B above holds the dict form constant in *both* arms, so it prices only the check —
  this plumbing win is on top of the 1.13×.)
- **Empty-pool reclamation hoisted out of `_pop_from_pool` into one sweep at the end of `update()`.** This is
  what kept item 1 from becoming a divergent duplicate path: the biggest shared block left both despawn
  functions instead of being copied into the new one. It also **fixes same-tick archetype blink for free** —
  an archetype emptied and refilled within one tick is no longer torn down and rebuilt, which is
  space-shooter finding 12 (747 pool builds in a chaos session). Pinned by
  `test_an_archetype_emptied_and_refilled_in_one_tick_is_never_torn_down`.

**Where the frame goes** (`w5_frame_breakdown.py`): on the #49-only branch, despawn 3% / **spawn 58%** /
update 33%; on the #44-only branch, despawn 2% / spawn 33% / **update 60%**. Opposite halves — #44 cuts the
buffered `add_entity`, #49 cuts the commit.

### Combined result (2026-07-27, after the merge — this is what shipped)

They compound, and it was checked rather than assumed: measured separately against the same 07-26 baseline,
#44 alone was −28.2% on w5 and #49 alone −29.3%; 0.72 × 0.71 = 0.51 against the **−50.2%** measured on the
merge. A churn pair went **12564 → 6556 ns (1.92×)**.

| | 07-26 baseline | #44 + #49 |
|---|--:|--:|
| w5 churn vs EnTT @100k | 0.82× | **1.65×** |
| w5 churn vs ecs-pattern @20k | 0.46× | **0.96×** (tie) |
| w7 migrate vs EnTT @100k | 0.77× | **1.01×** |
| w7 migrate vs snecs @1k | 0.89× | **1.23×** |
| churn pair | 12564 ns | **6556 ns** |

**microecs now wins all seven workloads at N=100k** for the first time, structural ones included. **w7
migrate −30.4% is entirely #49's and was nobody's target** — component migration runs through the same
`_pop_from_pool`/`_add_to_pool` pair as churn, so it paid the same per-entity costs twice per migration.
Confirmed off-benchmark: robosim's own physics tick is **2–18% faster (~11% mean, −16/−17% at 50/100
robots)** while its render tick — draw-bound, no ECS mutation — stayed flat at 0–3%, which is the control
that rules out drift. Both re-run on the merge (`f558640`); the earlier #49-only rows were dropped from
`baseline.csv` so this PR contributes exactly one entry per perf test.

**Two nits left for the dev** (both one-liners, neither affects behaviour):
- `pool.py:40` — `(shp := new_item.shape)` is now unused; the message reads `new_item.shape` directly.
  Pylint W0612, and it drops the file from 10.00 to 9.96.
- `world.py:194` — `_pop_from_pool`'s docstring still says it "removes them if they get empty". It doesn't
  any more; the sweep does.

**Tests** (494 passed, 8 xfailed): `test_pool.py`'s wrong-shape/wrong-dtype tests now expect `ValueError`
instead of `AssertionError`, plus `test_add_bad_field_error_message_reports_both_dtype_and_shape` — that trio
caught a real defect during the work (walrus + `or` short-circuit left `shp` unbound, so a wrong **dtype**
raised `UnboundLocalError` instead of `ValueError`; fixed in `3c16d06`). `test_world.py` gained
`test_the_two_despawn_paths_leave_identical_world_state` (7 cases) as the drift guard between the two
despawn functions, plus the three archetype-lifecycle tests above. All `pool.add_entity` call sites in
`test_pool.py` / `test_queryresult.py` were converted to the dict signature.

## Why

[#44](../../done/44-spawn-path-validates-twice/TASK.md) **landed 2026-07-27** (`3ba0e76`) and beat its own
estimate: 26% off a churn pair, not the ~16% predicted, and it flipped two cells — microecs now beats EnTT and
snecs on w5 at N=5,000. It did **not** flip the low-N cell: still ~10× behind ecs_pattern at N=200 (was 12×),
because there the fixed per-spawn cost dominates and #44 only removed one duplicated slice of it. That cell is
this task's target.

### Re-baselined post-#44 (2026-07-27) — where a w5 FRAME actually goes

`test/manual/churn/w5_frame_breakdown.py` splits one real w5 frame, so the next fix is aimed at the part that
is big rather than the part that is famous:

| N | b | despawn | spawn | **update()** | query | integrate |
|--:|--:|--:|--:|--:|--:|--:|
| 200 | 16 | 3% | 32% | **59%** | 3% | 3% |
| 1,000 | 16 | 3% | 33% | **58%** | 3% | 4% |
| 5,000 | 50 | 2% | 33% | **62%** | 1% | 2% |
| 20,000 | 200 | 2% | 33% | **64%** | 0% | 1% |

Churn is **93–99% of the frame** — so w5 really is a churn benchmark, and the two things to attack are
`update()`'s commit (~60%) and the buffered `add_entity` loop (~33%). **The cold query is NOT a factor**
(3–7 µs/frame): `#47`'s lazy `entity_ids` already removed the cost the space-shooter run flagged, so do not
re-file that. Everything below is priced on **current** code with `test/manual/churn/w5_next_wins.py`
(N=2000, min-of-7, one process, 3 fields — full spawn = 9261 ns):

| # | candidate | ns/entity | share of a spawn | effort |
|---|---|--:|--:|---|
| **2** | `np.issubdtype` → `==` in `Pool.add_entity` | **1909** | **21%** | **one line** |
| 1 | drop the discarded despawn copy | 1047 | 11% (43% of a *despawn*) | small |
| — | ~~memoize the per-spawn archetype key~~ | 150 | 2% | **rejected — not worth it** |
| ~~3~~ | ~~batch spawn (floor = 100 ns/e)~~ | 9160 | 99% | new API — **deferred, see below** |

**Outcome:** 2 and 1 landed (in that order — 2 had the best ratio in the library at the time). Item 3 would
subsume both and is by far the biggest number here, but it is **deferred** — see its section.

**Rejected candidate, recorded so it is not re-proposed:** `World._get_entity_pool` recomputes the archetype
bitmask key on every spawn even though a burst shares one archetype. Memoizing it looks like an obvious win
and is worth **150 ns/entity (2%)** — the key is already just a dict lookup over a 2-element loop. Not worth
the cache-invalidation surface.

## 1. Despawn copies data it throws away — 1047 ns/entity, **43% of a despawn**

`Pool.pop_entity` (`pool.py:58-62`) copies every field into a dict, then `World.update()`'s `REMOVE_ENTITY`
branch (`world.py:150`) discards `_pop_from_pool`'s return value. Only `_do_add_component` /
`_do_remove_component` need the data — they carry it to the new pool.

Re-measured post-#44 at 3 fields (`w5_next_wins.py`; the original 807 ns was 2 fields):

| | ns/entity |
|---|--:|
| `Pool.pop_entity` (copy + pop-swap) | 2436 |
| `Pool.remove_entity` (pop-swap only) | 1389 |
| **discarded on the despawn path** | **1047 (43%)** |

Scales with field count, so a fat archetype pays more. Fix: give `_pop_from_pool` a no-copy mode, or have
`update()` call the pop-swap directly for `REMOVE_ENTITY`.

## 2. The SoA insert's dtype assert uses `np.issubdtype` — 1909 ns/entity, **21% of a spawn, one line**

`pool.py:40-42` asserts `isinstance` / `shape ==` / `np.issubdtype` per field per spawn.

| check | ns/call |
|---|--:|
| `np.issubdtype(a.dtype, "float32")` | **653** |
| `a.dtype == "float32"` | 80 |
| `a.shape == shape` | 76 |
| `isinstance(a, np.ndarray)` | 65 |

The task proposed the narrow version: swap `np.issubdtype(new_item.dtype, field_dtype)` for
`new_item.dtype == field_dtype` (`pool.py:42`), keeping the assert. Measured **1909 ns/entity at 3 fields —
21% of a full spawn.**

**What actually landed goes one step further: the three asserts became a single `raise ValueError`.** That
does reopen [#34](../../done/34-asserts-to-raises-sweep/TASK.md)'s "keep as assert" decision, deliberately and
in the right direction — it matches the project's raise-over-assert principle, and this check guards data that
crosses a module boundary. The trade to be aware of: it is no longer free under `python -O`, so the pool now
pays it forever. That is affordable precisely *because* the `==` swap made it cheap; the two changes only make
sense together. `#34`'s entry should record that this one flipped.

**It is also the more correct assert, which is the real argument.** `World._validate_component`
(`world.py:228-229`) already **raises** on `field.dtype != dtype` — a plain `==`. So by the time data reaches
the pool, `==` is guaranteed by contract, and `issubdtype` is asserting something strictly *weaker* than what
upstream enforces: it would wave through a dtype the validator would have rejected. Matching the check to the
contract is the fix; being 8× faster is the bonus.

Equivalence checked over all 4 allowed dtypes (`float32/int32/bool/object`, `world.py:278`) × 9 real dtypes
(incl. `float64/int64/int8/uint32/float16`): **zero disagreements** between `issubdtype` and `==`. No abstract
dtype (`np.floating`) can reach here — `_check_components` rejects anything outside those four strings.

## 3. There is no batch spawn — measured ceiling 92× — **DEFERRED, see below**

B entities of one archetype, spawned one at a time (B=1000, min-of-5):

| | ns/entity | per frame at B=1000 |
|---|--:|--:|
| today — `add_entity()` × B + `update()` | 9261 | 9.26 ms |
| floor — validate once + block write + id map | 100 | 0.10 ms |

(Re-measured post-#44, 3 fields: **92×**. The pre-#44 figure was 10986 → 111 = 99×.)

The floor is ~all irreducible: two dict writes per id (`_eid_to_pool_ix`, `live_entities`). Everything else —
validation, the `Command`, the row-by-row numpy write — collapses into one pass: one shape/dtype check on the
`(B, ...)` arrays, one `pool.data[f][lo:lo+B] = arr` per field.

**The measurement is real** — w5 is not losing to archetype layout; **spawn is the last per-entity Python loop
in the library**. Every workload microecs wins, it wins by batching; the one API with no batch form is the
structural one, while the docs tell users to batch everything including random access.

### DEFERRED — not doing this now (decided 2026-07-27)

**A 92× ceiling is not the same as a 92× win, and the cost here is API surface, not implementation.** The
reasons, in order:

1. **The evidence is synthetic.** w5 is a benchmark workload we wrote. The spawn burst it models — B entities,
   one archetype, data already in `(B, ...)` arrays — is exactly the shape batch spawn is best at, so w5 is
   close to a best case rather than a representative one. Nothing in robosim or the space-shooter run spawns
   that way today. **Revisit when a non-synthetic workload asks for it.**
2. **The chaos-scene case is real, but it is not the same case.** Bullets and particles spawning and dying
   every tick is a genuine pattern, and paying an O(n) Python loop per frame for it is genuinely wasteful.
   But in that pattern the per-entity data is usually *computed* per entity (a spread angle, a jittered
   lifetime), so the caller would have to build the `(B, ...)` arrays first — which is the same loop, moved
   into user code. Batch spawn only pays when the arrays already exist.
3. **It is tedious against the current API.** Every field becomes a leading-axis array the caller assembles
   and keeps aligned, `add_entities` returns a range of ids the caller must map back to its own objects, and
   defaults have to broadcast. That is real ergonomic cost paid by every user, to speed up the users who have
   arrays already. Per "enabler, not solutioner": not worth adding until the demand is concrete.
4. **The two cheap items are banked.** Items 1 and 2 got 1.20× on a pair with no API change at all, and #44
   got 1.36× before them. The remaining gap is no longer embarrassing.

**Trigger to revisit — file a new task when any of these is true:** a real robosim (or other non-synthetic)
scene profiles with spawn as a top cost; a user hits the wall and asks; or batch **despawn** becomes needed
too, in which case design both together — a batch API with only one half is worse than none.

**Keep for that task, so the analysis is not re-done:** the 92× floor measurement above, and the design notes:

- **Signature.** `world.add_entities(components, **field_arrays) -> list[EntityId]` with leading axis B, vs. a
  batch-aware `add_entity`. Prefer the separate `add_entities` — no overload ambiguity between "a `(2,)` position"
  and "two scalars".
- **Contract stays deferred.** One `Command(ADD_ENTITIES, ...)` buffered, block write at `update()`. Validation
  stays eager at the call (#22). One `Command` per batch is ~270 ns total — it does not move the number.
- **Validate once, for the batch.** `arr.shape[1:] == field_shape` and `arr.dtype == field_dtype`, plus
  `len(arr) == B` agreement across fields. Defaults broadcast to `(B, *shape)`.
- **Ids are contiguous** (`_last_id + 1 … + B`), so the return can be a `range`. Do not build B `Entity` objects
  — `live_entities[eid] = None` as today.
- The prototype in the probe writes **eagerly** and bypasses the buffer; it measures the commit-side work only.
  It is a cost floor, not a design.
- Batch **despawn** is the harder half (pop-swap ordering when the victims are scattered through the pool) and
  was never in scope here — item 1 was the cheap part of that end.

## Validation (tester)

- Item 2 is behaviour-preserving **by contract** (`world.py:228` already enforces `==` upstream), so the whole
  suite stays green unchanged; add one pin that a wrong-dtype spawn still raises `TypeError` from
  `World.add_entity`, i.e. the pool assert is not what was catching it.
- Item 1 is behaviour-preserving: whole `test_pool.py` + `test_world.py` green, unchanged. It needs a
  pin that a despawn still leaves the *other* entities intact after the pop-swap (the copy removal touches the
  path that reorders rows), and that add/remove_component still carry data across pools.
- Item 3 is deferred, but keep its test plan with the design notes for the future task: batch spawn lands B
  rows with the right values and contiguous ids; a mismatched B across fields raises **at the call**, not at
  `update()`; wrong dtype/shape raises the same exception type as the single-entity path; defaults fill for
  omitted fields; a batch spawn interleaved with single spawns and despawns in one tick commits in order.
  Extend the randomized churn invariant check (`test_pool_ids_stay_aligned_through_random_churn`) to emit
  batch spawns too — that is what would catch an id/row-map desync.
- Re-run w5: `cd examples/05-benchmark-workloads && ./run_benchmark.sh 200 1000 5000`. **Compare ratios against
  the other libraries in the same run, never ms across runs** (absolutes drift ±10–25% — plan 1 Part 8).

## Relates

- [#44](../../done/44-spawn-path-validates-twice/TASK.md) — landed first, same path. It cuts the *buffered*
  half (`add_entity`) while this task cut the *commit* half, so the two compound rather than overlap. Note
  they were developed on separate branches off the same base and must be stacked before either set of
  absolute numbers is quoted together.
- [#34](../../done/34-asserts-to-raises-sweep/TASK.md) — item 2 keeps its decision, only makes the check cheaper.
- Plan 1 Part 8 P4 — the churn breakdown that started this. Its headline ("validation, not pop-swap") is right
  but too narrow; the fuller version is **per-entity Python, not layout**.
