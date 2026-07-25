# Optimize the entity read/write path — pay back the #29 physics-tick regression

**Created**: 2026-07-25
**Priority**: 2

## Why

#29 landed the single entity write path (`set_data`) and the read-only row view. Both are **keepers** — the
API and its guarantees stay. But they cost robosim **+7–14% on the physics tick**, measured across all robot
counts (robosim `test/e2e/perf-physics-tick`, 12 runs over 3 commits):

| N robots | before #29 (`faca352`, 2 runs) | after #29 (`35c8248`, 4 runs) | delta |
|---|---|---|---|
| 1 | 0.763 | 0.822 | +7.7% |
| 3 | 0.778 | 0.853 | +9.6% |
| 10 | 1.147 | 1.240 | +8.1% |
| 20 | 1.361 | 1.471 | +8.1% |
| 50 | 1.657 | 1.770 | +6.8% |
| 100 | 3.334 | 3.808 | +14.2% |

Ranges are fully disjoint at 5 of 6 N values, and thermal drift is ruled out (a baseline re-run *after* all the
post-#29 runs came back in line with its early ones). The four earlier microecs PRs (#28/#31/#32/#34)
contribute 0–5%, i.e. noise.

Two mechanisms, both micro-benchmarked:

1. **Every entity field read got 2.2× slower** — 491 ns vs 227 ns/op. `__getattr__` (`entity.py:88-91`) runs
   `isinstance(row, np.ndarray)` + `row.setflags(write=False)` on *every* read.
2. **Per-entity `set_data` in a loop is 357× slower than the batch column write** — 402 µs vs 1.1 µs for 100
   entities (`set_data` ×100 + `update()` vs one `qr.field = mask`). Partly a caller problem (robosim **#187**
   uses the wrong surface for a batch op), partly ours: the per-append cost is superlinear.

## What — four fixes, all verified feasible

### 1. Freeze the pool COLUMN once, not the row on every read (the big read win)

Keep a second, permanently read-only **view** of each pool column alongside the writable one. Indexing it
yields a read-only row for free — no `isinstance`, no `setflags` per read.

```python
ro = col.view()          # same memory as the writable pool column
ro.setflags(write=False)
row = ro[3]              # writeable=False, still a LIVE view, in-place writes raise
```

Measured: **75.7 ns/op** for `ro[ix]` vs the ~264 ns the isinstance+setflags pair adds today. Verified: the
writable column keeps `writeable=True` (the SoA/query path is untouched), the row still sees later writes
through the writable alias (it is a view, not a copy), and `row[:] = v` still raises — so #29's guarantee is
preserved exactly, including for shape-`()` fields (a numpy scalar is immutable anyway, so the `isinstance`
special case disappears with the branch).

Owner: `Pool` builds the alias next to `data[name]`, and **must rebuild it whenever a column is reallocated**
(capacity growth / `add_entity` resize) — a stale alias would point at freed memory. That invalidation hook is
the only real work here; get it wrong and reads silently read the old buffer.

### 2. `update()` must not clear the query cache for a SET_DATA-only buffer

`world.update()` (`world.py:153-155`) clears `_cache` whenever the buffer was non-empty. But SET_DATA writes
into existing rows — it changes no pool membership, no `entity_ids`, no pool list. Verified: after a
SET_DATA-only `update()`, a QueryResult held from before is still correct (same ids, len, pools) **and** sees
the new data. Only structural verbs (ADD/REMOVE ENTITY/COMPONENT) invalidate queries.

So: track whether the buffer contained a structural command and only clear `_cache` then. Today any mid-tick
`set_data` nukes every cached query, which is why the regression shows even at N=1.

### 3. Kill the O(buffer) projection scan per append

`CommandBuffer._get_components_state` walks the whole buffer backwards on every append, so N `set_data` calls
in one tick cost O(N²). Task **25** explicitly deferred this — *"optimize the CommandBuffer structure (e.g. an
incremental projected-state index) only if profiling shows a bottleneck — not now."* It now does. Maintain a
per-entity projected component-state map updated at append, making the check O(1).

### 4. Single-field fast path in `set_data`

`set_data` (`entity.py:51-66`) builds a dict-of-dicts to group by component even for the overwhelmingly common
one-field call. Skip the grouping when `len(data) == 1` — same semantics (one field is trivially one component,
and one command is already atomic), less allocation.

## Non-goals

- **The API does not change.** `set_data` stays the only entity write path; the row view stays read-only. This
  is purely about cost. Do not reintroduce eager `e.field = v` / `e.field[:] = v`.
- Batch writes staying on the QueryResult surface is **not** a regression to fix here — two surfaces on
  purpose (#25). Callers doing per-entity `set_data` in a loop over a whole query should use `qr.field = v`;
  that is robosim #187, not this task.

## Validation (tester)

- Correctness first — the whole #29 suite must stay green, unchanged: every eager idiom still raises, reads
  still work (incl. sliced `e.pose[0:3, 3]`), and a failed write still leaves the pool untouched.
- **Fix 1 needs a reallocation test**: force a pool capacity growth (spawn past capacity), then read through an
  Entity handle taken *before* the growth and assert it sees the current data and is still read-only. That is
  the one way the column-alias approach can break.
- **Fix 2 needs a cache-correctness test**: after a SET_DATA-only `update()`, a query must return the new data
  (cache reuse must not serve stale values); after a structural `update()`, the cache must still be dropped.
- Re-run robosim `test/e2e/perf-physics-tick` and confirm the p50s return to the `faca352` band. **Take medians
  of ≥3 runs** — single runs on this harness swing up to 55% on an unchanged commit (N=10 measured 1.085 and
  1.678 for the same code), so a single-run comparison cannot see a 10% effect.

## Relates

- Caused by **#29** (one write path) — the API it introduced is kept; only the cost is in scope.
- Fix 3 is the deferred perf note in **#25**.
- **#26** (low-N field overhead) — same "make the common path cheap" theme.
- Downstream: robosim **#187** (batch `is_colliding` write) — independent and the larger single win at high N.
