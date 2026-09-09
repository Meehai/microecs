# Optimize the entity read/write path — pay back the #29 physics-tick regression

**Created**: 2026-07-25
**Closed**: 2026-07-26
**Priority**: 2

## Verdict (2026-07-26): the cost is ACCEPTED — closing without the optimization

#29 (single entity write path + read-only row view) costs robosim a **flat +6-7% on the physics tick at N≥3**,
after robosim #187 recovered the rest. Real and systematic — but **below what this metric can resolve**, and with
plenty of headroom left, so it is not worth chasing. Medians of 4 runs on `134ed0e` vs the pre-#29 `faca352`,
next to the range the *same* p50 wandered over the previous month across commits with no perf intent:

| N robots | pre-#29 | after #187 | residual | [ms] | commit-to-commit range Jun22→Jul25 | % of the 16.67ms budget |
|---|---|---|---|---|---|---|
| 1 | 0.763 | 0.742 | −2.8% | −0.021 | 0.695 – 1.107 (59%) | 4% |
| 3 | 0.778 | 0.815 | +4.8% | +0.037 | 0.750 – 1.195 (59%) | 5% |
| 10 | 1.147 | 1.218 | +6.2% | +0.071 | 1.085 – 1.677 (55%) | 7% |
| 20 | 1.361 | 1.462 | +7.4% | +0.101 | 1.313 – 1.855 (41%) | 9% |
| 50 | 1.657 | 1.754 | +5.9% | +0.097 | 1.547 – 2.440 (58%) | 10% |
| 100 | 3.334 | 3.567 | +7.0% | +0.233 | 3.049 – 4.987 (64%) | 21% |

Why accepted:

- **Inside the metric's own wander.** Commits with no perf intent move this p50 by 40-64%; 6-7% does not clear
  that. It also flips sign with the choice of baseline: vs `329cadc` (best row ever, 1.085 at N=10) the residual
  reads +12.7%, vs `faca352` +6.2%, vs `c475ff0` two commits earlier **−10%**. Not a number to optimize against.
- **Headroom on the physics tick itself.** Worst case (N=100) uses 3.6 of the 16.67ms budget → **4.6×** spare;
  the default 2-3 robots use 0.86ms → **19×**. No cliff nearby.
- **Scale.** N=100 went 1142ms → 3.6ms since May (317×). This gives 0.23ms of that back.
- **It buys a guarantee.** One write path, and a pool no stray `e.field[:] = v` can corrupt.

Where a per-read tax *should* hurt most is low N, where fixed per-tick overhead dominates — and that is exactly
where it is gone (N=1 fully recovered, N=3 +4.8%), so the default config is clean.

Kept in mind: it is a **per-read tax on every entity field read** (2.2× slower: 491 vs 227 ns/op), so it scales
with any future code that walks per-entity handles in a hot loop. The four recipes below stay recorded for
whoever profiles next. **Reopen if** a profile puts the physics tick at the top — or if fix 3's superlinear cost
starts to bite (that one is a scaling hazard, not a fixed tax).

## Re-measured 2026-07-26 (post-`#42`) — two of the four fixes are moot, two new numbers stand

`#42` reverted `#29`'s design (eager `e.field = v` is back; the row view is no longer frozen), so **fix 1's
mechanism is gone** — `__getattr__` no longer runs `isinstance` + `setflags` — and **fix 4 shipped**
(`entity.py:54-57` is the single-field fast path). What is left, measured on `975097c`
(`test/manual/bench-compare/paths.py`, N=20k, ns/op):

| op | ns | note |
|---|--:|---|
| `pool.data[f][ix]` | 110 | the raw numpy floor |
| `world.get_entity(eid)` | 74 | dict lookup; `Entity` cached per id |
| `ent._locate(['position'])` | 144 | the shared guard `#42` introduced |
| `ent.position` (read) | 320 | vs 227 pre-`#29`, 491 under `#29` — **between the two** |
| `ent.position = v` (write) | 471 | |
| `ent.set_data(position=v)` | 610 | |
| `e.position = e.position + e.velocity*dt` | 2018 | one "entity tick" |

Two live items, both cheap, neither worth reopening this task on its own:

1. **`_locate` single-name fast path.** ~89 of its 144 ns is `pool.fields_set.issuperset([name])`; a plain
   `name in pool.fields_set` is **40 ns** (measured). `__getattr__`/`__setattr__` pass exactly one name, so a
   membership test with the existing error message on the miss branch is ~15% off every entity read/write.
2. **`set_data(f=v)` is now slower than `e.f = v`** (610 vs 471) for the identical effect — it pays for the kwargs
   dict. The docs already prefer the attribute for one field; the ~30% figure is now stated there
   (`docs/source/primitives.md`, `benchmarks.md`) so nobody has to guess.

Still accepted, same reasoning as above: both are single-digit-percent on a path the docs tell you to keep out of
hot loops. Recorded here so the next profiler starts from measurements, not from the `#29`-era table.

## The four fixes, all verified feasible

### 1. Freeze the pool COLUMN once, not the row on every read (the big read win)

`__getattr__` (`entity.py:88-91`) runs `isinstance(row, np.ndarray)` + `row.setflags(write=False)` on *every*
read. Instead keep a second, permanently read-only **view** of each pool column alongside the writable one;
indexing it yields a read-only row for free.

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
the only real work here; get it wrong and reads silently read the old buffer. This is the one fix with a
measured mechanism behind the residual above; do it first and re-measure before touching 2-4.

### 2. `update()` must not clear the query cache for a SET_DATA-only buffer

`world.update()` (`world.py:153-155`) clears `_cache` whenever the buffer was non-empty. But SET_DATA writes
into existing rows — it changes no pool membership, no `entity_ids`, no pool list. Verified: after a
SET_DATA-only `update()`, a QueryResult held from before is still correct (same ids, len, pools) **and** sees
the new data. Only structural verbs (ADD/REMOVE ENTITY/COMPONENT) invalidate queries. So: track whether the
buffer contained a structural command and only clear `_cache` then.

### 3. Kill the O(buffer) projection scan per append

`CommandBuffer._get_components_state` walks the whole buffer backwards on every append, so N `set_data` calls
in one tick cost O(N²). Task **25** deferred this explicitly — *"only if profiling shows a bottleneck"*.
Maintain a per-entity projected component-state map updated at append, making the check O(1). **This is the
one that can bite suddenly**: robosim no longer stages per-entity writes in a loop (#187), but any caller that
does pays superlinearly — measured 402µs for `set_data` ×100 + `update()` vs 1.1µs for one `qr.field = mask`.

### 4. Single-field fast path in `set_data`

`set_data` (`entity.py:51-66`) builds a dict-of-dicts to group by component even for the overwhelmingly common
one-field call. Skip the grouping when `len(data) == 1` — same semantics (one field is trivially one component,
and one command is already atomic), less allocation.

## Non-goals

- **The API does not change.** `set_data` stays the only entity write path; the row view stays read-only. This
  was always purely about cost. Do not reintroduce eager `e.field = v` / `e.field[:] = v`.
- Batch writes staying on the QueryResult surface is **not** a regression — two surfaces on purpose (#25).
  Callers doing per-entity `set_data` in a loop over a whole query should use `qr.field = v`.

## Validation (if reopened)

- Correctness first — the whole #29 suite must stay green, unchanged.
- **Fix 1 needs a reallocation test**: force a pool capacity growth (spawn past capacity), then read through an
  Entity handle taken *before* the growth and assert it sees the current data and is still read-only. That is
  the one way the column-alias approach can break.
- **Fix 2 needs a cache-correctness test**: after a SET_DATA-only `update()`, a query must return the new data;
  after a structural `update()`, the cache must still be dropped.
- Re-run robosim `test/e2e/perf-physics-tick`. **Take medians of ≥3 runs** — a single-run comparison cannot see
  a 10% effect on this harness (N=10 measured 1.085 and 1.678 for the same code).

## Relates

- Caused by **#29** (one write path) — the API it introduced is kept; only the cost was ever in scope.
- Fix 3 is the deferred perf note in **#25**.
- **#26** (low-N field overhead) — same "make the common path cheap" theme.
- Upstream: robosim **#187** (batch `is_colliding` write), closed 2026-07-25 — the larger single win at high N.
