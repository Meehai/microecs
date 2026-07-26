# What a real app found — space-shooter audit

**Created**: 2026-07-26
**Type**: Audit
**Re-run**: 2026-07-26 against `b6d2192` (v0.8.2). Every number below was re-measured on that commit;
nothing here is inherited from the first pass. The library is **846 LoC**, 459 unit+integration tests green
(8 xfail, all deliberate), pylint 10.00.
**Scope**: Build a throwaway 2D space shooter on microecs, on purpose, to (a) find corner cases a probe
script cannot reach and (b) grade both advertised patterns — the batch/ECS path and the `Entity`/OOP path —
from the seat of someone writing an app. Then decide whether anything is **structurally** wrong.
**Evidence**: `test/manual/space-shooter/{shooter.py,probes.py,fuzz.py}` and
`test/manual/13-query-cost/`. Both are **gitignored** (`test/manual/` in `.gitignore`), so all numbers and
repros are inlined here. The plan is the artifact; the scripts are disposable — finding 12's pool-cost
measurement was thrown away after its numbers landed in Part 2, and the table there is the record.
**Renumbered on this re-run.** The audit's one structural question — where the eager/deferred line belongs —
was answered and shipped, so it moved out of the body into **Appendix B**. Old → new: Part 3 → Appendix B,
Part 4 → Part 3, Part 5 → Part 4, Part 6 → Part 5, Part 7 → Part 6, Part 8 → Part 7. Finding numbers are
unchanged (`probes.py` uses them); the eight resolved ones are in **Appendix A**.

---

## TL;DR (grug verdict)

**The core is correct. The one structural defect the audit found is fixed. What is left is a query with no
lifetime, a `qr.field` with two types, and a short list of contracts and messages.**

- A shadow-model fuzzer replays **155,859 random structural ops** (600 seeds × 40 ticks) against plain dicts,
  comparing every entity's components and every field value after every `update()`, plus structural
  invariants. **Zero mismatches.** Pop-swap, archetype migration, buffer ordering, pool teardown: correct.
  Hand-rolled SoA ECSs usually have a pop-swap bug. This one does not.
- The app works and is fast enough: **864 live entities at 0.71 ms/tick**.
- **18 probes: 7 FINDING, 9 OK, 2 NOTE.** At the first pass it was 15 FINDING, 2 OK, 1 NOTE — eight findings
  closed the same day by #42, its `_locate` follow-up, and #43.
- **No finding, then or now, is corruption in the storage machinery.** Every one is a contract, an error
  message, composability, or scaling.
- What is left that is *structural*: **the query has no lifetime** (Part 3) and **`qr.field` has two types**
  (Part 4). Both are known ground in the tracker (#27, #37, #38); the app's contribution is showing they are
  not corner cases.
- Scorecard in Part 6: **~8/10 today** (from ~6.5 at the audit), **~9 with Parts 3–4 and the polish list**.

---

## Part 1 — The app, and what the runs measured

### Why a shooter

It forces everything at once, which is the point: mass spawn/despawn (bullets, debris), many archetypes alive
together, capabilities granted and revoked at runtime (shields), per-entity control flow (AI brains) mixed
with batch physics, and cross-archetype interaction (bullets hit anything with health). 18 components, 7 of
them tags.

### What it does

| mode | command | what it exercises |
|---|---|---|
| playable | `python shooter.py` | raylib window, `zip`-row rendering, one `update()` per tick |
| torture | `--headless -n 3000` | deterministic loop, structural invariants checked every 25 ticks |
| churn | `--headless --chaos` | random add/remove component/entity every tick, exceptions catalogued |
| scale | `--headless --heavy 400` | ~900 entities, pool reallocs |

### Numbers (2026-07-26, `b6d2192`)

```
plain    3000 ticks   0.24 ms/tick    87 live /   92 peak / 1027 ever spawned   4-8 archetypes,  40 pools, 0 reallocs
heavy    1500 ticks   0.71 ms/tick   864 live / 1107 peak / 4753 ever spawned   4-8 archetypes, 235 pools, 10 reallocs
chaos    3000 ticks   0.27 ms/tick    15 live /   21 peak /  453 ever spawned   3-10 archetypes, 777 pools
                                     104 exceptions caught: 81 uncommitted-read, 23 has_component-blind
fuzz      600 seeds x 40 ticks       155,859 ops, 0 mismatches vs the shadow model
probes     18 probes                 7 FINDING, 9 OK, 2 NOTE
```

Timings are the median of 2-3 runs (they wander ~5%); entity and pool counts are deterministic and identical
run to run.

Two notes on reading the chaos line. The **81** are a system picking a random id out of `world.live_entities`
and touching an entity spawned this tick — the id is public before the row exists, so it is the app's
mistake, and since the `_locate` follow-up it says so in one sentence (it used to be 98 bare `KeyError: <n>`).
The **23** are finding 2, still live: the guard the exception asks for is the one guard the app cannot write.

### The helpers an app still has to write

The ergonomic headline. None of them is app logic:

| helper | why it is needed | finding |
|---|---|---|
| `f32(shape, default)` | a bare field decl is 90 chars of ceremony for "two floats" | — |
| `join(dst_ids, src_ids, col)` | the only join key is `entity_ids` and there is no gather | 2 |
| `kill_all(world, eids)` | a bare `for` loop: there is no batch despawn | minor |

`Reaper` — the per-tick kill **set** every app had to grow — is **gone** (#43). It is worth naming what
replaced it, because it is the shape of a good fix: nothing was added. `remove_entity` became idempotent
within the tick, and because it drops the id from `live_entities` at the call, `eid in world.live_entities`
answers the kill set's *other* job ("has something already killed this?") for free. Four sites in
`shooter.py` swapped `eid in reaper.dead` for `eid in world.live_entities`, the class and its flush went, and
the deterministic run came out bit-identical.

### Grade on the two patterns

**Batch/ECS — excellent until two archetype sets have to talk.** Five systems are the promised 3 lines
(`qr.position = qr.position + qr.velocity * dt`). Then three walls: **reductions do not exist**
(finding 7), **subset writes are forbidden** by contract (fine, but it must be learnt), and **cross-query
math is banned**, so collision starts with `.numpy()` on both sides (a copy) and ends with a hand-rolled
id-join. The SoA advantage evaporates exactly where a game gets interesting: 30 lines of hand-rolled numpy
for one collision system. *Unchanged by this round — #42 and #43 were both on the entity/world side.*

**`Entity`/OOP — now pleasant, with one trap left.** RMW composes, `e.field = v` works, reads and writes
agree about what a field is, and a repeat despawn is a no-op. The one piece of app-side bookkeeping still
required is `granted_shield`: `has_component` cannot see the buffer (finding 2).

**Mixing them is no longer the sharp edge.** Both write paths now follow numpy's rules and both land
immediately; the split that remains is spawn-vs-write, and it is documented.

---

## Part 2 — The seven live findings

Probe ids are `probes.py`'s. The eight that were resolved on 2026-07-26 are in **Appendix A**, with their
original text, because *why* a trap existed is the part worth re-reading before re-introducing it.

| # | finding | kind | status |
|---|---|---|---|
| 2 | `has_component()` cannot see the command buffer | ergonomics | not filed |
| 6b | the removal leak is exactly "the last row of a surviving pool" | lifetime | sharpens [#40](../todos/open/40-object-dtype-lifetimes/TASK.md) |
| 7 | reductions work at 1 archetype, raise at 2 | contract | extends [#37](../todos/open/37-qrarray-qrfield-one-contract/TASK.md)/[#38](../todos/open/38-array-function-honour-check/TASK.md) |
| 9b | removing the last component reaches the state `add_entity` forbids | state machine | not filed — needs a decision, not a check |
| 12 | an emptied archetype is deleted and rebuilt from scratch | perf/design | not filed |
| 13 | a cold query is O(entities), and the cache never survives a tick | perf | not filed → Part 3 |
| 14 | a mid-tick `update()` voids the queries already taken | perf/correctness | game-shaped [#27](../todos/open/27-stale-queryresult-guard/TASK.md) |

### 2 — `has_component()` cannot see the command buffer

```python
e.add_component(Shield, shield=...)
e.has_component(Shield)       # False -- the staged add is invisible
e.add_component(Shield, ...)  # ValueError: either added twice or exists already
```

and the mirror: after `remove_component`, `has_component` still says `True` and removing again raises. **The
guard the exception tells you to write is the one guard you cannot write.** Two pickups on one tick, or one
system granting a shield while another expires it, are normal events — `shooter.py` carries a
`granted_shield` set purely for this, and the chaos run still produces **23** `either added/removed twice`
from the paths that trusted `has_component`.

`CommandBuffer._get_components_state` (`command_buffer.py:49-63`) already computes this answer and uses it
only to raise. **Fix**: expose it — `has_component(pending=True)`, or make it buffer-aware.

This is now the *only* app-side bookkeeping the shooter still needs, which is what promotes it to the top of
the polish list.

### 6b — the removal leak is exactly the pool's last row

**6b sharpens #40.** The leak happens **only when the removed row is the pool's last row**:
`Pool.remove_entity` does `data[i] = data[size-1]` (`pool.py:52`), which for `i == size-1` is a
self-assignment, so nothing overwrites the dead reference. Any other row is overwritten by the pop-swap and
freed at once; and if the removal empties the pool, the pool is deleted and takes the reference with it. So
the leak is "*the most recently added entity of a surviving archetype*", lasting until the pool refills past
that index. One `data[f][size-1] = <empty>` after the swap closes it — worth recording in #40, because it
narrows both the fix and the test.

### 7 — reductions work at one archetype and raise at two

| expression | 1 pool | 2 pools |
|---|---|---|
| `np.any(qr.health > 0)` | `True` | `TypeError: object of type 'numpy.bool' has no len()` |
| `np.sum(qr.health)` | `5.0` | `TypeError: object of type 'numpy.float32' has no len()` |
| `np.argmin(qr.health)` | `0` | `TypeError: object of type 'numpy.int64' has no len()` |
| `(qr.health > 0).sum()` | `1` | `AttributeError: 'QRField' object has no attribute 'sum'` |

Known ground (#37 type flip, #38 unhonourable functions) plus one addition: **reductions are a third class**,
distinct from #38's row-coupled-N→N and sequence-arg cases. They *are* caught — by
`assert len(part_result) == part.shape[0]` (`qr_field.py:44`) tripping over a scalar — but the message names
numpy's scalar type and nothing about the actual problem. Fold into #38's rejection list with a message like
*"np.sum is a reduction; call qr.f.numpy() first"*. "Is anyone alive?", "who is nearest?", "total damage" are
the questions every game asks, and the shooter's live archetype count swings **4→8 in normal play and 3→10
under churn**, so which branch a call site gets is decided by unrelated spawns elsewhere.

Note the guard is an `assert` on a user-reachable path, so under `python -O` it is gone and the call returns
a malformed `QRField` instead of raising. `primitives.md` already records this as a known gap.

### 9b — a reachable malformed state the library declares illegal

`add_entity([])` is explicitly forbidden, but `remove_component` of the last one gets there anyway: the
entity lives on with `components=[]`, `fields=set()`, invisible to every query except the no-arg one. And it
cannot be repaired — `add_component` on it hits `assert len(components) > 0` (`command_buffer.py:89`), an
**assert**, so under `python -O` the guard is gone and the state is silently reachable. Either forbid
removing the last component (symmetric with `add_entity`) or accept the empty archetype as legal — not both.

Its twin, finding 9 (duplicate components at spawn), was closed by #43 with a one-line `raise`. 9b was
deliberately **not** bundled with it: a check and a decision do not belong in one task.

### 12 — archetype blink rebuilds the pool from scratch

An archetype is deleted the moment its last entity leaves (`world.py:183`) and rebuilt on the next spawn.
One entity toggling one component 100× ⇒ **201 Pool objects**, each allocating `INITIAL_CAPACITY = 100` rows
per field at construction. Measured for the shooter's 6-field bullet archetype:

| | one Pool | 201 pools (100 toggles) | 777 pools (a chaos run) |
|---|---|---|---|
| build time | 4.82 µs | 0.97 ms | 3.74 ms |
| allocated | 4.3 KiB | 0.8 MiB | 3.3 MiB |

In the app: **40** builds for 7 live archetypes in normal play, **235** in the heavy run, **777** in chaos.

Absolute cost is small — under 4 ms and 3.3 MB over 50 s of game — so this is a **design smell, not an
emergency**: unbounded churn in the hot path with no reuse and no free list, scaling with the number of
entities that toggle capabilities, i.e. with the feature the docs advertise ("capabilities are additive").
Cheapest fix: keep an emptied pool (it already knows its fields) instead of deleting it, or start new pools
small and grow.

### 13 — a cold query is O(entities), and the cache never survives a tick

`update()` clears the whole cache whenever the buffer was non-empty (`world.py:158`) — **one structural
change from one entity anywhere in the frame is enough**. Measured in the shooter: **13.1 `query()` calls per
tick, 74% of them cache hits** — so the cache *does* work, but only *within* a tick, for repeats of the same
query. The ~3.4 distinct queries per tick are rebuilt from scratch, every tick, forever.

And a cold rebuild is not O(pools), it is O(entities), because of one line (`world.py:136`):

```python
entity_ids = np.array(sum((self._pool_ids[p] for p in res), []), dtype="int64")
```

a python list concat over every matching entity, then a python-list→ndarray conversion. Split into the four
things `world.query()` actually does, over 3 archetypes:

| N | mask scan | field dicts | **`entity_ids`** | QueryResult | total | vs one motion system |
|---|---|---|---|---|---|---|
| 100 | 0.3 µs | 1.4 µs | **4.2 µs** (39%) | 4.9 µs | 10.8 µs | 0.9× |
| 1000 | 0.2 µs | 0.9 µs | **26.0 µs** (85%) | 3.6 µs | 30.8 µs | 2.6× |
| 10000 | 0.2 µs | 1.0 µs | **257.8 µs** (98%) | 3.7 µs | 262.7 µs | **12.1×** |

A cached query is flat at **0.36 µs** at every N. So at N=10k a cold query costs **12× the system that
consumes it**, and 98% of that is a column most systems never read — motion, wrap, cooldown and decay never
touch `entity_ids`. See Part 3.

### 14 — a mid-tick `update()` silently voids the queries already taken

This is #27, but the game-shaped trigger deserves naming: a shooter *wants* mid-tick commits (spawn a bullet,
commit, let it collide this frame). The moment any system commits, every query an earlier system still holds
points at a freed buffer — stale reads, discarded writes, no error. Measured: 3 entities moved by `+100`
after a mid-tick commit, **0 actually moved**. App-side rule until #27 lands: exactly one `update()` per
tick, at the end.

### Came back clean

- **Tag components** (zero fields ⇒ zero-field pool) work end to end: query, `to_dict`, add/remove.
- **`world.query()` with no arguments returns every live entity.** Undocumented, useful; the invariant
  checker leans on it.
- **numpy `int64` ids** from `qr.entity_ids` work anywhere a python int does — `get_entity`, `remove_entity`,
  and the idempotency set — and do not pollute `live_entities` key types.
- **The storage core**, per the fuzzer: 155,859 ops, 0 mismatches.

---

## Part 3 — Structural #1: a query has no lifetime

### Why

`world.query()` is a memo, not an object: keyed by include+exclude, dropped wholesale when the buffer was
non-empty, holding raw pool slices (`query_result.py:32`), and eagerly building `entity_ids` for callers who
mostly never read it. Two consequences, one already in the tracker and one measured above:

- **#27 (stale qr)** is a *symptom* of this. If the world refreshed the queries it owns at `update()` instead
  of dropping them, staleness would be impossible **by construction** — no generation counter, no "re-query
  after every update" rule to remember. That is finding 14 too.
- **Finding 13**: 98% of a 263 µs query at N=10k is one line building a column nobody asked for, and the
  cache is thrown away every tick even though the world owns every live query.

### What

Two **independent** changes; either can land alone.

1. **Lazy `entity_ids`.** Make it a property built on first access. A cold query drops to O(pools) — from
   263 µs to ~5 µs at N=10k on the numbers above. This is the single highest-value perf change on this list
   for anything above ~1000 entities.
2. **Refresh instead of clear.** At `update()`, re-slice the cached queries the world already holds
   (`self._cache`) instead of `self._cache.clear()`. Held queries stay valid; the cache starts working across
   ticks, not just within one. This only works *because* the world owns every live query — which it already
   does.

Incremental extra, if #13's cost still shows: keep `_pool_ids` as a numpy array per pool with a size counter
(append and pop-swap stay O(1)), so `entity_ids` becomes `np.concatenate([p.ids[:len(p)] ...])` in C instead
of a python list concat.

### How

- `QueryResult.entity_ids` → `@property` with a `_entity_ids` cache; `world.py:136` stops running eagerly;
  the `assert len(entity_ids) == sum(len(p) ...)` in `QueryResult.__init__` moves into the property.
- `World.update()`: replace `self._cache.clear()` (`world.py:158`) with a loop asking each cached
  `QueryResult` to re-slice (`_data`, `_cache`, `_entity_ids`) — drop entries whose pools are gone.
- Then #27's generation guard is unnecessary; if it lands first, it is the right cheap stopgap.
- **Validation**: existing `test_queryresult.py` plus a new test that a qr held across `update()` (with
  growth, shrink, pool death and a new archetype) reads and writes correctly. The shooter's finding-14 repro
  becomes that test.

---

## Part 4 — Structural #2: one contract for `qr.field`

### Why

#37 already states the problem: `_QRArray` at 0–1 pools (whole numpy API live), `QRField` at 2+ (narrow
contract). The app's contribution is that the live archetype count **swings 4→8 in normal play and 3→10 under
churn**, so this is not a corner case — which branch a call site gets is decided by unrelated spawns
elsewhere. Plus finding 7: reductions are a third failing class beyond #38's two.

### What

#37 asks "how do we keep #26's perf win *and* one contract". Suggestion: **do not make `QRField` fast — make
`_QRArray` narrow.** Override `__getitem__`/`__setitem__` on `_QRArray` to enforce the same axis-0 predicate
(`QRField._selects_axis0`, already written and tested by
[#33](../todos/done/33-qrfield-one-key-predicate/TASK.md)), and add the array-protocol guards so reductions
are rejected identically. The fast path stays a real ndarray in memory layout and in every batch op; only the
entity-axis surface is closed. Both branches then behave identically, and #26's win survives untouched.

### How

- `_QRArray.__getitem__/__setitem__`: `if QRField._selects_axis0(key): raise TypeError(<same message>)`.
- `_QRArray.__array_function__` / `__array_ufunc__`: reject non-honourable functions with #38's message, so
  the 1-pool and 2-pool rejection sets are the same set.
- Keep `.numpy()` and `.parts` as they are — they are already the portable spelling.
- **Validation**: a parametrized test that runs the *same* expression against a 1-pool and a 2-pool world and
  asserts identical outcomes (value or exception type). That test is the contract.

---

## Part 5 — What is right (credit where due)

- **The storage core survives a model checker.** 155,859 structural ops, zero divergence against a shadow
  model that checks every component set and every field value after every commit.
- **Eager validation, deferred structural application** (#22). Errors point at the call site, not inside
  `update()`. Most ECS libraries get this backwards and hand you a mystery crash in `commands.apply()`.
- **One mutation-timing rule, and it is the right one**: structure is staged, data lands now. Drawn at
  *what the write does*, not *which object you reached for* — see Appendix B for why that distinction was
  the whole audit.
- **Bitmask archetype keys**, order-independent, query = a mask scan over pools. The standard right answer,
  in 4 lines (`world.py:128-131`), and measurably free: 0.2 µs of a query at any N.
- **Globally unique field names** (#23/#29) buying `set_data(**fields)` without naming the component.
  Constraint-buys-simplicity, the good kind of Tiger Style trade.
- **846 LoC, pylint 10.00.** The whole library is readable in ten minutes, after which pop-swap edge cases
  can be reasoned about from memory. This is the vendoring thesis paying off, and it is why an audit like
  this is even possible.

---

## Part 6 — Scorecard

Column *audit* is where this plan opened (pre-#42, 2026-07-25). Column **today** is `b6d2192`, re-measured.
Column *left* is what Parts 3–4 plus the Part 7 polish list would buy.

| axis | audit | **today** | left |
|---|---|---|---|
| Core correctness (pools, migration, pop-swap) | 9 | **9** | 9.5 |
| API coherence / conceptual integrity | 5 | **8** | 9.5 |
| Ergonomics — batch/ECS path | 7 | **7** | 8.5 |
| Ergonomics — `Entity`/OOP path | 5 | **8** | 9 |
| Error messages / debuggability | 4 | **7.5** | 9 |
| Performance as a game loads it | 7 | **7.5** | 9 |
| Docs | 8 | **8.5** | 9 |
| Size / auditability / deps | 10 | **10** | 10 |
| Fit for robosim (~10² entities, batch physics, 1 update/tick) | 8 | **9** | 9.5 |
| Fit for a game (this audit) | 6 | **7.5** | 9 |

Overall **~6.5 → ~8 → ~9**.

What moved, and why:

- **Coherence 5 → 8** is the whole bet of this plan paying off. The eager/deferred line is now drawn at
  structure-vs-data on every surface, both write paths follow numpy's rules, `_locate` is the single gate for
  every entity entry point, and a repeat despawn has a principled boundary (the tick) rather than an
  arbitrary one. It stops at 8 because a query still has no lifetime and `qr.field` still has two types.
- **`Entity` ergonomics 5 → 8**: RMW composes, `e.field = v` and `+=` work, reads and writes agree, and the
  `Reaper` is deleted. Held back by exactly one thing — finding 2's `granted_shield` bookkeeping.
- **Error messages 4 → 7.5**: findings 1, 4 and 15 all fixed; one message now covers both row-less states.
  What still reads badly: finding 7's reduction `TypeError`s naming a numpy scalar type, 9b's bare
  `AssertionError`, and three different phrasings for "there is no entity with this id" — `entity.py:76`
  "Entity N not in world", `world.py:90` "Entity: N is not in the world (stale)", `world.py:99` "Entity id: N
  not in the world".
- **Batch ergonomics 7 → 7** is the honest one: nothing in this round touched it. #42 and #43 were both on
  the entity/world side. Reductions, the missing gather, and the missing batch despawn are all still there.
- **Perf 7 → 7.5**: #42 deleted the per-read row-freeze tax (491 → 227 ns/op), but finding 13's cold query
  and finding 12's pool churn are untouched. Part 3.1 is what moves this axis.
- **robosim fit 8 → 9**: robosim sits in the sweet spot — ~100 entities, mostly batch physics, one `update()`
  per tick — so findings 13 and 14 do not bite, and the entity path got both faster and correct. None of
  what is left is urgent for robosim.

---

## Part 7 — Action list

Ranked, and only what is still open. Ids are this plan's findings unless a `#` says otherwise. Nothing here
is filed as a task yet.

| order | work | size | note |
|---|---|---|---|
| 1 | **Lazy `entity_ids`** (Part 3.1) | S | biggest measured win: 98% of a cold query at N=10k, which is 12× the system consuming it |
| 2 | **Buffer-aware `has_component`** (2) | S | the information already exists in `_get_components_state`; deletes the last app-side bookkeeping the shooter needs |
| 3 | **Close 9b** | S | an `assert` on a reachable path *and* a decision — forbid removing the last component, or accept the empty archetype. Not both |
| 4 | **Refresh-not-clear queries** (Part 3.2) | M | subsumes #27 and finding 14; do after 1 |
| 5 | **One contract for `qr.field`** (Part 4) | M | concrete resolution for open P1 #37 + #38, including finding 7's reductions |
| 6 | **Pool reuse** (12) | S | design smell, low absolute cost today (3.74 ms / 3.3 MiB across a chaos run) |
| 7 | **Docs: object fields are an escape hatch** (6) | S | name the one rule they opt out of: nothing can check what you stored |
| 8 | **One phrasing for "no such entity"** | XS | three today, across `entity.py` and two sites in `world.py` |

Append-only, no new task needed: **6b → #40** (the leak is only the pool's last row), **7 → #38** (reductions
are a third class), **14 → #27** (game-shaped trigger + repro).

Candidate new tasks: 2, 9b, 12, 13/Part 3 (lazy ids — the one worth filing first), Part 4 (or fold into #37
as the chosen approach).

---
---

# Appendix A — what was, and what is

The eight findings that were live when this audit opened and are not any more, each with its original text
plus what changed. Kept because *why* a trap existed is the part worth re-reading before re-introducing it.
All eight closed on 2026-07-26: six with [#42](../todos/done/42-set-data-eager-again/TASK.md) and its
`_locate` follow-up, two with
[#43](../todos/done/43-duplicate-components-and-idempotent-remove/TASK.md).

| # | finding | closed by |
|---|---|---|
| 1 | uncommitted entity: 4 of 5 reads raise a bare `KeyError(eid)` | the `_locate` follow-up |
| 3 | `remove_entity()` is not idempotent | #43 subtask 2 |
| 4 | a dangling `Entity` reports "not committed yet" | the `_locate` follow-up |
| 5 | RMW through `Entity` silently loses writes | #42 |
| 6 | an `object` field bypasses the write rules | restated by #42 (docs item survives) |
| 8 | the two write paths disagree on dtype validation | #42, by unification |
| 9 | `add_entity([Pos, Pos, Vel])` builds a pool with a duplicate field | #43 subtask 1 |
| 15 | a typo'd field name in `set_data` raises a bare `KeyError` | #42 |

### 9 — `add_entity([Pos, Pos, Vel])` builds a pool with a duplicate field

`add_entity([Pos, Pos, Vel])` is accepted and builds `pool.fields == ['position','position','velocity']`.
Every write to that field is then done twice, forever, for every entity of that archetype — **the first
caller's argument list decides the pool's shape**. Rejected by one `len(components) != len(set(components))`
check in `_validate_components`.

> **Fixed by #43 subtask 1**, exactly as proposed — one `raise` beside the empty/unknown checks, so both
> spawn gates (`add_entity`'s pre-pass and `CommandBuffer.append`) inherit it. Nothing was corrupted while it
> lasted: `pool.data` is a dict, so both writes hit one column. What it cost was double work per row forever,
> a `to_dict()` that round-tripped the malformed list back in, and an archetype whose shape was decided by
> whoever spawned into it first.

### 3 — `remove_entity()` is not idempotent

`ValueError: Entity: 0 not in live entities` on the second call in a tick. A kill can be decided by three
systems (damage, TTL, out-of-bounds) and two bullets hitting one asteroid is the most common event in the
game. Not a bug — but a remove of an already-dead id is a *no-op request*, not a programming error, and
making it idempotent deletes a class of app-side bookkeeping.

> **Fixed by #43 subtask 2** — but idempotent **within a tick only**, which is a sharper rule than this
> finding asked for. Within a tick system order is arbitrary, so a repeat kill is a race; across an
> `update()` it is a stale reference, and silence would hide it. `CommandBuffer.removed_this_tick` answers
> "killed since the last commit?" in O(1) and clears with the buffer. Two things worth keeping: the guard
> **cannot** live in `CommandBuffer.append` — that was the first attempt and it was dead code, because
> `append`'s blanket liveness gate fires before the dispatch and, more fundamentally, `append` can only raise
> or stage, never "do nothing". And an intermediate attempt gated on `0 <= eid <= _last_id` instead; the
> buffer-backed rule replaced it and is *smaller*, since never-spawned and long-dead collapse into one answer
> with no id arithmetic.

### 5 — read-modify-write through `Entity` does not compose

```python
def damage(e, n):
    e.set_data(health=e.health - np.float32([n]))
damage(e, 3); damage(e, 4); world.update()      # health 10 -> 6.0, not 3.0
```

Reads are eager (committed state), writes are buffered, so the second `set_data` reads the **pre-tick** value
and overwrites the first command. Silent. This is the general form of
[#1](../todos/open/1-bounce-impulse-accumulator/TASK.md)'s "flipping is non-composable": *any* accumulator
written through `Entity` keeps only the last contribution. Worst with physics substeps (one system running
twice per frame) and with two systems that both damage. This was the symptom of Appendix B's structural item,
not an isolated bug.

> **Fixed by #42.** Writes are eager, so the second `damage` reads the first one's result:
> `damage(3); damage(4)` leaves 3. `e.health -= np.float32(n)` composes too. Pinned by
> `test_entity_read_modify_write_composes_within_one_tick` and its `+=` twin.

### 15 — a typo'd field name gives a bare `KeyError`

`e.set_data(helth=...)` → `KeyError: 'helth'`. The same mistake one level up (`set_data(shield=...)` for a
component the entity lacks) gives a good `ValueError`. One `if field not in self._world_field_to_component:
raise` with the valid field list matches the rest of the library.

> **Fixed by #42.** `Entity._locate(names=...)` is the one gate for read, write and `set_data`: a name that
> is not a field of this entity's pool raises `AttributeError` listing the components *and* the valid fields.
> Both spellings of the mistake now land in the same place.

### 8 — two write paths, two validation contracts

```python
e.set_data(health=np.array([1.5]))   # TypeError: Expected dtype float32, got float64   (strict)
qr.health = np.array([[1.5]])        # accepted, silently cast
qr.color  = 3.9                      # int32 field -> stored [3,3,3,3], no warning
```

[#28](../todos/done/28-queryresult-setattr-validate/TASK.md) made `QueryResult.__setattr__` validate *names*;
dtypes fall through to numpy's unsafe casting. Whether the batch path *should* validate is a real call
(per-write dtype checks cost something in the hot path, and silent casting is what a numpy user expects). But
the two halves should not disagree in **silence** —
[#20](../todos/done/20-add-entity-eager-dtype-crash/TASK.md) exists because a silent shape surprise was
judged unacceptable on the other path.

> **Resolved by #42 — and the other way round than this finding assumed.** The entity path was loosened to
> match the batch path instead of the batch path being tightened: `e.field = v` now accepts exactly what
> `qr.field = v` accepts (converted to the field's dtype, broadcast into its shape — lists, tuples, scalars,
> a silent float64→float32 cast). Pinned as a *parity* test
> (`test_entity_write_agrees_with_the_batch_write_path`) so the two cannot drift again. Rationale: the batch
> path is where most writes happen and it has always followed numpy, so numpy's rules are the ones users
> already know. The disagreement did not vanish, it **moved**: `add_entity` / `add_component` are still
> strict (exact ndarray, exact dtype, exact shape), because those *declare* a row rather than update one.
> Deliberate, documented in `primitives.md`, and the remaining candidate for a follow-up if the split still
> annoys — but it is now spawn-vs-write, not `Entity`-vs-`QueryResult`.

### 6 — `object` fields are outside every rule

```python
e.brain.item()["state"] = "flee"     # in the pool immediately: no command, no update(), no validation
```

`row.setflags(write=False)` stops `e.brain[0] = x` but not mutation *through* the reference it just handed
out. `object` fields are also the only way to hold per-entity structure (an AI state dict), so this is
load-bearing, not exotic: the shooter's enemies rely on it every tick.

> **Restated by #42.** The timing half is gone: every data write is eager now, so mutating through the
> reference is no longer "outside the rule" — it *is* the rule, just without the dtype/shape check (there is
> nothing to check on an opaque object). `setflags(write=False)` no longer exists to be circumvented, and
> `e.brain = {...}` works directly. What remains is narrower and still worth documenting — item 7 on the
> action list: an object field is the one place where the library cannot tell you that you stored the wrong
> thing. #40 (lifetimes) and 6b (still live) are unaffected.

### 1 — an uncommitted entity answers four questions with a bare `KeyError`

`add_entity` puts the id in `world.live_entities` immediately and `get_entity()` hands out an `Entity`.
Reading a *field* gave the good message; nothing else did:

```
entity.position    -> AttributeError: Entity 0 not committed yet. Call `world.update()` (reading 'position')
has_component()    -> KeyError: 0
get_components()   -> KeyError: 0
get_fields()       -> KeyError: 0
to_dict()          -> KeyError: 0
```

The guard existed once, in `Entity.__getattr__`. This produced **98 undebuggable `KeyError: <number>`** in one
3000-tick chaos run, from a system that picked a random id out of `world.live_entities` — a public dict — and
asked `has_component`.

> **Fixed by the `_locate` follow-up to #42**, exactly as proposed. `get_components` and `get_fields` call
> `_locate(names=[])` instead of indexing `_eid_to_pool_ix`; `has_component` and `to_dict` inherit the fix for
> free, since they call through those two. All five now answer the same way, and the same chaos run reports
> 81 of them as one readable sentence.
>
> **Watch the nesting it creates**: `_locate`'s *other* raise (unknown field) builds its message by calling
> `get_components()`, which calls `_locate()`. Safe only because `names=[]` can never fail the field check. A
> call that *can* fail, added to that message, turns this into infinite recursion. Pinned by
> `test_entity_without_a_row_raises_attributeerror_everywhere` (7 entry points × 2 row-less states).

### 4 — a dangling `Entity` misdiagnoses itself

```python
e = world.get_entity(player); world.remove_entity(player); world.update()
e.position   # was: AttributeError: Entity 3 not committed yet. Call `world.update()`
```

Wrong in the most confusing way: the entity was dead and `update()` would never help.

> **Fixed by the same follow-up, but *not* by the branch this finding proposed.** Instead of asking
> `live_entities` and printing two messages, `_locate` prints one message that is true in both states, with
> the advice made conditional: *"Entity N not in world. Call `world.update()` if it was just added."* The
> false claim is gone, and the despawn case is not sent hunting for a missing commit. **Dev's call**: one
> accurate sentence beats two branches over a state the caller already knows.

### Reversed on purpose — spawn-then-modify in one tick

The audit's "came back clean" list had this: `add_entity` → `get_entity` → `set_data` → one `update()` works,
the buffer resolves the ordering, *keep this working*.

> **Deliberately reversed by #42 (dev's call).** It now raises: an uncommitted spawn has no row, and a write
> goes where the row is and never consults the buffer. Reads and writes agreeing was judged worth more than
> the convenience, the alternative being a buffer scan on every write. Spawn data belongs in `add_entity`'s
> kwargs; robosim never hit this path. Structural ops (`add_component`, `remove_component`, `remove_entity`)
> on an uncommitted spawn still work — those *do* go through the buffer.

---

# Appendix B — the mutation-timing decision (this plan's old Part 3)

Kept in full because it is the argument #42 was built on, and
[#42's task](../todos/done/42-set-data-eager-again/TASK.md) cites "Plan 2 Part 3" by name.

> **Decided and shipped 2026-07-26 as #42.** Accepted, and taken one step further than proposed: the
> read-only row view went too, so `e.field = v` / `e.field[:] = v` came back and `set_data` became the
> transactional wrapper over `setattr`. Reason: that view *was* the measured per-read tax (491 vs 227 ns/op),
> and the guarantee it bought was not worth a 2.2× read.

### The argument

The line was **`Entity` = deferred, `QueryResult` = eager**. It should be **structure = deferred, data =
eager**.

Buffering *structural* change is non-negotiable: moving a row between pools invalidates iteration in flight.
Every serious ECS defers it (Unity ECB, flecs, Bevy `Commands`) and so does microecs — correctly.

Buffering a *data* write buys nothing. `pool.data[f][ix] = v` moves no rows and invalidates nothing. The batch
path proved it: `qr.position = ...` wrote straight into pool memory, eagerly, and nobody complained.

And deferral did not buy the thing deferral usually buys — snapshot / simultaneity semantics — because
`update()` is deliberately **not atomic** ([#22](../todos/done/22-fully-eager-staging/TASK.md)), staged values
are **references, not snapshots** ([#39](../todos/open/39-staged-writes-snapshot/TASK.md)), and the batch path
was already order-dependent.

**So the cost of deferral was paid and the benefit was not collected.** That was the structural defect: not
"deferred is wrong", but "half-deferred delivers neither model". What it cost, concretely: finding 5 — silent
lost updates — and an `Entity` that was not a coherent view of a row but a write-only mailbox with a
read-through cache of the previous tick.

### What shipped, versus what this Part proposed

Four things landed differently. Read them as corrections to the proposal, not as separate decisions:

1. **`_locate(names=...)` became the single gate.** One helper resolves `(pool, row)` *and* checks that every
   name is a field of that pool; `__getattr__`, `__setattr__` and `set_data` all go through it. This Part
   only asked for eager writes; the gate is what made read and write stop disagreeing about what a field is,
   and it fixed findings 15 and (half of) 1 as a side effect.
2. **Writes follow numpy, not the strict schema check.** Pinned by a parity test. This resolved finding 8 by
   unification, in the opposite direction from what finding 8 proposed. `add_entity` / `add_component` stay
   strict.
3. **A write never consults the command buffer.** Spawn-then-write raises instead of patching the staged
   `ADD_ENTITY` command — the proposal wanted the opposite. A field whose component is only pending raises; a
   field pending removal is written and then dropped with the component. One rule, no scan.
4. **`set_data` kept the transaction, and pays for it only when it must.** One field → straight write
   (numpy converts and shape-checks the whole RHS before copying, so a single field cannot be half written):
   **free**, 1.04× the pre-change loop. Two or more → convert and fit-check every value first, then write:
   1.64× at three fields, all of it on a cold path. No rollback exists because after validation nothing can
   fail. The naive always-`np.broadcast_to` version measured **4.5–7.3×**.

What got deleted: the `SET_DATA` command type, its branch in `update()` and in `CommandBuffer.append`,
`_do_set_data`, and the rollback dance in `Entity.set_data`. Net −40 lines of library.

### The alternative, rejected

Commit to the *other* side properly: deep-copy staged values (#39), make `update()` atomic (reversing #22),
and give `Entity` reads a buffer-aware view so RMW composes. That is a bigger, slower library and the wrong
shape for a numpy-batch engine — but it is coherent. **The one option that should not have survived was the
middle**, and it did not.

---

# Appendix C — timeline

| date | what | effect on this plan |
|---|---|---|
| 2026-07-25 | audit run: shooter + 18 probes + fuzzer | 15 findings, 2 structural items, scorecard ~6.5 |
| 2026-07-26 | [#36](../todos/done/36-optimize-entity-read-write-path/TASK.md) closed | the #29 read tax accepted — then made moot by #42 |
| 2026-07-26 | [#42](../todos/done/42-set-data-eager-again/TASK.md) shipped | Appendix B's decision; findings 5, 8, 15 fixed, 6 restated |
| 2026-07-26 | `_locate` follow-up (untracked, 4 lines on #42) | findings 1 and 4 fixed |
| 2026-07-26 | [#43](../todos/done/43-duplicate-components-and-idempotent-remove/TASK.md) shipped | findings 9 and 3 fixed; the shooter's `Reaper` deleted |
| 2026-07-26 | plan re-run against `b6d2192` | 7 FINDING / 9 OK / 2 NOTE; scorecard ~8; this rewrite |

Two evidence-side repairs were needed for the re-run, both recorded here because a broken harness reports
"no findings" just as loudly as a fixed library:

- **`fuzz.py` had been dead since #42** — it wrote to entities spawned in the same tick, which now raises.
  Repairing it surfaced a *model* bug too: it applied ops in call order, but data writes land before the
  buffer replays, so a component removed and re-added in one tick re-imposes its staged data over an eager
  `set_data` to the same field. Not a storage bug; the shadow now models it. 155,859 ops, 0 mismatches.
- **`probes.py` had four hardcoded verdicts** (3, 4, 8, 15) that still printed FINDING against behaviour that
  had been fixed. All four now compute their verdict from what the library actually does, and probe 3 checks
  both halves of the tick-scoped rule rather than just the repeat.
