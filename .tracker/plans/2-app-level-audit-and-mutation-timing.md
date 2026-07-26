# What a real app found — space-shooter audit + the mutation-timing decision

**Created**: 2026-07-26
**Type**: Audit + design decision
**Scope**: Build a throwaway 2D space shooter on microecs, on purpose, to (a) find corner cases a probe
script cannot reach and (b) grade both advertised patterns — the batch/ECS path and the `Entity`/OOP path —
from the seat of someone writing an app. Then decide whether anything is **structurally** wrong.
**Code state**: verified against `pkg/microecs/microecs/*.py` at commit `cdab031` (files unchanged since
2026-07-25 17:17). Every number below was measured against that source.
**Takes as given**: [#36](../todos/done/36-optimize-entity-read-write-path/TASK.md) closed 2026-07-26 —
the #29 read/write tax is **accepted**, not to be chased. Part 3 is written to respect that decision.
**Superseded on exactly that point, 2026-07-26**: Part 3 was accepted and **shipped** as
[#42](../todos/done/42-set-data-eager-again/TASK.md), which *deletes* what #36 measured — the read-only row
view is gone, so there is no per-read tax left to accept. Findings **5, 8, 15** are resolved by it, **1** is
half resolved, **6** is restated, and the "keep spawn-then-write working" line in *Came back clean* was
deliberately reversed. Each is marked inline below; Part 3 carries the shipped-vs-proposed diff.
**Evidence**: `test/manual/space-shooter/{shooter.py,probes.py,fuzz.py,FINDINGS.md}`. That directory is
**gitignored** (`test/manual/` in `.gitignore`), so all numbers and repros are inlined here. The plan is the
artifact; the scripts are disposable.

---

## TL;DR (grug verdict)

**The core is correct. The library is sound and under-specified. Nothing needs rewriting; one line needs
re-drawing.**

- A shadow-model fuzzer replayed **151,008 random structural ops** (600 seeds × 40 ticks) against plain
  dicts, comparing every entity's components and every field value after every `update()`, plus structural
  invariants. **Zero mismatches.** Pop-swap, archetype migration, buffer ordering, pool teardown: correct.
  Hand-rolled SoA ECSs usually have a pop-swap bug. This one does not.
- The app works and is fast enough: **865 live entities at 0.75 ms/tick**, 8–14 archetypes live at once.
- **15 findings** from 18 probes. All are contracts, error messages, composability or scaling — none is
  corruption in the storage machinery.
- **One structural defect**: the eager/deferred line is drawn at *`Entity` vs `QueryResult`* when it should
  be drawn at *structure vs data*. Buffering a **data** write moves no rows, so it buys nothing — and it
  costs composability: two `set_data` calls on one field in a tick silently keep only the last. Part 3.
  **→ Shipped 2026-07-26 as #42.** The line is now structure-vs-data on every surface: `e.field = v` writes
  the pool at the call, `set_data` is the multi-field transaction over it, and the `SET_DATA` command is
  gone. Net −40 lines of library.
- Two smaller structural items: the query has no lifetime (Part 4), and `qr.field` has two types (Part 5).
- Scorecard in Part 7: **~6.5/10 today, ~7.5 after the polish list, ~9 with Parts 3–5.** The polish list
  does *not* move conceptual integrity. Only Part 3 does. **With Part 3 shipped: ~8 today** (the coherence
  axis moved; Parts 4–5 are what is left).

---

## Part 1 — The app, and what the runs measured

### Why a shooter

It forces everything at once, which is the point: mass spawn/despawn (bullets, debris), many archetypes
alive together, capabilities granted and revoked at runtime (shields), per-entity control flow (AI brains)
mixed with batch physics, and cross-archetype interaction (bullets hit anything with health). 18 components,
7 of them tags, 8–14 pools live.

### What it does

| mode | command | what it exercises |
|---|---|---|
| playable | `python shooter.py` | raylib window, `zip`-row rendering, one `update()` per tick |
| torture | `--headless -n 3000` | deterministic loop, structural invariants checked every 25 ticks |
| churn | `--headless --chaos` | random add/remove component/entity every tick, exceptions catalogued |
| scale | `--headless --heavy 400` | ~900 entities, pool reallocs |

### Numbers

```
plain    3000 ticks   0.33 ms/tick    87 live / 92 peak / 1027 ever spawned    4-8 archetypes, 40 pools built
heavy    1500 ticks   0.75 ms/tick   865 live / 1107 peak / 4732 ever spawned  232 pools built, 10 reallocs
chaos    3000 ticks   0.32 ms/tick   747 pools built; 98 bare KeyError + 33 "removed twice" caught
fuzz      600 seeds x 40 ticks       151,008 ops, 0 mismatches vs the shadow model
probes     18 probes                 15 FINDING, 2 OK, 1 NOTE
```

### The three helpers every app must write first

This is the ergonomic headline. None of them is app logic:

| helper | why it is needed | finding |
|---|---|---|
| `f32(shape, default)` | a bare field decl is 90 chars of ceremony for "two floats" | — |
| `Reaper` (a kill set) | `remove_entity` is not idempotent; two bullets hit one asteroid every tick | 3 |
| `join(dst_ids, src_ids, col)` | the only join key is `entity_ids` and there is no gather | 2 |

### Grade on the two patterns

**Batch/ECS — excellent until two archetype sets have to talk.** Five systems are the promised 3 lines
(`qr.position = qr.position + qr.velocity * dt`). Then three walls: **reductions do not exist**
(finding 7), **subset writes are forbidden** by contract (fine, but it must be learnt), and **cross-query
math is banned**, so collision starts with `.numpy()` on both sides (a copy) and ends with a hand-rolled
id-join. The SoA advantage evaporates exactly where a game gets interesting. 30 lines of hand-rolled numpy
for one collision system.

**`Entity`/OOP — pleasant for one entity, two silent traps.** The player and the AI read well. But RMW does
not compose (finding 5) and `has_component` cannot see the buffer (finding 2). The working discipline is
"one `set_data` per entity per tick, plus a shadow set of pending grants" — which is what `shooter.py` does.

**Mixing them is the sharp edge**: different timing *and* different validation (finding 8).

---

## Part 2 — The 15 findings

Probe ids are `probes.py`'s. "new" = not in the tracker or docs today.

| # | finding | kind | status |
|---|---|---|---|
| 2 | `has_component()` cannot see the command buffer | ergonomics | new |
| 3 | `remove_entity()` is not idempotent | ergonomics | new |
| 4 | a dangling `Entity` reports "not committed yet" | error quality | new |
| 6b | the removal leak is exactly "the last row of a surviving pool" | lifetime | sharpens [#40](../todos/open/40-object-dtype-lifetimes/TASK.md) |
| 7 | reductions work at 1 archetype, raise at 2 | contract | extends [#37](../todos/open/37-qrarray-qrfield-one-contract/TASK.md)/[#38](../todos/open/38-array-function-honour-check/TASK.md) |
| 9 | `add_entity([Pos, Pos, Vel])` builds a pool with a duplicate field | validation | new |
| 9b | removing the last component reaches the state `add_entity` forbids | state machine | new |
| 12 | an emptied archetype is deleted and rebuilt from scratch | perf/design | new |
| 13 | an uncached query is O(entities); every tick uncaches everything | perf | new |
| 14 | a mid-tick `update()` voids the queries already taken | perf/correctness | game-shaped [#27](../todos/open/27-stale-queryresult-guard/TASK.md) |
| | *— resolved by #42, detail at the end of this Part —* | | |
| 5 | RMW through `Entity` silently loses writes | correctness trap | **fixed** by #42 |
| 15 | a typo'd field name in `set_data` raises a bare `KeyError` | error quality | **fixed** by #42 |
| 8 | the two write paths disagree on dtype validation | contract | **fixed** by #42 (unified) |
| 6 | an `object` field bypasses the write rules | contract hole | restated by #42 |
| 1 | uncommitted entity: 4 of 5 reads raise a bare `KeyError(eid)` | error quality | **half fixed** by #42 |

### 2 — `has_component()` cannot see the command buffer

```python
e.add_component(Shield, shield=...)
e.has_component(Shield)       # False -- the staged add is invisible
e.add_component(Shield, ...)  # ValueError: either added twice or exists already
```

and the mirror: after `remove_component`, `has_component` still says `True` and removing again raises. **The
guard the exception tells you to write is the one guard you cannot write.** Two pickups on one tick, or one
system granting a shield while another expires it, are normal events — `shooter.py` carries a
`granted_shield` set purely for this, and the chaos run still produced **33 `either removed twice`** from
the paths that trusted `has_component`.

`CommandBuffer._get_components_state` (`command_buffer.py:48-64`) already computes this answer and uses it
only to raise. **Fix**: expose it — `has_component(pending=True)`, or make it buffer-aware.

### 3 — `remove_entity()` is not idempotent

`ValueError: Entity: 0 not in live entities` on the second call in a tick. A kill can be decided by three
systems (damage, TTL, out-of-bounds) and two bullets hitting one asteroid is the most common event in the
game. Not a bug — but a remove of an already-dead id is a *no-op request*, not a programming error, and
making it idempotent deletes a class of app-side bookkeeping.

### 4 — a dangling `Entity` misdiagnoses itself

```python
e = world.get_entity(player); world.remove_entity(player); world.update()
e.position   # AttributeError: Entity 3 not committed yet. Call `world.update()`
```

Wrong in the most confusing way: the entity is dead and `update()` will never help. The write path gets it
right (`ValueError: ... not in live entities`) because it goes through the buffer, which checks
`live_entities`. The read path can ask the same question — `Entity` holds `_world_command_buffer`, which
holds `.world` (verified). **Fix**: in `live_entities` ⇒ "not committed yet", else ⇒ "removed".

### 6b — the removal leak is exactly the pool's last row

**6b sharpens #40.** The removal leak happens **only when the removed row is the pool's last row**:
`Pool.remove_entity` does `data[i] = data[size-1]` (`pool.py:51-53`), which for `i == size-1` is a
self-assignment, so nothing overwrites the dead reference. Any other row is overwritten by the pop-swap and
freed at once; and if the removal empties the pool, the pool is deleted and takes the reference with it. So
the leak is "*the most recently added entity of a surviving archetype*", lasting until the pool refills past
that index. One `data[f][size-1] = <empty>` after the swap closes it — worth recording in #40, because it
narrows the fix and the test.

### 7 — reductions work at one archetype and raise at two

| expression | 1 pool | 2 pools |
|---|---|---|
| `np.any(qr.health > 0)` | `True` | `TypeError: object of type 'numpy.bool' has no len()` |
| `np.sum(qr.health)` | `5.0` | `TypeError: object of type 'numpy.float32' has no len()` |
| `np.argmin(qr.health)` | `0` | `TypeError: object of type 'numpy.int64' has no len()` |
| `(qr.health > 0).sum()` | `1` | `AttributeError: 'QRField' object has no attribute 'sum'` |

Known ground (#37 type flip, #38 unhonourable functions) plus one addition: **reductions are a third
class**, distinct from #38's row-coupled-N→N and sequence-arg cases. They *are* caught — by
`assert len(part_result) == part.shape[0]` (`qr_field.py:44`) tripping over a scalar — but the message names
numpy's scalar type and nothing about the actual problem. Fold into #38's rejection list with a message like
*"np.sum is a reduction; call qr.f.numpy() first"*. "Is anyone alive?", "who is nearest?", "total damage"
are the questions every game asks, and `len(world.pools)` swung 1→10 during normal play, so which branch you
get is decided by whatever else happens to be alive.

### 9 / 9b — two reachable malformed states

`add_entity([Pos, Pos, Vel])` is accepted and builds `pool.fields == ['position','position','velocity']`.
Every write to that field is then done twice, forever, for every entity of that archetype — **the first
caller's argument list decides the pool's shape**. Rejected by one `len(components) != len(set(components))`
check in `_validate_components`.

`add_entity([])` is explicitly forbidden, but `remove_component` of the last one gets there anyway: the
entity lives on with `components=[]`, `fields=[]`, invisible to every query except the no-arg one. And it
cannot be repaired — `add_component` on it hits `assert len(components) > 0` (`command_buffer.py:84`), an
**assert**, so under `python -O` the guard is gone and the state is silently reachable. Either forbid
removing the last component (symmetric with `add_entity`) or accept the empty archetype as legal — not both.

### 12 — archetype blink rebuilds the pool from scratch

An archetype is deleted the moment its last entity leaves (`world.py:178-181`) and rebuilt on the next
spawn. One entity toggling one component 100× ⇒ **201 Pool objects**, each allocating
`INITIAL_CAPACITY = 100` rows per field at construction (measured: 4.83 µs and 4.3 KiB for the shooter's
6-field bullet archetype). In the app: 40 builds for 7 live archetypes in normal play, **747 in chaos**.

Absolute cost is small — 3.6 ms and 3.3 MB over 50 s of game — so this is a **design smell, not an
emergency**: unbounded churn in the hot path with no reuse and no free list, scaling with the number of
entities that toggle capabilities, i.e. with the feature the docs advertise ("capabilities are additive").
Cheapest fix: keep an emptied pool (it already knows its fields) instead of deleting it, or start new pools
small and grow.

### 13 — the query cache never survives a tick, and a cold query is O(entities)

`update()` clears the whole cache whenever the buffer was non-empty (`world.py:153-155`) — **one `set_data`
from one entity anywhere in the frame is enough**. Measured in the shooter: **13 queries per tick, 13 cold,
0 cached.**

A cold query is not O(pools), it is O(entities), because `world.py:130` rebuilds the id column:

```python
entity_ids = np.array(sum((self._pool_ids[p] for p in res), []), dtype="int64")
```

a python list concat over every matching entity, then a python-list→ndarray conversion:

| N | cold query | cached query | one motion system | `entity_ids` share of the cold query |
|---|---|---|---|---|
| 100 | 11.1 µs | 0.48 µs | 11.0 µs | 33% |
| 1000 | 44.6 µs | 0.64 µs | 12.3 µs | 62% |
| 10000 | 228.5 µs | 0.29 µs | 15.5 µs | **96%** |

At N=10k **a query costs 15× the system that consumes it**, and most systems never read `entity_ids` at all
(motion, wrap, cooldown, decay never do). See Part 4.

### 14 — a mid-tick `update()` silently voids the queries already taken

This is #27, but the game-shaped trigger deserves naming: a shooter *wants* mid-tick commits (spawn a
bullet, commit, let it collide this frame). The moment any system commits, every query an earlier system
still holds points at a freed buffer — stale reads, discarded writes, no error. Measured: 3 entities moved
by `+100` after a mid-tick commit, **0 actually moved**. App-side rule until #27 lands: exactly one
`update()` per tick, at the end.


---

### Resolved by #42 — kept for the record

These five stop being live findings on 2026-07-26; they are below the line so the list above is
only what is still true. Each keeps its original text plus a note on what changed, because *why* a
trap existed is the part worth re-reading before re-introducing it.

### 5 — read-modify-write through `Entity` does not compose

```python
def damage(e, n):
    e.set_data(health=e.health - np.float32([n]))
damage(e, 3); damage(e, 4); world.update()      # health 10 -> 6.0, not 3.0
```

Reads are eager (committed state), writes are buffered, so the second `set_data` reads the **pre-tick**
value and overwrites the first command. Silent. This is the general form of
[#1](../todos/open/1-bounce-impulse-accumulator/TASK.md)'s "flipping is non-composable": *any* accumulator
written through `Entity` keeps only the last contribution. Worst with physics substeps (one system running
twice per frame) and with two systems that both damage. See Part 3 — this is the symptom of the structural
item, not an isolated bug.

> **Fixed by #42 (2026-07-26).** Writes are eager, so the second `damage` reads the first one's result:
> `damage(3); damage(4)` leaves 3. `e.health -= np.float32(n)` composes too. Pinned by
> `test_entity_read_modify_write_composes_within_one_tick` and its `+=` twin.

### 15 — a typo'd field name gives a bare `KeyError`

`e.set_data(helth=...)` → `KeyError: 'helth'` from `entity.py:58`. The same mistake one level up
(`set_data(shield=...)` for a component the entity lacks) gives a good `ValueError`. One `if field not in
self._world_field_to_component: raise` with the valid field list matches the rest of the library.

> **Fixed by #42 (2026-07-26).** `Entity._locate(names=...)` is the one gate for read, write and `set_data`:
> a name that is not a field of this entity's pool raises `AttributeError` listing the components and the
> valid fields. Both spellings of the mistake now land in the same place.

### 8 — two write paths, two validation contracts

```python
e.set_data(health=np.array([1.5]))   # TypeError: Expected dtype float32, got float64   (strict)
qr.health = np.array([[1.5]])        # accepted, silently cast
qr.color  = 3.9                      # int32 field -> stored [3,3,3,3], no warning
```

[#28](../todos/done/28-queryresult-setattr-validate/TASK.md) made `QueryResult.__setattr__` validate
*names*; dtypes fall through to numpy's unsafe casting. Whether the batch path *should* validate is a real
call (per-write dtype checks cost something in the hot path, and silent casting is what a numpy user
expects). But the two halves should not disagree in **silence** —
[#20](../todos/done/20-add-entity-eager-dtype-crash/TASK.md) exists because a silent shape surprise was
judged unacceptable on the other path.

> **Resolved by #42 (2026-07-26) — and the other way round than this finding assumed.** The entity path was
> loosened to match the batch path instead of the batch path being tightened: `e.field = v` now accepts
> exactly what `qr.field = v` accepts (converted to the field's dtype, broadcast into its shape — lists,
> tuples, scalars, a silent float64→float32 cast). Pinned as a *parity* test
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

`row.setflags(write=False)` (`entity.py:104-106`) stops `e.brain[0] = x` but not mutation *through* the
reference it just handed out. `object` fields are also the only way to hold per-entity structure (an AI
state dict), so this is load-bearing, not exotic: the shooter's enemies rely on it every tick. Mostly a
**docs** gap — `primitives.md` covers object-dtype *lifetimes* but not that object fields opt out of the
mutation-timing model entirely.

> **Restated by #42 (2026-07-26).** The timing half of this finding is gone: every data write is eager now,
> so mutating through the reference is no longer "outside the rule" — it *is* the rule, just without the
> dtype/shape check (there is nothing to check on an opaque object). `setflags(write=False)` no longer
> exists to be circumvented, and `e.brain = {...}` works directly. What remains is narrower and still worth
> documenting: an object field is the one place where the library cannot tell you that you stored the wrong
> thing. #40 (lifetimes) and 6b (still live, above the line) are unaffected.

### 1 — an uncommitted entity answers four questions with a bare `KeyError`

> **Half fixed by #42 (2026-07-26).** Everything that goes through `_locate` — a field read, a field write,
> `set_data` — now raises `AttributeError("Entity N not committed yet. Call world.update()")`. The four
> methods that still index `_eid_to_pool_ix` directly (`get_components`, `get_fields`, `to_dict`,
> `has_component`) still give a bare `KeyError: 0`. Verified 2026-07-26. Routing them through `_locate`
> (with `names=[]`) is a ~4-line finish.

`add_entity` puts the id in `world.live_entities` immediately and `get_entity()` hands out an `Entity`.
Reading a *field* gives the good message; nothing else does:

```
entity.position    -> AttributeError: Entity 0 not committed yet. Call `world.update()` (reading 'position')
has_component()    -> KeyError: 0
get_components()   -> KeyError: 0
get_fields()       -> KeyError: 0
to_dict()          -> KeyError: 0
```

The guard exists once, in `Entity.__getattr__` (`entity.py:93-97`). `get_components` (`entity.py:44`) and
`get_fields` (`entity.py:49`) index `_eid_to_pool_ix` raw; `has_component`/`to_dict` go through them. This
produced **98 undebuggable `KeyError: <number>`** in one 3000-tick chaos run, from a system that picked a
random id out of `world.live_entities` — a public dict — and asked `has_component`.
**Fix**: the same try/except in `get_components` and `get_fields`. 4 lines.

### Came back clean

- **Tag components** (zero fields ⇒ zero-field pool) work end to end: query, `to_dict`, add/remove.
- **`world.query()` with no arguments returns every live entity.** Undocumented, useful; the invariant
  checker leans on it.
- **numpy `int64` ids** from `qr.entity_ids` work anywhere a python int does, and do not pollute
  `live_entities` key types.
- **Spawn-then-modify in one tick** (`add_entity` → `get_entity` → `set_data` → one `update()`) works; the
  buffer resolves the ordering correctly. *Keep this working* — see Part 3's wrinkle.
  **→ Deliberately reversed by #42 (dev's call, 2026-07-26).** It now raises: an uncommitted spawn has no
  row, and a write goes where the row is and never consults the buffer. Reads and writes agreeing was
  judged worth more than the convenience, the alternative being a buffer scan on every write. Spawn data
  belongs in `add_entity`'s kwargs; robosim never hit this path.
- **The storage core**, per the fuzzer: 151,008 ops, 0 mismatches.

---

## Part 3 — Structural #1: the eager/deferred line is in the wrong place

> **Decided 2026-07-26 → task [#42](../todos/done/42-set-data-eager-again/TASK.md).** Accepted, and taken one
> step further: the read-only row view goes too, so `e.field = v` / `e.field[:] = v` come back and `set_data`
> becomes the transactional wrapper over `setattr`. Reason: that view *is* the measured per-read tax (491 vs
> 227 ns/op), and the guarantee it buys is not worth a 2.2× read — so mechanism (a) in the table below is
> **dropped**, not kept. #36's fix 1 is therefore moot rather than pending.
>
> **SHIPPED 2026-07-26.** 432 unit+integration tests green, pylint 10.00, docs updated. Four things landed
> differently from what is written below — read them as corrections to this Part, not as separate decisions:
>
> 1. **`_locate(names=...)` became the single gate.** One helper resolves `(pool, row)` *and* checks that
>    every name is a field of that pool; `__getattr__`, `__setattr__` and `set_data` all go through it. This
>    Part only asked for eager writes; the gate is what made read and write stop disagreeing about what a
>    field is, and it fixed findings 15 and (half of) 1 as a side effect.
> 2. **Writes follow numpy, not the strict schema check.** `e.field = v` accepts what `qr.field = v` accepts
>    (lists, tuples, scalars, `(1,)` fill, silent float64→float32) and is pinned by a parity test. This
>    resolves finding 8 by unification, in the opposite direction from what finding 8 proposed. `add_entity`
>    / `add_component` stay strict.
> 3. **A write never consults the command buffer.** Spawn-then-write raises instead of patching the staged
>    `ADD_ENTITY` command (the wrinkle below wanted the opposite — see *Came back clean*); a field whose
>    component is only pending raises; a field pending removal is written and then dropped with the
>    component. One rule, no scan, and reads and writes agree.
> 4. **`set_data` keeps the transaction, and pays for it only when it must.** One field → straight write
>    (numpy converts and shape-checks the whole RHS before copying, so a single field cannot be half
>    written): **free**, 1.04× the pre-change loop. Two or more → convert and fit-check every value first,
>    then write: 1.64× at three fields, all of it on a cold path (robosim's multi-field call sites are one
>    per TCP command and one per event). No rollback exists because after validation nothing can fail. The
>    naive always-`np.broadcast_to` version measured **4.5–7.3×** — `broadcast_to` alone is ~1.1 µs, so the
>    fast path is what makes the transaction affordable. Numbers:
>    `test/manual/42-set-data-eager/perf_probe.py`.
>
> Guarantees are pinned in `test/unit/test_entity.py` + `test/unit/test_entity_set_data.py`; that the tests
> *bite* is checked by `test/manual/42-set-data-eager/mutation_check.py` (3 mutants, caught 11/5/16).

### Why

Today the line is **`Entity` = deferred, `QueryResult` = eager**. It should be **structure = deferred, data =
eager**.

Buffering *structural* change is non-negotiable: moving a row between pools invalidates iteration in flight.
Every serious ECS defers it (Unity ECB, flecs, Bevy `Commands`) and so does microecs — correctly.

Buffering a *data* write buys nothing. `pool.data[f][ix] = v` moves no rows and invalidates nothing. The
batch path proves it: `qr.position = ...` writes straight into pool memory, eagerly, and nobody complains.

And deferral does not buy the thing deferral usually buys — snapshot / simultaneity semantics — because:

- `update()` is deliberately **not atomic** ([#22](../todos/done/22-fully-eager-staging/TASK.md));
- staged values are **references, not snapshots** ([#39](../todos/open/39-staged-writes-snapshot/TASK.md));
- the batch path is already order-dependent, and it is where most writes happen.

**So the cost of deferral is paid and the benefit is not collected.** That is the structural defect: not
"deferred is wrong", but "half-deferred delivers neither model".

What it costs, concretely: finding 5 — silent lost updates, the general form of #1 — and an `Entity` that is
not a coherent view of a row but a write-only mailbox with a read-through cache of the previous tick.

### What (and what this is *not*)

This is a **narrow, partial reversal of [#29](../todos/done/29-entity-single-write-path/TASK.md)** — its
*timing*, not its API. #29 bundled two mechanisms; separate them:

| #29 mechanism | keep? | why |
|---|---|---|
| (a) `set_data(**fields)` is the *only* entity write path; rows are read-only views | **keep** | this is what buys "no stray `e.field[:] = v` can corrupt a pool". Independent of timing |
| (b) `set_data` *defers* the write to `update()` | **drop** | this is the only part that breaks RMW, and it buys nothing |

So: `set_data` validates exactly as it does now, then writes straight to the pool. There is still exactly
one write path, `e.field = x` and `e.field[:] = x` still raise, and RMW composes.

**Correction to an earlier claim of mine:** the read-only row guard does *not* exist only to police eager
writes — it enforces the single write path regardless of timing, so it **stays** under this proposal. The
per-read tax stays with it, and #36 already has fix 1 ("freeze the pool column once, not the row per read")
recorded if it ever needs paying back.

**The perf argument is the weak leg and should not lead.** #36 measured the #29 tax at +6–7% inside a
40–64% commit-to-commit wander band with 4.6–19× headroom, and **accepted** it on 2026-07-26. That verdict
stands; nothing here reopens it. Dropping the `SET_DATA` command would incidentally return some of it (no
`Command` object, no kwargs dict, no buffer append per write, no `_do_set_data` pass at commit), but the
case rests on **composability and one mental model**, not on microseconds.

What gets deleted: the `SET_DATA` command type, its branch in `update()` (`world.py:148-149`) and in
`CommandBuffer.append` (`command_buffer.py:100-107`), `_do_set_data` (`world.py:201-204`), and the
rollback dance in `Entity.set_data` (`entity.py:63-70`). Net negative LoC.

### How (dev writes the code)

1. `Entity.set_data(**data)`: keep the field→component grouping and `_validate_component(strict=False)` per
   component — **validate every field first, then write none-or-all** (task
   [#25](../todos/done/25-set-component-data-validate-first/TASK.md)'s pattern). That preserves #24's
   transactional guarantee without a buffer.
2. Write path: `pool, ix = self._eid_to_pool_ix[self.entity_id]; pool.data[k][ix] = v`. Copy on write, which
   also closes #39 for this path.
3. **The one wrinkle** — an entity spawned this tick has no row yet, and today
   `add_entity` → `set_data` → `update()` works (verified, Part 2 "came back clean"). Keep it: if the eid is
   not in `_eid_to_pool_ix`, find its staged `ADD_ENTITY` command and update that command's args in place
   (`CommandBuffer` already scans the buffer for exactly this entity in `_get_entity_components`). ~10 lines.
   Reads of such an entity still raise, as today, so the deferral is invisible.
4. Docs: `primitives.md` "Mutation timing" becomes one rule — **data writes land now, structural changes land
   at `update()`** — and the `Entity`-is-buffered caveat disappears from `systems.md` §3.
5. **Validation**: this changes intra-tick visibility, so it is a robosim e2e parity question, not a unit
   test question. Run `bash test/e2e/run_all.sh` plus `perf-physics-tick` before and after. Tester adds:
   RMW composes (`damage(3); damage(4)` ⇒ 3), read-your-own-write, transaction still all-or-nothing on a bad
   field, spawn-then-`set_data` still works, `e.field = x` still raises.

### Alternative, if this was rejected (it was not — kept for the record)

Commit to the *other* side properly: deep-copy staged values (#39), make `update()` atomic (reversing #22),
and give `Entity` reads a buffer-aware view so RMW composes. That is a bigger, slower library and the wrong
shape for a numpy-batch engine — but it is coherent. **The one option that should not survive is today's
middle.** If neither is done, finding 5 must at least be documented as a rule ("never write the same field
twice in a tick through `Entity`"), because right now it is silent.

---

## Part 4 — Structural #2: a query has no lifetime

### Why

`world.query()` is a memo, not an object: keyed by include+exclude, dropped wholesale when the buffer was
non-empty, holding raw pool slices (`query_result.py:32`), and eagerly building `entity_ids` for callers who
mostly never read it. Two consequences already in the tracker or measured here:

- **#27 (stale qr)** is a *symptom* of this. If the world refreshed the queries it owns at `update()`
  instead of dropping them, staleness would be impossible **by construction** — no generation counter, no
  "re-query after every update" rule to remember.
- **Finding 13**: cold queries are O(entities) and the cache never survives a tick (13/13 cold in the app).
  96% of a 228 µs query at N=10k is one line building a column nobody asked for.

### What

Two **independent** changes; either can land alone.

1. **Lazy `entity_ids`.** Make it a property built on first access. Cold query drops to O(pools). This is the
   single highest-value perf change on this list for anything above ~1000 entities.
2. **Refresh instead of clear.** At `update()`, re-slice the cached queries the world already holds
   (`self._cache`) instead of `self._cache.clear()`. Held queries stay valid; the cache starts working
   across ticks. Note this only works *because* the world owns every live query — which it already does.

Incremental extra, if #13's cost still shows: keep `_pool_ids` as a numpy array per pool with a size counter
(append and pop-swap stay O(1)), so `entity_ids` becomes `np.concatenate([p.ids[:len(p)] ...])` in C instead
of a python list concat.

### How

- `QueryResult.entity_ids` → `@property` with a `_entity_ids` cache; `world.py:130` stops running eagerly;
  the `assert len(entity_ids) == sum(len(p) ...)` in `QueryResult.__init__` moves into the property.
- `World.update()`: replace `self._cache.clear()` with a loop asking each cached `QueryResult` to re-slice
  (`_data`, `_cache`, `_entity_ids`) — drop entries whose pools are gone.
- Then #27's generation guard is unnecessary; if it lands first, it is the right cheap stopgap.
- **Validation**: existing `test_queryresult.py` + a new test that a qr held across `update()` (with growth,
  shrink, pool death and a new archetype) reads and writes correctly. The shooter's finding-14 repro becomes
  that test.

---

## Part 5 — Structural #3: one contract for `qr.field`, resolved the other way round

### Why

#37 already states the problem: `_QRArray` at 0–1 pools (whole numpy API live), `QRField` at 2+ (narrow
contract). The app's contribution is that **`len(world.pools)` swung 1→10 during ordinary play**, so this is
not a corner case — which branch a call site gets is decided by unrelated spawns elsewhere. Plus finding 7:
reductions are a third failing class beyond #38's two.

### What

#37 asks "how do we keep #26's perf win *and* one contract". Suggestion: **do not make `QRField` fast — make
`_QRArray` narrow.** Override `__getitem__`/`__setitem__` on `_QRArray` to enforce the same axis-0 predicate
(`QRField._selects_axis0`, already written and tested by
[#33](../todos/done/33-qrfield-one-key-predicate/TASK.md)), and add the array-protocol guards so reductions
are rejected identically. The fast path stays a real ndarray in memory layout and in every batch op; only
the entity-axis surface is closed. Both branches then behave identically, and #26's win survives untouched.

### How

- `_QRArray.__getitem__/__setitem__`: `if QRField._selects_axis0(key): raise TypeError(<same message>)`.
- `_QRArray.__array_function__` / `__array_ufunc__`: reject non-honourable functions with #38's message, so
  the 1-pool and 2-pool rejection sets are the same set.
- Keep `.numpy()` and `.parts` as they are — they are already the portable spelling.
- **Validation**: a parametrized test that runs the *same* expression against a 1-pool and a 2-pool world and
  asserts identical outcomes (value or exception type). That test is the contract.

---

## Part 6 — What is right (credit where due)

- **Eager validation, deferred application** (#22). Errors point at the call site, not inside `update()`.
  Most ECS libraries get this backwards and hand you a mystery crash in `commands.apply()`.
- **Bitmask archetype keys**, order-independent, query = a mask scan over pools. The standard right answer,
  in 8 lines (`world.py:123-125`).
- **Globally unique field names** (#23/#29) buying `set_data(**fields)` without naming the component.
  Constraint-buys-simplicity, the good kind of Tiger Style trade.
- **850 LoC.** The whole library is readable in ten minutes, after which pop-swap edge cases can be reasoned
  about from memory. This is the vendoring thesis paying off, and it is why an audit like this is even
  possible.
- **It survives a model checker.** 151k structural ops, zero divergence.

---

## Part 7 — Scorecard

Column *now* is pre-#42. Part 3 shipped 2026-07-26, so today's real numbers are between *now* and
*+ Parts 3–5*: coherence **5 → 8**, `Entity` ergonomics **5 → 7.5**, error messages **4 → 6**
(findings 5, 8, 15 fixed; 1 half), overall **~6.5 → ~8**. Parts 4–5 are what is left of the coherence gap.

| axis | now (pre-#42) | after the Part 8 polish list | + Parts 3–5 |
|---|---|---|---|
| Core correctness (pools, migration, pop-swap) | **9** | 9.5 | 9.5 |
| API coherence / conceptual integrity | **5** | 6.5 | **9** |
| Ergonomics — batch/ECS path | **7** | 8 | 8.5 |
| Ergonomics — `Entity`/OOP path | **5** | 6.5 | **8.5** |
| Error messages / debuggability | **4** | 8 | 8.5 |
| Performance as a game loads it | **7** | 8.5 | 9 |
| Docs | **8** | 8.5 | 9 |
| Size / auditability / deps | **10** | 10 | 10 |
| Fit for robosim (~10² entities, batch physics, 1 update/tick) | **8** | 9 | 9.5 |
| Fit for a game (this audit) | **6** | 7 | 8.5 |

Overall **~6.5 → ~7.5 → ~9**. The polish list is worth doing and cheap, but note what it *cannot* do: the
coherence axis only moves with Part 3. robosim sits in the library's sweet spot either way — ~100 entities,
mostly batch physics, one `update()` per tick — which is why none of this is urgent for robosim.
**Post-#42**: the coherence axis did move (5 → 8), which was the whole bet of this plan.

---

## Part 8 — Action list

Ranked. Nothing here is filed as a task yet; ids are this plan's findings.

| order | work | size | note |
|---|---|---|---|
| 1 | **Lazy `entity_ids`** (Part 4.1) | S | biggest measured win: 96% of a cold query at N=10k |
| 2 | **Error messages: 1, 15, 4** | S | ~10 lines total; this is what a new user hits on day one. #42 did 15 and half of 1; what is left is `get_components`/`get_fields`/`to_dict`/`has_component` (~4 lines) plus 4 |
| 3 | **Buffer-aware `has_component`** (2) | S | information already exists; deletes app-side bookkeeping |
| ~~4~~ | ~~**Decide Part 3** (mutation timing)~~ | M | **done** — decided *and* shipped as #42 on 2026-07-26 |
| 5 | **Close 9b, reject 9** | S | two `raise`s; 9b is also an `assert` on a reachable path |
| 6 | **Refresh-not-clear queries** (Part 4.2) | M | subsumes #27; do after 1 |
| 7 | **One contract for `qr.field`** (Part 5) | M | concrete resolution for open P1 #37 + #38 |
| 8 | **Pool reuse** (12) | S | design smell, low absolute cost today |
| 9 | **Docs: object fields are an escape hatch** (6) | S | name the rules they opt out of |
| ~~10~~ | ~~**Decide 8** (batch-path dtype validation)~~ | S | **done** — #42 unified the two write paths on numpy's rules; the split that remains is spawn-vs-write, deliberate and documented |

Append-only, no new task needed: **6b → #40** (the leak is only the pool's last row), **7 → #38** (reductions
are a third class), **14 → #27** (game-shaped trigger + repro), **5 → #1** (same non-composability, general
form).

Candidate new tasks if this plan is accepted: 1+15+4 (one "entity error messages" task), 2, 9+9b (one
"reachable malformed states" task), 12, 13 (lazy ids), Part 3 (mutation timing), Part 5 (one contract —
or fold into #37 as the chosen approach).
