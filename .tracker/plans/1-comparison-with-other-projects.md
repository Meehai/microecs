# microecs vs. other ECS libraries — analysis & positioning

**Created**: 2026-06-04
**Type**: Reference / competitive analysis
**Scope**: Assess microecs as an ECS (efficient? good? nice?), compare against the relevant
Python-interface ECS libraries on GitHub / GitLab / Codeberg, and decide what features we
actually need.

---

## TL;DR (grug verdict)

microecs is a **good, correct, genuinely fast** little engine for its target — bulk numeric
updates over many entities. Its killer feature (numpy **SoA** + a cross-archetype **vectorized**
query view) is **rare**: of ~15 Python ECS libraries surveyed, only two others do it (`xecs`,
`manifoldx`), and both are heavier and less maintained. microecs owns a near-empty niche:
*standalone, pure-Python + numpy, ~300 LoC of logic, no compile step, vectorized, render-agnostic.*

- **Efficient?** Yes, for bulk same-archetype numeric work — 1–2 orders of magnitude over the
  per-entity Python loops that esper/snecs/tcod-ecs/ecs-pattern use.
- **Good?** Yes — clean single-responsibility split, strong tests, careful numpy-protocol impl.
- **Nice?** Mostly. Two ergonomic taxes: verbose component defs (shape/dtype metadata leaks),
  and AND-only queries.
- **Biggest gap:** query expressiveness (no `none_of` / optional / tags).
- **Best-in-class trait:** all four structural ops deferred through one commit point.

---

## Part 1 — Is microecs efficient / good / nice?

### Efficient — yes, for its target

| Mechanism | Where | Verdict |
|---|---|---|
| SoA storage, one numpy array per field per archetype | `pool.py:23` | ✅ cache-friendly, the whole point |
| Bitmask archetype keys, subset query `(arch & key) == key` | `world.py:90` | ✅ simple, correct |
| Query result **cached** between updates | `world.py:85`, `world.py:45` | ✅ steady-state queries are O(1) |
| Deferred command buffer (no iterator invalidation) | `world.py:38-45` | ✅ same approach as flecs/bevy |
| Vectorized cross-pool write | `query_result.py:56-71` | ✅ the differentiator |

For a motion/physics system over thousands of same-archetype entities, the work runs in numpy and
beats per-entity Python loops by **1–2 orders of magnitude**. That is the headline benchmark to
publish.

**Honest inefficiencies** (inherent to the approach, not bugs):

- **`_Field` loops over pools in Python** (`query_result.py:29-34`), allocates a new `_Field` per op,
  and does `np.broadcast_to` / `np.split` on writes (`query_result.py:69-71`). Vectorization only
  wins *within* a pool; a query over many small pools degenerates toward per-pool Python overhead.
  README already concedes the per-pool loop "maybe faster in extreme cases as it avoids the `_Field`
  obj."
- **Archetype fragmentation** — classic archetype-ECS tax: many component combos → many small pools.
- **`add/remove_component` copies the whole entity** to a new pool (`world.py:128-142`, `pool.py:52`).
  Fine occasionally, bad every tick — which is why the bounce task (`todos/open/1`) uses an impulse
  accumulator instead of per-tick component churn. Good instinct.
- **`get_entity` copies *all* fields** into a dict (`world.py:66`). No "read one component for one
  entity" fast path.
- Keys are `2**i` Python ints (`world.py:20`), arbitrary-precision → **no 64-component cap** that
  bites C engines. But a cache miss is a linear scan of all pools (`world.py:89`).

### Good — yes (design quality)

Clean single-responsibility split: `Pool` (SoA dynamic array, no id concept) / `World`
(ids + archetypes + command buffer) / `QueryResult` (cross-pool view) / `Component` (dataclass) /
`System` (convention). ~300 lines of logic, 2 deps.

Tests are genuinely strong: the 500-op randomized churn invariant check (`test_world.py:687`) and the
recarray-parity tests (`test_queryresult.py:458`) are things most ECS libraries lack. The
`__array_ufunc__` / `__array_function__` impl matches numpy broadcast + recarray write-through
semantics, with sharp edges rejected rather than silently wrong.

**Quality nits** (non-test files — dev's call):
- `_Field.__array_function__` whitelists only `{np.where, np.clip}` (`query_result.py:9`); everything
  else is `NotImplemented`. Documented, but hit often → forces `.numpy()`.
- `from typing import Callable, T` (`query_result.py:2`) imports the *private* `typing.T`. Works,
  fragile.
- **Resolved** (was an ABC-vs-convention nit): `system.py` and the `TickSystem` ABC were removed, so a
  system is now purely a convention — any callable taking `world` (examples use
  `class XSystem: def __call__(self, world)`). The old `on_tick(self, scene)` param-naming disagreement
  is gone; `microecs` exports only `Component, Pool, QueryResult, World` (4 primitives).

### Nice — mostly

The vectorized write idiom is the nicest thing here:

```python
qr.position[:] = qr.position + qr.velocity * DT   # updates every matched entity, in numpy
```

Two taxes:

**(a) Component defs leak numpy internals.**
```python
# microecs — shape/dtype metadata in every field
class HasPosition2D(Component):
    position: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})

# xecs — typed field, no metadata
class Velocity(xx.Component):
    value: xx.Vec2
```

**(b) Query is AND-only** (`query_and`). No `without` / optional / tags. Biggest gap (see Part 4).

---

## Part 2 — The landscape (top 10 relevant Python-interface ECS)

Surveyed GitHub, GitLab, Codeberg (2026-06-04). **Codeberg has nothing notable.** GitLab has only
microecs itself + a stale `pavouk-ecs`. Ranked by relevance = popularity × maintenance × design
distinctiveness.

| # | Library | Host | ★ (approx) | Status | Model | Vectorized? |
|---|---|---|---|---|---|---|
| 1 | **esper** | GitHub `benmoran56/esper` | ~690 | active (v3.7, 2026) | dict-of-objects, sparse, AoS | ❌ per-entity |
| 2 | **tcod-ecs** | GitHub `HexDecimal/python-tcod-ecs` | ~28 | very active | dict/set, **relationships + IsA**, typed queries | ❌ per-entity |
| 3 | **xecs** | GitHub `lukasturcani/xecs` | ~3 | stale (v0.9, 2023) | **Rust core + numpy SoA**, Bevy-style | ✅ |
| 4 | **manifoldx** (`manifold-gfx`) | GitHub `apiad/manifoldx` | ~6 | experimental (v0.1, 2026) | **numpy SoA archetype** (inside a wgpu engine) | ✅ |
| 5 | **snecs** | GitHub `the-moonwitch/snecs` (was `slavfox/snecs`) | ~19 | dead (~2020) | bitmask, **query algebra `& \| ~`**, serialization | ❌ per-entity |
| 6 | **ecs-pattern** | GitHub `ikvk/ecs_pattern` | ~54 | active | dataclass AoS, System/SystemManager | ❌ per-entity |
| 7 | **flecs** (via `pyflecs11`) | GitHub `SanderMertens/flecs` (C); `Wesxdz/pyflecs11` (py) | 8.4k C / 2 py | C active; **py binding is a toy** | C archetype; relationships/observers/query DSL/pipelines | ❌ (py path) |
| 8 | **entitas-python** | GitHub `Aenyhm/entitas-python` | ~49 | stale (2021) | port of Unity Entitas, **reactive groups** | ❌ per-entity |
| 9 | **seanfisk/ecs** | GitHub `seanfisk/ecs` | ~90 | abandoned (2015) | the historic reference impl | ❌ per-entity |
| 10 | **wecs** | GitHub `TheCheapestPixels/wecs` | ~14 | lightly active (2024) | Panda3D-oriented, system deps | ❌ per-entity |

Honorable mentions: `pavouk-ecs` (GitLab, ~2★, microecs's nearest host-neighbor), `pyriak`
(event-driven), `mecs` (minimalist), `pygame-ecs`. Phantoms (do **not** exist as usable Python ECS):
"ecstasy", "Adam" (that's Adam Martin, the blogger who defined the pattern), and any production-grade
flecs Python binding.

**Key finding:** microecs's vectorization is shared by only **#3 (xecs)** and **#4 (manifoldx)**, and
microecs beats both on *focus*:
- **xecs** — Rust core (compile step, hard install), Bevy-scale ambition, pre-1.0, ~abandoned, 13 open
  issues.
- **manifoldx** — welded to a wgpu renderer, v0.1, "not for production."
- **microecs** — standalone, pip-installable, render-agnostic, ~300 LoC, actively developed.

A real, defensible position.

---

## Part 3 — Ergonomics head-to-head, on the *same* features

### Vectorized update — microecs competitive, a touch more verbose
```python
# microecs
qr.position[:] = qr.position + qr.velocity * DT
# xecs        (transform, velocity) = query.result();  transform.translation += velocity.value
# manifoldx   query[Transform].pos += velocity * dt
```

### Structural-change safety — microecs is best-in-class
| Library | add/remove entity | add/remove component |
|---|---|---|
| **microecs** | ✅ deferred buffer | ✅ deferred buffer |
| esper | ✅ deferred delete | ⚠️ immediate |
| snecs | ✅ deferred delete | ⚠️ immediate |
| tcod-ecs | ⚠️ snapshot yourself | ⚠️ snapshot yourself |

Nobody else defers all four ops through one commit point. **Keep this — genuine strength.**

### Query expressiveness — microecs is weakest
```python
microecs:   world.query_and((A, B))                            # AND only
snecs:      query([A]).filter((B | C) & ~D)                    # AND / OR / NOT
tcod-ecs:   registry.Q.all_of([A]).none_of([B]).any_of([C])    # AND / NOT / OR + relations + tags
flecs:      world.query("A, B, !C, ?D")                        # full DSL + relationships
```

---

## Part 4 — Features we need vs. don't need

### Need (prioritized)

1. **Query exclusion (`none_of`)** — the #1 gap. Target API `query_and((A, B), none_of=(C,))`. The
   bitmask makes it cheap: a second `none_of` mask + require `(arch & none_mask) == 0` alongside the
   existing `(arch & key) == key` (`world.py:90`). Clean because exposed fields stay exactly the
   `all_of` fields → the contiguous `_Field` view is unaffected. High value, low complexity, grug.
   → **task 8.** Note: **`any_of` (OR) and optional are NOT this task** — an OR/optional component is
   present in some matched pools but not others, which breaks the aligned cross-pool column; that
   needs a separate design.
2. **Zero-size tag components** (`Player`, `Frozen`, `Dead`) — **already work** (verified by
   `test/manual/tag-components-probe/probe.py`): a field-less `@dataclass` component is a valid
   archetype bit with no pool array; pure-tag entities, tag-as-filter, and tag migration all pass
   today. So the work is **tests + README**, not implementation. → **task 9.**
3. **Single-component get/set by id** without copying all fields — completes the "object-like ops"
   story `get_entity` started. Minor.

### Don't need (scope discipline — matches CLAUDE.md minimalism)

- **Relationships / hierarchy** (tcod-ecs, flecs) — powerful but heavy; wrong fit for a numeric SoA
  engine unless a concrete use case demands it.
- **Event bus / observers / change-detection** (esper, entitas, flecs) — the bounce task *explicitly*
  prefers an impulse accumulator over events. Stay queue-free.
- **System scheduler / dependency graph / parallelism** — systems are deliberately a convention.
  xecs's Rust parallelism is exactly what makes it un-minimal.
- **Serialization** — nice-to-have, not core. `object` dtype + pickle covers escape hatches.

---

## Part 5 — Recommended follow-up tasks

1. **[task 8](../todos/open/8-query-exclusion-none-of/TASK.md)** — `none_of` exclusion on `query_and`
   (open, Priority 1). Bitmask `none_mask` + composite cache key. `any_of`/optional explicitly
   deferred (SoA field-misalignment). Implementation task for the IC; tester writes the specs.
2. **[task 9](../todos/open/9-tag-components/TASK.md)** — zero-field tag components (open, Priority 2).
   Already work; task is to pin them with tests + document them. Tester-heavy.
3. **README "Comparison / positioning" section** — *(not yet filed)* name xecs + manifoldx as the
   closest peers; state the honest perf caveat (vectorization wins *within* archetype; per-entity
   libs win on churn / branchy logic; native ECS win on raw structural ops).

---

## Sources (fetched 2026-06-04)

- esper — https://github.com/benmoran56/esper , https://esper.readthedocs.io
- tcod-ecs — https://github.com/HexDecimal/python-tcod-ecs , https://python-tcod-ecs.readthedocs.io
- xecs — https://github.com/lukasturcani/xecs , https://xecs.readthedocs.io
- manifoldx — https://github.com/apiad/manifoldx , https://blog.apiad.net/p/realtime-3d-in-pure-python-numpy
- snecs — https://github.com/the-moonwitch/snecs , https://snecs.slavfox.space
- ecs-pattern — https://github.com/ikvk/ecs_pattern
- flecs — https://github.com/SanderMertens/flecs ; py binding https://github.com/Wesxdz/pyflecs11
- entitas-python — https://github.com/Aenyhm/entitas-python
- seanfisk/ecs — https://github.com/seanfisk/ecs
- wecs — https://github.com/TheCheapestPixels/wecs

**Method note:** profiles assembled via web search + README/source fetches of each repo. Star counts
are approximate (drift daily) and reflect 2026-06-04. "Vectorized?" = whether bulk component math runs
as a numpy/array op over many entities vs. a per-entity Python loop.
