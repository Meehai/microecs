# The spawn path validates every entity twice — 20% of a spawn, on the workload we lose

**Created**: 2026-07-26
**Closed**: 2026-07-27
**Priority**: 2

## Outcome — shape 2 (world owns it), and it beat the estimate 3×

Landed in `3ba0e76`. `World.add_entity` keeps its pass; `CommandBuffer.append` does nothing for ADD_ENTITY
beyond the liveness check every verb gets. Shape 2 over shape 1 because `world.py:82` is the **sole**
construction site of an ADD_ENTITY command in the library — the "implicit contract between two files" the
task warned about is really a single producer feeding a single consumer, which is just ownership. (The
component verbs are unchanged: `Entity.add_component`/`remove_component` validate nothing, so `append`
stays their gate. It is a per-verb answer, not a global one.)

**Measured** (`test/manual/churn/task44_ab.py`, N=3000, min-of-7, both arms interleaved in one process, so
machine drift hits both equally — a 3-component archetype):

| workload | 2x (before) | 1x (after) | speedup |
|---|--:|--:|--:|
| `add_entity` only | 10474 ns | 5310 ns | **1.97×** |
| full spawn (`+ update()`) | 16576 ns | 11081 ns | **1.50×** |
| churn pair (spawn + despawn) | 19873 ns | 14666 ns | **1.36×** |

**w5 churn against the field** (`./run_benchmark.sh 200 1000 5000`; ratio = competitor ns ÷ microecs ns,
so >1 means microecs wins). Baseline is `test/manual/bench-compare/results_20260726_clean.json`; the
competitors' own absolutes barely moved between the two runs (EnTT 245.8→234.0 ns/e at N=200), which is
what makes the comparison safe:

| vs | N=200 | N=1,000 | N=5,000 |
|---|---|---|---|
| ecs_pattern | 0.078 → 0.098 | 0.216 → 0.319 | 0.363 → 0.502 |
| EnTT | 0.160 → 0.195 | 0.498 → 0.717 | 0.829 → **1.19** |
| snecs | 0.160 → 0.196 | 0.496 → 0.726 | 0.862 → **1.30** |
| esper | 0.267 → 0.316 | 0.927 → 1.34 | 1.64 → 2.41 |
| flecs | 0.387 → 0.437 | 1.379 → 1.92 | 2.13 → 3.33 |

+25% at N=200, +45% at 1k, +40–57% at 5k — against the 10–16% predicted below. **microecs now beats EnTT
and snecs on w5 at N=5,000**, two cells it used to lose. N=200 is still 10× behind ecs_pattern: the fixed
per-spawn cost dominates there, which is [#49](../../open/49-churn-headroom-beyond-44/TASK.md)'s territory.

Tests: `test_world.py::test_spawn_validates_exactly_once` /
`test_spawn_computes_defaults_exactly_once` / `test_rejected_spawn_validates_exactly_once_too` are the
counter-based regression guard, and they catch a duplicate re-appearing on *either* side. The five
rejection cases and the duplicate-components case moved from `test_command_buffer.py` to `test_world.py`
(the layer that now raises them); `test_buffer_stages_a_spawn_verbatim` and
`test_buffer_rejects_a_spawn_for_an_unregistered_id` are what remains of the buffer's own contract.
Suite: 483 passed, 8 xfailed.

Follow-ups this settles in **#23**: subtask 1 is done (resolved toward the world, not the buffer);
subtask 2 is **moot** — `add_entity` validates before it mutates, so a rejected spawn cannot burn an id;
subtask 3 (default-filling asymmetry) is still open but its "one story" is now *the producer fills
defaults*, not *append fills defaults*.

## Why

`World.add_entity` validates and fills defaults, then `CommandBuffer.append` does **both again** on the same
command:

```python
def add_entity(self, components, **kwargs):            # world.py:74-82
    self._validate_components(components, **kwargs)     #  <-- pass 1
    default_kwargs = self._defaults_for(components, **kwargs)
    ...
    self._command_buffer.append(Command(ADD_ENTITY, self._last_id,
                                       args={"components": components, **kwargs, **default_kwargs}))

def append(self, command):                             # command_buffer.py:74-77
    if command.command_type == CommandType.ADD_ENTITY:
        fk = {k: v for k, v in command.args.items() if k != "components"}   # dict rebuild
        world._validate_components(command.args["components"], **fk)        #  <-- pass 2, same data
        command.args.update(world._defaults_for(command.args["components"], **fk))  # returns {} now
```

Pass 2 is a superset-check of pass 1 (it re-validates the defaults pass 1 just inserted, and those were already
checked field-by-field at `World.__init__`). `_defaults_for` in pass 2 can only return `{}`.

**Measured** (`test/manual/bench-compare/spawn_breakdown.py`, N=3000, min-of-5, ns/entity):

| | ns | share of a full spawn |
|---|--:|--:|
| full spawn = `add_entity` + `update()` | 9964 | 100% |
| `_validate_components` | 1627 | 16% **×2** |
| `_defaults_for` | 383 | 4% **×2** |
| the `fk = {…}` dict rebuild | 261 | 3% |
| `Pool.add_entity` (the actual storage work) | 3227 | 32% |

**~2.0 µs/spawn is redundant — 34% of `add_entity`, 20% of a full spawn, 16% of a spawn+despawn pair.**

This lands exactly where it hurts. w5 churn is the **only** workload microecs loses at every N in the 7-library
benchmark (0.08× ecs-pattern at N=200, 0.46× at 20k, 0.82× EnTT at 100k), and the plan blamed the archetype
pop-swap for it. The pop-swap is 17% of a churn pair; **validation is 32%, half of it this duplicate.** See plan 1
Part 8, P4.

## What

Validate a spawn **once**. One owner, not two. No API change, no semantic change: the same inputs must raise the
same exceptions, at the same time (eagerly, at the `add_entity` call — that is `#22`'s contract and it stays).

## How (dev writes the code)

Two shapes, both fine — pick by who should own the invariant:

1. **Buffer owns it.** `World.add_entity` stops validating and stops computing defaults; it just assigns an id and
   appends `Command(ADD_ENTITY, id, args={"components": …, **kwargs})`. `append` keeps its existing pass and fills
   defaults. Fewest moving parts, and the buffer stays the single gate for everything it stages.
2. **World owns it.** `add_entity` keeps its pass; `append` skips validation for `ADD_ENTITY` (a comment saying the
   caller validated, since `World.add_entity` is the only producer of that command). Smaller diff, but it puts an
   implicit contract between the two files — worse per "no stub-that-lies".

Either way also drop the `fk = {k: v for …}` rebuild on the surviving path.

Note `ADD_COMPONENT` / `REMOVE_COMPONENT` are **not** affected: `Entity.add_component`/`remove_component` do no
validation themselves, so `append`'s single pass is already the only one. This task is `ADD_ENTITY` only.

While in there, the same file has a second, unmeasured scaling cost worth a look:
`CommandBuffer._get_components_state` walks the whole buffer per component op, so K component ops in one tick are
O(K²) — already recorded as fix 3 in [#36](../../done/36-optimize-entity-read-write-path/TASK.md) and deferred in
**#25**. Out of scope here; do not fold it in.

## Validation (tester)

The behaviour must not move, so the tests are the existing ones plus a "how many times" check:

- Whole `test/unit/test_world.py` + `test_command_buffer.py` green, unchanged — every `add_entity` rejection case
  (missing required field, wrong dtype, wrong shape, extra field, unknown component, duplicate component from
  `#43`) must still raise **the same exception type from the same call**, before `update()`.
- New: count the validation calls. Wrap `World._validate_components` with a counter, spawn one entity, assert
  it ran **once** (this is the regression guard — the whole point of the task).
- New: a spawn with omitted defaultable fields must still land its defaults after `update()` — that is the path
  where pass 2's `_defaults_for` currently runs on the merged kwargs.
- Re-run w5 churn before/after: `cd examples/05-benchmark-workloads && ./run_benchmark.sh 200 1000 5000`. Expect
  microecs w5 ~10–16% faster; **compare ratios against the other libraries in the same run, never ms across runs**
  (absolutes drift ±10–25% on this machine — plan 1 Part 8).

## Relates

- **#22** (buffer is a staging area, validated eagerly) — this keeps that contract, it only stops doing it twice.
- **#36** fix 3 / **#25** — the other known cost in `CommandBuffer.append`, deliberately not in scope.
- Plan 1 Part 8 (P4) — the churn breakdown that found this, and the w5 numbers it should move.
