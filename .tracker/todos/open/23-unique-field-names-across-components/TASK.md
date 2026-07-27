# Enforce globally-unique field names across components at World construction

**Created**: 2026-07-10
**Priority**: 2

## Status (2026-07-25): the guard is DONE — only subtasks 1–3 remain

The field-name guard landed with microecs **#29** (`world.py:49-51`), which needed it: `set_data(**fields)`
resolves a field name to its component via `world.field_to_component`, and that map is only well-defined if
names are unique. Raises `ValueError` naming the field, at construction, as specified below.

Consequence, as predicted: the commit-time `Duplicate keys` path is now **unreachable through the public API**,
so `test_add_component_with_field_name_clash_raises_at_update` was removed and
`test_world_rejects_duplicate_field_name_across_components` lost its xfail and is a live test.

**Still open: subtask 3 only.** Subtasks 1 and 2 closed on 2026-07-27 with
[#44](../../done/44-spawn-path-validates-twice/TASK.md) — 1 done (the duplicate pass is gone, resolved
toward the world), 2 moot (validation stayed ahead of the mutations, so a rejected spawn cannot burn an id).
See the subtask block at the bottom for what that changed about subtask 3's target.

## Why

The fully-eager command buffer (task 22) claims "nothing invalid ever reaches `update()`". One case slips
through: a component added via `add_component` whose field name collides with a field the entity already
has. The eager gate validates the new component **in isolation** (`world._validate_components([B], ...)`
only checks B's own metadata), so it never sees the clash. The bad command reaches commit and blows up:

```python
class A(Component): x: np.ndarray = field(metadata={"shape": (3,), "dtype": "float32", "default": None})
class B(Component): x: np.ndarray = field(metadata={"shape": (3,), "dtype": "float32", "default": None})  # same name

e = w.add_entity([A], x=...); w.update()
w.get_entity(e).add_component(B, x=...)        # eager gate PASSES (blind to the clash)
w.update()                                     # AssertionError: Duplicate keys ['x'] vs ['x']
```

Two problems:

- Breaks the 22 invariant — an invalid op reaches `update()` and crashes mid-loop.
- The crash is an **`assert`** → vanishes under `python -O` → silent field corruption. Contradicts the
  raise-over-assert principle (task 20), and here the assert is **user-reachable**, not an internal invariant.

The real fix is upstream of the gate: **components must have globally-unique field names.** This is already
an unstated assumption everywhere — `query()` sums field names across components and a pool merges fields by
name; a collision would silently alias two components' data. Enforce it once, at construction.

## What

Add a guard in `World`'s construction-time component check that rejects any world whose components share a
field name. Turn a commit-time (or query-time) crash into a **construction-time `raise`** — the earliest,
loudest place, where the offending components are right in front of you. Per raise-over-assert, use a real
exception (`ValueError`) so it survives `-O`.

## How

While walking `fields(c)`, accumulate `field_name -> component_name` and raise if a name repeats across two
different components:

```python
seen: dict[str, str] = {}
for c in components:
    for f in fields(c):
        if f.name in seen:
            raise ValueError(f"Field name '{f.name}' declared by both '{seen[f.name]}' and '{c.__name__}'; "
                             "component field names must be globally unique")
        seen[f.name] = c.__name__
```

(Fold into the existing per-field loop — don't add a second pass.)

Once landed, the 178 exhaustiveness hole closes: `add_component` can never introduce a colliding field
because no two registered components can share one. The `_do_add_component` disjoint `assert` becomes a true
never-happens internal invariant.

## Tests — done for the guard

- `test_world.py::test_world_rejects_duplicate_field_name_across_components` — two components sharing a field
  name → `World([...])` raises `ValueError`. xfail marker dropped, live and green.
- `test_world.py::test_field_to_component_is_keyed_by_field_name` / `..._resolves_the_name_entity_set_data_passes`
  — the uniqueness rule is what makes the field→component map usable; these pin the consumer.

## Subtasks (redundancy cleanups from the 178 review — the gate works, these are tidy-ups)

These were written to make `CommandBuffer.append` the *single, only* validation gate. **#44 (2026-07-27)
settled the ownership question the other way**, and it was the right way: which side gates a verb depends
on whether the verb has a single producer. ADD_ENTITY has one (`world.py:82`) → the world owns it.
ADD_COMPONENT / REMOVE_COMPONENT do not (`Entity` queues a bare command) → `append` owns those. So the goal
is not "one gate for everything", it is **one gate per verb**, which is what the code now does.

1. **~~ADD_ENTITY is validated twice.~~ DONE (#44, `3ba0e76`)** — resolved toward `World.add_entity`, not
   `append`: the world validates and fills defaults, `append` stages the command verbatim. 1.97× on
   `add_entity`, 1.36× on a churn pair, and w5 improved 25–57% against the whole field.

2. **~~`add_entity` mutates before it validates.~~ MOOT (#44)** — the concern was that deleting the
   pre-validation (subtask 1 as originally written) would let `append` raise *after* `_last_id` and
   `live_entities[id]=None` were already committed. #44 kept the validation in `add_entity`, ahead of both
   mutations, and left `append` with nothing to raise for ADD_ENTITY but the liveness check `add_entity`
   just satisfied. The ordering is now load-bearing rather than incidental, and it is pinned by
   `test_world.py::test_rejected_add_entity_burns_no_id` + `..._leaves_no_live_entity_and_nothing_staged`.

3. **Default-filling is asymmetric.** ADD_ENTITY carries a complete arg set into the buffer; ADD_COMPONENT
   defers to commit (`_do_add_component`). Both correct, but pick one story. #44 changed *where* the one
   story would live: it is now **the producer fills defaults** (`World.add_entity` does), so the symmetric
   fix is `Entity.add_component` filling them too — not `append` filling them for both. Still open; pinned
   by the `xfail(strict=True)` `test_command_buffer.py::test_buffer_alone_fills_defaults_into_a_staged_add_component`,
   which flips to XPASS when it lands.

Land the field-name guard first; 1–3 are lower value.

### Tests (rewritten 2026-07-27 for the shape #44 actually took)

Subtask 1 has no observable behaviour of its own — deleting dead work is invisible — so the guard has to
count, not assert an outcome:

- `test_world.py::test_spawn_validates_exactly_once`, `..._computes_defaults_exactly_once`,
  `test_rejected_spawn_validates_exactly_once_too` — wrap the bound method on the world instance and assert
  one call. The buffer reaches the world through `self.world` (same instance), so a re-added pass on
  *either* side lands in the same counter. This is the regression guard.
- `test_command_buffer.py::test_buffer_stages_a_spawn_verbatim` — the buffer's half: the args dict it
  stages is the same object it was handed, with no default injected.
- `test_command_buffer.py::test_buffer_rejects_a_spawn_for_an_unregistered_id` — the liveness check is the
  one thing `append` still does for ADD_ENTITY, and with validation gone it is the only thing between a
  hand-built spawn command and `update()`.
- `test_world.py::test_rejected_add_entity_burns_no_id` and `..._leaves_no_live_entity_and_nothing_staged` —
  **still subtask 2's net**, now guarding the validate-before-mutate ordering rather than a planned refactor.
  Mutation-checked (`test/manual/23-single-gate/mutation_check.py`).

The five ADD_ENTITY rejection cases (shape / dtype / non-ndarray / unknown field / missing-required) and the
duplicate-components case **moved from `test_command_buffer.py` to `test_world.py`**: they are now raised by
`World.add_entity`, and testing them against the buffer would defend it from commands its only caller cannot
build. Two of them (missing-required, non-ndarray) had no `add_entity`-level test at all before the move.

Subtask 3 is the only one left with a real behaviour change, so it keeps the xfail:
- `test_command_buffer.py::test_buffer_alone_fills_defaults_into_a_staged_add_component` —
  `xfail(strict=True)`, flips to XPASS when `Entity.add_component` fills defaults like `World.add_entity` does.

## Relates

- **Closes the one confirmed exhaustiveness gap in #22.** 22's gate is otherwise complete.
- Same spirit as #20 (raise-over-assert): the current failure is a `-O`-erasable assert on user-reachable input.
