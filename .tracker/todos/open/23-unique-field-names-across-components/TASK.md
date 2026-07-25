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

**Still open: subtasks 1–3** (the "single gate" tidy-ups at the bottom). Re-verified 2026-07-25 — `add_entity`
(`world.py:76-77`) still runs `_validate_components` + `_defaults_for`, and `CommandBuffer.append`
(`command_buffer.py:73-76`) still runs both again on the same args; `_last_id` / `live_entities` are still
mutated before the `append` that could raise.

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

These make `CommandBuffer.append` the *single, only* validation gate. None are correctness bugs; they
undercut the "one gate" story and leave dead work / fragile coupling.

1. **ADD_ENTITY is validated twice.** `world.add_entity` runs `_validate_components` + `_defaults_for`, then
   `append` runs *both again* on the same args. The second pass is a no-op → dead work, and it contradicts
   "append is the single gate". Make `add_entity` a thin command-builder: build the `ADD_ENTITY` command,
   let `append` do the only validation + default-fill.

2. **`add_entity` mutates before it validates.** It bumps `_last_id` and inserts `live_entities[id]=None`
   *before* `append`. Safe **only because** subtask 1's redundant pre-validation guarantees append won't
   raise — remove that pre-validation and a rejected spawn leaks a dangling `live_entities` entry and a burnt
   id. Fix with 1: build → `append` (validate) → *then* commit `_last_id` / `live_entities`.

3. **Default-filling is asymmetric.** ADD_ENTITY fills defaults into `command.args` eagerly; ADD_COMPONENT
   defers to commit. Both correct, but pick one story — simplest is fill-at-append for both, so a staged
   command always carries a complete arg set and `update()` never computes defaults. (Do after 1/2.)

Land the field-name guard first; 1–3 are lower value.

### Tests for 1–3 (in place 2026-07-25)

Subtask 1 has no observable behaviour of its own (deleting dead work), so what protects it is the pair of
invariants around it — both **green today**, so they are a safety net, not a spec:

- `test_command_buffer.py::test_buffer_alone_fully_validates_a_raw_add_entity` (5 cases: shape / dtype /
  non-ndarray / unknown field / missing-required) — proves `append` is already a complete gate for ADD_ENTITY,
  so removing `add_entity`'s pre-pass loses no validation.
- `test_command_buffer.py::test_buffer_alone_fills_defaults_into_a_staged_add_entity` — same for `_defaults_for`.
- `test_world.py::test_rejected_add_entity_burns_no_id` and
  `..._leaves_no_live_entity_and_nothing_staged` — **subtask 2's net.** Mutation-checked
  (`test/manual/23-single-gate/mutation_check.py`): deleting the pre-pass *without* reordering fails exactly
  these two; deleting it *and* reordering (build → append → then commit `_last_id`/`live_entities`) passes.
  So they tell the good refactor from the bad one.

Subtask 3 is the only one with a real behaviour change, so it gets the xfail:
- `test_command_buffer.py::test_buffer_alone_fills_defaults_into_a_staged_add_component` —
  `xfail(strict=True)`, flips to XPASS when ADD_COMPONENT fills defaults at append like ADD_ENTITY does.

## Relates

- **Closes the one confirmed exhaustiveness gap in #22.** 22's gate is otherwise complete.
- Same spirit as #20 (raise-over-assert): the current failure is a `-O`-erasable assert on user-reachable input.
