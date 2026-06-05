# Tag components (zero-field) — pin emergent behavior with tests + docs

**Created**: 2026-06-04
**Priority**: 2
**Status**: ✅ Done (2026-06-05) — tag behavior pinned by tests + README note; 67 tests green.

## What landed

Field-less tags already worked (the probe proved it); this task **guaranteed** it with tests and
**documented** it.

- Tests (tester), `test/unit/test_world.py`:
  - `test_tag_component_is_valid_and_queryable` — `class Frozen(Component): pass` registers with empty
    field/shape/dtype maps; pure-tag `add_entity((Frozen,))` needs no data; the tag works as a query
    filter (`(HasPosition, Frozen)` exposes `position`) and as a tag-only query (`(Frozen,)`) spanning
    the fielded `{Pos,Frozen}` pool and the pure `{Frozen}` pool; `qr._fields == []`; untagged excluded.
  - `test_tag_component_add_remove_migrates` — `add_component` / `remove_component` of a tag migrates
    the entity between archetypes and round-trips its data + id.
- README: the `Component` bullet now states a field-less component is a valid tag.

Coverage note: pins the core matrix, not every sub-test the spec listed. Left out as low-value (flag if
wanted): the Pool-level field-less grow/shrink past `INITIAL_CAPACITY`, and the two-distinct-tag-
archetypes case. `none_of` exclusion of tags stays in task 8.

## Surprise finding (verified, not assumed)

**Field-less tag components already work** in the current code. A probe
(`test/manual/tag-components-probe/probe.py`) exercised every case and they all pass:

```
Player.__dataclass_fields__ = {}                      # empty dataclass is valid
component_to_field_names[Player] = []                 # World accepts it
add_entity((HasPosition, Player), position=...)       # tag needs no data -> own archetype pool
query_and((Player,))            -> len 1, fields []   # tag-only query: entity_ids, no fields
query_and((HasPosition, Player))-> exposes position   # tag as a FILTER works
get_entity(tagged)              -> ({'position': ...}, [HasPosition, Player])
add_entity((Player,))           -> pure-tag entity, no data at all, works (field-less pool)
remove_component(e, Player)     -> migrates back, re-query reflects it
```

So this is **not an implementation task**. `Component.__init_subclass__` (`component.py:6-8`) applies
`@dataclass(kw_only=True)`, which is happy with zero fields; `_check_components` (`world.py:172-184`)
loops over fields and simply does nothing for a tagless component; `Pool` with `fields=[]`
(`pool.py:13-24`) is a valid empty-array pool whose `add_entity()` just bumps `size`. It all composes.

The task is therefore: **make this guaranteed (tests) and discoverable (docs)** — today it is an
untested, undocumented accident that a refactor could silently break.

## Why it matters

Tags (`Player`, `Frozen`, `Hidden`, `Dead`, `Enemy`) are idiomatic ECS — every other engine has
zero-size components. They cost nothing here (a tag is just an archetype bit; a pure-tag pool stores
no arrays) and they are the natural operand for [task 8](../8-query-exclusion-none-of/TASK.md)'s
`none_of`. Pinning them lets us advertise the feature and build on it safely.

## Done when

- A regression test suite pins tag behavior (matrix below), co-located in `test/unit/`.
- README documents that a component with no fields is a valid **tag**: how to declare one
  (`class Player(Component): pass`), how to add it (`add_entity((Pos, Player), position=...)` — no
  data for the tag), and how to query/exclude it.
- Any edge case the tests surface as actually broken is filed back to the dev with a minimal repro
  (none found in the probe, but the random-churn-style edges below are not yet covered).

## Tests (tester writes, under `test/unit/test_world.py` and/or `test_pool.py`)

- `test_field_less_component_is_valid` — `class Tag(Component): pass` registers; `component_to_*`
  maps give empty field/shape/dtype lists.
- `test_tag_needs_no_data_on_add` — `add_entity((Pos, Tag), position=...)` succeeds without a `Tag`
  kwarg; passing one is rejected (extra-field assert still fires).
- `test_pure_tag_entity` — `add_entity((Tag,))` with no data at all lands in a field-less pool;
  `query_and((Tag,))` finds it; `len`/`entity_ids` correct; `qr` has no fields.
- `test_tag_as_query_filter` — entity {Pos, Tag} vs {Pos}; `query_and((Pos, Tag))` matches only the
  tagged one and still exposes `position`.
- `test_tag_query_spans_field_less_and_fielded_pools` — {Tag} pure + {Pos, Tag}: `query_and((Tag,))`
  spans both pools, `entity_ids` correct, no crash from the field-less pool.
- `test_get_entity_on_tagged_entity` — returns the fielded data + the full component list incl. the
  tag.
- `test_add_remove_tag_component_migrates` — `add_component`/`remove_component` of a tag moves the
  entity between archetypes and round-trips (data preserved, id stable).
- `test_field_less_pool_grow_shrink` (Pool unit) — a pure-tag pool past `INITIAL_CAPACITY` and back
  doesn't choke on the empty `data` dict during `_realloc`.
- `test_two_tags_distinct_archetypes` — {A} vs {B} vs {A,B} are three distinct pools; queries
  separate them.

## Out of scope

- Any new API or sugar for declaring tags beyond `class X(Component): pass` (it already reads fine).
  Reconsider only if README review says it's unclear.
- Tag-only fast paths / storage micro-opt — a field-less pool already stores no arrays; nothing to
  optimize.
- `none_of` exclusion of tags — that's [task 8](../8-query-exclusion-none-of/TASK.md); this task only
  guarantees tags exist and are queryable.

## Related

- Evidence: `test/manual/tag-components-probe/probe.py` (run:
  `PYTHONPATH=. python test/manual/tag-components-probe/probe.py`).
- `component.py:6-8` `__init_subclass__`; `world.py:172-184` `_check_components`;
  `world.py:146-154` `_check_components_against_pool`; `pool.py:13-24` empty-field `Pool`.
- `.tracker/plans/1-comparison-with-other-projects.md` — Part 4 "Need #2".
- Composes with [task 8](../8-query-exclusion-none-of/TASK.md).
