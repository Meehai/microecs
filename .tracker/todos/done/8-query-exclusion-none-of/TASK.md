# Query exclusion — `none_of` on `query_and`

**Created**: 2026-06-04
**Priority**: 1

## Why

`query_and` only does AND (`world.py:82-99`). There is no way to express "entities with A and B but
**not** C" — the single most common query feature every other ECS has (snecs `~C`, tcod-ecs
`none_of`, flecs `!C`). Concrete need: "move everything with Position+Velocity that is **not** Frozen",
"draw everything Drawable that is **not** Hidden". Today you must split archetypes by hand or filter
after the fact, which defeats the point of a query.

This is the **#1 library gap** from the competitive analysis
(`.tracker/plans/1-comparison-with-other-projects.md`). It is also cheap: the bitmask machinery is
already there.

## Design

The archetype key is a bitmask (`world.py:20`, `component_to_bit = {t: 2**i ...}`); the current match
is a subset test (`world.py:90`):

```python
if (archetype_key & key) == key:   # archetype has ALL of all_of
```

Add an exclusion mask and one more test:

```python
all_mask  = self._make_key(all_of)             # existing `key`
none_mask = self._make_key(none_of) if none_of else 0
...
if (archetype_key & all_mask) == all_mask and (archetype_key & none_mask) == 0:
    res.append(archetype_pool)
```

`(arch & none_mask) == 0` means "archetype shares no bit with the excluded set". `none_of=()` →
`none_mask=0` → always true → **identical to today** (back-compatible).

### Two things to get right

1. **Cache key must include `none_of`.** `_cache` is keyed by the all-of int today (`world.py:33`,
   `world.py:85`). Two queries with the same `all_of` but different `none_of` must not collide → key
   becomes a composite, e.g. `(all_mask, none_mask)`. `update()`'s whole-cache clear (`world.py:45`)
   is unaffected.
2. **Exposed fields are still exactly the `all_of` fields.** This is *why* `none_of` is the clean
   win: every matched pool has every `all_of` field by construction, so the contiguous `_Field` view
   (`query_result.py`) stays valid. `none_of` only filters which pools match; it never adds a field
   that is present in some matched pools but not others.

### API (dev's call — non-test file)

Minimal-churn option (keeps every existing call site working):

```python
def query_and(self, component_types, none_of=()) -> QueryResult: ...
```

Larger option: a new `query(all_of=(), none_of=())` with `query_and` kept as a thin alias. Either is
fine; the minimal kwarg is the grug choice.

## Done when

- `query_and((A, B), none_of=(C,))` returns exactly the entities with A and B and not C, across all
  matching archetypes.
- `none_of=()` (or omitted) behaves byte-for-byte like today (no behavior change for existing callers
  or tests).
- Result exposes the `all_of` fields only; vectorized writes through it still scatter correctly per
  pool.
- Same-`all_of`/different-`none_of` queries get independent cache entries (no collision); the cache
  still invalidates on a mutating `update()`.
- An unregistered `none_of` component raises a clear error (via `_make_key`'s existing assert).
- Contradictory `all_of ∩ none_of` (e.g. `all_of=(A,)`, `none_of=(A,)`) returns an empty result, no
  crash.

## Tests (tester writes, under `test/unit/test_world.py`)

- `test_none_of_excludes_archetype` — three archetypes {Pos}, {Pos,Vel}, {Pos,Vel,Frozen};
  `query_and((Pos,Vel), none_of=(Frozen,))` matches the middle one only.
- `test_none_of_empty_is_unchanged` — `none_of=()` equals plain `query_and` (same pools, same ids).
- `test_none_of_spans_multiple_pools` — exclusion across >1 surviving archetype; `entity_ids` +
  field write-back stay pool-aligned.
- `test_none_of_cache_keyed_separately` — `query_and((Pos,))` and `query_and((Pos,), none_of=(Frozen,))`
  are distinct cached objects; neither serves the other.
- `test_none_of_cache_invalidated_on_mutation` — after adding a Frozen entity + `update()`, the
  excluding query still excludes it.
- `test_none_of_unregistered_component_raises`.
- `test_none_of_contradiction_is_empty` — `all_of` and `none_of` overlap → empty, no crash.
- `test_none_of_with_tag_component` — `none_of` of a field-less tag (composes with task 9):
  `query_and((Pos,), none_of=(Hidden,))`.

## Out of scope

- **`any_of` (OR) and optional components.** Both break the contiguous `_Field` view: an `any_of` /
  optional component is present in *some* matched pools but not others, so its field can't form one
  aligned `(N, …)` column across the result. That needs a separate design (per-pool optional views,
  or a sentinel fill) and a separate decision — do **not** fold it into this task. `none_of` has no
  such problem and ships alone.
- Query DSL / string syntax (flecs-style). Keyword args only.
- Relationship/hierarchy queries.

## Related

- `world.py:82-99` `query_and`; `world.py:90` the subset test; `world.py:20` `component_to_bit`;
  `world.py:165-170` `_make_key`; `world.py:33`/`:45`/`:85` the query cache.
- `query_result.py` — the contiguous `_Field` view that stays valid because exposed fields == `all_of`
  fields.
- `.tracker/plans/1-comparison-with-other-projects.md` — Part 3 (query expressiveness gap) and Part 4.
- Composes with [task 9](../9-tag-components/TASK.md): tags are the natural thing to exclude
  (`none_of=(Frozen,)`).
