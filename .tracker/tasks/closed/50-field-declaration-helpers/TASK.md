# `f32`/`i32`/`b8`/`obj` field-declaration helpers: stop every app writing the same one

**Created**: 2026-07-27
**Priority**: 2
**Closed**: 2026-09-09

**Status**: ✅ Done (2026-09-09) — landed on branch `microecs-update`
(commits `2ebca20` + `dd4fa5f`), pending CI + merge to master. Builders + broadcastable
defaults + array-like data conversion; 513 microecs tests / 1079 repo tests green,
pylint 10.00 (microecs rcfile), ruff clean.

## What landed

Implemented in `microecs/component.py` (not a new `fields.py`), exported from `__init__`:

```python
f32((2, ), serializable=True, comment="...", default=zeros)          # float32
i32((2, ), serializable=True, comment="...", default=zeros)          # int32
boolf((1, ), serializable=True, comment="...", default=zeros)        # bool  (named boolf, not b8)
obj((1, ), serializable=True, comment="...", default=None)           # object
```

- `zeros` sentinel → `np.zeros(shape, dtype)` (all-zero default in one word).
- **Defaults broadcastable**: `World.__init__` converts list/tuple/scalar defaults to the field's
  dtype/shape (via `np.broadcast_to`) and validates dtype/shape there, exactly like explicit data.
- **`add_entity` data conversion** (`World._convert_components_data`): lists/tuples/scalars/strings are
  converted to the field's dtype/shape at the call; explicit ndarrays pass through untouched so the
  strict dtype/shape validation still fires.
- **`None` semantics pinned**: key present + `None` is DATA for `object` fields (`array([None])`,
  e.g. `texture_path: null`); `None` for a numeric field is a caller bug → `TypeError` (no silent NaN);
  key absent is omission → default or the required `KeyError`. Unknown kwargs are preserved through
  conversion so the "Extra fields" guard still raises (bug found in review, fixed before merge).
- Tests: `test/unit/test_component_builders.py` (6), `test/unit/test_add_entity_data_conversion.py` (6),
  and the old non-ndarray rejection test superseded by the new contract.

### Deviations from this plan

- **`**extra` passthrough → explicit `serializable`/`comment` kwargs.** robosim's two required extras
  made passthrough the common case anyway; explicit params keep `World` construction's
  `metadata.keys() == expected` contract obvious and call sites short.
- **`b8` → `boolf`** (fuller name; `bool` shadows the builtin either way).
- **`shape=int` convenience NOT taken** — shape must be a tuple.
- **Default conversion moved from declaration time to `World.__init__`** — the same broadcast rules
  now apply to defaults AND `add_entity` data, one code path (`_convert_components_data` / broadcast).
- **`add_entity` is NOT strict for array-likes** (plan said it stays strict) — the data tolerance is
  deliberate: maps/plugins/loaders pass JSON lists; strictness is preserved for explicit ndarrays.

### Leftover (not in this task's core)

- Docs block in `docs/source/primitives.md` — **done** (2026-09-09: "Declaring fields" + "Broadcasting
  data everywhere" sections). The `serializable` note in `example-3-serialization.md` — still a follow-up.
- Adopting the helpers in `examples/01..03` — explicitly optional per the plan.

### TODO (post-merge)

- `_convert_components_data`'s `TypeError` for numeric-`None` uses the bare field name
  (`'velocity': None is not a valid float32 value`) — every other validation message names the component
  too (`'HasMotion/velocity'`). The converter has the component in scope, so it is a one-token change at
  the raise site.

## Why

A component field is 90 characters of ceremony for "two floats":

```python
position: np.ndarray = field(metadata={"shape": (2, ), "dtype": "float32", "default": None})   # 90 chars
position: np.ndarray = f32((2, ))                                                             # 31
```

Measured in this repo: **138 field declarations, median 95 chars, longest 120**. And the tell — **11 files
independently wrote their own `f32`** (`shooter.py`, `probes.py`, `fuzz.py`, four `structural-audit` probes,
`13-query-cost/*`, `41-failed-update/probe.py`, `docs-example3/verify_snippets.py`). Three of them also wrote
`i32`, two wrote `obj`. When every consumer writes the same 3-line helper, the helper belongs in the library.
This is row 1 of plan 2's "helpers an app still has to write" table, and the only one there that is pure sugar.

With `extra_metadata` it gets worse, because the key is mandatory on **every** field. robosim requires two
(`serializable`, `comment`), so its 25 declarations are 4-line blocks:

```python
tag: np.ndarray = field(metadata={"shape": (1, ), "dtype": "object", "serializable": True,
                                  "default": None,
                                  "comment": "The tag of this entity (e.g. robot, ground floor etc.)"})
tag: np.ndarray = obj((1, ), serializable=True, comment="The tag of this entity (e.g. robot, ...)")
```

Second, smaller win: the four legal dtypes are currently a `set` literal inside `World._check_components`
(`world.py:296`) — invisible from the API. Four named helpers **are** that whitelist, discoverable by import.

## What

`microecs/fields.py`, one private builder + four public aliases, ~15 LoC, exported from `__init__`:

```python
def _decl(dtype: str, shape: tuple[int, ...] | int, default=None, **extra) -> Field:
    shape = (shape, ) if isinstance(shape, int) else shape
    default = None if default is None else np.broadcast_to(np.asarray(default, dtype), shape).copy()
    return field(metadata={"shape": shape, "dtype": dtype, "default": default, **extra})

f32 = partial(_decl, "float32");  i32 = partial(_decl, "int32")
b8  = partial(_decl, "bool");     obj = partial(_decl, "object")
```

Three deliberate conveniences, each one a rule the library already follows elsewhere:

1. **`**extra` passthrough** — non-negotiable, or the helper is unusable in any world with `extra_metadata`.
   A missing required key still raises at `World(...)` construction, unchanged.
2. **`shape=1` ⇒ `(1, )`** — an int is unambiguous, and `(1, )` is the most-typed shape in the repo.
3. **`default` accepts anything numpy accepts** (`f32(2, 0)`, `i32(4, [1, 2, 3, 4])`), converted and
   broadcast to `shape`. Same rule as a data write (#42: writes follow numpy); `add_entity` stays strict.

Names: `f32`/`i32`/`obj` are what the harnesses already chose. `b8` because `bool` shadows the builtin and
numpy's bool is one byte.

### Not this

- **No new declaration syntax** (`radius: f32[1]`, annotation scraping, a `Component` metaclass). That is a
  second way to declare a component, i.e. a divergent duplicate path.
- **`field(metadata=...)` stays first-class**, documented, and used in at least one example. The helper
  returns exactly the same `dataclasses.Field` — no new concept, no new code path, nothing downstream changes.

## How

- New `microecs/fields.py`; add the four names to `microecs/__init__.py` `__all__`.
- Docs: a short "declaring fields" block in [primitives.md](../../../docs/source/primitives.md) showing both
  spellings side by side, and the `**extra` rule in
  [example 3](../../../docs/source/example-3-serialization.md) where `serializable` first appears.
- Adopting it in `examples/01..03` is **optional and separate** (it changes what the tutorials teach). If you
  do, re-copy each file into its page's `## Full code` block —
  `test/integration/test_i_docs_full_code.py` fails until you do, so the two cannot drift.

## Validation

- **Parity is the whole test**: `f32((2, ), [0, 0]).metadata == {hand-written dict}`, per dtype, with and
  without `extra`. If the metadata dicts are equal, every downstream behaviour is equal by construction.
- A `World` built from wrapper-declared components and one built from hand-written ones agree on
  `component_to_shapes` / `_dtypes` / `_defaults`.
- Negative space: a wrapper-declared field **missing** a required `extra_metadata` key still raises at
  `World(...)`; `f32(2, [1, 2, 3])` raises (not broadcastable) at *declaration* time, which is earlier and
  better than at spawn time.

## Relates

- Plan 2 Part 1, "The helpers an app still has to write" — this is the `f32` row.
- [#21](../../closed/21-field-default-metadata/TASK.md) — added `default`, the field that made the line long.
- [#11](../../closed/11-extra-field-metadata/TASK.md) — added `extra_metadata`, which is why `**extra` is
  mandatory here.
