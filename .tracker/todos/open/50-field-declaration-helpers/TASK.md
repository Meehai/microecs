# `f32`/`i32`/`b8`/`obj` field-declaration helpers: stop every app writing the same one

**Created**: 2026-07-27
**Priority**: 2

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
- [#21](../../done/21-field-default-metadata/TASK.md) — added `default`, the field that made the line long.
- [#11](../../done/11-extra-field-metadata/TASK.md) — added `extra_metadata`, which is why `**extra` is
  mandatory here.
