# Remove the `str` dtype — keep numeric + `object`; python strings go in `dtype=object`

**Created**: 2026-06-11
**Priority**: 2
**Breaking**: yes (pre-1.0, accepted) — but no known consumer uses it (see Why)
**Status**: DONE (2026-06-11) — dev dropped `"str"` from `world.py:212` (now
`{"float32", "int32", "bool", "object"}`); tests + README updated; full suite green (232 passed).

## Why

`str` is one of the five allowed dtypes (`world.py:212`,
`dtypes = {"float32", "int32", "bool", "str", "object"}`) but it is **dead and broken**:

1. **Nobody uses it.** It appears in exactly one place in the whole repo — the allowed set above.
   No source, no example, no test, no README sample ever declares `"dtype": "str"`. It is a path
   that is permitted but never travelled.

2. **It can't work in an SoA pool anyway.** Numpy strings are fixed-width (`<U n`). The pool
   pre-allocates with `np.empty(shape, dtype="str")`, and `np.empty(dtype="str")` is **`<U1`** —
   width one. So every write truncates to the first character:

   ```python
   a = np.empty((4, 1), dtype="str")   # dtype is <U1
   a[0, 0] = "hello"                   # -> "h"   (silent data loss)
   ```

   A pre-allocated pool can never size the width ahead of the data. Fixed-width strings and a
   grow-by-`np.empty` pool are fundamentally incompatible. The feature was corrupting data, quietly.

3. **`object` already covers it, correctly.** A python `str` is just a python object. Declare the
   field `dtype="object"` and store `np.array(["hello"], dtype=object)` — no width, no truncation,
   full string. Serialization already does the right thing: `entity.to_dict` (`entity.py:48-51`)
   routes `object` through `.item()`, so a stored string round-trips as a plain string.

Grug rule: one obvious way to hold non-numeric data (`object`), not two — one of which lies.

## Fix (DEVELOPER — non-test code)

- `microecs/world.py:212` — drop `"str"` from the set:
  `dtypes = {"float32", "int32", "bool", "object"}`. Declaring `dtype="str"` then fails the
  existing assert at `world.py:224` (`f"{fmd} not in {dtypes}"`) at `World` construction — fail
  loud, fail early, exactly where the bad field is defined.

That is the whole code change. No pool change, no serialization change, no Field change.

## Migration (what a user with a string field does)

| old | new |
|---|---|
| `field(metadata={"shape": (1,), "dtype": "str"})` | `field(metadata={"shape": (1,), "dtype": "object"})` |
| `name=np.array(["hello"], dtype="str")` | `name=np.array(["hello"], dtype=object)` |

Behaviour is strictly better: no width cap, no truncation, serializes as a real string.

## Done when

- `dtype="str"` raises at `World(...)` construction (caught by the existing `not in dtypes` assert),
  with the message naming the four remaining dtypes.
- `object` fields holding python strings still add / read / write / serialize unchanged.
- README dtype count is corrected.
- Full suite green.

## Tests / docs (tester — me) — DONE

- [x] `test/unit/test_world.py::test_world_rejects_str_dtype_component` — a `dtype="str"` field raises
  `AssertionError` (`match="str not in"`) at `World([...])` construction. Sits in the object-dtype
  section next to the accept test.
- [x] `test/unit/test_world.py::test_object_field_holds_python_string_and_compares_by_equality` — the
  replacement path: `np.array(["enemy"], dtype=object)` in a `component_kind`-style field, asserts the
  full string survives (no `<U` truncation), and `==` works element-wise and vectorised
  (`pool.kind[:, 0] == "enemy"`).
- [x] Survivors `float32` / `int32` / `bool` / `object` still accepted — covered by the existing
  `test_world_accepts_object_dtype_component` + the numeric component tests throughout the file.
- [x] `README.md:24` — now "We support 4 dtypes only: `int32`, `float32`, `bool` and `object`" with a
  note that python strings go in `dtype=object` and *why* (numpy fixed-width strings truncate).

Full suite green: **232 passed** (was 224; +2 new tests, +6 others unrelated already present).

## Out of scope

- Any new "variable-length string" / `bytes` / `np.object_`-backed string helper. The answer is
  plain `object`; do not invent a typed string field.
- Touching numeric or `bool` dtype handling.
- The `object` serialization path (`.item()`) — already correct, unchanged.

## Related

- `microecs/world.py:212` (allowed dtypes set), `world.py:224` (the assert that will now reject str).
- `microecs/entity.py:48-51` — object vs numeric serialization split (the reason strings survive in
  `object`).
- `microecs/pool.py:25` — `np.empty(shape, dtype=...)` pre-alloc (the reason fixed-width str can't
  work here).
- README.md:24 — supported-dtypes sentence.
- Follows task `16-remove-qr-field-int-index` — same "delete a dead/asymmetric path, no deprecation"
  pattern.
