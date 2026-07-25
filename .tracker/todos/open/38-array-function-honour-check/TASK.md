# `QRField` accepts numpy functions it cannot honour: silent per-pool results + unbounded recursion

**Created**: 2026-07-25
**Priority**: 1

## Why

`_apply_fn_on_parts` (`qr_field.py:34-46`) runs *any* numpy function once per pool and stitches the parts back.
That is only valid for functions that are **row-independent along axis 0**. Nothing checks that. The only guard
is `assert len(part_result) == part.shape[0]` (`qr_field.py:44`) — a weak proxy that lets two bad classes through:

1. **Row-coupled N→N functions return silently wrong numbers.** Length matches, so nothing fires:

   ```
   input [5, 1, 4, 2] across 2 pools
   np.sort(f, axis=0)    correct [1,2,4,5]     microecs [1,5,2,4]   <- sorted per pool
   np.cumsum(f, axis=0)  correct [5,6,10,12]   microecs [5,6,4,6]   <- accumulated per pool
   ```
   Same for `argsort`, `diff`, `median(axis=0)`. **No error, no warning.** Documented as a footgun in
   `docs/source/primitives.md`, but a footgun that returns plausible wrong numbers should be a rejection.

2. **Sequence-arg functions recurse without bound.** `_chunk` (`qr_field.py:25-32`) unwraps a bare `QRField`
   or a same-length ndarray, but not a *list* of them. So `np.concatenate([f, f])` passes the list through
   unchanged, re-dispatches on the same `QRField`, and doubles the part count each level — 333 frames, then
   `RecursionError`. Same for `stack` / `vstack` / `hstack`. Violates "bounded everything".

3. **The guards are asserts on a user-reachable path.** `qr_field.py:44` and `:73` fire on ordinary user code
   and vanish under `python -O`, where `np.sum(qr.f, axis=0)` returns a malformed `QRField` instead of raising.
   `docs/source/primitives.md` states the policy this breaks.

Verified: `test/manual/structural-audit/probe_array_function.py`.

## What

`QRField` either honours a function correctly or rejects it. No third outcome.

## How (dev writes the code)

- **Allow-list the functions the per-pool decomposition is actually valid for** (elementwise ufuncs,
  `where`/`clip`, axis>=1 reductions) and `return NotImplemented` for the rest, with a message pointing at
  `.numpy()`. An allow-list is the honest shape here: row-independence is not something we can detect.
- Handle sequence args in `_chunk` (unwrap lists/tuples of `QRField`) **or** reject them explicitly — either
  is fine, but the recursion must be impossible, not merely unlikely.
- Turn `qr_field.py:44` and `:73` into `raise`.

## Validation (tester)

Table-driven: one case per numpy function in `probe_array_function.py`, asserting each is either
value-correct against `f.numpy()` or raises. Run the whole set under `-O` too.

## Relates

- **#37** — the same functions behave differently at 1 pool (where `_QRArray` gives real numpy semantics).
- **#34** (asserts-to-raises sweep) missed these two.
