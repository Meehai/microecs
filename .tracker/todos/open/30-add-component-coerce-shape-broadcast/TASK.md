# add_component / add_entity: coerce field data to the declared shape via numpy broadcast

**Created**: 2026-07-24
**Priority**: 3

## Why
Today a field whose shape doesn't **exactly** match the component metadata is rejected. `_validate_component`
(`world.py:219`) does `if field.shape != shape: raise ValueError`. So a scalar for a `(1,)` field crashes even
though it's obviously convertible:

    np.array(1, dtype="int32")          # shape () -> rejected for a (1,) field
    np.array(1, dtype="int32").reshape((1,))   # what the caller has to type instead

This is friction with no payoff: `()` -> `(1,)` is unambiguous under numpy's own rules. The feedback asks us to
"try to convert based on the metadata shape using numpy rules" instead of forcing the caller to reshape by hand.

## The tension with #20 (read this first)
`#20` made shape mismatch crash **eagerly and strictly** — the whole point was "no silent shape surprises, fail
fast." This task **softens** that: exact-equality → broadcast-compatible + coerce. That is a deliberate reversal
of part of #20, not an oversight. Keep #20's *eager* timing and its *dtype* strictness (see below); only the
*shape* rule loosens from `==` to `np.broadcast_to`.

## What
When a provided field is not the declared shape but **is broadcastable to it**, coerce it (a copy at the declared
shape) and accept. Only raise `ValueError` if numpy itself can't broadcast it. dtype stays **strict** — silent
`int`↔`float` coercion loses data, that's a different, riskier ask and is out of scope here.

- `np.array(1, "int32")` (`()`)  for a `(1,)` field  -> ok, becomes `[1]`.
- `(1,)` value for a `(6,)` field -> broadcastable, so accepted in principle (numpy fans it to six copies).
- `(2,)` value for a `(3,)` field -> still raises (not broadcastable).
- wrong dtype -> still raises (unchanged from #20).

## How
1. **Split check from coerce.** `_validate_component` (`world.py:198`) is documented pure ("No mutation, no
   return"). Broadcasting produces a new array, so the coercion can't live in the pure validator. Mirror #21's
   default-fill: the fill/coerce path returns the prepared arrays that actually get written to the pool; the
   validator keeps only the checks it can do without mutating.
2. **Replace the shape gate** at `world.py:219` with a broadcast attempt:
   `try: field = np.broadcast_to(field, shape).copy()` (or `np.reshape` if we keep it to size-preserving only —
   see open question) `except ValueError: raise ValueError("… not broadcastable to {shape}")`. `.copy()` is
   **required** — `broadcast_to` returns a read-only, stride-0 view; writing it into a pool row without copying
   either fails or aliases.
3. **Both entry points.** The validator is shared by `add_entity` and `add_component` (via `_do_add_component`),
   so both get the behavior — keep them symmetric, exactly as #20/#21 did. Feedback named `add_component`; don't
   make `add_entity` diverge.
4. **dtype untouched.** Leave the `world.py:217` dtype check as a hard `raise`.

## Decided: broadcastable is the rule
**Anything numpy can broadcast to the declared shape is accepted** — that's the semantics. Whether the impl
uses the permissive `np.broadcast_to` (allows `(1,)`->`(6,)` fan-out) or the narrower size-preserving
`np.reshape` (`()`<->`(1,)` only) is an **implementation-time call**, not a design blocker — both satisfy the
rule for the feedback's example. Watch the one footgun either way: `broadcast_to` returns a read-only stride-0
view, so `.copy()` before writing to the pool.

## Validation (tester)
- scalar `()` accepted for a `(1,)` field, stored value equals the reshaped array.
- non-broadcastable shape (`(2,)` for `(3,)`) still raises `ValueError` **at the call** (eager, per #20).
- wrong dtype still raises `TypeError` (dtype stays strict).
- same behavior via `add_entity` and `entity.add_component`.
- stored array is writable / not an alias of the caller's input (the `.copy()`).

## Relates
- **Softens** `#20` (eager wrong-shape crash) — this is the one place #20's strict `==` becomes broadcast.
- Reuses the fill-and-return plumbing pattern from `#21` (default fill in the coerce path).
- Shape check to change: `microecs/world.py:219` (`_validate_component`); dtype check to leave: `:217`.
- Source: robosim `183-feedback` item 2.
