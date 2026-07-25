# `object` dtype: defaults are shared, and removed entities keep their references alive

**Created**: 2026-07-25
**Priority**: 2

## Why

An `object` field holds a real Python object (socket, GPU handle, dict). Two lifetime bugs that numeric fields
cannot have:

1. **A `default=` on an `object` field is shared by every entity.** `_defaults_for` (`world.py:252`) does
   `default.copy()` — an `ndarray` copy, which is shallow. Two entities defaulted from
   `np.array([{...}], dtype=object)` mutate the *same* dict. Verified: entity A sets `hits=42`, entity B reads 42.
   Latent — robosim uses `default=None` on all 8 of its object fields — but it is a trap laid for the next user.

2. **Removing an entity does not drop its reference.** `Pool.remove_entity` (`pool.py:51-53`) pop-swaps the live
   rows and decrements `size`, but never clears the vacated slot, so `data[f][size:capacity]` keeps pointing at
   the removed entity's object. Verified with a weakref: alive after `remove_entity` + `update()` + `gc.collect()`.
   The shrink-realloc that would eventually overwrite it only fires when `capacity > INITIAL_CAPACITY (100)` —
   so at robosim's scale (~10² entities) capacity stays 100 and those slots are **never** cleared. Real for
   robosim: `model`, `fpv_camera`, `fpv_texture`, `channel` are object fields holding raylib handles and sockets.

Verified: `test/manual/structural-audit/probe_invariants.py` probe 6, and the weakref probe in the audit notes.

## What

An `object` field behaves like the reference it is: independent per entity, released on removal.

## How (dev writes the code)

1. Deep-copy `object`-dtype defaults (`copy.deepcopy` on the boxed object), **or** reject a non-`None` default
   on an `object` field at `World` construction. Rejecting is the grug option and probably right — a shared
   default is almost never what anyone means.
2. In `Pool.remove_entity`, after the pop-swap, write `None` into the vacated slot **for object columns only**
   (skip numeric ones — that is pure cost for no benefit).

## Validation (tester)

Weakref both: two entities defaulted from one object default must not alias; an object stored on a removed
entity must be collectable after `update()` + `gc.collect()`, with the pool still alive and below the
realloc-shrink threshold.

## Relates

- **#39** — snapshotting staged writes is shallow by design; the referenced object stays shared. Same theme,
  different call.
