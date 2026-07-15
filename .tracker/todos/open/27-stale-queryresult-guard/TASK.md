# Detect stale QueryResult use after world.update()

**Created**: 2026-07-15
**Priority**: 3

## Why
`QueryResult` snapshots per-pool array slices at construction (`query_result.py:108`). After
`update()` resizes/compacts pools, a **held** qr's slices point at old buffers -> silent stale reads
and no-op writes. `World` already clears its own query cache on `update()` (`world.py:147`), so
**re-querying is always safe** (fresh qr). The footgun is narrow: only when a caller stashes a qr and
reuses it *across* an `update()` instead of re-querying. Low exposure, but silent corruption when hit
-- and the Field `_cache` (#26) makes long-lived qr more tempting.

## What
O(1) generation guard. No per-query tracking / weakrefs (that would be overkill).

## How (dev writes the code)
- `World._generation` int; bump it right where `update()` already does `self._cache.clear()`
  (`world.py:147`).
- `query()` stamps the returned qr (`_gen` + a world ref).
- Cheap check in QueryResult access (`__getattr__`/`__setattr__`): `if self._gen != world._generation:
  raise` -- **raise, not assert** (survives `python -O`; it rejects bad state). One int compare,
  negligible vs the 0.18us cached access.

## Validation
- test: qr held across `update()` then accessed -> raises; normal re-query path unaffected.
- no measurable hot-path regression (int compare).

Relates: #26 (Field `_cache` extends qr lifetime, raising exposure).
