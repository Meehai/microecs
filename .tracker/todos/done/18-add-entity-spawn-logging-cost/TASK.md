# Kill the per-entity `logger.debug` in `add_entity` (entity creation was ~19× too slow)

**Created**: 2026-06-12
**Completed**: 2026-06-12
**Priority**: 2
**Status**: DONE — dev removed the line; confirmed via the new cross-library benchmark.

## Outcome

The per-entity `logger.debug(...)` in `add_entity` (old `world.py:70`) was deleted. Measured on this
machine (numpy 2.3, py3.12), N=100k, 2 archetypes:

| metric | before | after | speedup |
|---|---:|---:|---:|
| `add_entity` (per call) | ~137 µs | ~2.7 µs | **~45×** |
| full build (add loop + `update()`) for 100k | ~16.3 s | ~0.85 s | **~19×** |
| step / frame (vectorized update) | 0.16 ms | 0.16 ms | unchanged |

Build went from *catastrophic* (80× slower than any competitor) to *merely heaviest phase* (~4× the
pure-python ECSs, ~60× xecs' Rust bulk spawn). The step — microecs' actual selling point — was never
affected. Surfaced by `examples/05-benchmark-vs-similar` (also the validation tool).

## Why

`add_entity` is on the scene-build hot path (called once per entity). cProfile of 5 000 calls:

```
3.001 s total — 2.975 s (99%) in loggez .debug()/.log()/_loglevel_colorize/colorize
              (185 000 colorize() calls); the actual ECS work was ~0.026 s (~5 µs/entity).
```

Two compounding costs, both paid **every** call even though nothing is logged:

1. **`loggez` colorizes the message *before* gating on level.** Default level is `INFO` (value 1);
   `DEBUG` is value 2, so the message is **never emitted** — yet the full ANSI-colorize cost is paid
   per entity. This is a `loggez`-side design issue (gate-after-format).
2. **The f-string is built eagerly.** `f"Created entity. ID: {…}. Components: {[c.__name__ for c in components]}"`
   runs a list comprehension + format on every call before `debug()` is even entered (the classic
   `logger.debug(f"...")` anti-pattern). Small next to (1), but free to avoid.

A per-entity spawn log is low value anyway — at 100k entities it's 100k debug lines nobody reads.

## Fix

Delete the line (done). The other two `logger.debug` calls in the package are **fine, left as-is** —
both are rare, not per-entity:

- `world.py:50` — once per `World` (scene creation).
- `pool.py:31` — once per pool capacity-doubling (~log₂N times, ~17 for 100k).

## Tests (tester — to add)

- `test/unit/test_world.py` — regression guard: monkeypatch `microecs.utils.logger.debug` to a counter,
  spawn N entities, assert it is **not** called per entity (count stays O(1), not O(N)). Cheap,
  deterministic, and pins the hot path against a re-introduced log call. (Offered; not yet committed.)
- Full suite must stay green — the fix is a pure deletion; no behavior change.

## Out of scope / follow-ups

- **The `update()` commit is now the larger half of build** (~0.4 s / 100k = ~4 µs/entity): `_add_to_pool`
  appends to each pool's numpy arrays one entity at a time, with a `len(...)==len(...)` assert per entity.
  A **bulk add path** (append M entities to a pool in one numpy op) would close most of the remaining gap
  to xecs' 14 ms. Separate task if entity-creation throughput matters.
- **Durable root-cause fix lives in `loggez`** (gate on level *before* colorizing): would make every
  hot-path `.debug()` across the dev's projects cheap-when-disabled, not just this one. Separate repo.

## Related

- `microecs/world.py:63` `add_entity` (removed log was the last line of the body); `world.py:136`
  `_add_to_pool` (the commit-side per-entity cost noted above).
- `microecs/pool.py:31`, `microecs/world.py:50` — the two rare logs left in place.
- `microecs/utils.py:8` `logger = make_logger("MICROECS")`; `loggez` colorizes before gating.
- `examples/05-benchmark-vs-similar/` — found it, validates it (run `python bench_microecs.py`).
- `.tracker/plans/1-comparison-with-other-projects.md` — competitive analysis this benchmark feeds.
