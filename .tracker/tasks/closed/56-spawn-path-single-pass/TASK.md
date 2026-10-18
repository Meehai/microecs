# Spawn path: one pass per field, one copy per field

**Created**: 2026-10-18
**Priority**: 2
**Closed**: 2026-10-18

## Why

`#50` (array-like conversion) and `#39` (staged writes are snapshots) are both correct and both stay. But
together they made every spawn ~25% dearer. In-process A/B against `6fc0d21`: churn step +24–31%, `build`
+17–32% on every workload, migrate +5–16%. microecs' churn lead over EnTT at 100k fell from 1.65× to 1.33×.
See `test/benchmarks/workloads-vs-other-libraries/FINDINGS.md`.

The cost is the shape of the path, not the features. Per field, a spawn now does:

- a convert loop;
- a validate loop;
- a defaults loop that copies;
- a second copy loop in `CommandBuffer.append`.

Defaults get copied twice. A caller's ndarray is checked in one pass and copied in another. Of a ~6.3 µs
`add_entity`, conversion is ~860 ns and the `append` copy ~680 ns.

**Making arrays read-only instead of copying is rejected.** The flag guards one array object, not its memory,
and it guards neither metadata nor other views. A wrong value lands silently when:

- the caller passes a row view and then writes `big[i] = …`;
- a system does `query(...).position += …` in the same tick;
- two worlds stage the same array;
- other code flips the flag back to writable.

Six `#39` tests fail, and the scratch-buffer spawn loop raises.
Evidence: `test/manual/spawn-path/readonly_probe.py`.

## What

- Behavior is unchanged. The full suite passes untouched. That covers the `#39` snapshot tests, the `#50`
  conversion tests, exception types for every rejection, and `#40` object-dtype shallow copies.
- Every staged array is private (owns its data, shares no memory with an input) and is copied **once**.
- Churn (w5) and migrate (w7) step/build return to the `6fc0d21` level within noise.

## How (dev writes the code)

Fold conversion and the snapshot into the existing per-field loop of `World._validate_component`:

1. `None` check;
2. array-like → `np.broadcast_to(np.asarray(v, dtype), shape)`;
3. dtype/shape check;
4. `res[name] = field.copy()`.

It returns the private data, and `_validate_components` merges and returns it. Then:

- `add_entity` stages that data;
- the `ADD_COMPONENT` branch of `append` uses the same call;
- `_convert_components_data` and `append`'s tail copy are deleted, and so is `numpy` in `command_buffer.py`.

That is one loop and one copy per field, and the library shrinks by 16 lines. Defaults stay as they are:
one copy in `_defaults_for`.

The diff was checked on a scratch copy of the library: suite green, pylint 10.00, ruff clean. Its speed was
measured with `test/manual/spawn-path/compare.py` (old tree vs current vs candidate, interleaved, median of 8):

| metric | old `6fc0d21` | current | candidate |
|---|--:|--:|--:|
| `add_entity` ns | 4972 | 7560 (1.52×) | 5611 (1.13×) |
| w5 churn step ms @20k | 2.051 | 2.603 (1.27×) | 2.002 (0.98×) |
| w7 migrate step ms @20k | 2.053 | 2.135 (1.04×) | 1.799 (0.88×) |

Copying once without merging the loops recovers only ~40% (`add_entity` 1.29×). Most of the cost is the
separate conversion loop. A deeper fused variant (`fused.py`, which also preserves the exact raise order) is
no faster, so it is not worth its complexity.

## Validation (tester)

Tests pin behavior, not the copy. The buffer-level test that asserted `append` copies (`is not`) is gone.
Covered at the public API: mutate, resize, scratch loop, object fields, and an `array.array` source that
shares memory, for both `add_entity` and `add_component`. Removing the copy fails 7 of them. Done when the
suite is green on the change, and the benchmark rerun plus `ab_versions.py` show w5/w7 back at the
`6fc0d21` level. Update FINDINGS.md.

## What landed

`f2bf8cb` + `4ed0043` + `befc0f5`. `World._validate_component` converts, validates and snapshots in one pass
per field, and returns the staged data:

- array-likes go through `np.array(field, dtype)` + `broadcast_to`; `np.array` always builds a new array, so
  that one allocation is the snapshot (no second copy for lists);
- ndarrays are checked first, then `.copy()`d.

Also removed: `_convert_components_data`, `append`'s tail copy, `numpy` in `command_buffer.py`, and the dead
`strict` parameter (and its reserved name). Suite 559 passed / 12 xfailed, pylint 10.00, ruff clean, robosim
e2e green.

Verified beyond the suite:

- an aliasing hunt (list, nested list, scalar, `array.array`, memoryview, `__array__` / `__array_interface__`
  objects, torch): no commit shares memory with the caller;
- a differential check against `da6616d` over 120k single-problem inputs: same accept/reject, exception types
  and committed values, except `add_entity` with an unknown component, which now raises `ValueError` instead of
  a bare `KeyError`.

Perf, in-process A/B: churn step 0.98× and migrate 0.88–0.99× of `6fc0d21`, i.e. back to pre-#50/#39 speed.
FINDINGS.md has w5/w7 extrapolated from the A/B against the published run (churn lead at 100k 1.33× → 1.47×
EnTT; winner map unchanged).

Left as-is, small follow-ups if they ever matter:

- `add_component(Unknown)` still raises a bare `KeyError`;
- objects with an old-style `__array__` (no `copy` keyword) now trigger a numpy `DeprecationWarning`;
- `_defaults_for` copies a default that the pool write copies again, on the in-`update()` add-component path.

## Relates

- **#39** snapshot semantics, **#50** conversion, **#40** object-dtype lifetimes: the contracts to preserve.
- **#44 / #49**: same lesson. Per-entity Python passes, not storage, decide spawn cost. The `#49` batch-spawn
  API stays deferred.
