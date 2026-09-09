# Retire `[:]` from the user-facing docs, messages and examples

**Created**: 2026-07-25
**Closed**: 2026-07-25
**Priority**: 3

## Resolution
`qr.f = value` is the one taught write. Swept 43 sites, reword only, no logic change:
- `microecs/`: `qr_field.py:49`, `:62`, `:65`, `:87` and `:69` (which also dropped its unreachable `[i][...]`
  advice); `query_result.py:65-66`.
- `docs/source/`: `systems.md:9,20,30` · `example-1-hello-world.md:46` + `:49` (prose) ·
  `example-2-moving-colliding-balls.md:26,35,40` · `primitives.md:26` · `benchmarks.md:12` (row label).
- `examples/`: `01:42` · `02:78,91,97` · `03:86,99` · `04:100` · `05-benchmark-workloads/` `w1:39,41`
  `w2:40,42,45` `w3:41,42,43` `w5:50` `w6:51,52,53,92,94` `w7:44`.
- robosim `src/robolib/systems.py:44,48,49,53,59,60,65,76,77,81,82,84` and `src/robosim/robosim.py:257`
  (robosim commit `8e51542`).

Verified: microecs suite 295 passed / 10 xfailed; `examples/04` runs; all six edited benchmark workloads verify
against `common.references` at N=200 (`ok=True`); robosim `test/e2e/run_all.sh` green except the four
protocol-fuzz payloads that already fail on master (stale corpus, separate task).

## Deliberately left alone
- **Pool writes** — `pool.f = v` *raises* and tells you to use `[:]` (`pool.py:70-72`): `systems.md:24`,
  `04-benchmark-ecs-vs-oop.py:107`, `benchmarks.md:11`. Consequence to own: `qr.f = v` works while `pool.f = v`
  raises, which breaks task 5's "QueryResult mirrors Pool's muscle memory" claim. Accepted (Pool is the internal
  layer, QueryResult the user layer); a follow-up could let Pool scatter too.
- **Entity writes** (`e.position[:]`, `robot.channel[:]` in `simulator_object.py:148-153`) — task 29's path.
- **Raw numpy** `[:]` in `05-benchmark-workloads/probes/boundary.py:67,68,73`.
- **The mechanism** — `query_result.py:61`, `qr_field.py:81`. `qr.f[:] = v` still works; it is just not taught.
- **`primitives.md:44`** — lists `qr.f[:]` among the forms that raise. True for a *read* today; becomes wrong
  once task 33 lands, so it moved there.

## Spun off
- **Task 33** — `__getitem__` rejects the bare `slice(None)` that `__setitem__` accepts, so `qr.f[:] += 1` raises
  while `qr.f[:] = v` works. Found by this sweep; the retired idiom was never uniformly available.

## Relates
- Split from task 28 (the `__setattr__` guard, closed 2026-07-25).
- Source: robosim `183-feedback` item 11.
