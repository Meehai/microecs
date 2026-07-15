# Cut low-N QueryResult/Field per-op overhead (win the 1k–10k range)

**Created**: 2026-07-15
**Closed**: 2026-07-15
**Priority**: 2

## Outcome — DONE, goal exceeded
microecs now **beats xecs across the entire 200–20k band** (was: lost below ~10k) and sits within
2–3.7× of the raw-numpy floor (was 22×). The 500-UAV robosim driver is met with huge headroom
(N=500: 5.4us/frame vs the 16.6ms budget). Benchmark + its deps are archived in this dir (`run.py`,
`requirements.txt`, full 4-stage log in `bench.txt`); see "Re-run" below.

## Why (original)
Cost per frame = `fixed_python_overhead + numpy_work(N)`. At low N the fixed per-op Python object
churn in `QueryResult`/`Field` (a fresh `Field` per `qr.field` access, more per arithmetic op/write)
dwarfed the numpy work, so xecs won the 500–10k band. Isolated in `06-benchmark-vs-xecs-low-entities`.

## What we did (this branch) — no large-N regression
1. **`QRField._cache`** — cache the field object per query (persists across frames via World's query
   cache, cleared on `update()`). Field access 2.96us → 0.18us.
2. **Lazy `_bounds`** (`qr_field.py`) — defer `np.cumsum` (was 69% of every build) to the first
   ndarray-operand `_chunk`; it's unused by scalar/field ops. −43% at low N.
3. **File split** — `Field` → `QRField` in its own `qr_field.py` (one-job file).
4. **`_QRArray`** (the big one) — a single-pool query (`len(parts) ∈ {0,1}`, the common one-archetype
   case) returns a thin `np.ndarray` **subclass view** instead of a `QRField`. It overrides nothing,
   so `+`/`*` run as native C numpy — no per-operator `__array_ufunc__` dispatch. `.numpy()`/`.parts`
   shims keep it swappable with `QRField`. Multi-pool (≥2 pools) still uses `QRField` gather/scatter.
5. Fallout fixed: `motion.py` `Field`→`QRField`; empty-query build (`arr = parts[0] if parts else
   np.empty(...)`); test suite migrated (single-pool = raw numpy, guard tests moved to multi-pool).
   274 pass.

**Trade-off (see review):** `_QRArray` drops `QRField`'s entity-axis index guards for the single-pool
case — `qr.x[3]` is now plain numpy, not a `TypeError`. Intended (valid on one contiguous buffer),
documented, and pinned by tests. Spun off **microecs #27** (stale-qr-after-`update()` guard).

## Results — mecs/xecs ratio (≤1.0 = microecs wins), this machine
| N     | initial | +cache | +lazy cumsum | **+_QRArray (final)** |
|------:|--------:|-------:|-------------:|----------------------:|
| 500   | 4.43×   | 2.73×  | 1.56×        | **0.69×**             |
| 1000  | 3.46×   | 2.11×  | 1.26×        | **0.55×**             |
| 2000  | 2.35×   | 1.42×  | 0.92×        | **0.41×**             |
| 5000  | 1.22×   | 0.78×  | 0.52×        | **0.30×**             |
| 10000 | 0.72×   | 0.51×  | 0.36×        | **0.23×**             |
| 20000 | 0.41×   | 0.38×  | 0.30×        | **0.24×**             |

Final absolute (ms/frame), all verified against a float64 reference (`ok`): N=500 microecs 0.0054 vs
xecs 0.0077 vs np-floor 0.0015 (mecs/floor 3.46×); N=10k 0.0145 vs 0.0634 vs 0.0074 (1.96×).

## Re-run
```bash
cd pkg/microecs/.tracker/todos/done/26-low-n-field-overhead
pip install -r requirements.txt      # numpy, xecs
pip install -e ../../../../           # microecs (pkg/microecs root)
python run.py                         # N sweep 200->20k + per-frame breakdown
```
Re-check the large-N path with `pkg/microecs/examples/05-benchmark-workloads/` (must not regress).
