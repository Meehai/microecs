# microecs vs xecs at low entity counts — the 1k–10k gap

> **Status: RESOLVED (task #26 done).** microecs now wins the whole 200–20k band. This README is the
> original problem statement; see `TASK.md` (same dir) for the outcome, results, and re-run steps.

A focused tracker for **microecs task #26**. The broad benchmark (`../../../../examples/05-benchmark-workloads/`) found
that microecs wins columnar physics at N≥20k but **loses to xecs in the 500–10k band** — the columnar
crossover sits at ~10k. This isolates that one gap so it can be optimized and watched for regressions.

**Why this band matters:** robosim needs ~500 UAVs at 60 fps (a **16.6 ms** frame budget) and can't
today — N=500 is not "small enough to ignore," it's the real target. This is the driver for task #26.

## The issue being tracked

microecs' cost per frame is `fixed_python_overhead + numpy_work(N)`. At large N the numpy work
dominates and microecs wins (it mutates pool arrays in place, zero-copy). At small N the fixed
overhead dominates — and it is **per-op Python object churn in `QueryResult`/`Field`**: every
`qr.field` access allocates a new `Field`, every arithmetic op and write allocates more
(`microecs/query_result.py`). xecs, with a leaner per-op path, wins the small-N band today.

## What it measures

Single archetype (every entity has Pos+Vel+Acc → one pool, exercising the single-pool path we want
to make fast). One frame = `vel += acc·dt; pos += vel·dt`. At each N, three contestants:

- **microecs** — `QueryResult` write-through (the lib under test)
- **xecs** — Rust SoA, in-place `view.x += …` (wins this band today)
- **numpy-floor** — the same math in-place on raw numpy arrays — the **theoretical floor** microecs
  should approach; the `mecs/floor` column *is* the overhead to remove

Every result is verified against a float64 reference. A per-frame **breakdown** at N=1k and 5k splits
microecs' time into `query()` lookup / one `Field` access / full step / raw-numpy floor — so the fix
has a concrete target (see task #26's "suspected hot spots").

## Running it

```bash
pip install -r requirements.txt   # numpy, xecs
pip install -e ../../../../        # microecs (pkg/microecs root)
python run.py                     # default N sweep (200 → 20k) + breakdown
python run.py 1000 5000           # custom N list
```

## Success criterion (task #26)

The `mecs/xecs` ratio in the **500–10k** band drops toward ≤1.0, `mecs/floor` shrinks, every result
stays verified, and the large-N win (re-check `../../../../examples/05-benchmark-workloads/`) does **not** regress.
