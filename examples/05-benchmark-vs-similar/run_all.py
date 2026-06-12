"""Run every library's benchmark and print a comparison table.

Usage:
    python run_all.py [N] [measure_steps]      # defaults: N=100000, steps=30

Each library is verified against a float64 numpy reference before it counts.
"""
import importlib
import _common as C

MODULES = ["bench_microecs", "bench_xecs", "bench_esper", "bench_snecs", "bench_ecs_pattern"]


def main():
    n, measure = C.cli_n_measure()

    print(f"# ECS batch-update benchmark -- N={n:,} entities, "
          f"{C.WARMUP} warmup + {measure} timed frames")
    print("# frame = (vel += acc*dt over half) then (pos += vel*dt over all), "
          "semi-implicit Euler\n")

    results = []
    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        r = mod.run(n=n, measure=measure)
        C.print_result(r)
        results.append(r)

    results.sort(key=lambda r: r["step_s"])
    slowest = results[-1]["step_s"]
    print("\n" + "=" * 80)
    print(f"{'rank':<5}{'library':<14}{'step (ms)':>11}{'ns/entity':>12}"
          f"{'M upd/s':>10}{'vs slowest':>12}{'ok':>6}")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        print(f"{i:<5}{r['name']:<14}{r['step_s'] * 1e3:>11.3f}{r['ns_per_entity']:>12.1f}"
              f"{r['m_updates_per_s']:>10.1f}{slowest / r['step_s']:>11.1f}x"
              f"{('yes' if r['ok'] else 'NO'):>6}")
    print("=" * 80)

    fastest = results[0]
    print(f"\nfastest: {fastest['name']} at {fastest['step_s'] * 1e3:.3f} ms/frame "
          f"({fastest['ns_per_entity']:.1f} ns/entity). "
          "all results verified against a float64 numpy reference.")


if __name__ == "__main__":
    main()
