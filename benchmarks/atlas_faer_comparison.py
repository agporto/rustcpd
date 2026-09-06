"""Alternate reference/candidate binaries; compare every saved numeric output.

Both binaries must use the identical atlas_fit_bench.rs harness and release
settings. Build the reference from 57ed868, then build the candidate separately.
The report records raw paired times as well as medians; no changed EM settings
or reduced model sizes are used to obtain a speedup.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess

import numpy as np


def snapshot(path):
    groups = {}
    for line in path.read_text().splitlines():
        key, value = line.split()
        group = key.rsplit("[", 1)[0]
        groups.setdefault(group, []).append(float(value))
    return {key: np.asarray(value) for key, value in groups.items()}


def compare(reference, candidate):
    a, b = snapshot(reference), snapshot(candidate)
    if a.keys() != b.keys():
        raise ValueError("Snapshot fields differ")
    output = {}
    for name in a:
        av, bv = a[name], b[name]
        if av.shape != bv.shape:
            raise ValueError(f"Shape differs for {name}")
        bitwise_equal = np.array_equal(av.view(np.uint64), bv.view(np.uint64))
        finite = np.isfinite(av) & np.isfinite(bv)
        nonfinite_equal = np.array_equal(np.isnan(av), np.isnan(bv)) and np.array_equal(
            av[~finite], bv[~finite], equal_nan=True
        )
        av, bv = av[finite], bv[finite]
        error = np.abs(av - bv)
        exact_field = name.endswith(".iterations") or name.endswith(".diagnostics")
        # Fixed in advance: exact discrete decisions, tight double-precision
        # tolerances for continuous outputs, and report absolute errors too.
        close = np.array_equal(av, bv) if exact_field else np.allclose(av, bv, rtol=1e-8, atol=1e-8)
        output[name] = {
            "count": int(av.size),
            "max_abs_error": float(error.max(initial=0.0)),
            "relative_l2_error": float(np.linalg.norm(error) / max(np.linalg.norm(av), 1e-300)),
            "bitwise_equal": bool(bitwise_equal),
            "passed": bool(close and nonfinite_equal),
        }
    return output


def run(binary, scenario, rank, points, parallel, output, threads):
    command = [str(binary), scenario, str(rank), str(points), "1", "parallel" if parallel else "serial"]
    if output is not None:
        command.append(str(output))
    env = {**os.environ, "RAYON_NUM_THREADS": str(threads)}
    result = subprocess.run(command, env=env, text=True, capture_output=True, check=True, timeout=240)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--require-bitwise", action="store_true",
                        help="Require identical floating-point bits in all successful snapshots")
    parser.add_argument("--case", help="Run a single scenario,rank,points,parallel case")
    args = parser.parse_args()
    if args.pairs < 1 or args.threads < 1:
        parser.error("--pairs and --threads must be positive")
    if args.validate_only and args.timing_only:
        parser.error("Choose only one of --validate-only and --timing-only")
    args.reference = args.reference.resolve()
    args.candidate = args.candidate.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference": str(args.reference.resolve()), "candidate": str(args.candidate.resolve()),
        "binary_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in [("reference", args.reference), ("candidate", args.candidate)]},
        "platform": platform.platform(), "threads": args.threads,
        "python": platform.python_version(), "numpy": np.__version__,
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "tolerance": {"rtol": 1e-8, "atol": 1e-8, "discrete_fields": "exact",
                      "require_bitwise": args.require_bitwise},
        "accuracy": [], "timings": [],
    }
    validation_cases = [
        ("fragment", rank, 768, parallel) for rank in [8, 12, 32, 64, 128] for parallel in [False, True]
    ] + [(scenario, 64, 768, parallel)
         for scenario in ["cold", "fragment_raw", "stress", "stress_weak", "stress_landmark",
                          "fragment_sparse", "fragment_converged", "fragment_offset", "pose_landmark"]
         for parallel in [False, True]]
    timing_cases = [(scenario, rank, 2000, parallel)
                    for scenario in ["fragment", "cold"] for rank in [12, 32, 64, 128]
                    for parallel in [False, True]]
    timing_cases += [("fragment_sparse", 128, 5000, True), ("pose_landmark", 64, 2000, True)]
    if args.case:
        scenario, rank, points, parallel = args.case.split(",")
        validation_cases = timing_cases = [(scenario, int(rank), int(points), parallel.lower() == "true")]
    if not args.timing_only:
        for scenario, rank, points, parallel in validation_cases:
            label = f"{scenario}_r{rank}_m{points}_p{parallel}"
            before, after = [args.output / f"{label}_{part}.txt" for part in ["reference", "candidate"]]
            failures = []
            for binary, target in [(args.reference, before), (args.candidate, after)]:
                try:
                    run(binary, scenario, rank, points, parallel, target, args.threads)
                    failures.append(None)
                except subprocess.CalledProcessError as error:
                    failures.append(error.stderr)
            if any(failures):
                matched = all(error and "SingularSystem" in error for error in failures)
                report["accuracy"].append({"case": label, "passed": bool(matched),
                                           "matched_failure": bool(matched), "errors": failures})
                print(f"accuracy {label}: {'MATCHED SINGULAR FAILURE' if matched else 'ERROR MISMATCH'}", flush=True)
                (args.output / "report.json").write_text(json.dumps(report, indent=2))
                continue
            comparison = compare(before, after)
            if args.require_bitwise:
                for field in comparison.values():
                    field["passed"] = field["passed"] and field["bitwise_equal"]
            passed = all(v["passed"] for v in comparison.values())
            report["accuracy"].append({"case": label, "passed": passed, "fields": comparison})
            print(f"accuracy {label}: {'PASS' if passed else 'FAIL'}", flush=True)
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
    if not args.validate_only:
        for scenario, rank, points, parallel in timing_cases:
            before, after = [], []
            ratios = []
            for pair in range(args.pairs):
                order = [(args.reference, before), (args.candidate, after)]
                if pair % 2:
                    order.reverse()
                for binary, times in order:
                    result = run(binary, scenario, rank, points, parallel, None, args.threads)
                    times.append(result["median_s"])
                ratios.append(before[-1] / after[-1])
            data = {"scenario": scenario, "rank": rank, "points": points, "parallel": parallel,
                    "reference_s": before, "candidate_s": after,
                    "reference_median_s": statistics.median(before),
                    "candidate_median_s": statistics.median(after),
                    "paired_speedup_median": statistics.median(ratios), "paired_speedups": ratios}
            report["timings"].append(data)
            print(f"timing {scenario} r={rank} m={points} parallel={parallel}: {data['paired_speedup_median']:.3f}x", flush=True)
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
    failed = [case["case"] for case in report["accuracy"] if not case["passed"]]
    if failed:
        raise SystemExit("Accuracy failures: " + ", ".join(failed))


if __name__ == "__main__":
    main()
