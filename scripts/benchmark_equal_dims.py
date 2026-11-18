#!/usr/bin/env python3
"""
Strict Equal-Useful-Dimension Benchmark Suite
Compares:
  1. DataSeal (Single-Key)
  2. DataSeal (Multi-Key, ell=8)
  3. CATCH-I  (Sequential Setting II, ell=8)
  4. CATCH-II (Batched Setting II, ell=8)

Crucial Property:
  All four schemes evaluate the EXACT SAME USEFUL MATRIX DIMENSION
  (e.g. usefulDim = 16x16, 32x32, 64x64, 128x128).

Demonstrates the "Power-of-Two Cliff":
  - DataSeal appends rows to an already power-of-two matrix, forcing
    the ciphertext grid to double in dimension (quadruple in plaintext slots).
  - Ours evaluates verification in a second round, preserving the exact
    native power-of-two grid and using Jiang Alg 3 Rectangular MatMul
    with only ell=8 homomorphic multiplications.
"""

import argparse
import csv
import os
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run strict equal-useful-dimension verification benchmarks across multiple trials."
    )
    parser.add_argument(
        "--build_dir",
        type=str,
        default="./build",
        help="Path to CMake build directory containing bin/ (default: ./build)",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="16,32,64",
        help="Comma-separated useful matrix dimensions to evaluate (default: 16,32,64)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of trials per configuration for statistical rigor (default: 5)",
    )
    parser.add_argument(
        "--ell_keys",
        type=int,
        default=8,
        help="Security parameter ell (keys / challenges) (default: 8)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save CSV data files (default: ./results)",
    )
    parser.add_argument(
        "--csv_stdout",
        action="store_true",
        help="Print the raw consolidated CSV to stdout in addition to Markdown tables",
    )
    parser.add_argument(
        "--recompute_only",
        action="store_true",
        help="Only aggregate and display existing raw CSV trial data without re-running executions",
    )
    return parser.parse_args()


def parse_val(v):
    if v is None:
        return v
    v_str = str(v).strip()
    if v_str.lower() == "true":
        return True
    if v_str.lower() == "false":
        return False
    try:
        return int(v_str)
    except ValueError:
        try:
            return float(v_str)
        except ValueError:
            return v_str


def run_command_csv(cmd):
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        print(
            f"[ERROR] Command failed: {' '.join(str(c) for c in cmd)}", file=sys.stderr
        )
        print(result.stderr, file=sys.stderr)
        return []

    lines = result.stdout.strip().split("\n")
    data_lines = []
    found_header = False
    for line in lines:
        stripped = line.strip()
        if not found_header:
            if stripped.startswith("scheme,") or "usefulDim," in stripped:
                found_header = True
                data_lines.append(stripped)
        else:
            if (
                stripped
                and "," in stripped
                and not stripped.startswith("---")
                and not stripped.startswith("Matrix")
            ):
                data_lines.append(stripped)

    if not data_lines:
        return []

    reader = csv.DictReader(data_lines)
    parsed = []
    for r in reader:
        row_clean = {k.strip(): parse_val(v) for k, v in r.items() if k is not None}
        parsed.append(row_clean)
    return parsed


def load_csv_file(filepath):
    if not filepath.exists():
        return []
    with open(filepath, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{k: parse_val(v) for k, v in row.items()} for row in reader]


def export_csv_file(data, filepath):
    if not data:
        return
    fieldnames = list(data[0].keys())
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"[Export] Saved CSV: {filepath}")


def run_trials(bin_path, sizes, ell_keys, trials_count):
    raw = []
    print(
        f"\n[Benchmarking] Running Strict Equal-Useful-Dimension Trials ({len(sizes)} sizes, {trials_count} trials each)..."
    )
    for size in sizes:
        print(f"  -> Useful Dimension {size}x{size} (ell={ell_keys}):")
        for t in range(1, trials_count + 1):
            sys.stdout.write(f"     Trial {t}/{trials_count}... ")
            sys.stdout.flush()
            cmd = [
                str(bin_path),
                "--useful",
                str(size),
                "--ell",
                str(ell_keys),
                "--csv",
            ]
            data = run_command_csv(cmd)
            if data:
                for item in data:
                    item["targetBase"] = size
                    item["trial_idx"] = t
                    raw.append(item)
                print("done.")
            else:
                print("FAILED.")
    return raw


def aggregate_comparisons(raw_results):
    grouped = {}
    for entry in raw_results:
        target = entry.get("usefulDim", entry.get("targetBase", 16))
        key = (target, entry["scheme"], entry["lambda"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(entry)

    scheme_order = {
        "DataSeal-SingleKey": 1,
        "DataSeal-MultiKey": 2,
        "Ours-SettingII-Sequential": 3,
        "Ours-SettingII-Batched": 4,
    }
    sorted_keys = sorted(
        grouped.keys(), key=lambda k: (k[0], scheme_order.get(k[1], 99), k[2])
    )

    summary = []
    for key in sorted_keys:
        target, scheme, ell_val = key
        trials = grouped[key]
        n = len(trials)

        def get_stats(field):
            vals = [float(t[field]) for t in trials]
            return statistics.mean(vals), (statistics.stdev(vals) if n > 1 else 0.0)

        srv_m, srv_s = get_stats("serverComputeTime_s")
        cli_m, cli_s = get_stats("totalClientTime_s")
        wal_m, wal_s = get_stats("totalWallTime_s")

        first = trials[0]
        comm_kb = float(first["totalBytes"]) / 1024.0

        summary.append(
            {
                "usefulDim": f"{target}x{target}",
                "scheme": scheme,
                "paddedDim": f"{first['paddedDim']}x{first['paddedDim']}",
                "ell": ell_val,
                "soundnessSecurity": first["soundnessSecurity"],
                "trials": n,
                "serverTime_mean_s": srv_m,
                "serverTime_std_s": srv_s,
                "clientTime_mean_s": cli_m,
                "clientTime_std_s": cli_s,
                "wallTime_mean_s": wal_m,
                "wallTime_std_s": wal_s,
                "totalComm_KB": comm_kb,
                "allVerified": all(t["verificationPassed"] for t in trials),
            }
        )
    return summary


def print_markdown_comparison_table(summary):
    print("\n" + "=" * 140)
    print(
        "STRICT EQUAL-USEFUL-DIMENSION BENCHMARK: APPLES-TO-APPLES SYSTEMS COMPARISON"
    )
    print("Exact Same Useful Matrix Computation Across All Four Schemes")
    print("=" * 140)
    headers = [
        "Useful Dim",
        "Verification Scheme",
        "Padded FHE Grid",
        "Security / Soundness",
        "Server Compute (s)",
        "Client Time (s)",
        "Total Wall (s)",
        "Comm (KB)",
        "Pass?",
    ]
    widths = [12, 26, 17, 30, 20, 16, 16, 12, 7]
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep_line = "| " + " | ".join("-" * w for w in widths) + " |"
    print(header_line)
    print(sep_line)

    current_useful = ""
    for r in summary:
        if current_useful and r["usefulDim"] != current_useful:
            print("|" + "-" * (sum(widths) + 3 * len(widths) - 1) + "|")
        current_useful = r["usefulDim"]

        srv_str = f"{r['serverTime_mean_s']:.4f} ± {r['serverTime_std_s']:.4f}"
        cli_str = f"{r['clientTime_mean_s']:.4f}s"
        wal_str = f"{r['wallTime_mean_s']:.4f}s"
        passed = "YES" if r["allVerified"] else "NO"

        cols = [
            f"**{r['usefulDim']}**".ljust(widths[0]),
            f"**{r['scheme']}**".ljust(widths[1]),
            r["paddedDim"].ljust(widths[2]),
            r["soundnessSecurity"][: widths[3]].ljust(widths[3]),
            srv_str.ljust(widths[4]),
            cli_str.ljust(widths[5]),
            wal_str.ljust(widths[6]),
            f"{r['totalComm_KB']:.1f}".ljust(widths[7]),
            passed.ljust(widths[8]),
        ]
        print("| " + " | ".join(cols) + " |")
    print("=" * 140 + "\n")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    if (
        args.build_dir == "./build"
        and not Path("./build").exists()
        and (repo_root / "build").exists()
    ):
        build_dir = repo_root / "build"
    else:
        build_dir = Path(args.build_dir).resolve()

    bin_dir = build_dir / "bin"

    if (
        args.output_dir == "./results"
        and not Path("./results").exists()
        and (repo_root / "results").exists()
    ):
        output_dir = repo_root / "results"
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_path = bin_dir / "bench_matmul_equal_dims"
    raw_csv_path = output_dir / "paper_benchmarks_raw_equal_dims.csv"
    summary_csv_path = output_dir / "table_equal_useful_dims.csv"

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]

    print("=================================================================")
    print("  STRICT EQUAL-USEFUL-DIMENSION BENCHMARK SUITE")
    print(f"  Trials per configuration : {args.trials}")
    print(f"  Useful matrix dimensions : {sizes}")
    print(f"  ell (keys/challenges)    : {args.ell_keys}")
    print(f"  Output directory         : {output_dir}")
    print("=================================================================")

    if args.recompute_only:
        raw_results = load_csv_file(raw_csv_path)
        if not raw_results:
            print(
                f"[ERROR] Cannot recompute: {raw_csv_path} does not exist.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[Recompute] Loaded {len(raw_results)} raw trials from {raw_csv_path}")
    else:
        if not bin_path.exists():
            print(f"[ERROR] Binary not found at: {bin_path}", file=sys.stderr)
            print(
                f"Please build it first: ninja -C {build_dir} bench_matmul_equal_dims",
                file=sys.stderr,
            )
            sys.exit(1)

        raw_results = run_trials(bin_path, sizes, args.ell_keys, args.trials)
        if raw_results:
            export_csv_file(raw_results, raw_csv_path)

    summary = aggregate_comparisons(raw_results)
    if summary:
        export_csv_file(summary, summary_csv_path)
        print_markdown_comparison_table(summary)

    if args.csv_stdout and summary:
        print("\n=== CONSOLIDATED CSV SUMMARY (STDOUT) ===")
        writer = csv.DictWriter(sys.stdout, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
        print("=== END CSV ===\n")

    print(f"[Done] Equal-dimension benchmark results archived in: {output_dir}")
    print(f"  - Summary Table CSV: {summary_csv_path}")
    print(f"  - Raw Trials CSV   : {raw_csv_path}\n")


if __name__ == "__main__":
    main()
