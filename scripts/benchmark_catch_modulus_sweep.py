#!/usr/bin/env python3
"""
CATCH Modulus-versus-Challenge-Count Sweep Benchmark
Evaluates the trade-off between plaintext modulus bit-width t, challenge count s,
soundness bound s * log2(t) >= 128 bits, and execution latency across network environments:
  - Sequential CATCH-II: 1 + s sequential round trips
  - Batched CATCH-II: Exactly 2 round trips

By default benchmarks across matrix dimensions 16, 32, and 64.
"""

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CATCH modulus-vs-rounds sweep benchmarks across matrix dimensions."
    )
    parser.add_argument(
        "--build_dir",
        type=str,
        default="./build",
        help="Path to CMake build directory (default: ./build)",
    )
    parser.add_argument(
        "--dims",
        "--sizes",
        "--dim",
        dest="dims",
        type=str,
        default="16,32,64",
        help="Comma-separated matrix dimensions to benchmark (default: 16,32,64)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of benchmark trials per configuration (default: 1)",
    )
    parser.add_argument(
        "--bits",
        type=str,
        default="16,24,30,32,34",
        help="Comma-separated target bit widths (default: 16,24,30,32,34)",
    )
    parser.add_argument(
        "--no-s9",
        "--no_s9",
        dest="no_s9",
        action="store_true",
        help="Exclude the conservative 16-bit s=9 case",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save CSV data files (default: ./results)",
    )
    return parser.parse_args()


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
            if stripped.startswith("matrixDim,"):
                found_header = True
                data_lines.append(stripped)
        else:
            if (
                stripped
                and "," in stripped
                and not stripped.startswith("[")
                and not stripped.startswith("=")
            ):
                data_lines.append(stripped)
    if not data_lines:
        return []

    reader = csv.DictReader(data_lines)
    return [dict(r) for r in reader]


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


def aggregate_trials(all_runs):
    """
    Groups raw runs by (matrixDim, targetBits, modulusP, challenges_s, protocol)
    and computes average timings, metrics, and relative speedups.
    """
    grouped = defaultdict(list)
    for r in all_runs:
        key = (
            int(r["matrixDim"]),
            int(r["targetBits"]),
            int(r["modulusP"]),
            int(r["challenges_s"]),
            r["protocol"],
        )
        grouped[key].append(r)

    raw_agg = {}
    numeric_timing_fields = [
        "clientSetupTime_s",
        "clientVerifyTime_s",
        "clientTotalTime_s",
        "serverRound1Time_s",
        "serverVerifyTime_s",
        "serverTotalTime_s",
        "totalComputeTime_s",
        "latencyLAN_1ms_s",
        "latencyWAN_50ms_s",
        "latencyWAN_150ms_s",
    ]

    for key, runs in grouped.items():
        base = dict(runs[0])
        # Average timing fields
        for field in numeric_timing_fields:
            vals = [float(item[field]) for item in runs]
            base[field] = f"{sum(vals) / len(vals):.6f}"

        base["trialsCount"] = len(runs)
        raw_agg[key] = base

    # Now pair up Sequential and Batched to compute accurate speedup ratios
    # Config key: (matrixDim, targetBits, modulusP, challenges_s)
    configs = set((k[0], k[1], k[2], k[3]) for k in raw_agg.keys())
    aggregated = []

    for cfg in configs:
        seq_row = None
        batch_row = None
        for k, v in raw_agg.items():
            if (k[0], k[1], k[2], k[3]) == cfg:
                if "Sequential" in k[4]:
                    seq_row = v
                elif "Batched" in k[4]:
                    batch_row = v

        if seq_row and batch_row:
            s_time_seq = float(seq_row["serverTotalTime_s"])
            s_time_batch = float(batch_row["serverTotalTime_s"])
            server_speedup = s_time_seq / s_time_batch if s_time_batch > 0 else 1.0

            wan50_seq = float(seq_row["latencyWAN_50ms_s"])
            wan50_batch = float(batch_row["latencyWAN_50ms_s"])
            wan50_speedup = wan50_seq / wan50_batch if wan50_batch > 0 else 1.0

            wan150_seq = float(seq_row["latencyWAN_150ms_s"])
            wan150_batch = float(batch_row["latencyWAN_150ms_s"])
            wan150_speedup = wan150_seq / wan150_batch if wan150_batch > 0 else 1.0

            bytes_seq = float(seq_row["totalCommKB"])
            bytes_batch = float(batch_row["totalCommKB"])
            comm_reduction = bytes_seq / bytes_batch if bytes_batch > 0 else 1.0

            # Store on Batched row
            batch_row["serverSpeedup"] = f"{server_speedup:.2f}x"
            batch_row["wan50Speedup"] = f"{wan50_speedup:.2f}x"
            batch_row["wan150Speedup"] = f"{wan150_speedup:.2f}x"
            batch_row["commReduction"] = f"{comm_reduction:.2f}x"

            # Store on Sequential baseline row
            seq_row["serverSpeedup"] = "1.00x (baseline)"
            seq_row["wan50Speedup"] = "1.00x (baseline)"
            seq_row["wan150Speedup"] = "1.00x (baseline)"
            seq_row["commReduction"] = "1.00x (baseline)"

            aggregated.append(batch_row)
            aggregated.append(seq_row)
        elif seq_row:
            aggregated.append(seq_row)
        elif batch_row:
            aggregated.append(batch_row)

    # Sort logically by matrixDim, targetBits, challenges_s, protocol
    aggregated.sort(
        key=lambda x: (
            int(x["matrixDim"]),
            int(x["targetBits"]),
            -int(x["challenges_s"]),
            0 if "Batched" in x["protocol"] else 1,
        )
    )
    return aggregated


def print_markdown_parameters_table(rows):
    print("\n" + "=" * 130)
    print("TABLE 1: FHE PARAMETERS & SOUNDNESS MAPPING (Target: >= 128-bit Soundness)")
    print("=" * 130)
    headers = [
        "Dim",
        "Target Bits",
        "Prime Modulus (t)",
        "Bits (log2 t)",
        "Challenges (s)",
        "Achieved Soundness",
        "Ring Dim (N)",
        "log2(Q)",
        "Limbs",
        "CT Size (KB)",
    ]
    widths = [7, 12, 20, 14, 15, 20, 13, 9, 7, 13]
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep_line = "| " + " | ".join("-" * w for w in widths) + " |"
    print(header_line)
    print(sep_line)

    seen_configs = set()
    for r in rows:
        cfg_key = (r["matrixDim"], r["targetBits"], r["modulusP"], r["challenges_s"])
        if cfg_key in seen_configs:
            continue
        seen_configs.add(cfg_key)

        target_lbl = f"{r['targetBits']}-bit"
        if r["targetBits"] == "16" and r["challenges_s"] == "9":
            target_lbl = "16-bit (t < 2^16)"

        dim_lbl = f"{r['matrixDim']}x{r['matrixDim']}"

        cols = [
            dim_lbl.ljust(widths[0]),
            target_lbl.ljust(widths[1]),
            r["modulusP"].ljust(widths[2]),
            f"{float(r['modulusBits']):.2f} bits".ljust(widths[3]),
            f"s = {r['challenges_s']}".ljust(widths[4]),
            f"**{float(r['achievedSoundnessBits']):.1f} bits**".ljust(widths[5]),
            r["ringDim_N"].ljust(widths[6]),
            f"{r['log2Q']} b".ljust(widths[7]),
            r["numLimbs"].ljust(widths[8]),
            f"{float(r['ciphertextSizeKB']):.1f} KB".ljust(widths[9]),
        ]
        print("| " + " | ".join(cols) + " |")
    print("=" * 130)


def print_markdown_performance_table(rows):
    print("\n" + "=" * 175)
    print(
        "TABLE 2: PERFORMANCE & SPEEDUP BREAKDOWN: SEQUENTIAL (1+s RTT) VS. BATCHED (2-RTT OURS)"
    )
    print("=" * 175)
    print("> Speedup Formulas:")
    print(">   * Server Speedup    = Server_Time(Sequential) / Server_Time(Batched)")
    print(
        ">   * WAN-50ms Speedup  = Latency_WAN50ms(Sequential) / Latency_WAN50ms(Batched)"
    )
    print(">   * Comm Reduction    = Comm_Bytes(Sequential) / Comm_Bytes(Batched)\n")

    headers = [
        "Dim",
        "Modulus Target",
        "Modulus t",
        "s",
        "Protocol",
        "RTTs",
        "Server (s)",
        "Client (s)",
        "WAN-50ms (s)",
        "Server Speedup",
        "WAN-50ms Speedup",
        "Comm (KB)",
        "Comm Reduction",
        "CTs",
    ]
    widths = [7, 16, 14, 4, 15, 5, 11, 11, 13, 16, 17, 11, 15, 5]
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep_line = "| " + " | ".join("-" * w for w in widths) + " |"
    print(header_line)
    print(sep_line)

    curr_group = None
    curr_dim = None
    for r in rows:
        dim = r["matrixDim"]
        grp = (dim, r["targetBits"], r["challenges_s"])

        if curr_dim is not None and curr_dim != dim:
            print("|" + "=" * (sum(widths) + 3 * len(widths) - 1) + "|")
        elif curr_group is not None and curr_group != grp:
            print("|" + "-" * (sum(widths) + 3 * len(widths) - 1) + "|")
        curr_group = grp
        curr_dim = dim

        target_lbl = f"{r['targetBits']}-bit"
        if r["targetBits"] == "16" and r["challenges_s"] == "9":
            target_lbl = "16-bit (t < 2^16)"

        dim_lbl = f"{r['matrixDim']}x{r['matrixDim']}"
        proto_str = "**Batched (Ours)**" if "Batched" in r["protocol"] else "Sequential"
        srv_sp = (
            f"**{r['serverSpeedup']}**"
            if "Batched" in r["protocol"]
            else r["serverSpeedup"]
        )
        wan_sp = (
            f"**{r['wan50Speedup']}**"
            if "Batched" in r["protocol"]
            else r["wan50Speedup"]
        )
        comm_red = (
            f"**{r['commReduction']}**"
            if "Batched" in r["protocol"]
            else r["commReduction"]
        )

        cols = [
            dim_lbl.ljust(widths[0]),
            target_lbl.ljust(widths[1]),
            r["modulusP"].ljust(widths[2]),
            r["challenges_s"].ljust(widths[3]),
            proto_str.ljust(widths[4]),
            r["roundTrips"].ljust(widths[5]),
            f"{float(r['serverTotalTime_s']):.4f}s".ljust(widths[6]),
            f"{float(r['clientTotalTime_s']):.4f}s".ljust(widths[7]),
            f"{float(r['latencyWAN_50ms_s']):.4f}s".ljust(widths[8]),
            srv_sp.ljust(widths[9]),
            wan_sp.ljust(widths[10]),
            f"{float(r['totalCommKB']):.1f}".ljust(widths[11]),
            comm_red.ljust(widths[12]),
            r["ciphertextCount"].ljust(widths[13]),
        ]
        print("| " + " | ".join(cols) + " |")
    print("=" * 175 + "\n")


def print_markdown_speedup_summary_table(rows):
    print("=" * 145)
    print("TABLE 3: DIRECT SPEEDUP COMPARISON SUMMARY (BATCHED GAINS OVER SEQUENTIAL)")
    print("=" * 145)
    print("> Exact Computation Formulas:")
    print(">   Speedup_server     = T_server(Sequential) / T_server(Batched)")
    print(
        ">   Speedup_WAN50ms    = [T_total(Seq) + (1+s)*50ms] / [T_total(Batch) + 2*50ms]"
    )
    print(
        ">   Speedup_WAN150ms   = [T_total(Seq) + (1+s)*150ms] / [T_total(Batch) + 2*150ms]"
    )
    print(">   Bandwidth_Saved    = CommBytes(Sequential) / CommBytes(Batched)\n")

    headers = [
        "Dim",
        "Modulus Target",
        "Prime (t)",
        "s",
        "Soundness",
        "Seq Server",
        "Batch Server",
        "Server Speedup",
        "WAN-50ms Speedup",
        "WAN-150ms Speedup",
        "Bandwidth Saved",
    ]
    widths = [7, 16, 14, 4, 11, 12, 13, 16, 17, 18, 16]
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep_line = "| " + " | ".join("-" * w for w in widths) + " |"
    print(header_line)
    print(sep_line)

    by_cfg = defaultdict(dict)
    for r in rows:
        cfg = (int(r["matrixDim"]), r["targetBits"], r["challenges_s"])
        if "Batched" in r["protocol"]:
            by_cfg[cfg]["batched"] = r
        else:
            by_cfg[cfg]["seq"] = r

    # Sort keys by matrixDim, targetBits, -challenges_s
    sorted_cfgs = sorted(by_cfg.keys(), key=lambda k: (k[0], int(k[1]), -int(k[2])))

    curr_dim = None
    for cfg in sorted_cfgs:
        pair = by_cfg[cfg]
        dim, targetBits, s = cfg
        if curr_dim is not None and curr_dim != dim:
            print("|" + "-" * (sum(widths) + 3 * len(widths) - 1) + "|")
        curr_dim = dim

        if "seq" in pair and "batched" in pair:
            seq = pair["seq"]
            batch = pair["batched"]

            dim_lbl = f"{dim}x{dim}"
            target_lbl = f"{targetBits}-bit"
            if targetBits == "16" and s == "9":
                target_lbl = "16-bit (t < 2^16)"

            cols = [
                dim_lbl.ljust(widths[0]),
                target_lbl.ljust(widths[1]),
                seq["modulusP"].ljust(widths[2]),
                str(s).ljust(widths[3]),
                f"{float(seq['achievedSoundnessBits']):.0f} bits".ljust(widths[4]),
                f"{float(seq['serverTotalTime_s']):.3f}s".ljust(widths[5]),
                f"{float(batch['serverTotalTime_s']):.3f}s".ljust(widths[6]),
                f"**{batch['serverSpeedup']}**".ljust(widths[7]),
                f"**{batch['wan50Speedup']}**".ljust(widths[8]),
                f"**{batch['wan150Speedup']}**".ljust(widths[9]),
                f"**{batch['commReduction']} less**".ljust(widths[10]),
            ]
            print("| " + " | ".join(cols) + " |")
    print("=" * 145 + "\n")


def print_tradeoff_summary(rows):
    print("=" * 115)
    print("KEY TAKEAWAYS & TRADE-OFF ANALYSIS (SOUNDNESS VS. RUNTIME COST)")
    print("=" * 115)

    by_dim = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        dim = int(r["matrixDim"])
        cfg = (r["targetBits"], r["challenges_s"])
        if "Batched" in r["protocol"]:
            by_dim[dim][cfg]["batched"] = r
        else:
            by_dim[dim][cfg]["seq"] = r

    for dim in sorted(by_dim.keys()):
        print(f"\n>>> Matrix Dimension {dim}x{dim}:")
        for (targetBits, s), pair in by_dim[dim].items():
            if "seq" in pair and "batched" in pair:
                seq = pair["seq"]
                batch = pair["batched"]
                lbl = f"{targetBits}-bit (P={seq['modulusP']}, s={s})"
                print(f"  • [{lbl}]:")
                print(
                    f"    - Sequential : {seq['roundTrips']} RTTs, Server={float(seq['serverTotalTime_s']):.3f}s, WAN-50ms={float(seq['latencyWAN_50ms_s']):.3f}s, Comm={float(seq['totalCommKB']):.1f} KB"
                )
                print(
                    f"    - Batched    : {batch['roundTrips']} RTTs, Server={float(batch['serverTotalTime_s']):.3f}s, WAN-50ms={float(batch['latencyWAN_50ms_s']):.3f}s, Comm={float(batch['totalCommKB']):.1f} KB"
                )
                print(
                    f"    -> Speedup   : Batched is {batch['serverSpeedup']} faster compute, {batch['wan50Speedup']} faster on WAN-50ms, with {batch['commReduction']} less bandwidth!\n"
                )
    print("=" * 115 + "\n")


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

    bin_path = build_dir / "bin" / "bench_catch_modulus_sweep"

    if (
        args.output_dir == "./results"
        and not Path("./results").exists()
        and (repo_root / "results").exists()
    ):
        output_dir = repo_root / "results"
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = output_dir / "paper_benchmarks_raw_catch_modulus.csv"
    table_csv_path = output_dir / "table_catch_modulus_vs_rounds.csv"

    if not bin_path.exists():
        print(f"[ERROR] Binary not found at: {bin_path}", file=sys.stderr)
        print(
            f"Please build it first: ninja -C {build_dir} bench_catch_modulus_sweep",
            file=sys.stderr,
        )
        sys.exit(1)

    dims = [int(d.strip()) for d in args.dims.split(",") if d.strip()]

    print("==========================================================================")
    print("  CATCH MODULUS-VERSUS-CHALLENGE-COUNT EXPERIMENTAL SWEEP")
    print(f"  Matrix Dimensions  : {dims}")
    print(f"  Trials per config  : {args.trials}")
    print(f"  Target Bit Widths  : {args.bits}")
    print(f"  Output directory   : {output_dir}")
    print("==========================================================================")

    all_raw_runs = []
    for dim in dims:
        print(f"\n========================================================")
        print(f"  BENCHMARKING MATRIX DIMENSION: {dim}x{dim}")
        print(f"========================================================")

        cmd_base = [
            str(bin_path),
            "--dim",
            str(dim),
            "--bits",
            str(args.bits),
            "--csv",
        ]
        if args.no_s9:
            cmd_base.append("--no-s9")

        for t in range(1, args.trials + 1):
            print(f"--> [{dim}x{dim}] Executing Trial {t}/{args.trials}...", flush=True)
            trial_runs = run_command_csv(cmd_base)
            if not trial_runs:
                print(
                    f"[ERROR] [{dim}x{dim}] Trial {t} failed to return data.",
                    file=sys.stderr,
                )
                sys.exit(1)
            for r in trial_runs:
                r["trialIndex"] = t
            all_raw_runs.extend(trial_runs)

    # Export raw data
    export_csv_file(all_raw_runs, raw_csv_path)

    # Aggregate over trials
    aggregated_table = aggregate_trials(all_raw_runs)
    export_csv_file(aggregated_table, table_csv_path)

    # Print publication Markdown tables
    print_markdown_parameters_table(aggregated_table)
    print_markdown_performance_table(aggregated_table)
    print_markdown_speedup_summary_table(aggregated_table)
    print_tradeoff_summary(aggregated_table)

    print(f"[Complete] Raw trial data saved to: {raw_csv_path}")
    print(f"[Complete] Aggregated paper table saved to: {table_csv_path}\n")


if __name__ == "__main__":
    main()
