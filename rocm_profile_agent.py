#!/usr/bin/env python3
"""ROCm Profile Agent - CLI tool that profiles ROCm GPU applications and generates HTML reports.

Usage:
    python3 rocm_profile_agent.py [options] -- <application> [application args...]
"""

import argparse
import datetime
import os
import sys

from profiler import run_profiling
from parser import (
    parse_kernel_trace, parse_hip_trace, rank_kernels,
    parse_counter_csv, aggregate_counters_by_kernel,
)
from gpu_specs import detect_gpu_from_agent_info, detect_gpu_from_rocminfo
from roofline import compute_roofline, compute_utilization, compute_occupancy
from report import generate_report, generate_markdown_report


def main():
    parser = argparse.ArgumentParser(
        description="Profile a ROCm GPU application and generate reports in various format.",
        usage="%(prog)s [options] -- <application> [args...]",
    )
    parser.add_argument("-o", "--output", default=None,
                        help="Output HTML file path (default: profile_report_<timestamp>.html)")
    parser.add_argument("-w", "--workdir", default=None,
                        help="Working directory for intermediate files (default: temp dir)")
    parser.add_argument("-n", "--top-n", type=int, default=5,
                        help="Number of top kernels to analyze (default: 5)")
    parser.add_argument("--keep-workdir", action="store_true",
                        help="Keep intermediate profiling files after report generation")
    parser.add_argument("--format", choices=["html", "md", "all"], default="html",
                        help="Output format: html, md, or all (default: html)")
    parser.add_argument("--timeline-only", action="store_true",
                        help="Only collect traces (skip PMC counters). "
                             "Produces timeline + top kernels without instruction mix, "
                             "roofline, or occupancy data.")

    # Everything after -- is the user command
    args, user_cmd = parser.parse_known_args()

    # Strip leading '--' if present
    if user_cmd and user_cmd[0] == "--":
        user_cmd = user_cmd[1:]

    if not user_cmd:
        parser.error("No application command specified. Use: %(prog)s -- <command>")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output is None:
        ext = ".md" if args.format == "md" else ".html"
        args.output = f"profile_report_{timestamp}{ext}"

    # Derive base name (strip extension) for generating both formats
    output_base = args.output
    for ext in (".html", ".md"):
        if output_base.endswith(ext):
            output_base = output_base[:-len(ext)]
            break

    print(f"ROCm Profile Agent", file=sys.stderr)
    print(f"Command: {' '.join(user_cmd)}", file=sys.stderr)
    print(f"Output:  {args.output}", file=sys.stderr)
    print(file=sys.stderr)

    # Step 1: Run profiling passes
    prof_results = run_profiling(user_cmd, workdir=args.workdir,
                                 timeline_only=args.timeline_only)
    workdir = prof_results["workdir"]
    print(f"\nProfiling data in: {workdir}", file=sys.stderr)

    # Step 2: Detect GPU specs from agent_info.csv
    agent_info_path = prof_results["agent_info"]
    if os.path.exists(agent_info_path):
        gpu_specs = detect_gpu_from_agent_info(agent_info_path)
    else:
        print("Warning: agent_info.csv not found, falling back to rocminfo", file=sys.stderr)
        gpu_specs = detect_gpu_from_rocminfo()
    print(f"GPU: {gpu_specs.get('name', '?')} ({gpu_specs.get('gfx_target', '?')})", file=sys.stderr)

    # Step 3: Parse trace data
    kernel_events = []
    hip_events = []

    kernel_trace_path = prof_results["kernel_trace"]
    if os.path.exists(kernel_trace_path) and os.path.getsize(kernel_trace_path) > 0:
        kernel_events = parse_kernel_trace(kernel_trace_path)
        print(f"Parsed {len(kernel_events)} kernel dispatches", file=sys.stderr)
    else:
        print("Warning: No kernel trace data found", file=sys.stderr)

    hip_trace_path = prof_results["hip_trace"]
    if os.path.exists(hip_trace_path) and os.path.getsize(hip_trace_path) > 0:
        hip_events = parse_hip_trace(hip_trace_path)
        print(f"Parsed {len(hip_events)} HIP API calls", file=sys.stderr)
    else:
        print("Warning: No HIP trace data found", file=sys.stderr)

    # Step 4: Rank kernels
    top_kernels = rank_kernels(kernel_events, top_n=args.top_n)
    if top_kernels:
        print(f"\nTop {len(top_kernels)} kernels:", file=sys.stderr)
        for i, k in enumerate(top_kernels):
            print(f"  {i+1}. {k['kernel_name'][:60]} — {k['pct']:.1f}% ({k['count']} calls)",
                  file=sys.stderr)
    else:
        print("Warning: No kernels found to rank", file=sys.stderr)

    top_names = [k["kernel_name"] for k in top_kernels]

    # Steps 5-7: Parse counters and compute derived metrics (skipped in timeline-only mode)
    insts_by_kernel = {}
    roofline_by_kernel = {}
    compute_by_kernel = {}
    occupancy_by_kernel = {}

    if not args.timeline_only:
        # Step 5: Parse and aggregate instruction counters
        insts_path = prof_results["insts_counters"]
        if insts_path and os.path.exists(insts_path) and os.path.getsize(insts_path) > 0:
            insts_dispatches = parse_counter_csv(insts_path)
            insts_by_kernel = aggregate_counters_by_kernel(insts_dispatches, top_names)

        # Step 6: Parse and aggregate memory counters (merge two passes)
        mem_by_kernel = {}
        for mem_path in [prof_results["mem_hbm_l2_counters"],
                         prof_results["mem_l1_lds_counters"]]:
            if mem_path and os.path.exists(mem_path) and os.path.getsize(mem_path) > 0:
                mem_dispatches = parse_counter_csv(mem_path)
                partial = aggregate_counters_by_kernel(mem_dispatches, top_names)
                for kname in top_names:
                    if kname not in mem_by_kernel:
                        mem_by_kernel[kname] = {}
                    if kname in partial:
                        mem_by_kernel[kname].update(partial[kname])

        # Step 6b: Parse and aggregate occupancy counters
        occ_by_kernel = {}
        occ_path = prof_results["occupancy_counters"]
        if occ_path and os.path.exists(occ_path) and os.path.getsize(occ_path) > 0:
            occ_dispatches = parse_counter_csv(occ_path)
            occ_by_kernel = aggregate_counters_by_kernel(occ_dispatches, top_names)

        # Step 6c: Compute actual occupancy
        occupancy_by_kernel = compute_occupancy(occ_by_kernel, kernel_events, top_kernels, gpu_specs)

        # Step 7: Compute roofline utilization
        for kname in top_names:
            counters = mem_by_kernel.get(kname, {})
            roofline_by_kernel[kname] = compute_roofline(counters, gpu_specs)

        # Step 7b: Compute FLOPS/IOPS utilization from instruction counters
        for kname in top_names:
            insts = insts_by_kernel.get(kname, {})
            compute_by_kernel[kname] = compute_utilization(insts, gpu_specs)

    # Step 8: Generate report(s)
    print(f"\nGenerating report...", file=sys.stderr)
    command_str = " ".join(user_cmd)
    report_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_kwargs = dict(
        command=command_str,
        gpu_specs=gpu_specs,
        kernel_events=kernel_events,
        hip_events=hip_events,
        top_kernels=top_kernels,
        insts_by_kernel=insts_by_kernel,
        roofline_by_kernel=roofline_by_kernel,
        compute_by_kernel=compute_by_kernel,
        occupancy_by_kernel=occupancy_by_kernel,
        timestamp=report_ts,
    )

    if args.format in ("html", "all"):
        html_path = output_base + ".html"
        html_content = generate_report(**report_kwargs)
        with open(html_path, "w") as f:
            f.write(html_content)
        print(f"HTML report written to: {html_path}", file=sys.stderr)

    if args.format in ("md", "all"):
        md_path = output_base + ".md"
        md_content = generate_markdown_report(**report_kwargs)
        with open(md_path, "w") as f:
            f.write(md_content)
        print(f"Markdown report written to: {md_path}", file=sys.stderr)

    # Cleanup
    if not args.keep_workdir and args.workdir is None:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)
        print(f"Cleaned up temp dir: {workdir}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
