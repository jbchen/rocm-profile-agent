"""Runs rocprofv3 to collect traces and hardware performance counters."""

import os
import subprocess
import sys
import tempfile

ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm-7.1.1")
ROCPROFV3 = ROCM_PATH + "/bin/rocprofv3"

# Instruction counters — max 8 per pass on gfx942 (GDS excluded, rarely used)
INSTS_COUNTERS = [
    "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_SMEM",
    "SQ_INSTS_LDS", "SQ_INSTS_FLAT",
    "SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR", "SQ_INSTS_MFMA",
]

# Memory counters - split into two passes to avoid exceeding HW limits
MEM_COUNTERS_HBM_L2 = [
    "TCC_EA0_RDREQ_sum", "TCC_EA0_WRREQ_sum",
    "TCP_TCC_READ_REQ_sum", "TCP_TCC_WRITE_REQ_sum",
]

MEM_COUNTERS_L1_LDS = [
    "TCP_TOTAL_READ_sum", "TCP_TOTAL_WRITE_sum", "SQ_INSTS_LDS",
]

# Occupancy counters
OCCUPANCY_COUNTERS = [
    "SQ_WAVE_CYCLES", "GRBM_GUI_ACTIVE",
]


def _run_rocprofv3(args, user_cmd, env=None):
    """Run rocprofv3 with the given args and user command."""
    cmd = [ROCPROFV3] + args + ["--"] + user_cmd
    print(f"  >> {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        env={**os.environ, **(env or {})},
    )
    if result.returncode != 0:
        print(f"rocprofv3 stderr:\n{result.stderr}", file=sys.stderr)
        # Don't raise - rocprofv3 sometimes exits non-zero but still produces output
    return result


def run_profiling(user_cmd, workdir=None, timeline_only=False):
    """Run all profiling passes and return paths to output files.

    If timeline_only is True, only the tracing pass (pass 1) is run,
    skipping all PMC counter collection.

    Returns a dict with keys:
        workdir, kernel_trace, hip_trace, agent_info,
        insts_counters, mem_hbm_l2_counters, mem_l1_lds_counters,
        occupancy_counters
    """
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="rocprof_")
    os.makedirs(workdir, exist_ok=True)

    results = {"workdir": workdir}
    total_passes = 1 if timeline_only else 5

    # --- Run 1: Tracing (HIP API + kernel dispatches) ---
    print(f"[1/{total_passes}] Collecting traces (HIP API + kernel dispatches)...", file=sys.stderr)
    trace_dir = os.path.join(workdir, "trace")
    os.makedirs(trace_dir, exist_ok=True)
    _run_rocprofv3(
        ["--hip-trace", "--kernel-trace",
         "-T",  # truncate/basenames
         "-f", "csv",
         "-d", trace_dir, "-o", "trace"],
        user_cmd,
    )
    results["kernel_trace"] = os.path.join(trace_dir, "trace_kernel_trace.csv")
    results["hip_trace"] = os.path.join(trace_dir, "trace_hip_api_trace.csv")
    results["agent_info"] = os.path.join(trace_dir, "trace_agent_info.csv")

    # Set empty paths for counter files (may not exist in timeline-only mode)
    results["insts_counters"] = ""
    results["mem_hbm_l2_counters"] = ""
    results["mem_l1_lds_counters"] = ""
    results["occupancy_counters"] = ""

    if timeline_only:
        return results

    # --- Run 2: Instruction counters ---
    print(f"[2/{total_passes}] Collecting instruction counters...", file=sys.stderr)
    insts_dir = os.path.join(workdir, "insts")
    os.makedirs(insts_dir, exist_ok=True)
    _run_rocprofv3(
        ["--pmc"] + INSTS_COUNTERS +
        ["-T", "-f", "csv",
         "-d", insts_dir, "-o", "insts"],
        user_cmd,
    )
    results["insts_counters"] = os.path.join(insts_dir, "insts_counter_collection.csv")

    # --- Run 3: Memory counters (HBM + L2) ---
    print(f"[3/{total_passes}] Collecting memory counters (HBM + L2)...", file=sys.stderr)
    mem1_dir = os.path.join(workdir, "mem_hbm_l2")
    os.makedirs(mem1_dir, exist_ok=True)
    _run_rocprofv3(
        ["--pmc"] + MEM_COUNTERS_HBM_L2 +
        ["-T", "-f", "csv",
         "-d", mem1_dir, "-o", "mem"],
        user_cmd,
    )
    results["mem_hbm_l2_counters"] = os.path.join(mem1_dir, "mem_counter_collection.csv")

    # --- Run 4: Memory counters (L1 + LDS) ---
    print(f"[4/{total_passes}] Collecting memory counters (L1 + LDS)...", file=sys.stderr)
    mem2_dir = os.path.join(workdir, "mem_l1_lds")
    os.makedirs(mem2_dir, exist_ok=True)
    _run_rocprofv3(
        ["--pmc"] + MEM_COUNTERS_L1_LDS +
        ["-T", "-f", "csv",
         "-d", mem2_dir, "-o", "mem"],
        user_cmd,
    )
    results["mem_l1_lds_counters"] = os.path.join(mem2_dir, "mem_counter_collection.csv")

    # --- Run 5: Occupancy counters ---
    print(f"[5/{total_passes}] Collecting occupancy counters...", file=sys.stderr)
    occ_dir = os.path.join(workdir, "occupancy")
    os.makedirs(occ_dir, exist_ok=True)
    _run_rocprofv3(
        ["--pmc"] + OCCUPANCY_COUNTERS +
        ["-T", "-f", "csv",
         "-d", occ_dir, "-o", "occ"],
        user_cmd,
    )
    results["occupancy_counters"] = os.path.join(occ_dir, "occ_counter_collection.csv")

    return results
