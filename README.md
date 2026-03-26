# ROCm Profile Agent

A Python CLI tool that profiles ROCm GPU applications using `rocprofv3` and generates a self-contained HTML report. No external dependencies — stdlib only.

**Target hardware**: AMD Instinct MI300X (gfx942), MI350X (gfx950)

## Prerequisites

- ROCm 7.1.1 installed at `/opt/rocm-7.1.1` (or set `ROCPROFV3` env var to your `rocprofv3` path)
- Python 3.6+
- A ROCm-enabled GPU application to profile

## Usage

```bash
python3 rocm_profile_agent.py [options] -- <application> [application args...]
```

### Example

```bash
python3 rocm_profile_agent.py -- ./llama.cpp/build/bin/llama-bench \
    -m ./llama.cpp/models/tinyllama-1.1b-q4_0.gguf -t 1 -r 1 -p 0 -n 16
```

This produces `profile_report_<timestamp>.html` in the current directory.

### Options

| Flag | Description |
|------|-------------|
| `-o`, `--output` | Output HTML file path (default: `profile_report_<timestamp>.html`) |
| `-w`, `--workdir` | Directory for intermediate profiling files (default: temp dir) |
| `-n`, `--top-n` | Number of top kernels to analyze (default: 5) |
| `--keep-workdir` | Keep intermediate profiling files after report generation |

## What It Does

The tool runs your application **4 times** under `rocprofv3` to collect different data. Each pass executes the exact command shown below (with `<workdir>` as the intermediate output directory):

### Pass 1: HIP API + Kernel Dispatch Tracing

```bash
rocprofv3 --hip-trace --kernel-trace -T -f csv \
    -d <workdir>/trace -o trace \
    -- <application> [args...]
```

- `--hip-trace` — records all HIP runtime API calls (e.g. `hipLaunchKernel`, `hipMemcpy`) with timestamps
- `--kernel-trace` — records GPU kernel dispatch start/end timestamps
- `-T` — truncates kernel names to basenames (strips template parameters)
- `-f csv` — outputs CSV format (instead of default SQLite `.db`)
- **Produces**: `trace_kernel_trace.csv`, `trace_hip_api_trace.csv`, `trace_agent_info.csv`

### Pass 2: Instruction Counters

```bash
rocprofv3 --pmc SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_SMEM SQ_INSTS_LDS \
    SQ_INSTS_FLAT SQ_INSTS_VMEM_RD SQ_INSTS_VMEM_WR SQ_INSTS_MFMA \
    -T -f csv \
    -d <workdir>/insts -o insts \
    -- <application> [args...]
```

- `--pmc` — collects hardware performance counters (one pass, all counters must fit in a single HW pass)
- 8 counters measuring instruction types issued per kernel dispatch:
  - `SQ_INSTS_VALU` — vector ALU instructions
  - `SQ_INSTS_SALU` — scalar ALU instructions
  - `SQ_INSTS_SMEM` — scalar memory instructions
  - `SQ_INSTS_LDS` — local data share (shared memory) instructions
  - `SQ_INSTS_FLAT` — flat memory instructions
  - `SQ_INSTS_VMEM_RD` — vector memory read instructions
  - `SQ_INSTS_VMEM_WR` — vector memory write instructions
  - `SQ_INSTS_MFMA` — matrix fused multiply-add instructions
- **Produces**: `insts_counter_collection.csv` (long format: one row per counter per dispatch)

### Pass 3: Memory Counters — HBM + L2

```bash
rocprofv3 --pmc TCC_EA0_RDREQ_sum TCC_EA0_WRREQ_sum \
    TCP_TCC_READ_REQ_sum TCP_TCC_WRITE_REQ_sum \
    -T -f csv \
    -d <workdir>/mem_hbm_l2 -o mem \
    -- <application> [args...]
```

- `TCC_EA0_RDREQ_sum` / `TCC_EA0_WRREQ_sum` — 64-byte read/write requests to HBM (DRAM)
- `TCP_TCC_READ_REQ_sum` / `TCP_TCC_WRITE_REQ_sum` — 64-byte cache line requests from L1 to L2
- **Produces**: `mem_counter_collection.csv`

### Pass 4: Memory Counters — L1 + LDS

```bash
rocprofv3 --pmc TCP_TOTAL_READ_sum TCP_TOTAL_WRITE_sum SQ_INSTS_LDS \
    -T -f csv \
    -d <workdir>/mem_l1_lds -o mem \
    -- <application> [args...]
```

- `TCP_TOTAL_READ_sum` / `TCP_TOTAL_WRITE_sum` — total 64-byte L1 cache transactions (reads/writes)
- `SQ_INSTS_LDS` — LDS instructions (used to estimate LDS bandwidth)
- **Produces**: `mem_counter_collection.csv`

> **Note:** Counter groups are split across passes 2–4 because gfx942 hardware supports a maximum of 8 PMC counters per pass. Requesting more causes `rocprofv3` to hang.

## HTML Report Contents

The generated report is a single self-contained HTML file (no external dependencies) with:

1. **Header** — Command, timestamp, GPU info (auto-detected from rocprofv3 agent info)
2. **Timeline** — SVG swimlane chart with CPU (HIP API) and GPU (kernel dispatch) lanes
3. **Top Kernels** — Donut chart and summary table showing the top N kernels by cumulative GPU time
4. **Kernel Detail Cards** (one per top kernel):
   - Duration statistics (total, avg, min, max, call count)
   - Instruction mix (horizontal stacked bar chart)
   - Bandwidth utilization (bar chart showing % of peak at HBM, L2, L1, LDS levels)

One example report is [here](https://jbchen.github.io/rocm-profile-agent/report/llama.cpp_report.html).

## File Structure

```
rocm_profile_agent.py   # CLI entry point — orchestrates the pipeline
profiler.py             # Runs rocprofv3 passes, manages temp files
parser.py               # Parses rocprofv3 CSV output into structured data
roofline.py             # Computes bandwidth utilization % from counters + GPU specs
report.py               # Generates self-contained HTML with inline SVG charts
gpu_specs.py            # GPU spec lookup and auto-detection via agent_info.csv
```

## Bandwidth Utilization Calculations

| Level | Formula | MI300X Peak |
|-------|---------|-------------|
| HBM | `(TCC_EA0_RDREQ_sum + TCC_EA0_WRREQ_sum) * 64 / duration` | 5,300 GB/s |
| L2 | `(TCP_TCC_READ_REQ_sum + TCP_TCC_WRITE_REQ_sum) * 64 / duration` | 13,926 GB/s |
| L1 | `(TCP_TOTAL_READ_sum + TCP_TOTAL_WRITE_sum) * 64 / duration` | 40,857 GB/s |
| LDS | `SQ_INSTS_LDS * wavefront_size * 4 / duration` | 81,715 GB/s |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ROCPROFV3` | Path to `rocprofv3` binary | `/opt/rocm-7.1.1/bin/rocprofv3` |
