"""GPU specifications lookup and auto-detection via rocminfo / rocprofv3 agent_info."""

import csv
import io
import subprocess

# Known GPU specs: peak bandwidth in GB/s, clock in MHz, CUs, wavefront size
GPU_SPECS = {
    "gfx942": {
        "name": "AMD Instinct MI300X",
        "cu_count": 304,
        "max_clock_mhz": 2100,
        "wavefront_size": 64,
        "hbm_bw_gbs": 5300,
        # Estimated peaks for cache levels
        "l2_bw_gbs": 13926,
        "l1_bw_gbs": 40857,
        "lds_bw_gbs": 81715,
        # Compute peaks
        # VALU FP32: CUs * 4 SIMDs * 64 lanes * clock_MHz / 1e6
        "peak_valu_tflops": 304 * 4 * 64 * 2100 / 1e6,  # ~163.3
        # SALU: CUs * clock_MHz / 1e3 (1 scalar op per CU per cycle)
        "peak_salu_gops": 304 * 2100 / 1e3,  # ~638.4
        # MFMA FP16 (published AMD spec)
        "peak_mfma_f16_tflops": 1307.4,
        # FLOPs per MFMA instruction (representative: mfma_f32_16x16x16_f16)
        "mfma_flops_per_inst": 8192,
    },
    "gfx950": {
        "name": "AMD Instinct MI350X",
        "cu_count": 256,
        "max_clock_mhz": 2200,
        "wavefront_size": 64,
        "hbm_bw_gbs": 8000,
        # Estimated peaks for cache levels (scaled from MI300X)
        "l2_bw_gbs": 12283,
        "l1_bw_gbs": 36036,
        "lds_bw_gbs": 72073,
        # Compute peaks
        # VALU FP32: CUs * 128 FP32 lanes/CU * clock_MHz / 1e6
        "peak_valu_tflops": 256 * 128 * 2200 / 1e6,  # ~72.1
        # SALU: CUs * clock_MHz / 1e3
        "peak_salu_gops": 256 * 2200 / 1e3,  # ~563.2
        # MFMA FP16 (published: ~2306 TFLOPS per GPU)
        "peak_mfma_f16_tflops": 2306.9,
        # FLOPs per MFMA instruction (CDNA4 2x throughput vs CDNA3)
        "mfma_flops_per_inst": 16384,
    },
}

DEFAULT_GFX = "gfx942"


def detect_gpu_from_agent_info(agent_info_csv_path):
    """Parse rocprofv3 agent_info.csv to detect GPU target and specs."""
    with open(agent_info_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Agent_Type", "").strip('"') == "GPU":
                gfx = row.get("Gfx_Target_Version", "0")
                gfx = str(gfx).strip('"')
                # Convert numeric gfx version (e.g. 90402) to gfx string (gfx942)
                if gfx.isdigit() and len(gfx) >= 5:
                    major = int(gfx[0])
                    minor = int(gfx[1:3])
                    stepping = int(gfx[3:5])
                    gfx_name = f"gfx{major}{minor}{stepping}"
                else:
                    gfx_name = gfx

                cu_count = int(row.get("Cu_Count", 0))
                max_clock = int(row.get("Max_Engine_Clk_Fcompute", 0))
                wave_size = int(row.get("Wave_Front_Size", 64))
                product = row.get("Product_Name", "").strip('"')

                specs = GPU_SPECS.get(gfx_name, GPU_SPECS[DEFAULT_GFX]).copy()
                # Override with detected values where available
                if cu_count > 0:
                    specs["cu_count"] = cu_count
                if max_clock > 0:
                    specs["max_clock_mhz"] = max_clock
                if wave_size > 0:
                    specs["wavefront_size"] = wave_size
                if product:
                    specs["name"] = product
                specs["gfx_target"] = gfx_name
                return specs

    # Fallback
    specs = GPU_SPECS[DEFAULT_GFX].copy()
    specs["gfx_target"] = DEFAULT_GFX
    return specs


def detect_gpu_from_rocminfo():
    """Fallback: parse rocminfo output to detect GPU."""
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10
        )
        output = result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        specs = GPU_SPECS[DEFAULT_GFX].copy()
        specs["gfx_target"] = DEFAULT_GFX
        return specs

    gfx_name = DEFAULT_GFX
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Name:") and "gfx" in line:
            gfx_name = line.split(":")[-1].strip()
            break

    specs = GPU_SPECS.get(gfx_name, GPU_SPECS[DEFAULT_GFX]).copy()
    specs["gfx_target"] = gfx_name
    return specs
