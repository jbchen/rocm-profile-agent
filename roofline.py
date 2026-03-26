"""Compute roofline bandwidth and compute utilization from counter data and GPU specs."""


def compute_utilization(insts_counters, gpu_specs):
    """Compute FLOPS/IOPS utilization from instruction counters.

    insts_counters: dict with counter totals + _duration_ns for one kernel
    gpu_specs: dict with peak_valu_tflops, peak_salu_gops, wavefront_size, etc.

    Returns dict with:
        valu_tflops, valu_pct  — VALU ops/s and % of FP32 VALU peak
        salu_gops, salu_pct    — SALU ops/s and % of scalar peak
        mfma_tflops, mfma_pct  — estimated MFMA FLOPS and % of FP16 MFMA peak
    """
    duration_ns = insts_counters.get("_duration_ns", 0)
    if duration_ns == 0:
        return {
            "valu_tflops": 0, "valu_pct": 0,
            "salu_gops": 0, "salu_pct": 0,
            "mfma_tflops": 0, "mfma_pct": 0,
        }

    duration_sec = duration_ns / 1e9
    wavefront_size = gpu_specs.get("wavefront_size", 64)

    # VALU: each instruction operates on wavefront_size lanes
    valu_insts = insts_counters.get("SQ_INSTS_VALU", 0)
    valu_ops = valu_insts * wavefront_size
    valu_tflops = valu_ops / duration_sec / 1e12

    # SALU: scalar, 1 op per instruction
    salu_insts = insts_counters.get("SQ_INSTS_SALU", 0)
    salu_gops = salu_insts / duration_sec / 1e9

    # MFMA: each instruction does many FLOPs depending on variant.
    # We use a representative FLOPs-per-instruction value from gpu_specs
    # that matches the published peak when multiplied by the peak issue rate.
    # MI300X (CDNA3): 8192 (mfma_f32_16x16x16_f16, 16-cycle latency)
    # MI350X (CDNA4): 16384 (2x throughput improvement over CDNA3)
    mfma_insts = insts_counters.get("SQ_INSTS_MFMA", 0)
    mfma_flops_per_inst = gpu_specs.get("mfma_flops_per_inst", 8192)
    mfma_ops = mfma_insts * mfma_flops_per_inst
    mfma_tflops = mfma_ops / duration_sec / 1e12

    def pct(achieved, peak):
        if peak == 0:
            return 0
        return min(100.0, achieved / peak * 100)

    return {
        "valu_tflops": valu_tflops,
        "valu_pct": pct(valu_tflops, gpu_specs.get("peak_valu_tflops", 163.3)),
        "salu_gops": salu_gops,
        "salu_pct": pct(salu_gops, gpu_specs.get("peak_salu_gops", 638.4)),
        "mfma_tflops": mfma_tflops,
        "mfma_pct": pct(mfma_tflops, gpu_specs.get("peak_mfma_f16_tflops", 1307.4)),
    }


def compute_roofline(kernel_counters, gpu_specs):
    """Compute bandwidth utilization for each memory level.

    kernel_counters: dict with counter totals + _duration_ns for one kernel
    gpu_specs: dict with hbm_bw_gbs, l2_bw_gbs, l1_bw_gbs, lds_bw_gbs, wavefront_size

    Returns dict with utilization percentages:
        {hbm_pct, l2_pct, l1_pct, lds_pct,
         hbm_achieved_gbs, l2_achieved_gbs, l1_achieved_gbs, lds_achieved_gbs}
    """
    duration_ns = kernel_counters.get("_duration_ns", 0)
    if duration_ns == 0:
        return {
            "hbm_pct": 0, "l2_pct": 0, "l1_pct": 0, "lds_pct": 0,
            "hbm_achieved_gbs": 0, "l2_achieved_gbs": 0,
            "l1_achieved_gbs": 0, "lds_achieved_gbs": 0,
        }

    duration_sec = duration_ns / 1e9
    wavefront_size = gpu_specs.get("wavefront_size", 64)

    # HBM: TCC_EA0_RDREQ_sum + TCC_EA0_WRREQ_sum count 64-byte requests to DRAM
    hbm_rd = kernel_counters.get("TCC_EA0_RDREQ_sum", 0)
    hbm_wr = kernel_counters.get("TCC_EA0_WRREQ_sum", 0)
    hbm_bytes = (hbm_rd + hbm_wr) * 64
    hbm_gbs = hbm_bytes / duration_sec / 1e9

    # L2: TCP_TCC_READ_REQ_sum + TCP_TCC_WRITE_REQ_sum count 64-byte cache line requests
    l2_rd = kernel_counters.get("TCP_TCC_READ_REQ_sum", 0)
    l2_wr = kernel_counters.get("TCP_TCC_WRITE_REQ_sum", 0)
    l2_bytes = (l2_rd + l2_wr) * 64
    l2_gbs = l2_bytes / duration_sec / 1e9

    # L1: TCP_TOTAL_READ_sum + TCP_TOTAL_WRITE_sum count 64-byte cache line transactions
    l1_rd = kernel_counters.get("TCP_TOTAL_READ_sum", 0)
    l1_wr = kernel_counters.get("TCP_TOTAL_WRITE_sum", 0)
    l1_bytes = (l1_rd + l1_wr) * 64
    l1_gbs = l1_bytes / duration_sec / 1e9

    # LDS: each SQ_INSTS_LDS operates on wavefront_size (64) elements * 4 bytes
    lds_insts = kernel_counters.get("SQ_INSTS_LDS", 0)
    lds_bytes = lds_insts * wavefront_size * 4
    lds_gbs = lds_bytes / duration_sec / 1e9

    def pct(achieved, peak):
        if peak == 0:
            return 0
        return min(100.0, achieved / peak * 100)

    return {
        "hbm_pct": pct(hbm_gbs, gpu_specs["hbm_bw_gbs"]),
        "l2_pct": pct(l2_gbs, gpu_specs["l2_bw_gbs"]),
        "l1_pct": pct(l1_gbs, gpu_specs["l1_bw_gbs"]),
        "lds_pct": pct(lds_gbs, gpu_specs["lds_bw_gbs"]),
        "hbm_achieved_gbs": hbm_gbs,
        "l2_achieved_gbs": l2_gbs,
        "l1_achieved_gbs": l1_gbs,
        "lds_achieved_gbs": lds_gbs,
    }
