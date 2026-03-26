"""Generate a self-contained HTML report with embedded SVG charts."""

import html
import datetime

# ---------------------------------------------------------------------------
# Color palettes — each chart type has its own palette to avoid confusion.
# ---------------------------------------------------------------------------

# Kernel identification (timeline GPU lane, pie chart, summary table, detail cards)
KERNEL_COLORS = [
    "#2563eb",  # blue
    "#dc2626",  # red
    "#16a34a",  # green
    "#9333ea",  # purple
    "#ea580c",  # orange
    "#0891b2",  # cyan
    "#be185d",  # pink
    "#854d0e",  # brown
    "#4f46e5",  # indigo
    "#0d9488",  # teal
]
KERNEL_OTHER = "#b0b0b0"  # non-top kernels

# Instruction mix (stacked bar in kernel detail cards)
INSTS_COLORS = {
    "SQ_INSTS_VALU":    "#6366f1",  # indigo
    "SQ_INSTS_SALU":    "#8b5cf6",  # violet
    "SQ_INSTS_SMEM":    "#0891b2",  # cyan
    "SQ_INSTS_VMEM_RD": "#0d9488",  # teal
    "SQ_INSTS_VMEM_WR": "#059669",  # emerald
    "SQ_INSTS_LDS":     "#d97706",  # amber
    "SQ_INSTS_FLAT":    "#78716c",  # stone
    "SQ_INSTS_MFMA":    "#e11d48",  # rose
}

# Bandwidth roofline bars (blue sequential — "data movement")
BW_COLORS = {
    "HBM": "#1e3a5f",  # dark navy
    "L2":  "#2563eb",  # blue
    "L1":  "#0ea5e9",  # sky
    "LDS": "#14b8a6",  # teal
}

# Compute roofline bars (green sequential — "processing")
COMPUTE_COLORS = {
    "VALU": "#064e3b",  # dark green
    "MFMA": "#047857",  # emerald
    "SALU": "#34d399",  # green
}

# Memory transfer directions (timeline memory lane)
MEM_COLORS = {
    "H2D":   "#0369a1",  # dark sky (distinct from kernel blue)
    "D2H":   "#0f766e",  # teal
    "D2D":   "#92400e",  # dark amber
    "OTHER": "#6b7280",  # gray
}

# CPU lane event types (timeline CPU lane — intentionally muted)
CPU_COLORS = {
    "LAUNCH": "#7c3aed",  # purple
    "SYNC":   "#b91c1c",  # dark red
    "API":    "#94a3b8",  # slate
}


def _esc(text):
    return html.escape(str(text))


def _fmt_ns(ns):
    """Format nanoseconds to a human-readable string."""
    if ns >= 1e9:
        return f"{ns/1e9:.3f} s"
    if ns >= 1e6:
        return f"{ns/1e6:.3f} ms"
    if ns >= 1e3:
        return f"{ns/1e3:.1f} us"
    return f"{ns} ns"


def _fmt_gbs(gbs):
    return f"{gbs:.1f} GB/s"


def _short_name(name, max_len=40):
    if len(name) <= max_len:
        return name
    return name[:max_len - 3] + "..."


def _darken(hex_color, factor=0.25):
    """Blend a hex color toward black by *factor* (0 = unchanged, 1 = black)."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = int(r * (1 - factor))
    g = int(g * (1 - factor))
    b = int(b * (1 - factor))
    return f"#{r:02x}{g:02x}{b:02x}"


def _kernel_color(name, top_kernels):
    """Get color for a kernel based on its position in the top list."""
    for i, k in enumerate(top_kernels):
        if k["kernel_name"] == name:
            return KERNEL_COLORS[i % len(KERNEL_COLORS)]
    return KERNEL_OTHER


def _classify_hip_event(func_name):
    """Classify a HIP API function into a category and color."""
    f = func_name.lower()
    if "memcpy" in f or "memset" in f:
        if "dtoh" in f or "devicetohost" in f:
            return "mem", "D\u2192H", MEM_COLORS["D2H"]
        elif "htod" in f or "hosttodevice" in f:
            return "mem", "H\u2192D", MEM_COLORS["H2D"]
        elif "dtod" in f or "devicetodevice" in f:
            return "mem", "D\u2192D", MEM_COLORS["D2D"]
        return "mem", "Host \u2194 GPU", MEM_COLORS["OTHER"]
    if "launch" in f:
        return "cpu", "Launch", CPU_COLORS["LAUNCH"]
    if "sync" in f or "wait" in f:
        return "cpu", "Sync", CPU_COLORS["SYNC"]
    return "cpu", "API", CPU_COLORS["API"]


def _classify_kernel_event(kernel_name):
    """Check if a kernel event is actually a memory operation."""
    kl = kernel_name.lower()
    if "copybuffer" in kl or "fillbuffer" in kl:
        return "mem", "D\u2192D", MEM_COLORS["D2D"]
    return "kernel", None, None


def _generate_timeline_svg(kernel_events, hip_events, top_kernels, width=1200):
    """Generate an SVG swimlane timeline chart with 3 lanes.

    Returns (svg_string, layout_dict) where layout_dict contains
    gpu_y, lane_h, margin_left, chart_w for overlay positioning.
    """
    if not kernel_events and not hip_events:
        return "<p>No timeline data available.</p>", {}

    all_starts = []
    all_ends = []
    if kernel_events:
        all_starts.extend(e["start_ns"] for e in kernel_events)
        all_ends.extend(e["end_ns"] for e in kernel_events)
    if hip_events:
        all_starts.extend(e["start_ns"] for e in hip_events)
        all_ends.extend(e["end_ns"] for e in hip_events)

    t_min = min(all_starts)
    t_max = max(all_ends)
    t_range = t_max - t_min
    if t_range == 0:
        t_range = 1

    margin_left = 100
    margin_right = 20
    chart_w = width - margin_left - margin_right
    lane_h = 40
    lane_gap = 6
    header_h = 10

    # Filter HIP events to runtime API only
    hip_runtime = [e for e in hip_events
                   if "HIP_RUNTIME_API" in e.get("domain", "")]

    # 3 lanes: CPU, Memory, GPU
    cpu_y = header_h
    mem_y = cpu_y + lane_h + lane_gap
    gpu_y = mem_y + lane_h + lane_gap
    axis_y_base = gpu_y + lane_h
    legend_y = axis_y_base + 25
    # Legend height: top kernels + mem legend
    num_legend_items = min(len(top_kernels), 10) + 3  # +3 for mem directions
    legend_h = ((num_legend_items + 3) // 4) * 18 + 10  # 4 columns
    svg_h = legend_y + legend_h + 5

    def x_pos(ns):
        return margin_left + (ns - t_min) / t_range * chart_w

    lines = []
    lines.append(f'<svg id="timeline-svg" width="{width}" height="{svg_h}" '
                 f'xmlns="http://www.w3.org/2000/svg" '
                 f'style="background:#fff;font-family:sans-serif;">')

    # Defs: clip path for chart area
    lines.append(f'<defs>'
                 f'<clipPath id="chart-clip">'
                 f'<rect x="{margin_left}" y="0" width="{chart_w}" height="{axis_y_base}"/>'
                 f'</clipPath>'
                 f'</defs>')

    # Lane labels
    for label, y in [("CPU (HIP)", cpu_y), ("Memory", mem_y), ("GPU", gpu_y)]:
        lines.append(f'<text x="{margin_left - 8}" y="{y + lane_h//2 + 4}" '
                     f'font-size="11" fill="#555" text-anchor="end" font-weight="500">{label}</text>')

    # Lane backgrounds
    lane_colors = ["#f5f5f5", "#f0f0f0", "#fafafa"]
    for y, bg in [(cpu_y, lane_colors[0]),
                  (mem_y, lane_colors[1]),
                  (gpu_y, lane_colors[2])]:
        lines.append(f'<rect x="{margin_left}" y="{y}" width="{chart_w}" '
                     f'height="{lane_h}" fill="{bg}" stroke="#e5e7eb" stroke-width="0.5"/>')

    # --- Fixed content: CPU + Memory lanes (clipped, not zoomable) ---
    lines.append(f'<g clip-path="url(#chart-clip)">')

    for ev in hip_runtime:
        cat, cat_label, color = _classify_hip_event(ev["function"])
        if cat == "mem":
            continue
        x1 = x_pos(ev["start_ns"])
        x2 = x_pos(ev["end_ns"])
        w = max(x2 - x1, 0.5)
        func = _esc(ev["function"])
        dur = _fmt_ns(ev["duration_ns"])
        lines.append(
            f'<rect class="tl-ev" x="{x1:.1f}" y="{cpu_y + 3}" width="{w:.1f}" '
            f'height="{lane_h - 6}" fill="{color}" opacity="0.8" rx="1" '
            f'data-type="hip" data-name="{func}" data-dur="{dur}" '
            f'data-cat="{_esc(cat_label)}"/>')

    for ev in hip_runtime:
        cat, cat_label, color = _classify_hip_event(ev["function"])
        if cat != "mem":
            continue
        x1 = x_pos(ev["start_ns"])
        x2 = x_pos(ev["end_ns"])
        w = max(x2 - x1, 0.5)
        func = _esc(ev["function"])
        dur = _fmt_ns(ev["duration_ns"])
        lines.append(
            f'<rect class="tl-ev" x="{x1:.1f}" y="{mem_y + 3}" width="{w:.1f}" '
            f'height="{lane_h - 6}" fill="{color}" opacity="0.85" rx="1" '
            f'data-type="mem" data-name="{func}" data-dur="{dur}" '
            f'data-dir="{_esc(cat_label)}"/>')

    for ev in kernel_events:
        cat, cat_label, color = _classify_kernel_event(ev["kernel_name"])
        if cat != "mem":
            continue
        x1 = x_pos(ev["start_ns"])
        x2 = x_pos(ev["end_ns"])
        w = max(x2 - x1, 0.5)
        kname = _esc(ev["kernel_name"])
        dur = _fmt_ns(ev["duration_ns"])
        lines.append(
            f'<rect class="tl-ev" x="{x1:.1f}" y="{mem_y + 3}" width="{w:.1f}" '
            f'height="{lane_h - 6}" fill="{color}" opacity="0.85" rx="1" '
            f'data-type="mem" data-name="{kname}" data-dur="{dur}" '
            f'data-dir="{_esc(cat_label)}"/>')

    lines.append('</g>')  # end fixed content

    # --- GPU lane (zoomable, clipped) ---
    lines.append(f'<g clip-path="url(#chart-clip)">')
    lines.append(f'<g id="gpu-content">')

    # Pre-compute dispatch counts per kernel for alternating shades + tooltip
    gpu_dispatch_counts = {}
    for ev in kernel_events:
        cat, _, _ = _classify_kernel_event(ev["kernel_name"])
        if cat != "mem":
            gpu_dispatch_counts[ev["kernel_name"]] = \
                gpu_dispatch_counts.get(ev["kernel_name"], 0) + 1

    dispatch_idx = {}
    for ev in kernel_events:
        cat, _, _ = _classify_kernel_event(ev["kernel_name"])
        if cat == "mem":
            continue
        kname_raw = ev["kernel_name"]
        idx = dispatch_idx.get(kname_raw, 0)
        dispatch_idx[kname_raw] = idx + 1
        total = gpu_dispatch_counts[kname_raw]

        base_color = _kernel_color(kname_raw, top_kernels)
        # Alternate between base and darkened shade for repeated dispatches
        color = _darken(base_color, 0.35) if idx % 2 == 1 else base_color

        x1 = x_pos(ev["start_ns"])
        x2 = x_pos(ev["end_ns"])
        w = max(x2 - x1, 0.5)
        kname = _esc(kname_raw)
        dur = _fmt_ns(ev["duration_ns"])
        gx = ev.get("grid_size_x", 0)
        gy = ev.get("grid_size_y", 0)
        gz = ev.get("grid_size_z", 0)
        wx = ev.get("workgroup_size_x", 0)
        wy = ev.get("workgroup_size_y", 0)
        wz = ev.get("workgroup_size_z", 0)
        lines.append(
            f'<rect class="tl-ev gpu-ev" x="{x1:.1f}" y="{gpu_y + 3}" width="{w:.1f}" '
            f'height="{lane_h - 6}" fill="{color}" '
            f'stroke="#444" stroke-width="1" vector-effect="non-scaling-stroke" rx="1" '
            f'data-type="kernel" data-name="{kname}" data-dur="{dur}" '
            f'data-grid="{gx}x{gy}x{gz}" data-wg="{wx}x{wy}x{wz}" '
            f'data-idx="{idx + 1}" data-cnt="{total}"/>')

    lines.append('</g>')  # end gpu-content
    lines.append('</g>')  # end gpu clip group

    # Time axis
    lines.append(f'<g id="tl-axis">')
    n_ticks = 10
    for i in range(n_ticks + 1):
        frac = i / n_ticks
        t = t_min + t_range * frac
        x = margin_left + frac * chart_w
        label = _fmt_ns(int(t - t_min))
        lines.append(f'<line x1="{x:.1f}" y1="{axis_y_base}" '
                     f'x2="{x:.1f}" y2="{axis_y_base + 5}" stroke="#aaa"/>')
        lines.append(f'<text x="{x:.1f}" y="{axis_y_base + 16}" font-size="9" '
                     f'fill="#666" text-anchor="middle">{label}</text>')
    lines.append('</g>')

    # Legend
    lines.append(f'<g id="tl-legend">')
    leg_items = []
    for i, k in enumerate(top_kernels):
        c = KERNEL_COLORS[i % len(KERNEL_COLORS)]
        leg_items.append((c, _short_name(k["kernel_name"], 25)))
    leg_items.append((KERNEL_OTHER, "Other kernels"))
    leg_items.append((MEM_COLORS["H2D"], "H\u2192D (Host\u2192Device)"))
    leg_items.append((MEM_COLORS["D2H"], "D\u2192H (Device\u2192Host)"))
    leg_items.append((MEM_COLORS["D2D"], "D\u2192D (Device\u2192Device)"))

    cols = 4
    col_w = (width - margin_left) // cols
    for idx, (color, label) in enumerate(leg_items):
        col = idx % cols
        row = idx // cols
        lx = margin_left + col * col_w
        ly = legend_y + row * 18
        lines.append(f'<rect x="{lx}" y="{ly}" width="10" height="10" fill="{color}" rx="1"/>')
        lines.append(f'<text x="{lx + 14}" y="{ly + 9}" font-size="10" fill="#444">'
                     f'{_esc(label)}</text>')
    lines.append('</g>')

    # Zoom level indicator
    lines.append(f'<text id="tl-zoom-label" x="{width - 10}" y="{header_h + 12}" '
                 f'text-anchor="end" font-size="10" fill="#999">1.0x</text>')

    lines.append('</svg>')

    layout = {
        "gpu_y": gpu_y,
        "lane_h": lane_h,
        "margin_left": margin_left,
        "chart_w": chart_w,
    }
    return "\n".join(lines), layout


def _generate_pie_svg(top_kernels, width=400, height=300):
    """Generate an SVG donut chart for kernel time distribution."""
    if not top_kernels:
        return "<p>No kernel data.</p>"

    cx, cy = width // 2, height // 2 - 10
    r_outer = 100
    r_inner = 55

    total = sum(k["total_ns"] for k in top_kernels)
    if total == 0:
        return "<p>No kernel data.</p>"

    lines = []
    lines.append(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
                 f'style="font-family:sans-serif;">')

    import math
    angle = -math.pi / 2  # start at top

    for i, k in enumerate(top_kernels):
        frac = k["total_ns"] / total
        sweep = frac * 2 * math.pi
        color = KERNEL_COLORS[i % len(KERNEL_COLORS)]

        # Arc path for donut segment
        x1_o = cx + r_outer * math.cos(angle)
        y1_o = cy + r_outer * math.sin(angle)
        x2_o = cx + r_outer * math.cos(angle + sweep)
        y2_o = cy + r_outer * math.sin(angle + sweep)
        x1_i = cx + r_inner * math.cos(angle + sweep)
        y1_i = cy + r_inner * math.sin(angle + sweep)
        x2_i = cx + r_inner * math.cos(angle)
        y2_i = cy + r_inner * math.sin(angle)

        large = 1 if sweep > math.pi else 0

        d = (f"M {x1_o:.1f} {y1_o:.1f} "
             f"A {r_outer} {r_outer} 0 {large} 1 {x2_o:.1f} {y2_o:.1f} "
             f"L {x1_i:.1f} {y1_i:.1f} "
             f"A {r_inner} {r_inner} 0 {large} 0 {x2_i:.1f} {y2_i:.1f} Z")

        kname = _esc(_short_name(k["kernel_name"], 30))
        pct = k["pct"]
        lines.append(f'<path d="{d}" fill="{color}" stroke="#fff" stroke-width="1">'
                     f'<title>{kname}: {pct:.1f}%</title></path>')

        angle += sweep

    # Center text
    lines.append(f'<text x="{cx}" y="{cy - 5}" text-anchor="middle" font-size="14" fill="#333">Top {len(top_kernels)}</text>')
    lines.append(f'<text x="{cx}" y="{cy + 12}" text-anchor="middle" font-size="11" fill="#666">Kernels</text>')

    # Legend below
    leg_y = cy + r_outer + 25
    for i, k in enumerate(top_kernels):
        ly = leg_y + i * 16
        color = KERNEL_COLORS[i % len(KERNEL_COLORS)]
        kname = _esc(_short_name(k["kernel_name"], 35))
        lines.append(f'<rect x="10" y="{ly - 9}" width="10" height="10" fill="{color}"/>')
        lines.append(f'<text x="25" y="{ly}" font-size="10" fill="#333">'
                     f'{kname} ({k["pct"]:.1f}%)</text>')

    # Adjust SVG height for legend
    final_h = leg_y + len(top_kernels) * 16 + 10
    lines[0] = (f'<svg width="{width}" height="{final_h}" xmlns="http://www.w3.org/2000/svg" '
                f'style="font-family:sans-serif;">')

    lines.append('</svg>')
    return "\n".join(lines)


def _generate_insts_bar_svg(insts_counters, width=500, height=30):
    """Generate a horizontal stacked bar for instruction mix."""
    insts_keys = [
        ("SQ_INSTS_VALU", "VALU"),
        ("SQ_INSTS_SALU", "SALU"),
        ("SQ_INSTS_SMEM", "SMEM"),
        ("SQ_INSTS_VMEM_RD", "VMEM_RD"),
        ("SQ_INSTS_VMEM_WR", "VMEM_WR"),
        ("SQ_INSTS_LDS", "LDS"),
        ("SQ_INSTS_FLAT", "FLAT"),
        ("SQ_INSTS_MFMA", "MFMA"),
    ]
    insts_order = [(k, l, INSTS_COLORS[k]) for k, l in insts_keys]

    total = sum(insts_counters.get(key, 0) for key, _, _ in insts_order)
    if total == 0:
        return "<p>No instruction data.</p>"

    bar_y = 0
    bar_h = height
    lines = [f'<svg width="{width}" height="{bar_h + 20}" xmlns="http://www.w3.org/2000/svg" '
             f'style="font-family:sans-serif;">']

    x = 0
    for key, label, color in insts_order:
        val = insts_counters.get(key, 0)
        if val == 0:
            continue
        w = val / total * width
        pct = val / total * 100
        lines.append(f'<rect x="{x:.1f}" y="{bar_y}" width="{w:.1f}" height="{bar_h}" fill="{color}">'
                     f'<title>{label}: {int(val):,} ({pct:.1f}%)</title></rect>')
        if w > 30:
            lines.append(f'<text x="{x + w/2:.1f}" y="{bar_y + bar_h/2 + 4}" '
                         f'text-anchor="middle" font-size="9" fill="#fff">{label}</text>')
        x += w

    lines.append('</svg>')
    return "\n".join(lines)


def _generate_roofline_bars_svg(roofline, gpu_specs, width=400, height=160):
    """Generate bar chart for roofline utilization."""
    levels = [
        ("HBM", roofline["hbm_pct"], roofline["hbm_achieved_gbs"], gpu_specs["hbm_bw_gbs"], BW_COLORS["HBM"]),
        ("L2", roofline["l2_pct"], roofline["l2_achieved_gbs"], gpu_specs["l2_bw_gbs"], BW_COLORS["L2"]),
        ("L1", roofline["l1_pct"], roofline["l1_achieved_gbs"], gpu_specs["l1_bw_gbs"], BW_COLORS["L1"]),
        ("LDS", roofline["lds_pct"], roofline["lds_achieved_gbs"], gpu_specs["lds_bw_gbs"], BW_COLORS["LDS"]),
    ]

    bar_w = 60
    gap = 30
    margin_left = 50
    margin_bottom = 40
    chart_h = height - margin_bottom

    total_w = margin_left + len(levels) * (bar_w + gap)
    lines = [f'<svg width="{total_w}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
             f'style="font-family:sans-serif;">']

    # Y axis
    for pct_tick in [0, 25, 50, 75, 100]:
        y = chart_h - (pct_tick / 100 * chart_h)
        lines.append(f'<line x1="{margin_left}" y1="{y:.0f}" x2="{total_w}" y2="{y:.0f}" '
                     f'stroke="#eee" stroke-width="1"/>')
        lines.append(f'<text x="{margin_left - 5}" y="{y + 4:.0f}" text-anchor="end" '
                     f'font-size="9" fill="#999">{pct_tick}%</text>')

    for i, (label, pct, achieved, peak, color) in enumerate(levels):
        x = margin_left + i * (bar_w + gap) + gap // 2
        bar_h = max(pct / 100 * chart_h, 1)
        y = chart_h - bar_h

        lines.append(f'<rect x="{x}" y="{y:.0f}" width="{bar_w}" height="{bar_h:.0f}" '
                     f'fill="{color}" rx="2">'
                     f'<title>{label}: {pct:.1f}% ({_fmt_gbs(achieved)} / {_fmt_gbs(peak)})</title></rect>')

        # Percentage label
        label_y = y - 5 if y > 15 else y + 15
        lines.append(f'<text x="{x + bar_w/2}" y="{label_y:.0f}" text-anchor="middle" '
                     f'font-size="10" fill="#333" font-weight="bold">{pct:.1f}%</text>')

        # X axis label
        lines.append(f'<text x="{x + bar_w/2}" y="{chart_h + 15}" text-anchor="middle" '
                     f'font-size="10" fill="#333">{label}</text>')
        lines.append(f'<text x="{x + bar_w/2}" y="{chart_h + 27}" text-anchor="middle" '
                     f'font-size="8" fill="#999">{_fmt_gbs(achieved)}</text>')

    lines.append('</svg>')
    return "\n".join(lines)


def _fmt_tflops(tflops):
    if tflops >= 1:
        return f"{tflops:.1f} TFLOPS"
    if tflops >= 0.001:
        return f"{tflops*1000:.1f} GFLOPS"
    return f"{tflops*1e6:.1f} MFLOPS"


def _fmt_gops(gops):
    if gops >= 1000:
        return f"{gops/1000:.1f} TOPS"
    if gops >= 1:
        return f"{gops:.1f} GOPS"
    return f"{gops*1000:.1f} MOPS"


def _generate_compute_bars_svg(compute, gpu_specs, width=500, height=160):
    """Generate bar chart for compute (FLOPS/IOPS) utilization."""
    levels = [
        ("VALU\n(FP32)", compute["valu_pct"], _fmt_tflops(compute["valu_tflops"]),
         _fmt_tflops(gpu_specs.get("peak_valu_tflops", 0)), COMPUTE_COLORS["VALU"]),
        ("MFMA\n(FP16)", compute["mfma_pct"], _fmt_tflops(compute["mfma_tflops"]),
         _fmt_tflops(gpu_specs.get("peak_mfma_f16_tflops", 0)), COMPUTE_COLORS["MFMA"]),
        ("SALU\n(INT)", compute["salu_pct"], _fmt_gops(compute["salu_gops"]),
         _fmt_gops(gpu_specs.get("peak_salu_gops", 0)), COMPUTE_COLORS["SALU"]),
    ]

    bar_w = 60
    gap = 30
    margin_left = 50
    margin_bottom = 50
    chart_h = height - margin_bottom

    total_w = margin_left + len(levels) * (bar_w + gap)
    lines = [f'<svg width="{total_w}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
             f'style="font-family:sans-serif;">']

    # Y axis
    for pct_tick in [0, 25, 50, 75, 100]:
        y = chart_h - (pct_tick / 100 * chart_h)
        lines.append(f'<line x1="{margin_left}" y1="{y:.0f}" x2="{total_w}" y2="{y:.0f}" '
                     f'stroke="#eee" stroke-width="1"/>')
        lines.append(f'<text x="{margin_left - 5}" y="{y + 4:.0f}" text-anchor="end" '
                     f'font-size="9" fill="#999">{pct_tick}%</text>')

    for i, (label, pct, achieved_str, peak_str, color) in enumerate(levels):
        x = margin_left + i * (bar_w + gap) + gap // 2
        bar_h = max(pct / 100 * chart_h, 1)
        y = chart_h - bar_h

        lines.append(f'<rect x="{x}" y="{y:.0f}" width="{bar_w}" height="{bar_h:.0f}" '
                     f'fill="{color}" rx="2">'
                     f'<title>{label.replace(chr(10), " ")}: {pct:.1f}% ({achieved_str} / {peak_str})</title></rect>')

        # Percentage label
        label_y = y - 5 if y > 15 else y + 15
        lines.append(f'<text x="{x + bar_w/2}" y="{label_y:.0f}" text-anchor="middle" '
                     f'font-size="10" fill="#333" font-weight="bold">{pct:.1f}%</text>')

        # X axis label (two lines)
        label_parts = label.split("\n")
        lines.append(f'<text x="{x + bar_w/2}" y="{chart_h + 15}" text-anchor="middle" '
                     f'font-size="10" fill="#333">{_esc(label_parts[0])}</text>')
        if len(label_parts) > 1:
            lines.append(f'<text x="{x + bar_w/2}" y="{chart_h + 27}" text-anchor="middle" '
                         f'font-size="9" fill="#888">{_esc(label_parts[1])}</text>')
        lines.append(f'<text x="{x + bar_w/2}" y="{chart_h + 40}" text-anchor="middle" '
                     f'font-size="8" fill="#999">{achieved_str}</text>')

    lines.append('</svg>')
    return "\n".join(lines)


def generate_report(command, gpu_specs, kernel_events, hip_events,
                    top_kernels, insts_by_kernel, roofline_by_kernel,
                    compute_by_kernel=None, timestamp=None):
    """Generate a self-contained HTML report string."""
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    gpu_name = gpu_specs.get("name", "Unknown GPU")
    gfx = gpu_specs.get("gfx_target", "")

    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ROCm Profile Report</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #f8f9fa; color: #333; padding: 20px; line-height: 1.5; }}
.container {{ max-width: 1300px; margin: 0 auto; }}
h1 {{ font-size: 24px; margin-bottom: 5px; color: #1a1a1a; }}
h2 {{ font-size: 18px; margin: 25px 0 10px 0; color: #2c3e50; border-bottom: 2px solid #e0e0e0; padding-bottom: 5px; }}
h3 {{ font-size: 15px; margin: 15px 0 8px 0; color: #34495e; }}
.header {{ background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 20px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.header .meta {{ font-size: 13px; color: #666; margin-top: 5px; }}
.header .cmd {{ font-family: monospace; background: #f0f0f0; padding: 8px 12px;
               border-radius: 4px; margin-top: 10px; font-size: 13px; word-break: break-all; }}
.card {{ background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 15px;
         box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
               gap: 10px; margin: 10px 0; }}
.stat {{ background: #f8f9fa; border-radius: 6px; padding: 10px; text-align: center; }}
.stat .label {{ font-size: 11px; color: #888; text-transform: uppercase; }}
.stat .value {{ font-size: 16px; font-weight: 600; color: #2c3e50; margin-top: 2px; }}
.kernel-name {{ font-family: monospace; font-size: 13px; word-break: break-all;
                background: #f0f0f0; padding: 4px 8px; border-radius: 3px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #eee; }}
th {{ background: #f8f9fa; font-weight: 600; color: #555; }}
.timeline-container {{ overflow-x: auto; position: relative; }}
.section-row {{ display: flex; gap: 20px; flex-wrap: wrap; }}
.section-col {{ flex: 1; min-width: 300px; }}
#tl-tooltip {{ position: absolute; display: none; pointer-events: none;
    background: rgba(30,30,30,0.92); color: #f0f0f0; border-radius: 6px;
    padding: 8px 12px; font-size: 12px; line-height: 1.6; z-index: 100;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3); max-width: 400px;
    white-space: nowrap; font-family: -apple-system, sans-serif; }}
#tl-tooltip .tt-title {{ font-weight: 600; color: #fff; margin-bottom: 2px;
    white-space: normal; word-break: break-all; }}
#tl-tooltip .tt-row {{ color: #ccc; }}
#tl-tooltip .tt-val {{ color: #7dd3fc; font-weight: 500; }}
.tl-ev:hover {{ stroke: #000; stroke-width: 1.5; opacity: 1 !important; }}
</style>
</head>
<body>
<div class="container">
""")

    # Header
    parts.append(f"""<div class="header">
<h1>ROCm Profile Report</h1>
<div class="meta">{_esc(timestamp)} &mdash; {_esc(gpu_name)} ({_esc(gfx)})
 &mdash; {gpu_specs.get('cu_count', '?')} CUs @ {gpu_specs.get('max_clock_mhz', '?')} MHz</div>
<div class="cmd">{_esc(command)}</div>
</div>
""")

    # Timeline
    timeline_svg, tl_layout = _generate_timeline_svg(kernel_events, hip_events, top_kernels)
    gpu_y = tl_layout.get("gpu_y", 102)
    lane_h = tl_layout.get("lane_h", 40)
    ml = tl_layout.get("margin_left", 100)
    cw = tl_layout.get("chart_w", 1080)

    parts.append('<div class="card">')
    parts.append('<h2>Timeline</h2>')
    parts.append('<div class="timeline-container" id="tl-container">')
    parts.append(timeline_svg)
    # HTML overlay on GPU lane for brush-select (cursor + event capture)
    parts.append(f'<div id="gpu-overlay" style="position:absolute;'
                 f'left:{ml}px;top:{gpu_y}px;width:{cw}px;height:{lane_h}px;'
                 f'cursor:crosshair;background:transparent;z-index:5;"></div>')
    # Brush highlight div (shown during drag)
    parts.append(f'<div id="tl-brush" style="position:absolute;display:none;'
                 f'top:{gpu_y}px;height:{lane_h}px;'
                 f'background:rgba(59,130,246,0.15);border:1px dashed #3b82f6;'
                 f'pointer-events:none;z-index:6;"></div>')
    parts.append('<div id="tl-tooltip"></div>')
    parts.append('</div>')
    parts.append(f"""<div style="font-size:11px;color:#999;margin-top:4px;">
Drag on the GPU lane to select a region to zoom &bull; Scroll to pan when zoomed &bull; Double-click to reset</div>""")
    parts.append("""<script>
(function() {
  var svg = document.getElementById('timeline-svg');
  if (!svg) return;
  var gpuContent = document.getElementById('gpu-content');
  var container = document.getElementById('tl-container');
  var tooltip = document.getElementById('tl-tooltip');
  var zoomLabel = document.getElementById('tl-zoom-label');
  var gpuOverlay = document.getElementById('gpu-overlay');
  var brushDiv = document.getElementById('tl-brush');
  var ML = """ + str(ml) + """;
  var CW = """ + str(cw) + """;
  var scale = 1, tx = 0;
  var brushing = false, brushStartClientX = 0;

  function clampTx() {
    var maxTx = ML * (1 - scale);
    var minTx = (ML + CW) * (1 - scale);
    tx = Math.max(minTx, Math.min(maxTx, tx));
  }
  function applyTransform() {
    gpuContent.setAttribute('transform',
      'translate(' + tx + ',0) scale(' + scale + ',1)');
    if (zoomLabel) zoomLabel.textContent = scale.toFixed(1) + 'x';
  }
  function toSvgX(clientX) {
    var pt = svg.createSVGPoint();
    pt.x = clientX; pt.y = 0;
    return pt.matrixTransform(svg.getScreenCTM().inverse()).x;
  }
  function containerPx(clientX) {
    return clientX - container.getBoundingClientRect().left + container.scrollLeft;
  }

  // --- Brush-select on the GPU overlay (HTML div — reliable in all browsers) ---
  gpuOverlay.onmousedown = function(e) {
    if (e.button !== 0) return;
    brushing = true;
    brushStartClientX = e.clientX;
    var px = containerPx(e.clientX);
    brushDiv.style.left = px + 'px';
    brushDiv.style.width = '0px';
    brushDiv.style.display = 'block';
    tooltip.style.display = 'none';
    e.preventDefault();
  };

  document.onmousemove = function(e) {
    if (!brushing) return;
    e.preventDefault();
    var startPx = containerPx(brushStartClientX);
    var curPx = containerPx(e.clientX);
    brushDiv.style.left = Math.min(startPx, curPx) + 'px';
    brushDiv.style.width = Math.abs(curPx - startPx) + 'px';
  };

  document.onmouseup = function(e) {
    if (!brushing) return;
    brushing = false;
    brushDiv.style.display = 'none';

    var svgX1 = toSvgX(Math.min(brushStartClientX, e.clientX));
    var svgX2 = toSvgX(Math.max(brushStartClientX, e.clientX));
    svgX1 = Math.max(ML, Math.min(ML + CW, svgX1));
    svgX2 = Math.max(ML, Math.min(ML + CW, svgX2));
    if (svgX2 - svgX1 < 3) return;

    var cLeft = (svgX1 - tx) / scale;
    var cRight = (svgX2 - tx) / scale;
    scale = CW / (cRight - cLeft);
    scale = Math.max(1, Math.min(5000, scale));
    tx = ML - cLeft * scale;
    clampTx();
    applyTransform();
  };

  // Scroll wheel pans when zoomed
  gpuOverlay.onwheel = function(e) {
    if (scale <= 1) return;
    e.preventDefault();
    tx -= e.deltaY * 2;
    if (e.deltaX) tx -= e.deltaX * 2;
    clampTx();
    applyTransform();
  };

  // Double-click resets
  gpuOverlay.ondblclick = function() {
    scale = 1; tx = 0; applyTransform();
  };

  // --- GPU tooltip via overlay (peek through to SVG elements below) ---
  var lastHovered = null;
  gpuOverlay.addEventListener('mousemove', function(e) {
    if (brushing) return;
    // Temporarily hide overlay to find SVG element below
    gpuOverlay.style.pointerEvents = 'none';
    var el = document.elementFromPoint(e.clientX, e.clientY);
    gpuOverlay.style.pointerEvents = '';

    if (lastHovered && lastHovered !== el) {
      lastHovered.style.stroke = '';
      lastHovered.style.strokeWidth = '';
      lastHovered.style.opacity = '';
    }
    if (el && el.classList && el.classList.contains('gpu-ev')) {
      lastHovered = el;
      el.style.stroke = '#000';
      el.style.strokeWidth = '1.5';
      el.style.opacity = '1';
      var d = el.dataset, html = '';
      html += '<div class="tt-title">' + d.name + '</div>';
      html += '<div class="tt-row">Duration: <span class="tt-val">' + d.dur + '</span></div>';
      if (d.cnt && parseInt(d.cnt) > 1)
        html += '<div class="tt-row">Dispatch: <span class="tt-val">' + d.idx + ' / ' + d.cnt + '</span></div>';
      if (d.grid && d.grid !== '0x0x0')
        html += '<div class="tt-row">Grid: <span class="tt-val">' + d.grid + '</span></div>';
      if (d.wg && d.wg !== '0x0x0')
        html += '<div class="tt-row">Workgroup: <span class="tt-val">' + d.wg + '</span></div>';
      tooltip.innerHTML = html;
      tooltip.style.display = 'block';
      var cr = container.getBoundingClientRect();
      var x = e.clientX - cr.left + 12;
      var y = e.clientY - cr.top - 10;
      if (x + tooltip.offsetWidth > cr.width) x = e.clientX - cr.left - tooltip.offsetWidth - 8;
      if (y < 0) y = e.clientY - cr.top + 20;
      tooltip.style.left = x + 'px';
      tooltip.style.top = y + 'px';
    } else {
      lastHovered = null;
      tooltip.style.display = 'none';
    }
  });
  gpuOverlay.addEventListener('mouseleave', function() {
    if (lastHovered) {
      lastHovered.style.stroke = '';
      lastHovered.style.strokeWidth = '';
      lastHovered.style.opacity = '';
      lastHovered = null;
    }
    tooltip.style.display = 'none';
  });

  // --- Tooltip for CPU/Memory events (SVG elements not under overlay) ---
  svg.querySelectorAll('.tl-ev:not(.gpu-ev)').forEach(function(el) {
    el.addEventListener('mouseenter', function(e) {
      var d = el.dataset, html = '';
      html += '<div class="tt-title">' + d.name + '</div>';
      html += '<div class="tt-row">Duration: <span class="tt-val">' + d.dur + '</span></div>';
      if (d.type === 'mem' && d.dir)
        html += '<div class="tt-row">Direction: <span class="tt-val">' + d.dir + '</span></div>';
      if (d.type === 'hip' && d.cat)
        html += '<div class="tt-row">Type: <span class="tt-val">' + d.cat + '</span></div>';
      tooltip.innerHTML = html;
      tooltip.style.display = 'block';
    });
    el.addEventListener('mousemove', function(e) {
      var cr = container.getBoundingClientRect();
      var x = e.clientX - cr.left + 12;
      var y = e.clientY - cr.top - 10;
      if (x + tooltip.offsetWidth > cr.width) x = e.clientX - cr.left - tooltip.offsetWidth - 8;
      if (y < 0) y = e.clientY - cr.top + 20;
      tooltip.style.left = x + 'px';
      tooltip.style.top = y + 'px';
    });
    el.addEventListener('mouseleave', function() {
      tooltip.style.display = 'none';
    });
  });
})();
</script>""")
    parts.append('</div>')

    # Top Kernels Overview
    parts.append('<div class="section-row">')

    # Pie chart
    parts.append('<div class="section-col"><div class="card">')
    parts.append('<h2>Top Kernels by GPU Time</h2>')
    parts.append(_generate_pie_svg(top_kernels))
    parts.append('</div></div>')

    # Summary table
    parts.append('<div class="section-col"><div class="card">')
    parts.append('<h2>Kernel Summary</h2>')
    parts.append('<table><thead><tr><th>#</th><th>Kernel</th><th>Calls</th>'
                 '<th>Total</th><th>Avg</th><th>%</th></tr></thead><tbody>')
    for i, k in enumerate(top_kernels):
        color = KERNEL_COLORS[i % len(KERNEL_COLORS)]
        kname = _esc(_short_name(k["kernel_name"], 40))
        parts.append(f'<tr>'
                     f'<td><span style="color:{color};">&#9632;</span> {i+1}</td>'
                     f'<td class="kernel-name">{kname}</td>'
                     f'<td>{k["count"]}</td>'
                     f'<td>{_fmt_ns(k["total_ns"])}</td>'
                     f'<td>{_fmt_ns(k["avg_ns"])}</td>'
                     f'<td>{k["pct"]:.1f}%</td></tr>')
    parts.append('</tbody></table>')
    parts.append('</div></div>')
    parts.append('</div>')  # end section-row

    # Kernel Detail Cards
    for i, k in enumerate(top_kernels):
        kname = k["kernel_name"]
        color = KERNEL_COLORS[i % len(KERNEL_COLORS)]

        parts.append(f'<div class="card" style="border-left: 4px solid {color};">')
        parts.append(f'<h2 style="border-bottom-color:{color};">#{i+1}: {_esc(_short_name(kname, 60))}</h2>')
        parts.append(f'<div class="kernel-name" style="margin-bottom:10px;">{_esc(kname)}</div>')

        # Duration stats
        parts.append('<div class="stats-grid">')
        for label, val in [("Total", _fmt_ns(k["total_ns"])),
                           ("Avg", _fmt_ns(k["avg_ns"])),
                           ("Min", _fmt_ns(k["min_ns"])),
                           ("Max", _fmt_ns(k["max_ns"])),
                           ("Calls", str(k["count"])),
                           ("% GPU Time", f"{k['pct']:.1f}%")]:
            parts.append(f'<div class="stat"><div class="label">{label}</div>'
                         f'<div class="value">{val}</div></div>')
        parts.append('</div>')

        # Instruction mix
        insts = insts_by_kernel.get(kname, {})
        if insts:
            parts.append('<h3>Instruction Mix</h3>')
            parts.append(_generate_insts_bar_svg(insts))

            # Also show a small table
            insts_order = [
                ("SQ_INSTS_VALU", "VALU"), ("SQ_INSTS_SALU", "SALU"),
                ("SQ_INSTS_SMEM", "SMEM"), ("SQ_INSTS_VMEM_RD", "VMEM_RD"),
                ("SQ_INSTS_VMEM_WR", "VMEM_WR"), ("SQ_INSTS_LDS", "LDS"),
                ("SQ_INSTS_FLAT", "FLAT"),
                ("SQ_INSTS_MFMA", "MFMA"),
            ]
            total_insts = sum(insts.get(key, 0) for key, _ in insts_order)
            if total_insts > 0:
                parts.append('<table><thead><tr>')
                for _, label in insts_order:
                    parts.append(f'<th>{label}</th>')
                parts.append('</tr></thead><tbody><tr>')
                for key, _ in insts_order:
                    val = insts.get(key, 0)
                    parts.append(f'<td>{int(val):,}</td>')
                parts.append('</tr></tbody></table>')

        # Roofline utilization
        roof = roofline_by_kernel.get(kname)
        comp = (compute_by_kernel or {}).get(kname)

        if roof or comp:
            parts.append('<h3>Roofline Utilization</h3>')
            parts.append('<div class="section-row">')
            if roof:
                parts.append('<div class="section-col">')
                parts.append('<div style="font-size:12px;color:#666;margin-bottom:4px;">Bandwidth</div>')
                parts.append(_generate_roofline_bars_svg(roof, gpu_specs))
                parts.append('</div>')
            if comp:
                parts.append('<div class="section-col">')
                parts.append('<div style="font-size:12px;color:#666;margin-bottom:4px;">Compute (FLOPS / IOPS)</div>')
                parts.append(_generate_compute_bars_svg(comp, gpu_specs))
                parts.append('</div>')
            parts.append('</div>')

        parts.append('</div>')

    # Footer
    parts.append("""
<div style="text-align:center;color:#999;font-size:11px;margin-top:30px;padding:10px;">
Generated by rocm-profile-agent using rocprofv3
</div>
</div>
</body>
</html>""")

    return "\n".join(parts)
