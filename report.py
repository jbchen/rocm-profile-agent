"""Generate a self-contained HTML report with embedded SVG charts."""

import html
import datetime

# Color palette for kernels
COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]


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


def _kernel_color(name, top_kernels):
    """Get color for a kernel based on its position in the top list."""
    for i, k in enumerate(top_kernels):
        if k["kernel_name"] == name:
            return COLORS[i % len(COLORS)]
    return "#cccccc"


def _generate_timeline_svg(kernel_events, hip_events, top_kernels, width=1200):
    """Generate an SVG swimlane timeline chart."""
    if not kernel_events and not hip_events:
        return "<p>No timeline data available.</p>"

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

    margin_left = 80
    margin_right = 20
    chart_w = width - margin_left - margin_right
    lane_h = 30
    lane_gap = 10
    header_h = 30

    # Filter HIP events to only runtime API (skip compiler API noise)
    hip_runtime = [e for e in hip_events
                   if e.get("domain", "") == "HIP_RUNTIME_API"]

    num_lanes = 2  # CPU (HIP) + GPU (kernel)
    svg_h = header_h + num_lanes * (lane_h + lane_gap) + 40

    def x_pos(ns):
        return margin_left + (ns - t_min) / t_range * chart_w

    lines = []
    lines.append(f'<svg id="timeline-svg" width="{width}" height="{svg_h}" '
                 f'xmlns="http://www.w3.org/2000/svg" style="background:#fff;font-family:sans-serif;">')

    # Clip path for chart area
    lines.append(f'<defs><clipPath id="chart-clip">'
                 f'<rect x="{margin_left}" y="0" width="{chart_w}" height="{svg_h}"/>'
                 f'</clipPath></defs>')
    lines.append(f'<g id="timeline-content" clip-path="url(#chart-clip)">')

    # Lane labels
    cpu_y = header_h
    gpu_y = header_h + lane_h + lane_gap
    lines.append(f'<text x="5" y="{cpu_y + lane_h//2 + 4}" font-size="11" fill="#333">CPU (HIP)</text>')
    lines.append(f'<text x="5" y="{gpu_y + lane_h//2 + 4}" font-size="11" fill="#333">GPU</text>')

    # Lane backgrounds
    lines.append(f'<rect x="{margin_left}" y="{cpu_y}" width="{chart_w}" height="{lane_h}" fill="#f5f5f5" stroke="#ddd"/>')
    lines.append(f'<rect x="{margin_left}" y="{gpu_y}" width="{chart_w}" height="{lane_h}" fill="#f0f0f0" stroke="#ddd"/>')

    # CPU events (HIP runtime API)
    for ev in hip_runtime:
        x1 = x_pos(ev["start_ns"])
        x2 = x_pos(ev["end_ns"])
        w = max(x2 - x1, 0.5)
        func = _esc(ev["function"])
        dur = _fmt_ns(ev["duration_ns"])
        lines.append(f'<rect x="{x1:.1f}" y="{cpu_y + 2}" width="{w:.1f}" height="{lane_h - 4}" '
                     f'fill="#7bafd4" opacity="0.7">'
                     f'<title>{func}\n{dur}</title></rect>')

    # GPU events (kernel dispatches)
    for ev in kernel_events:
        x1 = x_pos(ev["start_ns"])
        x2 = x_pos(ev["end_ns"])
        w = max(x2 - x1, 0.5)
        color = _kernel_color(ev["kernel_name"], top_kernels)
        kname = _esc(ev["kernel_name"])
        dur = _fmt_ns(ev["duration_ns"])
        lines.append(f'<rect x="{x1:.1f}" y="{gpu_y + 2}" width="{w:.1f}" height="{lane_h - 4}" '
                     f'fill="{color}" opacity="0.85">'
                     f'<title>{kname}\n{dur}</title></rect>')

    lines.append('</g>')

    # Time axis
    axis_y = gpu_y + lane_h + 15
    n_ticks = 8
    for i in range(n_ticks + 1):
        t = t_min + t_range * i / n_ticks
        x = x_pos(t)
        label = _fmt_ns(int(t - t_min))
        lines.append(f'<line x1="{x:.1f}" y1="{gpu_y + lane_h}" x2="{x:.1f}" y2="{gpu_y + lane_h + 5}" stroke="#999"/>')
        lines.append(f'<text x="{x:.1f}" y="{axis_y}" font-size="9" fill="#666" text-anchor="middle">{label}</text>')

    lines.append('</svg>')
    return "\n".join(lines)


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
        color = COLORS[i % len(COLORS)]

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
        color = COLORS[i % len(COLORS)]
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
    insts_order = [
        ("SQ_INSTS_VALU", "VALU", "#4e79a7"),
        ("SQ_INSTS_SALU", "SALU", "#f28e2b"),
        ("SQ_INSTS_SMEM", "SMEM", "#e15759"),
        ("SQ_INSTS_VMEM_RD", "VMEM_RD", "#76b7b2"),
        ("SQ_INSTS_VMEM_WR", "VMEM_WR", "#59a14f"),
        ("SQ_INSTS_LDS", "LDS", "#edc948"),
        ("SQ_INSTS_FLAT", "FLAT", "#b07aa1"),
        ("SQ_INSTS_MFMA", "MFMA", "#9c755f"),
    ]

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
        ("HBM", roofline["hbm_pct"], roofline["hbm_achieved_gbs"], gpu_specs["hbm_bw_gbs"], "#4e79a7"),
        ("L2", roofline["l2_pct"], roofline["l2_achieved_gbs"], gpu_specs["l2_bw_gbs"], "#f28e2b"),
        ("L1", roofline["l1_pct"], roofline["l1_achieved_gbs"], gpu_specs["l1_bw_gbs"], "#e15759"),
        ("LDS", roofline["lds_pct"], roofline["lds_achieved_gbs"], gpu_specs["lds_bw_gbs"], "#76b7b2"),
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


def generate_report(command, gpu_specs, kernel_events, hip_events,
                    top_kernels, insts_by_kernel, roofline_by_kernel,
                    timestamp=None):
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
.timeline-container {{ overflow-x: auto; }}
.section-row {{ display: flex; gap: 20px; flex-wrap: wrap; }}
.section-col {{ flex: 1; min-width: 300px; }}
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
    parts.append('<div class="card">')
    parts.append('<h2>Timeline</h2>')
    parts.append('<div class="timeline-container">')
    parts.append(_generate_timeline_svg(kernel_events, hip_events, top_kernels))
    parts.append('</div></div>')

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
        color = COLORS[i % len(COLORS)]
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
        color = COLORS[i % len(COLORS)]

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
        if roof:
            parts.append('<h3>Bandwidth Utilization</h3>')
            parts.append(_generate_roofline_bars_svg(roof, gpu_specs))

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
