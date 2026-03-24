"""Parses rocprofv3 CSV output files into structured data."""

import csv
from collections import defaultdict


def _read_csv(path):
    """Read a CSV file, stripping quotes from field names."""
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        # Strip quotes from fieldnames
        reader.fieldnames = [fn.strip('"') for fn in reader.fieldnames]
        rows = []
        for row in reader:
            cleaned = {k.strip('"'): v.strip('"') if isinstance(v, str) else v
                       for k, v in row.items()}
            rows.append(cleaned)
    return rows


def parse_kernel_trace(kernel_trace_path):
    """Parse kernel dispatch trace CSV.

    Returns list of dicts with keys:
        kernel_name, start_ns, end_ns, duration_ns, agent_id,
        grid_size_x/y/z, workgroup_size_x/y/z
    """
    rows = _read_csv(kernel_trace_path)
    events = []
    for row in rows:
        start = int(row["Start_Timestamp"])
        end = int(row["End_Timestamp"])
        events.append({
            "kernel_name": row["Kernel_Name"],
            "start_ns": start,
            "end_ns": end,
            "duration_ns": end - start,
            "agent_id": row.get("Agent_Id", ""),
            "queue_id": row.get("Queue_Id", ""),
            "grid_size_x": int(row.get("Grid_Size_X", 0)),
            "grid_size_y": int(row.get("Grid_Size_Y", 0)),
            "grid_size_z": int(row.get("Grid_Size_Z", 0)),
            "workgroup_size_x": int(row.get("Workgroup_Size_X", 0)),
            "workgroup_size_y": int(row.get("Workgroup_Size_Y", 0)),
            "workgroup_size_z": int(row.get("Workgroup_Size_Z", 0)),
        })
    return events


def parse_hip_trace(hip_trace_path):
    """Parse HIP API trace CSV.

    Returns list of dicts with keys:
        domain, function, start_ns, end_ns, duration_ns, thread_id
    """
    rows = _read_csv(hip_trace_path)
    events = []
    for row in rows:
        start = int(row["Start_Timestamp"])
        end = int(row["End_Timestamp"])
        events.append({
            "domain": row.get("Domain", ""),
            "function": row.get("Function", ""),
            "start_ns": start,
            "end_ns": end,
            "duration_ns": end - start,
            "thread_id": row.get("Thread_Id", ""),
        })
    return events


def rank_kernels(kernel_events, top_n=5):
    """Aggregate kernel dispatches by name and return top N by cumulative time.

    Returns list of dicts with keys:
        kernel_name, total_ns, count, avg_ns, min_ns, max_ns, pct
    """
    by_name = defaultdict(list)
    for ev in kernel_events:
        by_name[ev["kernel_name"]].append(ev["duration_ns"])

    total_gpu_ns = sum(d for durations in by_name.values() for d in durations)
    if total_gpu_ns == 0:
        return []

    ranked = []
    for name, durations in by_name.items():
        total = sum(durations)
        ranked.append({
            "kernel_name": name,
            "total_ns": total,
            "count": len(durations),
            "avg_ns": total // len(durations),
            "min_ns": min(durations),
            "max_ns": max(durations),
            "pct": total / total_gpu_ns * 100,
        })

    ranked.sort(key=lambda x: x["total_ns"], reverse=True)
    return ranked[:top_n]


def parse_counter_csv(counter_csv_path):
    """Parse rocprofv3 counter collection CSV (long format).

    rocprofv3 outputs one row per (dispatch, counter) pair:
        Correlation_Id, Dispatch_Id, Kernel_Name, Counter_Name, Counter_Value,
        Start_Timestamp, End_Timestamp, ...

    Returns list of dicts, each representing one dispatch with all its counters:
        {kernel_name, start_ns, end_ns, duration_ns, counters: {name: value}}
    """
    rows = _read_csv(counter_csv_path)

    # Group by (Correlation_Id, Dispatch_Id) to reassemble per-dispatch records
    dispatches = defaultdict(lambda: {"counters": {}})
    for row in rows:
        key = (row["Correlation_Id"], row.get("Dispatch_Id", ""))
        d = dispatches[key]
        d["kernel_name"] = row["Kernel_Name"]
        d["start_ns"] = int(row["Start_Timestamp"])
        d["end_ns"] = int(row["End_Timestamp"])
        d["duration_ns"] = d["end_ns"] - d["start_ns"]
        counter_name = row["Counter_Name"]
        counter_value = float(row["Counter_Value"])
        d["counters"][counter_name] = counter_value

    return list(dispatches.values())


def aggregate_counters_by_kernel(dispatch_records, kernel_names):
    """Aggregate counter values across dispatches for each named kernel.

    Returns dict: kernel_name -> {counter_name: total_value, _duration_ns: total, _count: n}
    """
    result = {}
    for kname in kernel_names:
        agg = defaultdict(float)
        count = 0
        total_dur = 0
        for d in dispatch_records:
            if d["kernel_name"] == kname:
                count += 1
                total_dur += d["duration_ns"]
                for cname, cval in d["counters"].items():
                    agg[cname] += cval
        result[kname] = dict(agg)
        result[kname]["_duration_ns"] = total_dur
        result[kname]["_count"] = count
    return result
