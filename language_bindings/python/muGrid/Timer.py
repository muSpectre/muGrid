#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Timer.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   25 Dec 2024

@brief  Hierarchical timing utility with nested context manager support

Copyright © 2024 Lars Pastewka

µGrid is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µGrid is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µGrid; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.
"""

import json
import time
from contextlib import contextmanager

# Try to import pypapi for hardware counter access (optional)
try:
    import pypapi
    from pypapi import events as papi_events

    PAPI_AVAILABLE = True
except ImportError:
    PAPI_AVAILABLE = False


class Timer:
    """
    Hierarchical timing utility with nested context manager support.

    This class provides fine-grained timing of code sections with support for:
    - Nested timing contexts that track parent-child relationships
    - Accumulation of time across multiple calls to the same timer
    - Call counting for repeated operations
    - Hierarchical summary output in tabular format
    - Optional PAPI hardware counter integration for FLOP measurement

    Example usage:
        from muGrid import Timer

        timer = Timer()

        with timer("outer"):
            # some code
            with timer("inner"):
                # nested code
            with timer("inner"):  # called again - time accumulates
                # more nested code

        timer.print_summary()

    Output:
        ==============================================================================
        Timing Summary
        ==============================================================================
        Name                                  Total    Calls      Average   % Parent
        ------------------------------ ------------ -------- ------------ ----------
        outer                             10.00 ms        1            -          -
          inner                            5.00 ms        2     2.50 ms      50.0%
          (other)                          5.00 ms        -            -      50.0%
        ==============================================================================

    With PAPI enabled (use_papi=True), additional columns show GFLOP/s:
        ==============================================================================
        Timing Summary (with PAPI hardware counters)
        ==============================================================================
        Name                            Total    Calls    GFLOP/s        IPC
        ------------------------------ -------- -------- ---------- ----------
        outer                          10.00 ms        1       2.50       1.85
          inner                         5.00 ms        2       3.10       2.01
        ==============================================================================
    """

    # PAPI events we track (in order)
    PAPI_EVENTS = [
        "PAPI_TOT_CYC",  # Total cycles
        "PAPI_TOT_INS",  # Total instructions
        "PAPI_FP_OPS",   # Floating point operations
        "PAPI_L1_DCM",   # L1 data cache misses
        "PAPI_L2_DCM",   # L2 data cache misses
        "PAPI_L3_TCM",   # L3 total cache misses
    ]

    def __init__(self, use_papi=False):
        """
        Initialize the timer.

        Args:
            use_papi: If True, enable PAPI hardware counter measurement.
                      Requires pypapi to be installed. Only works on CPU.
        """
        self._timers = {}  # name -> {"total": float, "calls": int, "children": list, "papi": dict}
        self._stack = []   # stack of (name, start_time, papi_start) for nesting
        self._roots = []   # top-level timer names in order of first use

        # Initialize PAPI if requested
        self._use_papi = use_papi and PAPI_AVAILABLE
        self._papi_enabled = False

        if use_papi and not PAPI_AVAILABLE:
            import warnings
            warnings.warn(
                "PAPI requested but pypapi not available. "
                "Install with: pip install pypapi"
            )

        if self._use_papi:
            try:
                self._papi_events_list = [
                    getattr(papi_events, name) for name in self.PAPI_EVENTS
                ]
                # Start PAPI counters - they run continuously
                pypapi.papi_high.start_counters(self._papi_events_list)
                self._papi_enabled = True
            except Exception as e:
                import warnings
                warnings.warn(f"Could not initialize PAPI: {e}")
                self._use_papi = False

    def __del__(self):
        """Clean up PAPI counters on destruction."""
        if self._papi_enabled:
            try:
                pypapi.papi_high.stop_counters()
            except Exception:
                pass  # Ignore errors during cleanup

    def _read_papi_counters(self):
        """Read current PAPI counter values."""
        if not self._papi_enabled:
            return None
        try:
            # Read counters without stopping them
            counters = pypapi.papi_high.read_counters()
            return {
                "cycles": counters[0],
                "instructions": counters[1],
                "fp_ops": counters[2],
                "l1_dcm": counters[3],
                "l2_dcm": counters[4],
                "l3_tcm": counters[5],
            }
        except Exception:
            return None

    @contextmanager
    def __call__(self, name):
        """
        Context manager for timing a named code section.

        Args:
            name: Identifier for this timing section. Repeated calls with the
                  same name accumulate time and increment the call counter.

        Yields:
            None

        Example:
            with timer("my_operation"):
                # code to time
                pass
        """
        # Build hierarchical name based on current stack
        if self._stack:
            parent_name = self._stack[-1][0]
            full_name = f"{parent_name}/{name}"
        else:
            full_name = name

        # Initialize timer if first call
        if full_name not in self._timers:
            self._timers[full_name] = {
                "total": 0.0,
                "calls": 0,
                "children": [],
                "papi": {
                    "cycles": 0,
                    "instructions": 0,
                    "fp_ops": 0,
                    "l1_dcm": 0,
                    "l2_dcm": 0,
                    "l3_tcm": 0,
                } if self._papi_enabled else None
            }
            # Track as root or as child of parent
            if self._stack:
                parent_full = self._stack[-1][0]
                if full_name not in self._timers[parent_full]["children"]:
                    self._timers[parent_full]["children"].append(full_name)
            else:
                if full_name not in self._roots:
                    self._roots.append(full_name)

        # Read PAPI counters at start
        papi_start = self._read_papi_counters()

        # Push onto stack and start timing
        start = time.perf_counter()
        self._stack.append((full_name, start, papi_start))

        try:
            yield
        finally:
            # Pop from stack and accumulate time
            elapsed = time.perf_counter() - start
            _, _, papi_start = self._stack.pop()
            self._timers[full_name]["total"] += elapsed
            self._timers[full_name]["calls"] += 1

            # Accumulate PAPI counters
            if papi_start is not None:
                papi_end = self._read_papi_counters()
                if papi_end is not None:
                    papi_data = self._timers[full_name]["papi"]
                    for key in papi_data:
                        papi_data[key] += papi_end[key] - papi_start[key]

    def get_time(self, name):
        """
        Get total accumulated time for a timer.

        Args:
            name: Timer name (can be short name or full hierarchical name)

        Returns:
            Total time in seconds, or 0.0 if timer not found
        """
        # Try exact match first, then search for it in hierarchy
        if name in self._timers:
            return self._timers[name]["total"]
        # Search for timer ending with this name
        for full_name in self._timers:
            if full_name.endswith("/" + name) or full_name == name:
                return self._timers[full_name]["total"]
        return 0.0

    def get_calls(self, name):
        """
        Get call count for a timer.

        Args:
            name: Timer name (can be short name or full hierarchical name)

        Returns:
            Number of calls, or 0 if timer not found
        """
        if name in self._timers:
            return self._timers[name]["calls"]
        for full_name in self._timers:
            if full_name.endswith("/" + name) or full_name == name:
                return self._timers[full_name]["calls"]
        return 0

    def get_papi(self, name):
        """
        Get PAPI counter data for a timer.

        Args:
            name: Timer name (can be short name or full hierarchical name)

        Returns:
            Dictionary with PAPI counter values, or None if not available
        """
        timer_info = None
        if name in self._timers:
            timer_info = self._timers[name]
        else:
            for full_name in self._timers:
                if full_name.endswith("/" + name) or full_name == name:
                    timer_info = self._timers[full_name]
                    break

        if timer_info is None or timer_info.get("papi") is None:
            return None

        papi = timer_info["papi"]
        total_time = timer_info["total"]

        # Calculate derived metrics
        result = dict(papi)  # Copy the raw counters
        if total_time > 0:
            result["gflops"] = (papi["fp_ops"] / 1e9) / total_time
        else:
            result["gflops"] = 0.0

        if papi["cycles"] > 0:
            result["ipc"] = papi["instructions"] / papi["cycles"]
        else:
            result["ipc"] = 0.0

        return result

    def reset(self):
        """Clear all timing data (PAPI counters continue running if enabled)."""
        self._timers.clear()
        self._stack.clear()
        self._roots.clear()

    def _format_time(self, seconds):
        """Format time with appropriate units, fixed width."""
        if seconds < 1e-6:
            return f"{seconds * 1e9:8.2f} ns"
        elif seconds < 1e-3:
            return f"{seconds * 1e6:8.2f} us"
        elif seconds < 1:
            return f"{seconds * 1e3:8.2f} ms"
        else:
            return f"{seconds:8.4f} s "

    def _collect_rows(self, name, indent=0, parent_time=None, rows=None):
        """Recursively collect timing data as rows for tabular display."""
        if rows is None:
            rows = []

        info = self._timers[name]
        total = info["total"]
        calls = info["calls"]
        avg = total / calls if calls > 0 else 0

        # Calculate percentage of parent time
        if parent_time and parent_time > 0:
            pct = 100.0 * total / parent_time
        else:
            pct = None

        # Calculate PAPI-derived metrics
        papi = info.get("papi")
        gflops = None
        ipc = None
        if papi is not None and total > 0:
            gflops = (papi["fp_ops"] / 1e9) / total
            if papi["cycles"] > 0:
                ipc = papi["instructions"] / papi["cycles"]

        # Extract short name (last component)
        short_name = name.split("/")[-1]

        # Add row for this timer
        rows.append({
            "indent": indent,
            "name": short_name,
            "total": total,
            "calls": calls,
            "avg": avg if calls > 1 else None,
            "pct": pct,
            "gflops": gflops,
            "ipc": ipc,
            "papi": papi,
        })

        # Collect children
        for child in info["children"]:
            self._collect_rows(child, indent + 1, total, rows)

        # Add "other" time if there are children
        if info["children"]:
            children_time = sum(self._timers[c]["total"] for c in info["children"])
            other_time = total - children_time
            if other_time > 1e-9:  # Only show if meaningful
                other_pct = 100.0 * other_time / total if total > 0 else 0
                # Calculate "other" PAPI by subtracting children from parent
                other_papi = None
                other_gflops = None
                other_ipc = None
                if papi is not None:
                    other_papi = {}
                    for key in papi:
                        children_val = sum(
                            self._timers[c].get("papi", {}).get(key, 0) or 0
                            for c in info["children"]
                        )
                        other_papi[key] = papi[key] - children_val
                    if other_time > 0:
                        other_gflops = (other_papi["fp_ops"] / 1e9) / other_time
                        if other_papi["cycles"] > 0:
                            other_ipc = other_papi["instructions"] / other_papi["cycles"]
                rows.append({
                    "indent": indent + 1,
                    "name": "(other)",
                    "total": other_time,
                    "calls": None,
                    "avg": None,
                    "pct": other_pct,
                    "gflops": other_gflops,
                    "ipc": other_ipc,
                    "papi": other_papi,
                })

        return rows

    def print_summary(self, title=None, name_width=30):
        """
        Print hierarchical timing summary in tabular format.

        Args:
            title: Title for the summary section (auto-detected if None)
            name_width: Width of the name column (default: 30)
        """
        if not self._timers:
            print("No timing data collected.")
            return

        # Collect all rows
        all_rows = []
        for root in self._roots:
            self._collect_rows(root, rows=all_rows)

        # Check if PAPI data is available
        has_papi = self._papi_enabled and any(row.get("papi") for row in all_rows)

        # Determine title
        if title is None:
            title = "Timing Summary (with PAPI)" if has_papi else "Timing Summary"

        # Print header - different format depending on PAPI availability
        if has_papi:
            line_width = 94
            print(f"\n{'=' * line_width}")
            print(title)
            print(f"{'=' * line_width}")
            print(f"{'Name':<{name_width}} {'Total':>12} {'Calls':>8} "
                  f"{'% Parent':>10} {'GFLOP/s':>10} {'IPC':>8} {'FP ops':>12}")
            print(f"{'-' * name_width} {'-' * 12} {'-' * 8} "
                  f"{'-' * 10} {'-' * 10} {'-' * 8} {'-' * 12}")
        else:
            line_width = 78
            print(f"\n{'=' * line_width}")
            print(title)
            print(f"{'=' * line_width}")
            print(f"{'Name':<{name_width}} {'Total':>12} {'Calls':>8} "
                  f"{'Average':>12} {'% Parent':>10}")
            print(f"{'-' * name_width} {'-' * 12} {'-' * 8} {'-' * 12} {'-' * 10}")

        # Print rows
        for row in all_rows:
            # Build indented name
            prefix = "  " * row["indent"]
            name = f"{prefix}{row['name']}"
            if len(name) > name_width:
                name = name[:name_width - 3] + "..."

            # Format columns
            total_str = self._format_time(row["total"])

            if row["calls"] is not None:
                calls_str = f"{row['calls']:>8}"
            else:
                calls_str = f"{'-':>8}"

            if row["pct"] is not None:
                pct_str = f"{row['pct']:>9.1f}%"
            else:
                pct_str = f"{'-':>10}"

            if has_papi:
                # PAPI-enabled output
                if row.get("gflops") is not None:
                    gflops_str = f"{row['gflops']:>10.2f}"
                else:
                    gflops_str = f"{'-':>10}"

                if row.get("ipc") is not None:
                    ipc_str = f"{row['ipc']:>8.2f}"
                else:
                    ipc_str = f"{'-':>8}"

                papi = row.get("papi")
                if papi is not None and papi.get("fp_ops", 0) > 0:
                    fp_ops = papi["fp_ops"]
                    if fp_ops >= 1e9:
                        fp_ops_str = f"{fp_ops / 1e9:>10.2f} G"
                    elif fp_ops >= 1e6:
                        fp_ops_str = f"{fp_ops / 1e6:>10.2f} M"
                    elif fp_ops >= 1e3:
                        fp_ops_str = f"{fp_ops / 1e3:>10.2f} K"
                    else:
                        fp_ops_str = f"{fp_ops:>12}"
                else:
                    fp_ops_str = f"{'-':>12}"

                print(f"{name:<{name_width}} {total_str} {calls_str} "
                      f"{pct_str} {gflops_str} {ipc_str} {fp_ops_str}")
            else:
                # Standard output (no PAPI)
                if row["avg"] is not None:
                    avg_str = self._format_time(row["avg"])
                else:
                    avg_str = f"{'-':>12}"

                print(f"{name:<{name_width}} {total_str} {calls_str} {avg_str} {pct_str}")

        print(f"{'=' * line_width}")

    def _build_tree(self, name):
        """Recursively build a tree structure for a timer and its children."""
        info = self._timers[name]
        total = info["total"]
        calls = info["calls"]

        result = {
            "name": name.split("/")[-1],
            "total_seconds": total,
            "calls": calls,
            "avg_seconds": total / calls if calls > 0 else 0,
        }

        # Add PAPI data if available
        papi = info.get("papi")
        if papi is not None:
            result["papi"] = dict(papi)
            # Add derived metrics
            if total > 0:
                result["gflops"] = (papi["fp_ops"] / 1e9) / total
            else:
                result["gflops"] = 0.0
            if papi["cycles"] > 0:
                result["ipc"] = papi["instructions"] / papi["cycles"]
            else:
                result["ipc"] = 0.0

        if info["children"]:
            result["children"] = [
                self._build_tree(child) for child in info["children"]
            ]
            children_time = sum(self._timers[c]["total"] for c in info["children"])
            other_time = total - children_time
            if other_time > 1e-9:
                result["other_seconds"] = other_time

        return result

    def to_dict(self):
        """
        Return timing data as a hierarchical dictionary.

        Returns:
            Dictionary with hierarchical timer structure suitable for JSON export.
        """
        result = {
            "timers": [self._build_tree(root) for root in self._roots]
        }
        if self._papi_enabled:
            result["papi_enabled"] = True
        return result

    def to_json(self, indent=2):
        """
        Return timing data as a JSON string.

        Args:
            indent: Indentation level for pretty printing (default: 2)

        Returns:
            JSON string representation of timing data.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def summary_dict(self):
        """
        Return timing data as a flat dictionary for programmatic access.

        Returns:
            Dictionary with timer names as keys and dicts containing
            'total', 'calls', 'children', and optionally 'papi' as values.
        """
        result = {}
        for name, info in self._timers.items():
            entry = {
                "total": info["total"],
                "calls": info["calls"],
                "avg": info["total"] / info["calls"] if info["calls"] > 0 else 0,
                "children": list(info["children"])
            }
            papi = info.get("papi")
            if papi is not None:
                entry["papi"] = dict(papi)
                total = info["total"]
                if total > 0:
                    entry["gflops"] = (papi["fp_ops"] / 1e9) / total
                else:
                    entry["gflops"] = 0.0
                if papi["cycles"] > 0:
                    entry["ipc"] = papi["instructions"] / papi["cycles"]
                else:
                    entry["ipc"] = 0.0
            result[name] = entry
        return result
