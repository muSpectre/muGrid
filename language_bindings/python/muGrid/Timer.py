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


class Timer:
    """
    Hierarchical timing utility with nested context manager support.

    This class provides fine-grained timing of code sections with support for:
    - Nested timing contexts that track parent-child relationships
    - Accumulation of time across multiple calls to the same timer
    - Call counting for repeated operations
    - Hierarchical summary output in tabular format

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
    """

    def __init__(self, use_papi=False):
        """
        Initialize the timer.

        Args:
            use_papi: Ignored (PAPI support has been removed).
        """
        self._timers = {}  # name -> {"total": float, "calls": int, "children": list}
        self._stack = []  # stack of (name, start_time) for nesting
        self._roots = []  # top-level timer names in order of first use

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
            }
            # Track as root or as child of parent
            if self._stack:
                parent_full = self._stack[-1][0]
                if full_name not in self._timers[parent_full]["children"]:
                    self._timers[parent_full]["children"].append(full_name)
            else:
                if full_name not in self._roots:
                    self._roots.append(full_name)

        # Push onto stack and start timing
        start = time.perf_counter()
        self._stack.append((full_name, start))

        try:
            yield
        finally:
            # Pop from stack and accumulate time
            elapsed = time.perf_counter() - start
            self._stack.pop()
            self._timers[full_name]["total"] += elapsed
            self._timers[full_name]["calls"] += 1

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

    def reset(self):
        """Clear all timing data."""
        self._timers.clear()
        self._stack.clear()
        self._roots.clear()

    def _format_time(self, seconds):
        """Format time with appropriate units, fixed width (12 chars)."""
        if seconds < 1e-6:
            return f"{seconds * 1e9:9.2f} ns"
        elif seconds < 1e-3:
            return f"{seconds * 1e6:9.2f} us"
        elif seconds < 1:
            return f"{seconds * 1e3:9.2f} ms"
        else:
            return f"{seconds:9.4f} s "

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

        # Extract short name (last component)
        short_name = name.split("/")[-1]

        # Add row for this timer
        rows.append(
            {
                "indent": indent,
                "name": short_name,
                "total": total,
                "calls": calls,
                "avg": avg if calls > 1 else None,
                "pct": pct,
            }
        )

        # Collect children
        for child in info["children"]:
            self._collect_rows(child, indent + 1, total, rows)

        # Add "other" time if there are children
        if info["children"]:
            children_time = sum(self._timers[c]["total"] for c in info["children"])
            other_time = total - children_time
            if other_time > 1e-9:  # Only show if meaningful
                other_pct = 100.0 * other_time / total if total > 0 else 0
                rows.append(
                    {
                        "indent": indent + 1,
                        "name": "(other)",
                        "total": other_time,
                        "calls": None,
                        "avg": None,
                        "pct": other_pct,
                    }
                )

        return rows

    def print_summary(self, title=None, name_width=30):
        """
        Print hierarchical timing summary in tabular format.

        Args:
            title: Title for the summary section (default: "Timing Summary")
            name_width: Width of the name column (default: 30)
        """
        if not self._timers:
            print("No timing data collected.")
            return

        # Collect all rows
        all_rows = []
        for root in self._roots:
            self._collect_rows(root, rows=all_rows)

        # Determine title
        if title is None:
            title = "Timing Summary"

        line_width = 78
        print(f"\n{'=' * line_width}")
        print(title)
        print(f"{'=' * line_width}")
        print(
            f"{'Name':<{name_width}} {'Total':>12} {'Calls':>8} "
            f"{'Average':>12} {'% Parent':>10}"
        )
        print(f"{'-' * name_width} {'-' * 12} {'-' * 8} {'-' * 12} {'-' * 10}")

        # Print rows
        for row in all_rows:
            # Build indented name
            prefix = "  " * row["indent"]
            name = f"{prefix}{row['name']}"
            if len(name) > name_width:
                name = name[: name_width - 3] + "..."

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

            if row["avg"] is not None:
                avg_str = self._format_time(row["avg"])
            else:
                avg_str = f"{'-':>12}"

            print(
                f"{name:<{name_width}} {total_str} {calls_str} {avg_str} {pct_str}"
            )

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
        return {"timers": [self._build_tree(root) for root in self._roots]}

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
            'total', 'calls', and 'children' as values.
        """
        result = {}
        for name, info in self._timers.items():
            entry = {
                "total": info["total"],
                "calls": info["calls"],
                "avg": info["total"] / info["calls"] if info["calls"] > 0 else 0,
                "children": list(info["children"]),
            }
            result[name] = entry
        return result
