"""
Public interface to muGrid's Field allocation profiler.

The profiler records every muGrid ``Field`` buffer allocation per memory pool
("cpu", "cuda:0", ...), tracking current/peak/time-averaged bytes and, for
device pools, the physical total/available memory of the device. Use it as a
context manager around the code of interest::

    import muGrid

    with muGrid.memory_profile(include_cupy_pool=True) as prof:
        fc = muGrid.GlobalFieldCollection((512, 512), device=muGrid.Device.gpu())
        ...  # run the workload

    print(prof.summary())   # human-readable per-pool table
    report = prof.report()  # nested dict for programmatic use

or via the ``muGrid.memory_profiler`` namespace for non-scoped use::

    muGrid.memory_profiler.enable()
    ...
    print(muGrid.memory_profiler.summary())
    muGrid.memory_profiler.disable()

Two caveats worth knowing:

* Recording is only active when muGrid was built with
  ``-DMUGRID_PROFILE_ALLOCATIONS=ON``. The calls below are always safe — they
  are no-ops (returning empty data) in a non-instrumented build.
* Only muGrid ``Field`` buffers are tracked. Array libraries such as cupy draw
  from their own caching pool, which on a GPU workload is frequently the larger
  consumer. Pass ``include_cupy_pool=True`` to fold the cupy pool's
  used/reserved bytes and the device's physical free/total into the report, so
  a single call shows the full device picture rather than only the muGrid part.
"""

from . import _muGrid

__all__ = [
    "enable",
    "disable",
    "reset",
    "is_enabled",
    "report",
    "summary",
    "memory_profile",
]


def enable():
    """Start recording muGrid Field allocations."""
    _muGrid.enable_allocation_profiling()


def disable():
    """Stop recording. Accumulated statistics are retained until :func:`reset`."""
    _muGrid.disable_allocation_profiling()


def reset():
    """Discard accumulated statistics and restart the measurement window."""
    _muGrid.reset_allocation_profiling()


def is_enabled():
    """Return ``True`` if Field allocation recording is currently active."""
    return _muGrid.allocation_profiling_enabled()


def _augment_with_cupy(report_dict):
    """Add the cupy memory pool's used/reserved bytes and the device's physical
    free/total to ``report_dict`` under the ``"cupy_pool"`` key, if cupy and a
    GPU are available. Best-effort: silently does nothing otherwise."""
    try:
        import cupy
    except ImportError:
        return
    try:
        pool = cupy.get_default_memory_pool()
        device = cupy.cuda.runtime.getDevice()
        free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
    except Exception:  # no GPU, no pool, driver error -- leave report untouched
        return
    report_dict.setdefault("cupy_pool", {})[f"cuda:{device}"] = {
        "used_bytes": pool.used_bytes(),
        "reserved_bytes": pool.total_bytes(),
        "device_free_bytes": free_bytes,
        "device_total_bytes": total_bytes,
    }


def report(include_cupy_pool=False):
    """Return the recorded Field memory usage as a nested dict.

    The structure is ``{"elapsed_seconds": float, "pools": {name: {...}}}``,
    where each pool carries ``current_bytes``, ``peak_bytes``,
    ``average_bytes`` (time-weighted), the physical ``total_bytes`` /
    ``available_bytes`` of the device, and a ``buffers`` list of per-Field
    ``{label, space, current_bytes, peak_bytes}`` records.

    With ``include_cupy_pool=True``, a top-level ``"cupy_pool"`` entry reports
    the cupy pool's live and reserved bytes and the device's physical
    free/total (see module docstring).
    """
    rep = _muGrid.allocation_profile()
    if include_cupy_pool:
        _augment_with_cupy(rep)
    return rep


def summary():
    """Return a human-readable multi-line per-pool summary string."""
    return _muGrid.format_allocation_profile()


class memory_profile:
    """Context manager that records muGrid Field allocations within a block.

    On entry it (optionally) resets the profiler and enables recording; on exit
    it captures a final report and disables recording, so :meth:`report` and
    :meth:`summary` are valid both inside and after the ``with`` block.

    Parameters
    ----------
    reset : bool, optional
        Reset accumulated statistics on entry (default: ``True``).
    include_cupy_pool : bool, optional
        Fold cupy pool and physical device memory into :meth:`report`
        (default: ``False``). See the module docstring.
    """

    def __init__(self, reset=True, include_cupy_pool=False):
        self._reset = reset
        self._include_cupy_pool = include_cupy_pool
        self._final = None

    def __enter__(self):
        if self._reset:
            reset()
        enable()
        return self

    def __exit__(self, *exc):
        # Capture at the high-water point of the block, before any later code
        # (or the cupy pool) changes the picture.
        self._final = report(include_cupy_pool=self._include_cupy_pool)
        disable()
        return False  # do not suppress exceptions

    def report(self, include_cupy_pool=None):
        """The captured report after the block, or a live one inside it."""
        if self._final is not None:
            return self._final
        inc = (self._include_cupy_pool if include_cupy_pool is None
               else include_cupy_pool)
        return report(include_cupy_pool=inc)

    def summary(self):
        """Human-readable per-pool summary string."""
        return summary()
