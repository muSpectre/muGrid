"""
Device memory allocator integration.

muGrid's C++ core allocates GPU memory with raw cudaMalloc/hipMalloc,
while Python-side array libraries like cupy draw from a caching memory
pool that never returns freed blocks to the driver. Two independent
allocators on one device starve each other: near capacity, muGrid's raw
allocation fails even though plenty of memory sits "free" inside the
pool. Registering cupy's allocator with muGrid makes one allocator own
every device byte and removes this failure mode by construction.
"""

from . import _muGrid


def use_cupy_allocator():
    """
    Route all muGrid device allocations through cupy's memory pool.

    Call this once, before creating GPU field collections, in programs
    that use both muGrid GPU fields and cupy arrays::

        import muGrid
        muGrid.use_cupy_allocator()
        fc = muGrid.GlobalFieldCollection(nb_grid_pts, device=muGrid.Device.gpu())

    Works with both CUDA and ROCm builds of cupy. Call
    :func:`muGrid.clear_device_allocator` to restore the default
    allocator (allocations made in the meantime are still freed through
    the pool).
    """
    import cupy

    # Keep the MemoryPointer objects alive until muGrid frees them; the
    # integer pointer value is the lookup key.
    keepalive = {}

    def allocate(nbytes):
        mem = cupy.cuda.alloc(nbytes)
        keepalive[int(mem.ptr)] = mem
        return int(mem.ptr)

    def deallocate(ptr):
        keepalive.pop(int(ptr), None)

    _muGrid.set_device_allocator(allocate, deallocate)


# Keep the active routing objects alive for cupy's lifetime (the closures and
# the pool are referenced by cupy's global allocator, but we hold them too so
# they are never collected out from under it).
_cupy_routing = None


def route_cupy_through_mugrid(pooled=True, label="<cupy>"):
    """
    Route cupy's device allocations through muGrid's allocator.

    This is the inverse of :func:`use_cupy_allocator`: instead of muGrid
    drawing from cupy's pool, cupy draws from muGrid's allocator, making
    **muGrid the single owner** of every device byte and recording cupy's
    allocations in the allocation profiler under ``label`` (so they appear
    alongside Fields and library scratch). This matches the HPC use case where
    the application wants full, observable control over device memory.

    Parameters
    ----------
    pooled : bool, optional
        If True (default), cupy keeps a caching :class:`cupy.cuda.MemoryPool`
        on top of muGrid: muGrid provides the (few, large) pool blocks while
        cupy serves array allocations from them — fast, but cupy still caches
        freed blocks (visible as ``label`` in the profiler). If False, every
        cupy allocation goes straight to muGrid (raw cudaMalloc/hipMalloc, no
        caching): fully deterministic and each allocation individually
        accounted, but slow for code that churns temporaries — pair it with
        fused kernels that avoid per-operation temporaries.
    label : str, optional
        Profiler label for cupy's allocations (default ``"<cupy>"``).

    Returns
    -------
    cupy.cuda.MemoryPool or None
        The pool when ``pooled`` is True (call ``pool.free_all_blocks()`` to
        release cached blocks back to muGrid), else None.

    Notes
    -----
    Mutually exclusive with :func:`use_cupy_allocator` — enabling both would
    make the two allocators call each other and recurse. This raises if an
    external muGrid allocator is already registered; call
    :func:`clear_device_allocator` first if you really need to switch.
    """
    import cupy

    if _muGrid.device_allocator_is_external():
        raise RuntimeError(
            "route_cupy_through_mugrid() conflicts with use_cupy_allocator(): "
            "the former routes cupy -> muGrid, the latter muGrid -> cupy, so "
            "enabling both would recurse. Call clear_device_allocator() first."
        )

    def _malloc(size, device_id):
        return _muGrid.device_allocate(size, label)

    def _free(ptr, device_id):
        _muGrid.device_deallocate(ptr)

    allocator = cupy.cuda.PythonFunctionAllocator(_malloc, _free)
    pool = None
    if pooled:
        pool = cupy.cuda.MemoryPool(allocator.malloc)
        cupy.cuda.set_allocator(pool.malloc)
    else:
        cupy.cuda.set_allocator(allocator.malloc)

    global _cupy_routing
    _cupy_routing = (allocator, pool, _malloc, _free)
    return pool


def restore_default_cupy_allocator():
    """
    Undo :func:`route_cupy_through_mugrid`, restoring cupy's own default
    caching pool. Allocations made while routed are still freed through
    muGrid; only new cupy allocations use the restored pool.
    """
    import cupy

    cupy.cuda.set_allocator(cupy.cuda.MemoryPool().malloc)
    global _cupy_routing
    _cupy_routing = None


set_device_allocator = _muGrid.set_device_allocator
clear_device_allocator = _muGrid.clear_device_allocator
