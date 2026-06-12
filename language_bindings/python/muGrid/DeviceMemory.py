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


set_device_allocator = _muGrid.set_device_allocator
clear_device_allocator = _muGrid.clear_device_allocator
