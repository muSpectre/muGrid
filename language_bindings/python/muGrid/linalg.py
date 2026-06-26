"""
Linear algebra operations for muGrid fields.

This module provides efficient linear algebra operations that operate
directly on muGrid fields, avoiding the overhead of creating non-contiguous
views. Operations follow the Array API specification where applicable:
https://data-apis.org/array-api/latest/

For fields with ghost regions (GlobalFieldCollection), reduction operations
(vecdot, norm_sq) only iterate over the interior region to ensure correct
MPI-parallel semantics. Update operations (axpy, scal, copy) operate on
the full buffer for efficiency.

Note: These functions return local (process-local) results for parallel
computations. Use comm.sum() to reduce across MPI ranks.
"""

from . import _muGrid

# Get the underlying C++ linalg module
_linalg = _muGrid.linalg


def _get_cpp(field):
    """Extract the underlying C++ field object from a Python wrapper."""
    if hasattr(field, "_cpp"):
        return field._cpp
    return field


def vecdot(a, b):
    """
    Compute vector dot product of two fields (interior only).

    Computes sum_i(a[i] * b[i]) over all interior pixels and components.
    Ghost regions are excluded for correct MPI-parallel semantics.

    Parameters
    ----------
    a : Field
        First field
    b : Field
        Second field (must have same shape as a)

    Returns
    -------
    float or complex
        Local dot product (not MPI-reduced)

    Notes
    -----
    Following Array API vecdot semantics:
    https://data-apis.org/array-api/latest/API_specification/generated/array_api.vecdot.html
    """
    return _linalg.vecdot(_get_cpp(a), _get_cpp(b))


def norm_sq(x):
    """
    Compute squared L2 norm of a field (interior only).

    Equivalent to vecdot(x, x).

    Parameters
    ----------
    x : Field
        Input field

    Returns
    -------
    float or complex
        Local squared norm (not MPI-reduced)
    """
    return _linalg.norm_sq(_get_cpp(x))


def pipelined_cg_dots(r, u, w):
    """
    Fused interior reduction for pipelined CG.

    Returns ``[(r, u), (w, u), (r, r)]`` computed in a single pass over the
    interior region (one GPU kernel and one device-to-host copy), replacing the
    three separate reductions of standard preconditioned CG.

    Parameters
    ----------
    r, u, w : Field
        Fields on the same collection.

    Returns
    -------
    list of float
        ``[(r, u), (w, u), (r, r)]`` (local, not MPI-reduced).
    """
    return _linalg.pipelined_cg_dots(_get_cpp(r), _get_cpp(u), _get_cpp(w))


def axpy(alpha, x, y):
    """
    AXPY operation: y = alpha * x + y (full buffer).

    Parameters
    ----------
    alpha : float or complex
        Scalar multiplier
    x : Field
        Input field
    y : Field
        Input/output field (modified in place)
    """
    _linalg.axpy(alpha, _get_cpp(x), _get_cpp(y))


def scal(alpha, x):
    """
    Scale operation: x = alpha * x (full buffer).

    Following BLAS *scal, but alpha may also be a real field on the same
    collection, applied per pixel: a single-component alpha is broadcast
    over the components of x (e.g. the inverse symbol of an operator in
    a Fourier-space preconditioner), an alpha with x's number of
    components is applied elementwise (e.g. a per-component Jacobi
    diagonal). Works on host and device fields; x may be real or
    complex.

    Parameters
    ----------
    alpha : float, complex or Field
        Scalar multiplier, or a real field of per-pixel multipliers with
        one or x's number of components
    x : Field
        Input/output field (modified in place)
    """
    if hasattr(alpha, "_cpp") or not isinstance(alpha, (int, float, complex)):
        _linalg.scal(_get_cpp(alpha), _get_cpp(x))
    else:
        _linalg.scal(alpha, _get_cpp(x))


def axpby(alpha, x, beta, y):
    """
    AXPBY operation: y = alpha * x + beta * y (full buffer).

    Combined scale-and-add that is more efficient than separate scal + axpy
    because it reads and writes each element only once.

    Parameters
    ----------
    alpha : float or complex
        Scalar multiplier for x
    x : Field
        Input field
    beta : float or complex
        Scalar multiplier for y
    y : Field
        Input/output field (modified in place)
    """
    _linalg.axpby(alpha, _get_cpp(x), beta, _get_cpp(y))


def copy(src, dst):
    """
    Copy operation: dst = src (full buffer).

    Parameters
    ----------
    src : Field
        Source field
    dst : Field
        Destination field (modified in place)
    """
    _linalg.copy(_get_cpp(src), _get_cpp(dst))


def axpy_norm_sq(alpha, x, y):
    """
    Fused AXPY + norm_sq: y = alpha * x + y, returns ||y||² (interior only).

    This fused operation computes both the AXPY update and the squared norm
    of the result in a single pass through memory. More efficient than
    separate axpy() + norm_sq() calls because:
    - axpy + norm_sq: 2 reads of x, 2 reads of y, 1 write of y
    - axpy_norm_sq:   1 read of x, 1 read of y, 1 write of y

    Parameters
    ----------
    alpha : float or complex
        Scalar multiplier
    x : Field
        Input field
    y : Field
        Input/output field (modified in place)

    Returns
    -------
    float or complex
        Squared norm of y after update (local, not MPI-reduced)
    """
    return _linalg.axpy_norm_sq(alpha, _get_cpp(x), _get_cpp(y))


def cross(a, b, out):
    """
    Per-pixel three-vector cross product: out = a x b (full buffer).

    Fused single-pass kernel for three-component fields (e.g. the vorticity
    ``ik x u`` and the Lamb vector ``u x omega`` of a pseudo-spectral solver),
    avoiding the temporaries of an array-expression cross product. Runs on host
    or device. ``out`` must be a field distinct from ``a`` and ``b``.

    Parameters
    ----------
    a : Field
        First field (exactly 3 components)
    b : Field
        Second field (3 components, same collection as a)
    out : Field
        Output field (3 components, modified in place)
    """
    _linalg.cross(_get_cpp(a), _get_cpp(b), _get_cpp(out))


def leray_project(k, invk, N, out):
    """
    Fused Leray projection update: ``out[c] -= k[c] * sum_d(invk[d] * N[d])``.

    Removes the longitudinal (compressible) part of a Fourier-space vector
    field in a single pass: with ``k`` the wavevector and ``invk = k/|k|**2``,
    this subtracts ``k (k.N)/|k|**2``, projecting ``out`` onto the
    divergence-free subspace, without the intermediate ``(invk.N)`` field of
    the array form. Runs on host or device; ``out`` may alias ``N``.

    Parameters
    ----------
    k : RealField
        Wavevector field (3 components)
    invk : RealField
        Field k/|k|**2 (3 components; the k=0 mode regularised by the caller)
    N : ComplexField
        Source vector field (3 components)
    out : ComplexField
        Field updated in place (3 components)
    """
    _linalg.leray_project(_get_cpp(k), _get_cpp(invk), _get_cpp(N),
                          _get_cpp(out))


__all__ = ["vecdot", "norm_sq", "axpy", "scal", "axpby", "copy", "axpy_norm_sq",
           "cross", "leray_project"]
