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

    Parameters
    ----------
    alpha : float or complex
        Scalar multiplier
    x : Field
        Input/output field (modified in place)
    """
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


__all__ = ["vecdot", "norm_sq", "axpy", "scal", "axpby", "copy", "axpy_norm_sq"]
