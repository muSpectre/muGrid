#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Wrappers.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   22 Dec 2024

@brief  Python wrappers for muGrid C++ classes with Pythonic interfaces

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

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import ArrayLike

# Import C++ module
try:
    from . import _muGrid
except ImportError:
    import _muGrid

from .Field import Field

if TYPE_CHECKING:
    from .Parallel import Communicator

# Type aliases
DeviceStr = Literal["cpu", "cuda", "rocm", "gpu", "host", "device"]
SubPtMap = Dict[str, int]
Shape = Union[Tuple[int, ...], List[int], Sequence[int]]


def _typed_field(collection: Any, name: str, components: Shape, sub_pt: str,
                 dtype: Any, allowed: Dict[Any, str], double_method: str):
    """Get-or-create a field of the requested ``dtype`` on ``collection``.

    The double-precision case delegates to the collection's get-or-create
    convenience method (``real_field`` / ``complex_field``); the
    single-precision case mirrors those semantics around the
    ``register_*32_field`` methods (which are register-only) by returning the
    existing field when one of the given name is already registered. ``allowed``
    maps each supported NumPy dtype to the name of the register-only method to
    use for it (the double dtype maps to ``double_method``)."""
    dt = np.dtype(dtype)
    method = allowed.get(dt)
    if method is None:
        kinds = ", ".join(str(np.dtype(d)) for d in allowed)
        raise ValueError(
            f"{double_method} supports dtype {kinds}, got {dt}"
        )
    if method == double_method:
        # Collection's own get-or-create convenience (handles existing names).
        return getattr(collection, double_method)(name, components, sub_pt)
    # Single precision: emulate get-or-create around the register-only method.
    if collection.field_exists(name):
        return collection.get_field(name)
    return getattr(collection, method)(name, components, sub_pt)


class FieldCollectionMixin:
    """
    Mixin providing field creation methods for classes that have an underlying
    C++ field collection.

    Classes using this mixin must implement `_get_field_collection()` which
    returns the C++ field collection object to use for field creation.
    """

    def _get_field_collection(self) -> Any:
        """
        Get the C++ field collection for field creation.

        Subclasses must override this method. The default implementation
        returns self._cpp, which works for GlobalFieldCollection and
        LocalFieldCollection.
        """
        return self._cpp

    def real_field(
        self,
        name: str,
        components: Shape = (),
        sub_pt: str = "pixel",
        dtype: Any = np.float64,
    ) -> Field:
        """
        Create a real-valued field.

        Parameters
        ----------
        name : str
            Unique name for the field.
        components : tuple of int, optional
            Shape of field components. Default is () for scalar.
        sub_pt : str, optional
            Sub-point type. Default is "pixel".
        dtype : data-type, optional
            Floating-point precision of the field: ``np.float64`` (default,
            double precision) or ``np.float32`` (single precision). Single
            precision halves the memory footprint.

        Returns
        -------
        Field
            Wrapped field with .s, .p, .sg, .pg accessors.
        """
        cpp_field = _typed_field(
            self._get_field_collection(), name, components, sub_pt, dtype,
            {np.dtype(np.float64): "real_field",
             np.dtype(np.float32): "register_real32_field"},
            "real_field",
        )
        return Field(cpp_field)

    def complex_field(
        self,
        name: str,
        components: Shape = (),
        sub_pt: str = "pixel",
        dtype: Any = np.complex128,
    ) -> Field:
        """
        Create a complex-valued field.

        Parameters
        ----------
        name : str
            Unique name for the field.
        components : tuple of int, optional
            Shape of field components. Default is () for scalar.
        sub_pt : str, optional
            Sub-point type. Default is "pixel".
        dtype : data-type, optional
            Precision of the field: ``np.complex128`` (default, double
            precision) or ``np.complex64`` (single precision). Single
            precision halves the memory footprint.

        Returns
        -------
        Field
            Wrapped field with .s, .p, .sg, .pg accessors.
        """
        cpp_field = _typed_field(
            self._get_field_collection(), name, components, sub_pt, dtype,
            {np.dtype(np.complex128): "complex_field",
             np.dtype(np.complex64): "register_complex32_field"},
            "complex_field",
        )
        return Field(cpp_field)

    def int_field(
        self,
        name: str,
        components: Shape = (),
        sub_pt: str = "pixel",
    ) -> Field:
        """
        Create an integer field.

        Parameters
        ----------
        name : str
            Unique name for the field.
        components : tuple of int, optional
            Shape of field components. Default is () for scalar.
        sub_pt : str, optional
            Sub-point type. Default is "pixel".

        Returns
        -------
        Field
            Wrapped field with .s, .p, .sg, .pg accessors.
        """
        cpp_field = self._get_field_collection().int_field(name, components, sub_pt)
        return Field(cpp_field)

    def uint_field(
        self,
        name: str,
        components: Shape = (),
        sub_pt: str = "pixel",
    ) -> Field:
        """
        Create an unsigned integer field.

        Parameters
        ----------
        name : str
            Unique name for the field.
        components : tuple of int, optional
            Shape of field components. Default is () for scalar.
        sub_pt : str, optional
            Sub-point type. Default is "pixel".

        Returns
        -------
        Field
            Wrapped field with .s, .p, .sg, .pg accessors.
        """
        cpp_field = self._get_field_collection().uint_field(name, components, sub_pt)
        return Field(cpp_field)


def _unwrap(obj: Any) -> Any:
    """
    Extract the C++ object from a Python wrapper.

    Parameters
    ----------
    obj : Any
        A Python wrapper object with a `_cpp` attribute, or a raw C++ object.

    Returns
    -------
    Any
        The underlying C++ object.
    """
    return getattr(obj, "_cpp", obj)


# Valid real<->complex precision pairings for an FFT: the transform runs in the
# precision of the fields, so a real field must be paired with the complex field
# of matching precision. A mismatched pair (e.g. float32 real with complex128
# Fourier) does not raise in the C++ layer -- it silently produces garbage/NaN
# -- so it is caught here with an actionable message.
_FFT_REAL_TO_COMPLEX = {
    np.dtype(np.float32): np.dtype(np.complex64),
    np.dtype(np.float64): np.dtype(np.complex128),
}


def _check_fft_precision(real_field: Any, complex_field: Any,
                         direction: str) -> None:
    """Raise if the real- and Fourier-space fields have mismatched precision.

    ``real_field`` is the real-space field and ``complex_field`` the
    Fourier-space field of a forward (``direction="fft"``) or inverse
    (``direction="ifft"``) transform. The FFT pairs ``float32`` with
    ``complex64`` and ``float64`` with ``complex128``; any other combination is
    rejected here rather than silently transforming into a NaN-filled buffer.
    """
    try:
        real_dtype = np.dtype(real_field.dtype)
        complex_dtype = np.dtype(complex_field.dtype)
    except (AttributeError, TypeError):
        # dtype not introspectable (e.g. a raw C++ field); leave it to C++.
        return
    expected = _FFT_REAL_TO_COMPLEX.get(real_dtype)
    if expected is None or complex_dtype != expected:
        raise ValueError(
            f"FFT precision mismatch in {direction}(): the real-space field is "
            f"{real_dtype} but the Fourier-space field is {complex_dtype}. The "
            f"transform pairs float32 with complex64 and float64 with "
            f"complex128 -- create both fields at the same precision, e.g. "
            f"real_space_field(..., dtype=np.float32) together with "
            f"fourier_space_field(..., dtype=np.complex64)."
        )


def _parse_device(
    device: Optional[Union[str, "_muGrid.Device"]],
) -> "_muGrid.Device":
    """
    Parse a device specification to a Device instance.

    Parameters
    ----------
    device : str, Device, or None
        Device specification. Can be:
        - None: defaults to CPU
        - "cpu" or "host": CPU device
        - "gpu" or "device": Default GPU (auto-detects CUDA or ROCm)
        - "gpu:N": Default GPU with device ID N
        - "cuda": CUDA GPU (device 0)
        - "cuda:N": CUDA GPU with device ID N
        - "rocm": ROCm GPU (device 0)
        - "rocm:N": ROCm GPU with device ID N
        - Device instance: used directly

    Returns
    -------
    Device
        The C++ Device instance.

    Raises
    ------
    ValueError
        If an invalid device string is provided.
    RuntimeError
        If GPU is requested but not available.
    """
    Device = _muGrid.Device
    # If it's already a Device instance, return it directly
    if isinstance(device, Device):
        return device
    if device is None or device == "host" or device == "cpu":
        return Device.cpu()
    elif (
        device == "gpu"
        or device.startswith("gpu:")
        or device == "device"
        or device.startswith("cuda")
        or device.startswith("rocm")
    ):
        if not _muGrid.has_gpu:
            raise RuntimeError(
                "GPU support is not available. "
                "Rebuild muGrid with MUGRID_ENABLE_CUDA=ON or MUGRID_ENABLE_HIP=ON."
            )
        # Parse device:id format
        if ":" in device:
            parts = device.split(":")
            device_id = int(parts[1]) if len(parts) > 1 else 0
        else:
            device_id = 0
        # Auto-detect GPU backend for "gpu" or "device"
        if device == "gpu" or device.startswith("gpu:") or device == "device":
            return Device.gpu(device_id)
        elif device.startswith("cuda"):
            return Device.cuda(device_id)
        else:
            return Device.rocm(device_id)
    else:
        raise ValueError(
            f"Invalid device: {device!r}. "
            f"Must be 'cpu', 'gpu', 'cuda', 'cuda:N', 'rocm', or 'rocm:N'."
        )


def _ghost_requirement(op: Any) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    Return the (left, right) ghost requirement reported by an operator,
    or None if the object does not report one.

    Any object exposing a `ghost_requirement` attribute of the form
    ((left...), (right...)) qualifies; this includes all muGrid stencil
    operators and user-defined operators following the same protocol.
    """
    return getattr(_unwrap(op), "ghost_requirement", None)


def _as_ghost_shape(value: Any, nb_dims: int, name: str) -> List[int]:
    """Normalize an int or per-dimension sequence to a list of ghost counts."""
    if isinstance(value, (int, np.integer)):
        return [int(value)] * nb_dims
    counts = [int(v) for v in value]
    if len(counts) != nb_dims:
        raise ValueError(
            f"{name} has {len(counts)} entries, but the grid is {nb_dims}D"
        )
    return counts


def _resolve_ghosts(
    ghosts: Any,
    nb_ghosts_left: Optional[Shape],
    nb_ghosts_right: Optional[Shape],
    nb_dims: int,
) -> Tuple[List[int], List[int]]:
    """
    Resolve the `ghosts` argument of CartesianDecomposition/FFTEngine to
    explicit (nb_ghosts_left, nb_ghosts_right) lists.

    Parameters
    ----------
    ghosts : operator, sequence of operators, int, or (left, right) pair
        - An operator (anything exposing `ghost_requirement`): use the
          ghost layers the operator reports for all of its operations.
        - A sequence of operators: elementwise maximum of their
          requirements (one decomposition serving several stencils).
        - An int n: n ghost layers on both sides of every dimension.
        - A pair (left, right), each an int or a per-dimension sequence:
          explicit ghost counts (expert override).
    nb_ghosts_left, nb_ghosts_right : sequence of int, optional
        Legacy explicit ghost counts; mutually exclusive with `ghosts`.
    nb_dims : int
        Spatial dimension of the grid, used for validation.

    Returns
    -------
    (left, right) : pair of lists of int
        Ghost layers per dimension.
    """
    if ghosts is None:
        left = [0] * nb_dims if nb_ghosts_left is None else list(nb_ghosts_left)
        right = (
            [0] * nb_dims if nb_ghosts_right is None else list(nb_ghosts_right)
        )
        return left, right
    if nb_ghosts_left is not None or nb_ghosts_right is not None:
        raise ValueError(
            "Specify either `ghosts` or `nb_ghosts_left`/`nb_ghosts_right`, "
            "not both"
        )

    # Uniform ghost count on both sides of every dimension
    if isinstance(ghosts, (int, np.integer)):
        return [int(ghosts)] * nb_dims, [int(ghosts)] * nb_dims

    # Single operator reporting its own requirement
    requirement = _ghost_requirement(ghosts)
    if requirement is not None:
        requirements = [requirement]
    else:
        items = list(ghosts)
        item_requirements = [_ghost_requirement(item) for item in items]
        if items and all(r is not None for r in item_requirements):
            # Sequence of operators
            requirements = item_requirements
        elif any(r is not None for r in item_requirements):
            raise TypeError(
                "Cannot mix operators and explicit ghost counts in `ghosts`; "
                "pass either operators or a (left, right) pair"
            )
        elif len(items) == 2:
            # Explicit (left, right) pair
            return (
                _as_ghost_shape(items[0], nb_dims, "ghosts[0] (left)"),
                _as_ghost_shape(items[1], nb_dims, "ghosts[1] (right)"),
            )
        else:
            raise TypeError(
                "`ghosts` must be an operator, a sequence of operators, an "
                "int, or a (left, right) pair of ghost counts"
            )

    left = [0] * nb_dims
    right = [0] * nb_dims
    for requirement in requirements:
        req_left, req_right = requirement
        if len(req_left) != nb_dims:
            raise ValueError(
                f"Operator reports a {len(req_left)}D ghost requirement, "
                f"but the grid is {nb_dims}D"
            )
        left = [max(a, int(b)) for a, b in zip(left, req_left)]
        right = [max(a, int(b)) for a, b in zip(right, req_right)]
    return left, right


class GlobalFieldCollection(FieldCollectionMixin):
    """
    Python wrapper for muGrid GlobalFieldCollection.

    A GlobalFieldCollection manages a set of fields that share the same
    global grid structure. It can allocate fields in either host (CPU)
    or device (GPU) memory.

    Parameters
    ----------
    nb_grid_pts : Sequence[int]
        Grid dimensions, e.g., [64, 64] for 2D or [32, 32, 32] for 3D.
    nb_sub_pts : dict, optional
        Number of sub-points per pixel for each sub-point type.
        Default is {}. Alias: sub_pts.
    nb_ghosts_left : Sequence[int], optional
        Ghost cells on low-index side. Default is no ghosts. Prefer the
        `ghosts` argument, which sizes the buffers from the operators that
        will run on this collection.
    nb_ghosts_right : Sequence[int], optional
        Ghost cells on high-index side. Default is no ghosts. Prefer the
        `ghosts` argument.
    device : str or Device, optional
        Device for field allocation: "cpu", "cuda", "cuda:N", "rocm:N",
        or a Device instance. Default is "cpu".
    ghosts : operator, sequence of operators, int, or (left, right), optional
        Ghost-buffer specification; mutually exclusive with
        `nb_ghosts_left`/`nb_ghosts_right`. Pass the stencil operator (or a
        list of operators) that will run on this collection to size the
        ghost buffers from the requirement the operators report. An int n
        means n ghost layers on both sides of every dimension; a
        (left, right) pair gives explicit per-side counts.

    Examples
    --------
    >>> fc = GlobalFieldCollection([64, 64])
    >>> field = fc.real_field("temperature")
    >>> field.p[:] = 300.0  # Set temperature to 300K

    >>> # GPU field collection
    >>> fc_gpu = GlobalFieldCollection([64, 64], device="cuda")
    """

    # Expose Device class for setting device
    Device = _muGrid.Device

    def __init__(self, *args, **kwargs) -> None:
        # If called with positional args matching C++ signature, pass through directly
        if args and len(args) > 1:
            # User is using C++ constructor directly
            self._cpp = _muGrid.GlobalFieldCollection(*args, **kwargs)
            self._nb_grid_pts = list(args[0])
            return

        # Otherwise, use keyword-based simplified API
        nb_grid_pts = args[0] if args else kwargs.get("nb_grid_pts")
        if nb_grid_pts is None:
            raise TypeError("nb_grid_pts is required")

        nb_sub_pts = kwargs.get("nb_sub_pts")
        sub_pts = kwargs.get("sub_pts")
        nb_ghosts_left = kwargs.get("nb_ghosts_left")
        nb_ghosts_right = kwargs.get("nb_ghosts_right")
        ghosts = kwargs.get("ghosts")
        device = kwargs.get("device")
        nb_subdomain_grid_pts = kwargs.get("nb_subdomain_grid_pts")

        # Handle sub_pts alias
        if sub_pts is not None:
            if nb_sub_pts is not None and nb_sub_pts != sub_pts:
                raise ValueError(
                    "Cannot specify both 'nb_sub_pts' and 'sub_pts' with "
                    "different values"
                )
            nb_sub_pts = sub_pts
        if nb_sub_pts is None:
            nb_sub_pts = {}

        # Handle ghost defaults
        nb_ghosts_left, nb_ghosts_right = _resolve_ghosts(
            ghosts, nb_ghosts_left, nb_ghosts_right, len(nb_grid_pts)
        )

        # Build C++ constructor arguments
        cpp_kwargs = {
            "nb_domain_grid_pts": list(nb_grid_pts),
            "sub_pts": nb_sub_pts,
            "nb_ghosts_left": list(nb_ghosts_left),
            "nb_ghosts_right": list(nb_ghosts_right),
        }

        # Handle device parameter
        if device is not None:
            cpp_kwargs["device"] = _parse_device(device)

        # Handle nb_subdomain_grid_pts if provided
        if nb_subdomain_grid_pts is not None:
            cpp_kwargs["nb_subdomain_grid_pts"] = list(nb_subdomain_grid_pts)

        self._cpp = _muGrid.GlobalFieldCollection(**cpp_kwargs)
        self._nb_grid_pts = list(nb_grid_pts)

    @property
    def nb_grid_pts(self) -> List[int]:
        """Grid dimensions."""
        return self._nb_grid_pts

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying C++ object."""
        return getattr(self._cpp, name)

    def __repr__(self) -> str:
        return f"GlobalFieldCollection({self._nb_grid_pts})"


class LocalFieldCollection(FieldCollectionMixin):
    """
    Python wrapper for muGrid LocalFieldCollection.

    A LocalFieldCollection manages fields on a subset of pixels, typically
    used for material-specific data in heterogeneous simulations.

    Parameters
    ----------
    spatial_dim : int
        Spatial dimension (2 or 3).
    name : str, optional
        Name for the collection.
    nb_sub_pts : dict, optional
        Number of sub-points per pixel for each sub-point type.
    device : str or Device, optional
        Device for field allocation: "cpu", "cuda", "cuda:N", "rocm:N",
        or a Device instance. Default is "cpu".
    """

    # Expose Device class for setting device
    Device = _muGrid.Device

    def __init__(self, *args, **kwargs) -> None:
        # If called with positional args matching C++ signature, pass through directly
        if len(args) >= 2:
            # User is using C++ constructor directly:
            # LocalFieldCollection(dim, name, ...)
            self._cpp = _muGrid.LocalFieldCollection(*args, **kwargs)
            self._spatial_dim = args[0]
            return

        # Otherwise use keyword/simplified API
        spatial_dim = args[0] if args else kwargs.get("spatial_dim")
        if spatial_dim is None:
            raise TypeError("spatial_dim is required")

        name = kwargs.get("name", "")
        nb_sub_pts = kwargs.get("nb_sub_pts", {})
        device = kwargs.get("device")

        cpp_kwargs = {
            "spatial_dimension": spatial_dim,
            "name": name,
            "nb_sub_pts": nb_sub_pts,
        }

        if device is not None:
            cpp_kwargs["device"] = _parse_device(device)

        self._cpp = _muGrid.LocalFieldCollection(**cpp_kwargs)
        self._spatial_dim = spatial_dim

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying C++ object."""
        return getattr(self._cpp, name)

    def __repr__(self) -> str:
        return f"LocalFieldCollection(spatial_dim={self._spatial_dim})"


class CartesianDecomposition(FieldCollectionMixin):
    """
    Python wrapper for muGrid CartesianDecomposition.

    CartesianDecomposition manages domain decomposition for MPI-parallel
    computations on structured grids, including ghost buffer regions for
    stencil operations.

    Parameters
    ----------
    communicator : Communicator
        MPI communicator for parallel execution.
    nb_domain_grid_pts : Sequence[int]
        Global domain grid dimensions.
    nb_subdivisions : Sequence[int], optional
        Number of subdivisions in each dimension. Default is automatic.
    nb_ghosts_left : Sequence[int], optional
        Ghost cells on low-index side. Default is no ghosts. Prefer the
        `ghosts` argument, which sizes the buffers from the operators that
        will run on this decomposition.
    nb_ghosts_right : Sequence[int], optional
        Ghost cells on high-index side. Default is no ghosts. Prefer the
        `ghosts` argument.
    nb_sub_pts : dict, optional
        Number of sub-points per pixel for each sub-point type.
    device : Device or str, optional
        Device for field allocation: Device instance, "host", "device",
        "cpu", "cuda:N", or "rocm:N". Default is CPU.
    ghosts : operator, sequence of operators, int, or (left, right), optional
        Ghost-buffer specification; mutually exclusive with
        `nb_ghosts_left`/`nb_ghosts_right`. Pass the stencil operator (or a
        list of operators) that will run on this decomposition to size the
        ghost buffers from the requirement the operators report. An int n
        means n ghost layers on both sides of every dimension; a
        (left, right) pair gives explicit per-side counts.

    Examples
    --------
    >>> from muGrid import Communicator, CartesianDecomposition, LaplaceOperator
    >>> comm = Communicator()
    >>> laplace = LaplaceOperator(2)
    >>> decomp = CartesianDecomposition(
    ...     comm,
    ...     nb_domain_grid_pts=[128, 128],
    ...     ghosts=laplace,
    ... )
    >>> field = decomp.real_field("displacement", components=(3,))
    """

    # Expose Device class for setting device
    Device = _muGrid.Device

    def __init__(
        self,
        communicator: "Communicator",
        nb_domain_grid_pts: Shape,
        nb_subdivisions: Optional[Shape] = None,
        nb_ghosts_left: Optional[Shape] = None,
        nb_ghosts_right: Optional[Shape] = None,
        nb_sub_pts: Optional[SubPtMap] = None,
        device: Optional[Union[DeviceStr, "_muGrid.Device"]] = None,
        ghosts: Any = None,
    ) -> None:
        from .Parallel import Communicator as CommFactory

        # Ensure we have a proper C++ Communicator object
        if not isinstance(communicator, _muGrid.Communicator):
            communicator = CommFactory(communicator)

        # Handle defaults
        nb_dims = len(nb_domain_grid_pts)
        if nb_subdivisions is None:
            nb_subdivisions = [0] * nb_dims
        nb_ghosts_left, nb_ghosts_right = _resolve_ghosts(
            ghosts, nb_ghosts_left, nb_ghosts_right, nb_dims
        )
        if nb_sub_pts is None:
            nb_sub_pts = {}

        mem_loc = _parse_device(device)
        # C++ constructor signature:
        # CartesianDecomposition(comm, nb_domain_grid_pts, nb_subdivisions,
        #                        nb_ghosts_left, nb_ghosts_right, sub_pts={},
        #                        device=Device::cpu())
        self._cpp = _muGrid.CartesianDecomposition(
            _unwrap(communicator),
            list(nb_domain_grid_pts),
            list(nb_subdivisions),
            list(nb_ghosts_left),
            list(nb_ghosts_right),
            nb_sub_pts,
            mem_loc,
        )

    def _get_field_collection(self) -> Any:
        """Get the C++ field collection for field creation."""
        return self._cpp.collection

    @property
    def collection(self) -> GlobalFieldCollection:
        """
        Get the underlying GlobalFieldCollection.

        Returns
        -------
        GlobalFieldCollection
            Wrapped field collection with Pythonic interfaces.
        """
        # Create a wrapper around the C++ collection
        wrapper = GlobalFieldCollection.__new__(GlobalFieldCollection)
        wrapper._cpp = self._cpp.collection
        # Get grid pts from decomposition, not from collection
        wrapper._nb_grid_pts = list(self._cpp.nb_domain_grid_pts)
        return wrapper

    @property
    def nb_grid_pts(self) -> List[int]:
        """Local subdomain grid dimensions (alias for nb_subdomain_grid_pts)."""
        return list(self._cpp.nb_subdomain_grid_pts)

    def set_nb_sub_pts(self, sub_pt_type: str, nb_sub_pts: int) -> None:
        """
        Set the number of sub-points for a given sub-point type.

        Parameters
        ----------
        sub_pt_type : str
            Name of the sub-point type (e.g., "quad").
        nb_sub_pts : int
            Number of sub-points per pixel for this type.
        """
        self._cpp.collection.set_nb_sub_pts(sub_pt_type, nb_sub_pts)

    def communicate_ghosts(self, field: Field) -> None:
        """
        Exchange ghost buffer data for a field.

        Parameters
        ----------
        field : Field
            The field whose ghost buffers should be filled from neighbors.
        """
        self._cpp.communicate_ghosts(_unwrap(field))

    def reduce_ghosts(self, field: Field) -> None:
        """
        Accumulate ghost buffer contributions back to the interior domain.

        This is the adjoint operation of communicate_ghosts and is needed
        for transpose operations (e.g., divergence) with periodic BCs.
        After the operation, ghost buffers are zeroed.

        Parameters
        ----------
        field : Field
            The field whose ghost buffers should be reduced to interior.
        """
        self._cpp.reduce_ghosts(_unwrap(field))

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying C++ object."""
        return getattr(self._cpp, name)

    def __repr__(self) -> str:
        return (
            f"CartesianDecomposition("
            f"global={list(self._cpp.nb_domain_grid_pts)}, "
            f"local={list(self._cpp.nb_subdomain_grid_pts)})"
        )


class FFTEngine:
    """
    Python wrapper for muGrid FFTEngine.

    The FFTEngine provides distributed FFT operations on structured grids
    with MPI parallelization using pencil (2D) decomposition.

    Parameters
    ----------
    nb_domain_grid_pts : Sequence[int]
        Global grid dimensions [Nx, Ny] or [Nx, Ny, Nz].
    communicator : Communicator, optional
        MPI communicator. Default is serial execution.
    nb_ghosts_left : Sequence[int], optional
        Ghost cells on low-index side of each dimension. Prefer the
        `ghosts` argument, which sizes the buffers from the operators that
        will run on this engine's real-space fields.
    nb_ghosts_right : Sequence[int], optional
        Ghost cells on high-index side of each dimension. Prefer the
        `ghosts` argument.
    nb_sub_pts : dict, optional
        Number of sub-points per pixel.
    device : str or Device, optional
        Device for FFT execution: "cpu" (default), "cuda", "cuda:N", "gpu",
        or a Device instance. When a GPU device is specified, the FFT uses
        cuFFT and fields are allocated on GPU memory.
    ghosts : operator, sequence of operators, int, or (left, right), optional
        Ghost-buffer specification for the real-space fields; mutually
        exclusive with `nb_ghosts_left`/`nb_ghosts_right`. Pass the stencil
        operator (or a list of operators) that will run on this engine's
        real-space fields to size the ghost buffers from the requirement
        the operators report. An int n means n ghost layers on both sides
        of every dimension; a (left, right) pair gives explicit per-side
        counts.

    Examples
    --------
    >>> engine = FFTEngine([64, 64])
    >>> real_field = engine.real_space_field("displacement", components=(3,))
    >>> fourier_field = engine.fourier_space_field("displacement_k", components=(3,))
    >>> engine.fft(real_field, fourier_field)
    >>> engine.ifft(fourier_field, real_field)
    >>> real_field.s[:] *= engine.normalisation

    GPU example with multi-GPU MPI:

    >>> from mpi4py import MPI
    >>> comm = Communicator(MPI.COMM_WORLD)
    >>> engine = FFTEngine([64, 64], comm, device=f"cuda:{comm.rank}")
    """

    def __init__(
        self,
        nb_domain_grid_pts: Shape,
        communicator: Optional["Communicator"] = None,
        nb_ghosts_left: Optional[Shape] = None,
        nb_ghosts_right: Optional[Shape] = None,
        nb_sub_pts: Optional[SubPtMap] = None,
        device: Optional[Union[DeviceStr, "_muGrid.Device"]] = None,
        ghosts: Any = None,
        decomposition: str = "auto",
    ) -> None:
        from .Parallel import Communicator as CommFactory

        # Handle communicator - ensure we have a C++ Communicator object
        if communicator is None:
            comm = CommFactory()
        elif not isinstance(communicator, _muGrid.Communicator):
            comm = CommFactory(communicator)
        else:
            comm = communicator

        # Handle defaults
        nb_ghosts_left, nb_ghosts_right = _resolve_ghosts(
            ghosts, nb_ghosts_left, nb_ghosts_right, len(nb_domain_grid_pts)
        )
        if nb_sub_pts is None:
            nb_sub_pts = {}

        # Parse device and select appropriate engine
        parsed_device = _parse_device(device)
        if parsed_device.is_device:
            # GPU FFT engine (cuFFT or rocFFT)
            device_id = parsed_device.device_id
            if hasattr(_muGrid, "FFTEngineCUDA"):
                self._cpp = _muGrid.FFTEngineCUDA(
                    list(nb_domain_grid_pts),
                    _unwrap(comm),
                    list(nb_ghosts_left),
                    list(nb_ghosts_right),
                    nb_sub_pts,
                    device_id,
                    decomposition,
                )
            elif hasattr(_muGrid, "FFTEngineROCm"):
                self._cpp = _muGrid.FFTEngineROCm(
                    list(nb_domain_grid_pts),
                    _unwrap(comm),
                    list(nb_ghosts_left),
                    list(nb_ghosts_right),
                    nb_sub_pts,
                    device_id,
                    decomposition,
                )
            else:
                raise RuntimeError(
                    "GPU FFT requested but muGrid was compiled without "
                    "CUDA or ROCm support"
                )
        else:
            # CPU FFT engine (PocketFFT)
            self._cpp = _muGrid.FFTEngine(
                list(nb_domain_grid_pts),
                _unwrap(comm),
                list(nb_ghosts_left),
                list(nb_ghosts_right),
                nb_sub_pts,
                decomposition,
            )

    def fft(self, input_field: Field, output_field: Field) -> None:
        """
        Forward FFT: real space -> Fourier space.

        The transform is unnormalized. To recover original data after
        ifft(fft(x)), multiply by `normalisation`.

        Parameters
        ----------
        input_field : Field
            Real-space field (must be in this engine's real collection).
        output_field : Field
            Fourier-space field (must be in this engine's Fourier collection).
        """
        _check_fft_precision(input_field, output_field, "fft")
        self._cpp.fft(_unwrap(input_field), _unwrap(output_field))

    def ifft(self, input_field: Field, output_field: Field) -> None:
        """
        Inverse FFT: Fourier space -> real space.

        The transform is unnormalized. To recover original data after
        ifft(fft(x)), multiply by `normalisation`.

        Parameters
        ----------
        input_field : Field
            Fourier-space field (must be in this engine's Fourier collection).
        output_field : Field
            Real-space field (must be in this engine's real collection).
        """
        _check_fft_precision(output_field, input_field, "ifft")
        self._cpp.ifft(_unwrap(input_field), _unwrap(output_field))

    def communicate_ghosts(self, field: Field) -> None:
        """
        Exchange ghost buffer data for a real-space field.

        The FFT engine is also a CartesianDecomposition; when constructed
        with ghost buffers, real-space fields support the same ghost
        communication as a stand-alone decomposition. This lets stencil
        operators and FFTs share one set of fields (e.g. for spectral
        preconditioning of a finite-difference solve).

        Parameters
        ----------
        field : Field
            The field whose ghost buffers should be filled from neighbors.
        """
        self._cpp.communicate_ghosts(_unwrap(field))

    def reduce_ghosts(self, field: Field) -> None:
        """
        Accumulate ghost buffer contributions back to the interior domain.

        Adjoint of communicate_ghosts; ghost buffers are zeroed afterwards.

        Parameters
        ----------
        field : Field
            The field whose ghost buffers should be reduced to interior.
        """
        self._cpp.reduce_ghosts(_unwrap(field))

    def register_real_space_field(
        self, name: str, components: Shape = (), dtype: Any = np.float64
    ) -> Field:
        """
        Register a new real-space field.

        Raises an error if a field with the given name already exists.

        Parameters
        ----------
        name : str
            Unique field name.
        components : tuple of int, optional
            Shape of field components. Default is () for scalar.
        dtype : data-type, optional
            ``np.float64`` (default) or ``np.float32``. Single precision pairs
            with single-precision FFTs (the engine picks the transform
            precision from the field dtype) and halves the memory footprint.

        Returns
        -------
        Field
            Wrapped real-valued field with array accessors.

        Raises
        ------
        RuntimeError
            If a field with the given name already exists.
        """
        if np.dtype(dtype) == np.float64:
            return Field(self._cpp.register_real_space_field(name, components))
        # Single precision: register on the engine's real-space collection.
        return self.real_space_collection.real_field(
            name, components, dtype=dtype
        )

    def register_fourier_space_field(
        self, name: str, components: Shape = (), dtype: Any = np.complex128
    ) -> Field:
        """
        Register a new Fourier-space field.

        Raises an error if a field with the given name already exists.

        Parameters
        ----------
        name : str
            Unique field name.
        components : tuple of int, optional
            Shape of field components. Default is () for scalar.
        dtype : data-type, optional
            ``np.complex128`` (default) or ``np.complex64`` (single precision,
            the Fourier-space counterpart of a ``np.float32`` real field).

        Returns
        -------
        Field
            Wrapped complex-valued field with array accessors.

        Raises
        ------
        RuntimeError
            If a field with the given name already exists.
        """
        if np.dtype(dtype) == np.complex128:
            return Field(
                self._cpp.register_fourier_space_field(name, components)
            )
        return self.fourier_space_collection.complex_field(
            name, components, dtype=dtype
        )

    def real_space_field(
        self, name: str, components: Shape = (), dtype: Any = np.float64
    ) -> Field:
        """
        Get or create a real-space field for FFT operations.

        If a field with the given name already exists, returns it.
        Otherwise creates a new field with the specified component shape.

        Parameters
        ----------
        name : str
            Unique field name.
        components : tuple of int, optional
            Shape of field components. Default is () for scalar.
        dtype : data-type, optional
            ``np.float64`` (default) or ``np.float32`` (single precision).

        Returns
        -------
        Field
            Wrapped real-valued field with array accessors.
        """
        if np.dtype(dtype) == np.float64:
            return Field(self._cpp.real_space_field(name, components))
        collection = self.real_space_collection
        if collection.field_exists(name):
            return Field(collection.get_field(name))
        return collection.real_field(name, components, dtype=dtype)

    def fourier_space_field(
        self, name: str, components: Shape = (), dtype: Any = np.complex128
    ) -> Field:
        """
        Get or create a Fourier-space field for FFT operations.

        If a field with the given name already exists, returns it.
        Otherwise creates a new field with the specified component shape.

        Parameters
        ----------
        name : str
            Unique field name.
        components : tuple of int, optional
            Shape of field components. Default is () for scalar.
        dtype : data-type, optional
            ``np.complex128`` (default) or ``np.complex64`` (single precision).

        Returns
        -------
        Field
            Wrapped complex-valued field with array accessors.
        """
        if np.dtype(dtype) == np.complex128:
            return Field(self._cpp.fourier_space_field(name, components))
        collection = self.fourier_space_collection
        if collection.field_exists(name):
            return Field(collection.get_field(name))
        return collection.complex_field(name, components, dtype=dtype)

    @property
    def real_space_collection(self) -> GlobalFieldCollection:
        """
        Get the real-space field collection.

        Returns
        -------
        GlobalFieldCollection
            Wrapped field collection for real-space fields.
        """
        wrapper = GlobalFieldCollection.__new__(GlobalFieldCollection)
        wrapper._cpp = self._cpp.real_space_collection
        wrapper._nb_grid_pts = list(self._cpp.nb_domain_grid_pts)
        return wrapper

    @property
    def fourier_space_collection(self) -> GlobalFieldCollection:
        """
        Get the Fourier-space field collection.

        Returns
        -------
        GlobalFieldCollection
            Wrapped field collection for Fourier-space fields.
        """
        wrapper = GlobalFieldCollection.__new__(GlobalFieldCollection)
        wrapper._cpp = self._cpp.fourier_space_collection
        wrapper._nb_grid_pts = list(self._cpp.nb_fourier_grid_pts)
        return wrapper

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying C++ object."""
        return getattr(self._cpp, name)

    def __repr__(self) -> str:
        return (
            f"FFTEngine("
            f"real={list(self._cpp.nb_domain_grid_pts)}, "
            f"fourier={list(self._cpp.nb_fourier_grid_pts)})"
        )


class GenericLinearOperator:
    """
    Python wrapper for muGrid GenericLinearOperator.

    Applies convolution (stencil) operations to fields. Useful for computing
    gradients, Laplacians, and other discrete differential operators.

    Parameters
    ----------
    offset : Sequence[int]
        Offset of the stencil origin relative to the current pixel.
    stencil : array_like
        Stencil coefficients. Shape determines the stencil size.

    Examples
    --------
    >>> # Create a 2D Laplacian stencil
    >>> import numpy as np
    >>> stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    >>> laplace = GenericLinearOperator([-1, -1], stencil)
    >>> laplace.apply(input_field, output_field)
    """

    def __init__(
        self,
        offset: Shape,
        stencil: ArrayLike,
    ) -> None:
        stencil_arr = np.asarray(stencil, dtype=np.float64, order="F")
        self._cpp = _muGrid.GenericLinearOperator(list(offset), stencil_arr)

    def apply(self, nodal_field: Field, quadrature_point_field: Field) -> None:
        """
        Apply convolution to fields.

        Parameters
        ----------
        nodal_field : Field
            Input field.
        quadrature_point_field : Field
            Output field.
        """
        self._cpp.apply(_unwrap(nodal_field), _unwrap(quadrature_point_field))

    def transpose(
        self,
        quadrature_point_field: Field,
        nodal_field: Field,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Apply transpose convolution to fields.

        Parameters
        ----------
        quadrature_point_field : Field
            Input field.
        nodal_field : Field
            Output field.
        weights : sequence of float, optional
            Weights for the transpose operation.
        """
        if weights is None:
            weights = []
        self._cpp.transpose(
            _unwrap(quadrature_point_field), _unwrap(nodal_field), list(weights)
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying C++ object."""
        return getattr(self._cpp, name)

    def __repr__(self) -> str:
        return (
            f"GenericLinearOperator("
            f"spatial_dim={self._cpp.spatial_dim}, "
            f"nb_output_components={self._cpp.nb_output_components})"
        )


class LaplaceOperator:
    """
    Python wrapper for muGrid LaplaceOperator.

    A hard-coded, optimized Laplacian stencil operator using the standard
    5-point (2D) or 7-point (3D) finite difference stencil.

    Parameters
    ----------
    spatial_dim : int
        Spatial dimension (2 or 3).
    scale : float, optional
        Scaling factor for the Laplacian. Default is 1.0.

    Examples
    --------
    >>> # Create a 2D Laplacian
    >>> laplace = LaplaceOperator(2, scale=-1.0)
    >>> laplace.apply(input_field, output_field)
    """

    def __init__(self, spatial_dim: int, scale: float = 1.0) -> None:
        if spatial_dim == 2:
            self._cpp = _muGrid.LaplaceOperator2D(scale)
        elif spatial_dim == 3:
            self._cpp = _muGrid.LaplaceOperator3D(scale)
        else:
            raise ValueError(
                f"spatial_dim must be 2 or 3, got {spatial_dim}"
            )

    def apply(self, input_field: Field, output_field: Field) -> None:
        """
        Apply the Laplacian operator.

        Parameters
        ----------
        input_field : Field
            Input field.
        output_field : Field
            Output field.
        """
        self._cpp.apply(_unwrap(input_field), _unwrap(output_field))

    def apply_increment(
        self, input_field: Field, alpha: float, output_field: Field
    ) -> None:
        """
        Apply Laplacian and add scaled result to output: output += alpha * L(input).

        Parameters
        ----------
        input_field : Field
            Input field.
        alpha : float
            Scaling factor.
        output_field : Field
            Output field (updated in-place).
        """
        self._cpp.apply_increment(_unwrap(input_field), alpha, _unwrap(output_field))

    def transpose(
        self,
        input_field: Field,
        output_field: Field,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Apply transpose operator. For Laplacian, this is the same as apply.

        Parameters
        ----------
        input_field : Field
            Input field.
        output_field : Field
            Output field.
        weights : sequence of float, optional
            Weights (unused for Laplacian, included for API compatibility).
        """
        if weights is None:
            weights = []
        self._cpp.transpose(_unwrap(input_field), _unwrap(output_field), list(weights))

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying C++ object."""
        return getattr(self._cpp, name)

    def __repr__(self) -> str:
        return f"LaplaceOperator(spatial_dim={self._cpp.spatial_dim})"


class FEMGradientOperator:
    """
    Python wrapper for muGrid FEMGradientOperator.

    A hard-coded, optimized gradient operator using linear finite element
    shape functions on triangles (2D) or tetrahedra (3D).

    Parameters
    ----------
    spatial_dim : int
        Spatial dimension (2 or 3).
    grid_spacing : sequence of float, optional
        Grid spacing in each direction. Default is [1.0, ...] for each dimension.

    Examples
    --------
    >>> # Create a 2D gradient operator with default grid spacing
    >>> grad = FEMGradientOperator(2)
    >>> grad.apply(nodal_field, quadrature_point_gradient_field)
    """

    def __init__(
        self, spatial_dim: int, grid_spacing: Optional[Sequence[float]] = None,
        element=_muGrid.FEMElement.q1
    ) -> None:
        if grid_spacing is None:
            grid_spacing = []
        gs = list(grid_spacing)
        q1 = element == _muGrid.FEMElement.q1
        if spatial_dim == 2:
            self._cpp = (_muGrid.FEMGradientOperatorQ1_2D(gs) if q1
                         else _muGrid.FEMGradientOperator2D(gs))
        elif spatial_dim == 3:
            self._cpp = (_muGrid.FEMGradientOperatorQ1_3D(gs) if q1
                         else _muGrid.FEMGradientOperator3D(gs))
        else:
            raise ValueError(
                f"spatial_dim must be 2 or 3, got {spatial_dim}"
            )

    def apply(self, nodal_field: Field, quadrature_point_field: Field) -> None:
        """
        Apply the gradient operator (nodal values → quadrature point gradients).

        Parameters
        ----------
        nodal_field : Field
            Input field at nodal points.
        quadrature_point_field : Field
            Output field at quadrature points.
        """
        self._cpp.apply(_unwrap(nodal_field), _unwrap(quadrature_point_field))

    def apply_increment(
        self, nodal_field: Field, alpha: float, quadrature_point_field: Field
    ) -> None:
        """
        Apply gradient and add scaled result to output: output += alpha * grad(input).

        Parameters
        ----------
        nodal_field : Field
            Input field at nodal points.
        alpha : float
            Scaling factor.
        quadrature_point_field : Field
            Output field at quadrature points (updated in-place).
        """
        self._cpp.apply_increment(
            _unwrap(nodal_field), alpha, _unwrap(quadrature_point_field)
        )

    def transpose(
        self,
        quadrature_point_field: Field,
        nodal_field: Field,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Apply transpose (divergence) operator (quadrature points → nodal values).

        Parameters
        ----------
        quadrature_point_field : Field
            Input field at quadrature points.
        nodal_field : Field
            Output field at nodal points.
        weights : sequence of float, optional
            Weights for the transpose operation.
        """
        if weights is None:
            weights = []
        self._cpp.transpose(
            _unwrap(quadrature_point_field), _unwrap(nodal_field), list(weights)
        )

    def transpose_increment(
        self,
        quadrature_point_field: Field,
        alpha: float,
        nodal_field: Field,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Apply transpose and add scaled result: output += alpha * div(input).

        Parameters
        ----------
        quadrature_point_field : Field
            Input field at quadrature points.
        alpha : float
            Scaling factor.
        nodal_field : Field
            Output field at nodal points (updated in-place).
        weights : sequence of float, optional
            Weights for the transpose operation.
        """
        if weights is None:
            weights = []
        self._cpp.transpose_increment(
            _unwrap(quadrature_point_field), alpha, _unwrap(nodal_field), list(weights)
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying C++ object."""
        return getattr(self._cpp, name)

    def __repr__(self) -> str:
        return f"FEMGradientOperator(spatial_dim={self._cpp.spatial_dim})"


class IsotropicStiffnessOperator:
    """
    Python wrapper for the fused isotropic linear-elastic stiffness operator.

    Computes ``force = K @ displacement`` with ``K = B^T C B`` for linear
    triangular (2D) or tetrahedral (3D) elements, where ``C`` is the isotropic
    elasticity tensor parameterized by per-element Lamé fields ``lambda`` and
    ``mu``. The geometry-only matrices are precomputed from the grid spacing at
    construction time.

    Parameters
    ----------
    spatial_dim : int
        Spatial dimension (2 or 3).
    grid_spacing : sequence of float
        Grid spacing in each direction, e.g. ``[hx, hy]`` (2D) or
        ``[hx, hy, hz]`` (3D).

    Examples
    --------
    >>> op = IsotropicStiffnessOperator(2, [0.1, 0.1])
    >>> op.apply(displacement, lambda_field, mu_field, force)
    """

    def __init__(self, spatial_dim: int, grid_spacing: Sequence[float],
                 element=_muGrid.FEMElement.q1) -> None:
        if spatial_dim == 2:
            self._cpp = _muGrid.IsotropicStiffnessOperator2D(
                list(grid_spacing), element)
        elif spatial_dim == 3:
            self._cpp = _muGrid.IsotropicStiffnessOperator3D(
                list(grid_spacing), element)
        else:
            raise ValueError(f"spatial_dim must be 2 or 3, got {spatial_dim}")
        self._spatial_dim = spatial_dim

    def apply(
        self,
        displacement: Field,
        lambda_field: Field,
        mu_field: Field,
        force: Field,
    ) -> None:
        """Compute ``force = K @ displacement``."""
        self._cpp.apply(
            _unwrap(displacement),
            _unwrap(lambda_field),
            _unwrap(mu_field),
            _unwrap(force),
        )

    def apply_increment(
        self,
        displacement: Field,
        lambda_field: Field,
        mu_field: Field,
        alpha: float,
        force: Field,
    ) -> None:
        """Compute ``force += alpha * K @ displacement``."""
        self._cpp.apply_increment(
            _unwrap(displacement),
            _unwrap(lambda_field),
            _unwrap(mu_field),
            alpha,
            _unwrap(force),
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying C++ object."""
        return getattr(self._cpp, name)

    def __repr__(self) -> str:
        return f"IsotropicStiffnessOperator(spatial_dim={self._spatial_dim})"


class IsotropicStiffnessOperator2D(IsotropicStiffnessOperator):
    """Convenience wrapper fixing the spatial dimension to 2."""

    def __init__(self, grid_spacing: Sequence[float],
                 element=_muGrid.FEMElement.q1) -> None:
        super().__init__(2, grid_spacing, element)


class IsotropicStiffnessOperator3D(IsotropicStiffnessOperator):
    """Convenience wrapper fixing the spatial dimension to 3."""

    def __init__(self, grid_spacing: Sequence[float],
                 element=_muGrid.FEMElement.q1) -> None:
        super().__init__(3, grid_spacing, element)


# FileIONetCDF wrapper (only if NetCDF is available)
if hasattr(_muGrid, "FileIONetCDF"):
    _OpenMode = _muGrid.FileIONetCDF.OpenMode

    class FileIONetCDF:
        """
        Python wrapper for muGrid FileIONetCDF.

        Provides NetCDF file I/O for muGrid fields with optional MPI support.

        Parameters
        ----------
        file_name : str
            Path to the NetCDF file.
        open_mode : str or OpenMode, optional
            File open mode: "read", "write", "overwrite", or "append".
            Default is "read".
        communicator : Communicator, optional
            MPI communicator for parallel I/O. Default is serial.

        Examples
        --------
        >>> file = FileIONetCDF("output.nc", open_mode="overwrite")
        >>> file.register_field_collection(field_collection)
        >>> file.append_frame().write()
        """

        # Class-level enum for backwards compatibility
        OpenMode = _OpenMode

        def __init__(
            self,
            file_name: str,
            open_mode: Union[str, "_OpenMode"] = "read",
            communicator: Optional["Communicator"] = None,
        ) -> None:
            from .Parallel import Communicator as CommFactory

            # Handle communicator - ensure we have a C++ Communicator object
            if communicator is None:
                comm = CommFactory()
            elif not isinstance(communicator, _muGrid.Communicator):
                comm = CommFactory(communicator)
            else:
                comm = communicator

            # Parse open mode string
            if isinstance(open_mode, str):
                mode_map = {
                    "read": _OpenMode.Read,
                    "write": _OpenMode.Write,
                    "overwrite": _OpenMode.Overwrite,
                    "append": _OpenMode.Append,
                }
                open_mode_lower = open_mode.lower()
                if open_mode_lower not in mode_map:
                    raise ValueError(
                        f"Invalid open_mode: {open_mode!r}. "
                        f"Must be one of: {list(mode_map.keys())}"
                    )
                open_mode = mode_map[open_mode_lower]

            self._cpp = _muGrid.FileIONetCDF(file_name, open_mode, _unwrap(comm))
            self._file_name = file_name

        def register_field_collection(
            self,
            collection: Union[
                GlobalFieldCollection, LocalFieldCollection, CartesianDecomposition, Any
            ],
            **kwargs,
        ) -> None:
            """
            Register a field collection for I/O.

            Parameters
            ----------
            collection : FieldCollection or CartesianDecomposition
                The field collection to register. If a CartesianDecomposition is
                passed, its underlying field collection is used.
            **kwargs
                Additional arguments passed to C++ register_field_collection.
            """
            # Handle CartesianDecomposition specially - extract its collection
            if isinstance(collection, CartesianDecomposition):
                cpp_collection = collection._cpp.collection
            else:
                cpp_collection = _unwrap(collection)
            self._cpp.register_field_collection(cpp_collection, **kwargs)

        def __getitem__(self, index: int) -> Any:
            """Access a frame by index."""
            return self._cpp[index]

        def __iter__(self):
            """Iterate over frames."""
            return iter(self._cpp)

        def __len__(self) -> int:
            """Return number of frames."""
            return len(self._cpp)

        def __getattr__(self, name: str) -> Any:
            """Delegate attribute access to the underlying C++ object."""
            return getattr(self._cpp, name)

        def __repr__(self) -> str:
            return f"FileIONetCDF({self._file_name!r})"

else:
    # Placeholder when NetCDF is not available
    class FileIONetCDF:  # type: ignore[no-redef]
        """FileIONetCDF is not available (NetCDF support not compiled)."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ModuleNotFoundError(
                "muGrid was installed without NetCDF support. "
                "Rebuild with NetCDF libraries available."
            )
