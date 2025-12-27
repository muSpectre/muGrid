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
MemoryLocationStr = Literal["host", "device"]
SubPtMap = Dict[str, int]
Shape = Union[Tuple[int, ...], List[int], Sequence[int]]


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

        Returns
        -------
        Field
            Wrapped field with .s, .p, .sg, .pg accessors.
        """
        cpp_field = self._get_field_collection().real_field(name, components, sub_pt)
        return Field(cpp_field)

    def complex_field(
        self,
        name: str,
        components: Shape = (),
        sub_pt: str = "pixel",
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

        Returns
        -------
        Field
            Wrapped field with .s, .p, .sg, .pg accessors.
        """
        cpp_field = self._get_field_collection().complex_field(name, components, sub_pt)
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


def _parse_memory_location(
    location: Optional[
        Union[MemoryLocationStr, "_muGrid.GlobalFieldCollection.MemoryLocation"]
    ],
) -> "_muGrid.GlobalFieldCollection.MemoryLocation":
    """
    Parse a memory location string to the C++ enum.

    Parameters
    ----------
    location : str, MemoryLocation enum, or None
        Memory location: "host", "device", a MemoryLocation enum value,
        or None (defaults to "host").

    Returns
    -------
    MemoryLocation
        The C++ MemoryLocation enum value.

    Raises
    ------
    ValueError
        If an invalid memory location string is provided.
    """
    MemoryLocation = _muGrid.GlobalFieldCollection.MemoryLocation
    # If it's already a MemoryLocation enum, return it directly
    if isinstance(location, MemoryLocation):
        return location
    if location is None or location == "host":
        return MemoryLocation.Host
    elif location == "device":
        if not _muGrid.has_gpu:
            raise RuntimeError(
                "GPU support is not available. "
                "Rebuild muGrid with MUGRID_ENABLE_CUDA=ON or MUGRID_ENABLE_HIP=ON."
            )
        return MemoryLocation.Device
    else:
        raise ValueError(
            f"Invalid memory_location: {location!r}. " f"Must be 'host' or 'device'."
        )


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
        Ghost cells on low-index side. Default is no ghosts.
    nb_ghosts_right : Sequence[int], optional
        Ghost cells on high-index side. Default is no ghosts.
    memory_location : str, optional
        Memory location for field data: "host" (CPU) or "device" (GPU).
        Default is "host".

    Examples
    --------
    >>> fc = GlobalFieldCollection([64, 64])
    >>> field = fc.real_field("temperature")
    >>> field.p[:] = 300.0  # Set temperature to 300K

    >>> # GPU field collection
    >>> fc_gpu = GlobalFieldCollection([64, 64], memory_location="device")
    """

    # Expose MemoryLocation enum for backwards compatibility
    MemoryLocation = _muGrid.GlobalFieldCollection.MemoryLocation

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
        memory_location = kwargs.get("memory_location")
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
        if nb_ghosts_left is None:
            nb_ghosts_left = []
        if nb_ghosts_right is None:
            nb_ghosts_right = []

        # Build C++ constructor arguments
        cpp_kwargs = {
            "nb_domain_grid_pts": list(nb_grid_pts),
            "sub_pts": nb_sub_pts,
            "nb_ghosts_left": list(nb_ghosts_left),
            "nb_ghosts_right": list(nb_ghosts_right),
        }

        # Handle memory_location
        if memory_location is not None:
            if isinstance(memory_location, str):
                cpp_kwargs["memory_location"] = _parse_memory_location(memory_location)
            else:
                cpp_kwargs["memory_location"] = memory_location

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
    memory_location : str, optional
        Memory location: "host" or "device". Default is "host".
    """

    # Expose MemoryLocation enum for backwards compatibility
    MemoryLocation = _muGrid.LocalFieldCollection.MemoryLocation

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
        memory_location = kwargs.get("memory_location")

        cpp_kwargs = {
            "spatial_dimension": spatial_dim,
            "name": name,
            "nb_sub_pts": nb_sub_pts,
        }

        if memory_location is not None:
            if isinstance(memory_location, str):
                cpp_kwargs["memory_location"] = _parse_memory_location(memory_location)
            else:
                cpp_kwargs["memory_location"] = memory_location

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
        Ghost cells on low-index side. Default is no ghosts.
    nb_ghosts_right : Sequence[int], optional
        Ghost cells on high-index side. Default is no ghosts.
    nb_sub_pts : dict, optional
        Number of sub-points per pixel for each sub-point type.
    memory_location : str, optional
        Memory location: "host" or "device". Default is "host".

    Examples
    --------
    >>> from muGrid import Communicator, CartesianDecomposition
    >>> comm = Communicator()
    >>> decomp = CartesianDecomposition(
    ...     comm,
    ...     nb_domain_grid_pts=[128, 128],
    ...     nb_ghosts_left=[1, 1],
    ...     nb_ghosts_right=[1, 1]
    ... )
    >>> field = decomp.real_field("displacement", components=(3,))
    """

    def __init__(
        self,
        communicator: "Communicator",
        nb_domain_grid_pts: Shape,
        nb_subdivisions: Optional[Shape] = None,
        nb_ghosts_left: Optional[Shape] = None,
        nb_ghosts_right: Optional[Shape] = None,
        nb_sub_pts: Optional[SubPtMap] = None,
        memory_location: Optional[MemoryLocationStr] = None,
    ) -> None:
        from .Parallel import Communicator as CommFactory

        # Ensure we have a proper C++ Communicator object
        if not isinstance(communicator, _muGrid.Communicator):
            communicator = CommFactory(communicator)

        # Handle defaults
        nb_dims = len(nb_domain_grid_pts)
        if nb_subdivisions is None:
            nb_subdivisions = [0] * nb_dims
        if nb_ghosts_left is None:
            nb_ghosts_left = [0] * nb_dims
        if nb_ghosts_right is None:
            nb_ghosts_right = [0] * nb_dims
        if nb_sub_pts is None:
            nb_sub_pts = {}

        mem_loc = _parse_memory_location(memory_location)
        # C++ constructor signature:
        # CartesianDecomposition(comm, nb_domain_grid_pts, nb_subdivisions,
        #                        nb_ghosts_left, nb_ghosts_right, sub_pts={},
        #                        memory_location=Host)
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
        Ghost cells on low-index side of each dimension.
    nb_ghosts_right : Sequence[int], optional
        Ghost cells on high-index side of each dimension.
    nb_sub_pts : dict, optional
        Number of sub-points per pixel.

    Examples
    --------
    >>> engine = FFTEngine([64, 64])
    >>> real_field = engine.real_space_field("displacement", nb_components=3)
    >>> fourier_field = engine.fourier_space_field("displacement_k", nb_components=3)
    >>> engine.fft(real_field, fourier_field)
    >>> engine.ifft(fourier_field, real_field)
    >>> real_field.s[:] *= engine.normalisation
    """

    def __init__(
        self,
        nb_domain_grid_pts: Shape,
        communicator: Optional["Communicator"] = None,
        nb_ghosts_left: Optional[Shape] = None,
        nb_ghosts_right: Optional[Shape] = None,
        nb_sub_pts: Optional[SubPtMap] = None,
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
        if nb_ghosts_left is None:
            nb_ghosts_left = []
        if nb_ghosts_right is None:
            nb_ghosts_right = []
        if nb_sub_pts is None:
            nb_sub_pts = {}

        self._cpp = _muGrid.FFTEngine(
            list(nb_domain_grid_pts),
            _unwrap(comm),
            list(nb_ghosts_left),
            list(nb_ghosts_right),
            nb_sub_pts,
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
        self._cpp.ifft(_unwrap(input_field), _unwrap(output_field))

    def real_space_field(self, name: str, nb_components: int = 1) -> Field:
        """
        Create a real-space field for FFT operations.

        Parameters
        ----------
        name : str
            Unique field name.
        nb_components : int, optional
            Number of components. Default is 1.

        Returns
        -------
        Field
            Wrapped real-valued field with array accessors.
        """
        cpp_field = self._cpp.real_space_field(name, nb_components)
        return Field(cpp_field)

    def fourier_space_field(self, name: str, nb_components: int = 1) -> Field:
        """
        Create a Fourier-space field for FFT operations.

        Parameters
        ----------
        name : str
            Unique field name.
        nb_components : int, optional
            Number of components. Default is 1.

        Returns
        -------
        Field
            Wrapped complex-valued field with array accessors.
        """
        cpp_field = self._cpp.fourier_space_field(name, nb_components)
        return Field(cpp_field)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying C++ object."""
        return getattr(self._cpp, name)

    def __repr__(self) -> str:
        return (
            f"FFTEngine("
            f"real={list(self._cpp.nb_domain_grid_pts)}, "
            f"fourier={list(self._cpp.nb_fourier_grid_pts)})"
        )


class ConvolutionOperator:
    """
    Python wrapper for muGrid ConvolutionOperator.

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
    >>> laplace = ConvolutionOperator([-1, -1], stencil)
    >>> laplace.apply(input_field, output_field)
    """

    def __init__(
        self,
        offset: Shape,
        stencil: ArrayLike,
    ) -> None:
        stencil_arr = np.asarray(stencil, dtype=np.float64, order="F")
        self._cpp = _muGrid.ConvolutionOperator(list(offset), stencil_arr)

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
            f"ConvolutionOperator("
            f"spatial_dim={self._cpp.spatial_dim}, "
            f"nb_operators={self._cpp.nb_operators})"
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
        self._cpp = _muGrid.LaplaceOperator(spatial_dim, scale)

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
        self, spatial_dim: int, grid_spacing: Optional[Sequence[float]] = None
    ) -> None:
        if grid_spacing is None:
            grid_spacing = []
        self._cpp = _muGrid.FEMGradientOperator(spatial_dim, list(grid_spacing))

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
            return f"FileIONetCDF({self._cpp.file_name!r})"

else:
    # Placeholder when NetCDF is not available
    class FileIONetCDF:  # type: ignore[no-redef]
        """FileIONetCDF is not available (NetCDF support not compiled)."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ModuleNotFoundError(
                "muGrid was installed without NetCDF support. "
                "Rebuild with NetCDF libraries available."
            )
