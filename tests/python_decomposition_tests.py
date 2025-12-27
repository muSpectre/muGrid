import os

import numpy as np
import pytest
from NuMPI import MPI
from NuMPI.Testing.Assertions import assert_all_allclose
from NuMPI.Testing.Subdivision import suggest_subdivisions

import muGrid

try:
    import netCDF4
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False

# Check if muGrid was built with NetCDF support
HAS_MUGRID_NETCDF = hasattr(muGrid, 'OpenMode')


def get_nb_subdivisions(nb_processes: int):
    subdivision_setup = {
        1: [(1,), (1, 1)],
        2: [
            (2,),
        ],
        4: [
            (4,),
            (2, 2),
        ],
        8: [
            (8,),
            (2, 2, 2),
            (4, 2, 1),
            (8, 1, 1),
        ],
    }
    if nb_processes in subdivision_setup:
        return subdivision_setup[nb_processes]
    else:
        raise NotImplementedError("Not planned for this number of processes.")


def make_subdivisions():
    try:
        from mpi4py import MPI

        comm = muGrid.Communicator(MPI.COMM_WORLD)
    except ImportError:
        comm = muGrid.Communicator()
    nb_processes = comm.size

    # Create a Cartesian decomposition
    nb_subdivisions = get_nb_subdivisions(nb_processes)

    return [(comm, s) for s in nb_subdivisions]


@pytest.mark.parametrize("comm,nb_subdivisions", make_subdivisions())
def test_communicate_ghosts(comm, nb_subdivisions):
    # Create a Cartesian decomposition
    spatial_dim = len(nb_subdivisions)
    nb_pts_per_dim = 5
    nb_domain_grid_pts = np.full(spatial_dim, nb_pts_per_dim)
    nb_ghosts_left = np.full(spatial_dim, 2)
    nb_ghosts_right = np.full(spatial_dim, 2)
    cart_decomp = muGrid.CartesianDecomposition(
        comm,
        nb_domain_grid_pts.tolist(),
        nb_subdivisions,
        nb_ghosts_left.tolist(),
        nb_ghosts_right.tolist(),
    )

    # Idiot check the subdivision
    for i in range(len(nb_subdivisions)):
        s = comm.sum(cart_decomp.nb_subdomain_grid_pts[i])
        assert (
            s
            == cart_decomp.nb_domain_grid_pts[i]
            * np.prod(cart_decomp.nb_subdivisions)
            / cart_decomp.nb_subdivisions[i]
        )

    # Create a field for testing
    field_name = "test_field"
    field = cart_decomp.real_field(field_name)

    # Create reference values
    global_coords = cart_decomp.icoordsg
    weights = np.arange(spatial_dim) + 1
    ref_values = np.einsum("i, i...->...", weights, global_coords)

    # Fill the field, non-ghost with reference values, ghost with some other value
    nb_subdomain_grid_pts = cart_decomp.nb_subdomain_grid_pts_with_ghosts
    for index in np.ndindex(*nb_subdomain_grid_pts):
        is_not_ghost = all(
            idx >= nb_ghosts_left[dim]
            and idx < nb_subdomain_grid_pts[dim] - nb_ghosts_right[dim]
            for dim, idx in enumerate(index)
        )
        if is_not_ghost:
            field.sg[(..., *index)] = ref_values[(..., *index)]
        else:
            field.sg[(..., *index)] = -1

    # Check accessors
    np.testing.assert_array_equal(
        field.s.shape[-spatial_dim:],
        np.array(field.sg.shape)[-spatial_dim:] - nb_ghosts_left - nb_ghosts_right,
    )
    np.testing.assert_array_equal(
        field.p.shape[-spatial_dim:],
        np.array(field.pg.shape)[-spatial_dim:] - nb_ghosts_left - nb_ghosts_right,
    )

    # Communicate ghost cells
    cart_decomp.communicate_ghosts(field)

    # Check values at all grid points
    for index in np.ndindex(*nb_subdomain_grid_pts):
        np.testing.assert_allclose(
            field.sg[(..., *index)],
            ref_values[(..., *index)],
            err_msg=f"Mismatch at {index}",
        )


def test_field_accessors(comm, nb_grid_pts=(128, 128)):
    s = suggest_subdivisions(len(nb_grid_pts), comm.size)

    decomposition = muGrid.CartesianDecomposition(comm, nb_grid_pts, s, (1, 1), (1, 1))

    field = decomposition.real_field("test-field")

    xg, yg = decomposition.coordsg
    field.pg[...] = xg + 100 * yg

    assert_all_allclose(MPI.COMM_WORLD, field.pg[..., 1:-1, 1:-1], field.p)
    assert_all_allclose(MPI.COMM_WORLD, field.sg[..., 1:-1, 1:-1], field.s)

    # Test in-place assignment
    field.pg[...] = np.random.random(field.pg.shape)

    assert_all_allclose(MPI.COMM_WORLD, field.pg[..., 1:-1, 1:-1], field.p)
    assert_all_allclose(MPI.COMM_WORLD, field.sg[..., 1:-1, 1:-1], field.s)


@pytest.mark.skipif(not HAS_MUGRID_NETCDF, reason="muGrid built without NetCDF support")
@pytest.mark.parametrize("comm,nb_subdivisions", make_subdivisions())
def test_io(comm, nb_subdivisions):
    filename = "test_io_output.nc"

    # Create a Cartesian decomposition
    spatial_dim = len(nb_subdivisions)
    nb_pts_per_dim = 5
    nb_domain_grid_pts = np.full(spatial_dim, nb_pts_per_dim)
    nb_ghost_left = np.full(spatial_dim, 1)
    nb_ghost_right = np.full(spatial_dim, 2)
    cart_decomp = muGrid.CartesianDecomposition(
        comm,
        nb_domain_grid_pts.tolist(),
        nb_subdivisions,
        nb_ghost_left.tolist(),
        nb_ghost_right.tolist(),
    )

    # Create a field for testing
    field_name = "test_field"
    field = cart_decomp.real_field(field_name)

    field.pg[...] = (cart_decomp.icoordsg**2).sum(axis=0)

    # Write to file
    try:
        f = muGrid.FileIONetCDF(filename, muGrid.OpenMode.Overwrite, comm)
    except RuntimeError as e:
        print(f"Opening file for write failed on rank {comm.rank}/{comm.size}")
        raise e
    f.register_field_collection(cart_decomp)
    f.append_frame().write()
    f.close()


@pytest.mark.parametrize("comm,nb_subdivisions", make_subdivisions())
def test_reduce_ghosts(comm, nb_subdivisions):
    """Test that reduce_ghosts correctly accumulates ghost contributions to interior."""
    # Create a Cartesian decomposition
    spatial_dim = len(nb_subdivisions)
    nb_pts_per_dim = 5
    nb_domain_grid_pts = np.full(spatial_dim, nb_pts_per_dim)
    nb_ghosts_left = np.full(spatial_dim, 2)
    nb_ghosts_right = np.full(spatial_dim, 2)
    cart_decomp = muGrid.CartesianDecomposition(
        comm,
        nb_domain_grid_pts.tolist(),
        nb_subdivisions,
        nb_ghosts_left.tolist(),
        nb_ghosts_right.tolist(),
    )

    # Create a field for testing
    field = cart_decomp.real_field("reduce_test")

    # Get subdomain dimensions
    nb_subdomain_grid_pts = cart_decomp.nb_subdomain_grid_pts_with_ghosts

    # Fill interior with ones, ghosts with a pattern that should accumulate
    # After reduce_ghosts:
    # - Interior boundary points should have accumulated contributions
    # - Ghost regions should be zeroed
    for index in np.ndindex(*nb_subdomain_grid_pts):
        is_interior = all(
            idx >= nb_ghosts_left[dim]
            and idx < nb_subdomain_grid_pts[dim] - nb_ghosts_right[dim]
            for dim, idx in enumerate(index)
        )
        if is_interior:
            field.sg[(..., *index)] = 1.0
        else:
            # Ghost points get value 2.0 - these should accumulate to boundary interior
            field.sg[(..., *index)] = 2.0

    # Store original interior values for comparison
    interior_before = field.s.copy()

    # Perform ghost reduction
    cart_decomp.reduce_ghosts(field)

    # Check that ghost regions are now zero
    for index in np.ndindex(*nb_subdomain_grid_pts):
        is_ghost = any(
            idx < nb_ghosts_left[dim]
            or idx >= nb_subdomain_grid_pts[dim] - nb_ghosts_right[dim]
            for dim, idx in enumerate(index)
        )
        if is_ghost:
            np.testing.assert_array_equal(
                field.sg[(..., *index)],
                0.0,
                err_msg=f"Ghost region at {index} not zeroed after reduce",
            )

    # For single-process periodic case, boundary interior points should have
    # accumulated ghost contributions
    if comm.size == 1:
        # The accumulated values depend on the ghost structure
        # At minimum, interior values should be >= original values
        assert np.all(field.s >= interior_before), \
            "Interior values should not decrease after reduction"


@pytest.mark.parametrize("comm,nb_subdivisions", make_subdivisions())
def test_reduce_ghosts_multicomponent(comm, nb_subdivisions):
    """Test reduce_ghosts with multi-component fields."""
    spatial_dim = len(nb_subdivisions)
    nb_pts_per_dim = 4
    nb_domain_grid_pts = np.full(spatial_dim, nb_pts_per_dim)
    nb_ghosts_left = np.full(spatial_dim, 1)
    nb_ghosts_right = np.full(spatial_dim, 1)

    cart_decomp = muGrid.CartesianDecomposition(
        comm,
        nb_domain_grid_pts.tolist(),
        nb_subdivisions,
        nb_ghosts_left.tolist(),
        nb_ghosts_right.tolist(),
    )

    # Create a multi-component field (e.g., vector field)
    nb_components = 3
    field = cart_decomp.real_field("vector_field", nb_components)

    # Fill with component-dependent values
    for comp in range(nb_components):
        field.sg[comp, ...] = comp + 1.0

    # Store interior before reduction
    interior_before = field.s.copy()

    # Reduce ghosts
    cart_decomp.reduce_ghosts(field)

    # Check ghost regions are zeroed for all components using slicing
    # Build slices for ghost regions
    # Note: field.sg has shape (nb_components, nb_sub_pts, *spatial_dims)
    nb_subdomain_grid_pts = cart_decomp.nb_subdomain_grid_pts_with_ghosts
    nb_prefix_dims = len(field.sg.shape) - spatial_dim  # components + sub_pts

    # Check left ghost regions
    for dim in range(spatial_dim):
        if nb_ghosts_left[dim] > 0:
            slices = [slice(None)] * len(field.sg.shape)
            slices[nb_prefix_dims + dim] = slice(0, nb_ghosts_left[dim])
            np.testing.assert_array_equal(
                field.sg[tuple(slices)],
                0.0,
                err_msg=f"Left ghost region in dim {dim} not zeroed",
            )

    # Check right ghost regions
    for dim in range(spatial_dim):
        if nb_ghosts_right[dim] > 0:
            slices = [slice(None)] * len(field.sg.shape)
            slices[nb_prefix_dims + dim] = slice(
                nb_subdomain_grid_pts[dim] - nb_ghosts_right[dim], None
            )
            np.testing.assert_array_equal(
                field.sg[tuple(slices)],
                0.0,
                err_msg=f"Right ghost region in dim {dim} not zeroed",
            )


@pytest.mark.parametrize("comm,nb_subdivisions", make_subdivisions())
def test_reduce_ghosts_asymmetric(comm, nb_subdivisions):
    """Test reduce_ghosts with asymmetric ghost buffer sizes."""
    spatial_dim = len(nb_subdivisions)
    nb_pts_per_dim = 6
    nb_domain_grid_pts = np.full(spatial_dim, nb_pts_per_dim)
    # Asymmetric: more ghosts on right than left
    nb_ghosts_left = np.full(spatial_dim, 1)
    nb_ghosts_right = np.full(spatial_dim, 2)

    cart_decomp = muGrid.CartesianDecomposition(
        comm,
        nb_domain_grid_pts.tolist(),
        nb_subdivisions,
        nb_ghosts_left.tolist(),
        nb_ghosts_right.tolist(),
    )

    field = cart_decomp.real_field("asymmetric_test")

    # Fill with pattern: interior=1, left_ghost=10, right_ghost=20
    nb_subdomain_grid_pts = cart_decomp.nb_subdomain_grid_pts_with_ghosts
    for index in np.ndindex(*nb_subdomain_grid_pts):
        is_left_ghost = any(idx < nb_ghosts_left[dim] for dim, idx in enumerate(index))
        is_right_ghost = any(
            idx >= nb_subdomain_grid_pts[dim] - nb_ghosts_right[dim]
            for dim, idx in enumerate(index)
        )
        if is_left_ghost:
            field.sg[(..., *index)] = 10.0
        elif is_right_ghost:
            field.sg[(..., *index)] = 20.0
        else:
            field.sg[(..., *index)] = 1.0

    # Reduce
    cart_decomp.reduce_ghosts(field)

    # Verify ghosts are zeroed
    for index in np.ndindex(*nb_subdomain_grid_pts):
        is_ghost = any(
            idx < nb_ghosts_left[dim]
            or idx >= nb_subdomain_grid_pts[dim] - nb_ghosts_right[dim]
            for dim, idx in enumerate(index)
        )
        if is_ghost:
            np.testing.assert_array_equal(
                field.sg[(..., *index)],
                0.0,
                err_msg=f"Ghost at {index} not zeroed (asymmetric case)",
            )


@pytest.mark.parametrize("comm,nb_subdivisions", make_subdivisions())
def test_reduce_ghosts_adjoint_property(comm, nb_subdivisions):
    """
    Test that reduce_ghosts is the adjoint of communicate_ghosts.

    For periodic BC, if we define:
    - C = communicate_ghosts (fills ghosts from interior)
    - R = reduce_ghosts (accumulates ghosts to interior, zeros ghosts)

    Then R should be the adjoint of C in the sense that:
    <C(x), y> = <x, R(y)>

    where <.,.> is the inner product over the full domain (including ghosts).
    """
    spatial_dim = len(nb_subdivisions)
    # Use enough points to ensure every rank has at least 1 interior point
    # even with maximum subdivision (8 in any dimension for 8 processes)
    nb_pts_per_dim = max(8, max(nb_subdivisions))
    nb_domain_grid_pts = np.full(spatial_dim, nb_pts_per_dim)
    nb_ghosts_left = np.full(spatial_dim, 1)
    nb_ghosts_right = np.full(spatial_dim, 1)

    cart_decomp = muGrid.CartesianDecomposition(
        comm,
        nb_domain_grid_pts.tolist(),
        nb_subdivisions,
        nb_ghosts_left.tolist(),
        nb_ghosts_right.tolist(),
    )

    # Create two fields
    field_x = cart_decomp.real_field("x")
    field_y = cart_decomp.real_field("y")

    # Initialize x with random interior values and zero ghosts
    # Initialize y with random values everywhere (interior + ghosts)
    np.random.seed(42 + comm.rank)
    field_x.sg[...] = np.random.rand(*field_x.sg.shape)
    field_y.sg[...] = np.random.rand(*field_y.sg.shape)

    # Zero out x's ghosts (x represents interior-only data)
    nb_subdomain_grid_pts = cart_decomp.nb_subdomain_grid_pts_with_ghosts
    for index in np.ndindex(*nb_subdomain_grid_pts):
        is_ghost = any(
            idx < nb_ghosts_left[dim]
            or idx >= nb_subdomain_grid_pts[dim] - nb_ghosts_right[dim]
            for dim, idx in enumerate(index)
        )
        if is_ghost:
            field_x.sg[(..., *index)] = 0.0

    # Store original values
    x_original = field_x.sg.copy()  # x with interior values, zero ghosts
    y_original = field_y.sg.copy()  # y with values everywhere

    # Compute C(x) = communicate_ghosts(x)
    cart_decomp.communicate_ghosts(field_x)

    # <C(x), y> - inner product over full ghosted domain
    inner_Cx_y = comm.sum(np.sum(field_x.sg * y_original))

    # Compute R(y) = reduce_ghosts(y)
    # Restore y to original values first
    field_y.sg[...] = y_original
    cart_decomp.reduce_ghosts(field_y)

    # <x, R(y)> - inner product with original x (zero ghosts)
    inner_x_Ry = comm.sum(np.sum(x_original * field_y.sg))

    # The adjoint property: <C(x), y> = <x, R(y)>
    np.testing.assert_allclose(
        inner_Cx_y,
        inner_x_Ry,
        rtol=1e-10,
        err_msg="reduce_ghosts is not adjoint of communicate_ghosts",
    )


@pytest.mark.parametrize("comm,nb_subdivisions", make_subdivisions())
def test_large_ghost_buffers(comm, nb_subdivisions):
    # Create a Cartesian decomposition
    spatial_dim = len(nb_subdivisions)
    nb_pts_per_dim = 5
    nb_domain_grid_pts = np.full(spatial_dim, nb_pts_per_dim)
    nb_ghosts_left = np.full(spatial_dim, 5)
    nb_ghosts_right = np.full(spatial_dim, 5)
    muGrid.CartesianDecomposition(
        comm,
        nb_domain_grid_pts.tolist(),
        nb_subdivisions,
        nb_ghosts_left.tolist(),
        nb_ghosts_right.tolist(),
    )


@pytest.mark.skipif(not HAS_MUGRID_NETCDF, reason="muGrid built without NetCDF support")
@pytest.mark.skipif(not HAS_NETCDF4, reason="netCDF4 not available")
@pytest.mark.parametrize("comm,nb_subdivisions", make_subdivisions())
def test_fileio_netcdf_ghost_offset(comm, nb_subdivisions):
    """Test that FileIONetCDF writes interior data, not ghost-shifted data."""
    spatial_dim = len(nb_subdivisions)
    nb_domain_grid_pts = np.full(spatial_dim, 4)
    nb_ghost_left = np.full(spatial_dim, 1)
    nb_ghost_right = np.full(spatial_dim, 1)

    cart_decomp = muGrid.CartesianDecomposition(
        comm,
        nb_domain_grid_pts.tolist(),
        nb_subdivisions,
        nb_ghost_left.tolist(),
        nb_ghost_right.tolist(),
    )

    field = cart_decomp.real_field("test_field")

    # Fill pg with weighted coordinate sums to make shifts detectable
    global_coords = cart_decomp.icoordsg
    weights = np.array([100**i for i in range(spatial_dim)])
    field.pg[...] = np.einsum("i, i...->...", weights, global_coords)

    expected_interior = field.p.copy()

    filename = "test_ghost_offset.nc"

    try:
        file_io = muGrid.FileIONetCDF(
            filename, muGrid.OpenMode.Overwrite, communicator=comm
        )
        file_io.register_field_collection(
            cart_decomp, field_names=["test_field"]
        )
        file_io.append_frame().write()
        file_io.close()

        comm.barrier()

        # Each rank reads full file and checks its own slice
        with netCDF4.Dataset(filename, "r") as nc:
            stored_data = np.asarray(nc.variables["test_field"][0])

        # Build slice for this rank's subdomain in global array
        slices = tuple(
            slice(loc, loc + size)
            for loc, size in zip(
                cart_decomp.subdomain_locations,
                cart_decomp.nb_subdomain_grid_pts,
            )
        )

        np.testing.assert_array_equal(
            stored_data[slices],
            expected_interior
        )
    finally:
        comm.barrier()
        if comm.rank == 0 and os.path.exists(filename):
            os.unlink(filename)
