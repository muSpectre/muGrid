import os

import numpy as np
import pytest
from conftest import (
    create_device,
    get_array_module,
    get_test_devices,
    skip_if_gpu_unavailable,
)
from NuMPI import MPI
from NuMPI.Testing.Assertions import assert_all_allclose
from NuMPI.Testing.Subdivision import suggest_subdivisions

import muGrid
from muGrid import GlobalFieldCollection

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


@pytest.mark.parametrize("comm,nb_subdivisions", make_subdivisions())
def test_ghost_communication_rejects_foreign_field(comm, nb_subdivisions):
    """Ghost communication must reject fields from other collections
    (issue #191).

    communicate_ghosts/reduce_ghosts combine the decomposition's own grid
    extents and ghost counts with the strides of the field they are handed.
    For a field from any other collection (e.g. a Fourier-space field of an
    FFT engine, or a stand-alone collection without ghosts) this scrambles
    interior data, so such fields must be rejected up front.
    """
    spatial_dim = len(nb_subdivisions)
    nb_domain_grid_pts = np.full(spatial_dim, 5)
    nb_ghosts = np.full(spatial_dim, 1)
    cart_decomp = muGrid.CartesianDecomposition(
        comm,
        nb_domain_grid_pts.tolist(),
        nb_subdivisions,
        nb_ghosts.tolist(),
        nb_ghosts.tolist(),
    )

    # A field on a separate collection of the same (ghost-padded) subdomain
    # shape, but the collection itself has no ghost regions. NB: the shape
    # must be non-zero on every rank. With more ranks than grid points some
    # ranks own no interior points, and a zero-sized collection would throw
    # on those ranks only, stranding the others in the MPI calls below.
    foreign_collection = GlobalFieldCollection(
        list(cart_decomp.nb_subdomain_grid_pts_with_ghosts)
    )
    foreign_field = foreign_collection.real_field("foreign-field")

    # The guard throws on every rank before any communication is issued,
    # so this cannot deadlock under MPI.
    with pytest.raises(RuntimeError):
        cart_decomp.communicate_ghosts(foreign_field)
    with pytest.raises(RuntimeError):
        cart_decomp.reduce_ghosts(foreign_field)

    # Fields of the decomposition's own collection keep working
    own_field = cart_decomp.real_field("own-field")
    own_field.sg[...] = 1.0
    cart_decomp.communicate_ghosts(own_field)
    cart_decomp.reduce_ghosts(own_field)


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


# Halo geometries for the reduce_ghosts tests below: (grid points per
# direction, left ghosts, right ghosts). With up to 8 ranks per direction the
# small grids give ranks with a single interior point or none at all, so the
# halo spans several ranks (or, for 3 points and 4 ghosts, wraps around the
# whole periodic domain) and the multi-step reduction cascade is exercised.
HALO_GEOMETRIES = [(8, 1, 1), (5, 2, 2), (5, 3, 1), (3, 4, 4)]


@pytest.mark.parametrize("nb_pts_per_dim,ghosts_left,ghosts_right", HALO_GEOMETRIES)
@pytest.mark.parametrize("comm,nb_subdivisions", make_subdivisions())
def test_reduce_ghosts_adjoint_property(
    comm, nb_subdivisions, nb_pts_per_dim, ghosts_left, ghosts_right
):
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
    nb_domain_grid_pts = np.full(spatial_dim, nb_pts_per_dim)
    nb_ghosts_left = np.full(spatial_dim, ghosts_left)
    nb_ghosts_right = np.full(spatial_dim, ghosts_right)

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


@pytest.mark.parametrize("nb_pts_per_dim,ghosts_left,ghosts_right", HALO_GEOMETRIES)
@pytest.mark.parametrize("comm,nb_subdivisions", make_subdivisions())
def test_reduce_ghosts_multiplicity(
    comm, nb_subdivisions, nb_pts_per_dim, ghosts_left, ghosts_right
):
    """Reducing all-ones ghosts yields, at each interior cell, the number of
    ghost cells on all ranks that alias it.

    Unlike the adjoint test this pins down the absolute result and is
    sensitive to contributions being routed to the wrong owner when the halo
    spans several ranks.
    """
    spatial_dim = len(nb_subdivisions)
    nb_domain_grid_pts = np.full(spatial_dim, nb_pts_per_dim)
    nb_ghosts_left = np.full(spatial_dim, ghosts_left)
    nb_ghosts_right = np.full(spatial_dim, ghosts_right)
    cart_decomp = muGrid.CartesianDecomposition(
        comm,
        nb_domain_grid_pts.tolist(),
        nb_subdivisions,
        nb_ghosts_left.tolist(),
        nb_ghosts_right.tolist(),
    )
    field = cart_decomp.real_field("multiplicity")

    # Mask of ghost cells on this rank (with-ghost local coordinates)
    nb_with_ghosts = np.array(cart_decomp.nb_subdomain_grid_pts_with_ghosts)
    local = np.indices(nb_with_ghosts)
    is_ghost = np.zeros(nb_with_ghosts, dtype=bool)
    for dim in range(spatial_dim):
        is_ghost |= local[dim] < nb_ghosts_left[dim]
        is_ghost |= local[dim] >= nb_with_ghosts[dim] - nb_ghosts_right[dim]

    # Ones in the ghosts, zeros in the interior
    field.sg[...] = 0.0
    field.sg[..., is_ghost] = 1.0
    cart_decomp.reduce_ghosts(field)

    # Expected: global histogram of the periodic images of all ghost cells
    location = np.array(cart_decomp.subdomain_locations_with_ghosts)
    global_coords = (local + location.reshape(-1, *([1] * spatial_dim))) % (
        nb_domain_grid_pts.reshape(-1, *([1] * spatial_dim))
    )
    counts = np.zeros(nb_domain_grid_pts, dtype=np.int64)
    np.add.at(counts, tuple(global_coords[:, is_ghost]), 1)
    counts = comm.sum(np.asfortranarray(counts.reshape(-1, 1))).reshape(
        counts.shape
    )

    # A rank that owns no grid points has an empty interior; the comparison
    # below is then trivially empty (ravel rather than reshape, which cannot
    # infer a dimension of an empty array), and only the ghosts are checked.
    interior = ~is_ghost
    expected = counts[tuple(global_coords[:, interior])]
    np.testing.assert_array_equal(
        field.sg[(..., *np.nonzero(interior))].ravel(),
        expected,
        err_msg="Ghost contributions were reduced to the wrong interior cells",
    )
    np.testing.assert_array_equal(
        field.sg[..., is_ghost], 0.0, err_msg="Ghosts not zeroed after reduce"
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


@pytest.mark.parametrize("comm,nb_subdivisions", make_subdivisions())
def test_collection_is_wrapped(comm, nb_subdivisions):
    """
    Test that CartesianDecomposition.collection returns a wrapped
    GlobalFieldCollection.
    """
    spatial_dim = len(nb_subdivisions)
    nb_pts_per_dim = 5
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

    collection = cart_decomp.collection

    # Check that it's the wrapped type, not the raw C++ type
    assert isinstance(collection, GlobalFieldCollection)
    # Check that the wrapper has the expected attributes
    assert hasattr(collection, "_cpp")
    assert hasattr(collection, "nb_grid_pts")


@pytest.mark.parametrize("comm,nb_subdivisions", make_subdivisions())
def test_collection_field_creation(comm, nb_subdivisions):
    """Test that fields created via collection property work correctly."""
    spatial_dim = len(nb_subdivisions)
    nb_pts_per_dim = 5
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

    # Create field via collection property
    collection = cart_decomp.collection
    field = collection.real_field("test_via_collection")

    # Check that field works correctly
    assert field is not None
    # Fill with data
    field.sg[...] = 1.0

    # Communicate ghosts should work with field from collection
    cart_decomp.communicate_ghosts(field)

    # Verify ghost values are properly filled (all ones for constant field)
    np.testing.assert_array_almost_equal(field.sg, 1.0)


@pytest.mark.parametrize("device", get_test_devices())
def test_ghost_operations_devices(comm, device):
    """Device communicate_ghosts and reduce_ghosts match the host results
    (exercises the contiguous-staging device communication, including the
    host-bounce fallback when MPI is not GPU-aware)."""
    skip_if_gpu_unavailable(device)
    xp = get_array_module(device)
    nb_domain_grid_pts = [8, 6, 4]
    subdivisions = [comm.size, 1, 1]
    ghosts = [1, 1, 1]

    buffers = {}
    for dev in ("cpu", device):
        dec = muGrid.CartesianDecomposition(
            comm,
            nb_domain_grid_pts,
            subdivisions,
            ghosts,
            ghosts,
            device=create_device(dev),
        )
        field = dec.real_field("ghost_ops")
        x, y, z = dec.coords
        values = 100 * x + 10 * y + z  # globally unique interior values
        arr = xp.asarray(values) if dev != "cpu" else values

        # communicate_ghosts: ghosts get filled from the periodic neighbors
        field.pg[...] = -1.0
        field.p[...] = arr
        dec.communicate_ghosts(field)
        comm_result = field.sg
        comm_result = (
            comm_result.get() if hasattr(comm_result, "get")
            else np.array(comm_result)
        )

        # reduce_ghosts: ghost values accumulate onto the interior
        field.pg[...] = 2.0
        field.p[...] = arr
        dec.reduce_ghosts(field)
        reduce_result = field.sg
        reduce_result = (
            reduce_result.get() if hasattr(reduce_result, "get")
            else np.array(reduce_result)
        )

        buffers[dev] = (comm_result, reduce_result)

    np.testing.assert_allclose(buffers[device][0], buffers["cpu"][0])
    np.testing.assert_allclose(buffers[device][1], buffers["cpu"][1])
