import numpy as np
import pytest
from NuMPI import MPI
from NuMPI.Testing.Assertions import assert_all_allclose
from NuMPI.Testing.Subdivision import suggest_subdivisions

import muGrid


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
    field = cart_decomp.collection.real_field(field_name)

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
    fc = decomposition.collection

    field = fc.real_field("test-field")

    xg, yg = decomposition.coordsg
    field.pg = xg + 100 * yg

    assert_all_allclose(MPI.COMM_WORLD, field.pg[..., 1:-1, 1:-1], field.p)
    assert_all_allclose(MPI.COMM_WORLD, field.sg[..., 1:-1, 1:-1], field.s)

    # Test setter
    field.pg = np.random.random(field.pg.shape)

    assert_all_allclose(MPI.COMM_WORLD, field.pg[..., 1:-1, 1:-1], field.p)
    assert_all_allclose(MPI.COMM_WORLD, field.sg[..., 1:-1, 1:-1], field.s)


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
    field = cart_decomp.collection.real_field(field_name)

    field.pg = (cart_decomp.icoordsg**2).sum(axis=0)

    # Write to file
    try:
        f = muGrid.FileIONetCDF(filename, muGrid.OpenMode.Overwrite, comm)
    except RuntimeError as e:
        print(f"Opening file for write failed on rank {comm.rank}/{comm.size}")
        raise e
    f.register_field_collection(cart_decomp.collection)
    f.append_frame().write()
    f.close()


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
