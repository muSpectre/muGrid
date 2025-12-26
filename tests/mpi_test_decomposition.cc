#include "mpi_context.hh"
#include "tests.hh"

#include "field/field_typed.hh"
#include "field/field_map.hh"
#include "grid/index_ops.hh"
#include "mpi/communicator.hh"
#include "mpi/cartesian_decomposition.hh"

namespace muGrid {
  BOOST_AUTO_TEST_SUITE(mpi_decomposition_test);

  // ----------------------------------------------------------------------
  BOOST_AUTO_TEST_CASE(domain_decomposition_test) {
    auto & comm{MPIContext::get_context().comm};

    // A large enough domain
    DynGridIndex nb_domain_grid_pts{1, 10};

    // Decomposition only along one dimension
    DynGridIndex nb_subdivisions{1, comm.size()};
    DynGridIndex nb_ghosts_left{1, 0};
    DynGridIndex nb_ghosts_right{1, 0};
    CartesianDecomposition cart_decomp{comm, nb_domain_grid_pts,
                                       nb_subdivisions, nb_ghosts_left,
                                       nb_ghosts_right};

    // Check the number of subdomain grid points adds up
    auto nb_subdomain_grid_pts{cart_decomp.get_nb_subdomain_grid_pts_with_ghosts()};
    int dim{1};
    auto res{comm.template sum<Index_t>(nb_subdomain_grid_pts[dim] -
                                        nb_ghosts_left[dim] -
                                        nb_ghosts_right[dim])};
    BOOST_CHECK_EQUAL(res, nb_domain_grid_pts[dim]);
  }

  // ----------------------------------------------------------------------
  BOOST_AUTO_TEST_CASE(inter_subdomain_communicate_test) {
    auto & comm{MPIContext::get_context().comm};

    // Decide the number of subdivisions according to number of processes.
    int nb_process{comm.size()};
    DynGridIndex nb_subdivisions{};
    if (nb_process == 1)
      nb_subdivisions = DynGridIndex{1};
    else if (nb_process == 2)
      nb_subdivisions = DynGridIndex{2};
    else if (nb_process == 4)
      nb_subdivisions = DynGridIndex{2, 2};
    else if (nb_process == 8)
      nb_subdivisions = DynGridIndex{2, 2, 2};
    else
      throw RuntimeError("Not planned for this number of processes.");

    // Decide the size of the whole domain
    int spatial_dims{nb_subdivisions.size()};
    int nb_grid_pts_per_dim{5};
    const DynGridIndex & nb_domain_grid_pts{
        DynGridIndex(spatial_dims, nb_grid_pts_per_dim)};

    // A function to get referrence values
    auto && get_ref_value{
        [nb_grid_pts_per_dim](const DynGridIndex & global_coords) {
          Index_t val{0};
          Index_t coeff{1};
          for (int dim{0}; dim < global_coords.size(); ++dim) {
            val += coeff * global_coords[dim];
            coeff *= nb_grid_pts_per_dim;
          }
          return val;
        }};

    // Create a Cartesian decomposition with ghost buffers
    const DynGridIndex & nb_ghosts_left{DynGridIndex(spatial_dims, 1)};
    const DynGridIndex & nb_ghosts_right{DynGridIndex(spatial_dims, 2)};
    CartesianDecomposition cart_decomp{comm, nb_domain_grid_pts,
                                       nb_subdivisions, nb_ghosts_left,
                                       nb_ghosts_right};

    // Create a field inside the collection for test
    auto & collection{cart_decomp.get_collection()};
    const Index_t nb_components{1};
    const std::string field_name{"test_field"};
    auto & field{dynamic_cast<TypedFieldBase<Real, HostSpace> &>(
        collection.real_field(field_name, nb_components))};

    // Fill the field with some values
    auto subdomain_locations{cart_decomp.get_subdomain_locations_with_ghosts()};
    auto nb_subdomain_grid_pts{cart_decomp.get_nb_subdomain_grid_pts_with_ghosts()};
    auto && field_map{field.get_sub_pt_map(Unknown)};
    CcoordOps::Pixels pixels{nb_subdomain_grid_pts};
    for (auto && pixel_id_coords : pixels.enumerate()) {
      auto && id{std::get<0>(pixel_id_coords)};
      auto && local_coords{std::get<1>(pixel_id_coords)};

      auto && left_check{local_coords - nb_ghosts_left};
      bool is_not_ghost_left{
          std::all_of(left_check.begin(), left_check.end(),
                      [](const auto & elem) { return elem >= 0; })};

      auto && right_check{local_coords + nb_ghosts_right -
                          nb_subdomain_grid_pts};
      bool is_not_ghost_right{
          std::all_of(right_check.begin(), right_check.end(),
                      [](const auto & elem) { return elem < 0; })};

      if (is_not_ghost_left && is_not_ghost_right) {
        auto && global_coords{(subdomain_locations + local_coords) %
                              nb_domain_grid_pts};
        field_map[id] << get_ref_value(global_coords);
      } else {
        field_map[id] << -1;
      }
    }

    // Communicate the ghost buffer
    cart_decomp.communicate_ghosts(field_name);

    // Check the values at the ghost cells are correct
    for (auto && pixel_id_coords : pixels.enumerate()) {
      auto && id(std::get<0>(pixel_id_coords));
      auto && local_coords{std::get<1>(pixel_id_coords)};
      auto && global_coords{(subdomain_locations + local_coords) %
                            nb_domain_grid_pts};
      BOOST_CHECK_EQUAL(field_map[id].coeffRef(0, 0),
                        get_ref_value(global_coords));
    }
  }

  // ----------------------------------------------------------------------
  // Test for multi-step ghost communication when halo > subdomain_size
  BOOST_AUTO_TEST_CASE(multi_step_ghost_communication_test) {
    auto & comm{MPIContext::get_context().comm};

    // This test is designed to exercise the multi-step communication feature.
    // We use a configuration where ghost buffer size > subdomain size in at
    // least one direction.

    int nb_process{comm.size()};
    DynGridIndex nb_subdivisions{};
    int nb_grid_pts_per_dim{10};  // Larger domain to allow realistic halo sizes
    int halo_size{6};             // Large halo that exceeds subdomain size

    if (nb_process == 1) {
      // Serial case: 1D decomposition
      nb_subdivisions = DynGridIndex{1};
    } else if (nb_process == 2) {
      // 1D decomposition with 2 processes
      nb_subdivisions = DynGridIndex{2};
      // Each process gets ~5 grid points, but halo is 6
      // => requires 2 communication steps
    } else if (nb_process == 4) {
      // 2D decomposition (2x2)
      nb_subdivisions = DynGridIndex{2, 2};
      // Each dimension: ~5 grid points per subdomain, halo is 6
      // => requires 2 communication steps per dimension
    } else if (nb_process == 8) {
      // 3D decomposition (2x2x2)
      nb_subdivisions = DynGridIndex{2, 2, 2};
      // Each dimension: ~5 grid points per subdomain, halo is 6
      // => requires 2 communication steps per dimension
    } else {
      throw RuntimeError("Not planned for this number of processes.");
    }

    int spatial_dims{nb_subdivisions.size()};
    const DynGridIndex & nb_domain_grid_pts{
        DynGridIndex(spatial_dims, nb_grid_pts_per_dim)};

    // A function to get reference values based on global coordinates
    auto && get_ref_value{
        [nb_grid_pts_per_dim](const DynGridIndex & global_coords) {
          Index_t val{0};
          Index_t coeff{1};
          for (int dim{0}; dim < global_coords.size(); ++dim) {
            val += coeff * (global_coords[dim] + 100);  // +100 to make values
                                                        // clearly distinct
            coeff *= (nb_grid_pts_per_dim + 100);
          }
          return val;
        }};

    // Create decomposition with large halo buffer
    const DynGridIndex & nb_ghosts_left{DynGridIndex(spatial_dims, halo_size)};
    const DynGridIndex & nb_ghosts_right{DynGridIndex(spatial_dims, halo_size)};
    CartesianDecomposition cart_decomp{comm, nb_domain_grid_pts,
                                       nb_subdivisions, nb_ghosts_left,
                                       nb_ghosts_right};

    // Create a field inside the collection for test
    auto & collection{cart_decomp.get_collection()};
    const Index_t nb_components{1};
    const std::string field_name{"multi_step_test_field"};
    auto & field{dynamic_cast<TypedFieldBase<Real, HostSpace> &>(
        collection.real_field(field_name, nb_components))};

    // Fill the field with reference values
    auto subdomain_locations{cart_decomp.get_subdomain_locations_with_ghosts()};
    auto nb_subdomain_grid_pts{cart_decomp.get_nb_subdomain_grid_pts_with_ghosts()};
    auto && field_map{field.get_sub_pt_map(Unknown)};
    CcoordOps::Pixels pixels{nb_subdomain_grid_pts};

    for (auto && pixel_id_coords : pixels.enumerate()) {
      auto && id{std::get<0>(pixel_id_coords)};
      auto && local_coords{std::get<1>(pixel_id_coords)};

      auto && left_check{local_coords - nb_ghosts_left};
      bool is_not_ghost_left{
          std::all_of(left_check.begin(), left_check.end(),
                      [](const auto & elem) { return elem >= 0; })};

      auto && right_check{local_coords + nb_ghosts_right -
                          nb_subdomain_grid_pts};
      bool is_not_ghost_right{
          std::all_of(right_check.begin(), right_check.end(),
                      [](const auto & elem) { return elem < 0; })};

      if (is_not_ghost_left && is_not_ghost_right) {
        // Real (non-ghost) cell: set to reference value
        auto && global_coords{(subdomain_locations + local_coords) %
                              nb_domain_grid_pts};
        field_map[id] << get_ref_value(global_coords);
      } else {
        // Ghost cell: initialize to invalid value
        field_map[id] << -999999;
      }
    }

    // Communicate the ghost buffer using multi-step communication
    cart_decomp.communicate_ghosts(field_name);

    // Verify that all cells (including ghost cells) have correct reference values
    for (auto && pixel_id_coords : pixels.enumerate()) {
      auto && id(std::get<0>(pixel_id_coords));
      auto && local_coords{std::get<1>(pixel_id_coords)};
      auto && global_coords{(subdomain_locations + local_coords) %
                            nb_domain_grid_pts};
      auto expected_value{get_ref_value(global_coords)};
      auto actual_value{field_map[id].coeffRef(0, 0)};

      BOOST_CHECK_EQUAL(actual_value, expected_value);
    }
  }

  // ----------------------------------------------------------------------
  // Test for edge case: zero grid points with ghost buffers
  BOOST_AUTO_TEST_CASE(zero_grid_points_with_ghosts_test) {
    auto & comm{MPIContext::get_context().comm};

    // This test handles the edge case where an MPI process has zero grid points
    // but still needs to receive ghost data from its neighbors.

    // Only run this test with 2+ processes
    if (comm.size() < 2) {
      return;
    }

    // Create a very small domain that will result in zero grid points on some
    // processes when decomposed
    int nb_process{comm.size()};
    DynGridIndex nb_subdivisions{};
    int nb_grid_pts_per_dim{2};  // Very small domain
    int halo_size{1};

    if (nb_process == 2) {
      nb_subdivisions = DynGridIndex{2};
    } else if (nb_process == 4) {
      nb_subdivisions = DynGridIndex{2, 2};
    } else if (nb_process == 8) {
      nb_subdivisions = DynGridIndex{2, 2, 2};
    } else {
      return;  // Skip for other process counts
    }

    int spatial_dims{nb_subdivisions.size()};
    const DynGridIndex & nb_domain_grid_pts{
        DynGridIndex(spatial_dims, nb_grid_pts_per_dim)};

    auto && get_ref_value{
        [nb_grid_pts_per_dim](const DynGridIndex & global_coords) {
          Index_t val{0};
          Index_t coeff{1};
          for (int dim{0}; dim < global_coords.size(); ++dim) {
            val += coeff * global_coords[dim];
            coeff *= nb_grid_pts_per_dim;
          }
          return val;
        }};

    // Create decomposition with ghost buffers
    const DynGridIndex & nb_ghosts_left{DynGridIndex(spatial_dims, halo_size)};
    const DynGridIndex & nb_ghosts_right{DynGridIndex(spatial_dims, halo_size)};
    CartesianDecomposition cart_decomp{comm, nb_domain_grid_pts,
                                       nb_subdivisions, nb_ghosts_left,
                                       nb_ghosts_right};

    // Create a field
    auto & collection{cart_decomp.get_collection()};
    const Index_t nb_components{1};
    const std::string field_name{"zero_grid_points_field"};
    auto & field{dynamic_cast<TypedFieldBase<Real, HostSpace> &>(
        collection.real_field(field_name, nb_components))};

    // Fill the field
    auto subdomain_locations{cart_decomp.get_subdomain_locations_with_ghosts()};
    auto nb_subdomain_grid_pts{cart_decomp.get_nb_subdomain_grid_pts_with_ghosts()};
    auto && field_map{field.get_sub_pt_map(Unknown)};
    CcoordOps::Pixels pixels{nb_subdomain_grid_pts};

    for (auto && pixel_id_coords : pixels.enumerate()) {
      auto && id{std::get<0>(pixel_id_coords)};
      auto && local_coords{std::get<1>(pixel_id_coords)};

      auto && left_check{local_coords - nb_ghosts_left};
      bool is_not_ghost_left{
          std::all_of(left_check.begin(), left_check.end(),
                      [](const auto & elem) { return elem >= 0; })};

      auto && right_check{local_coords + nb_ghosts_right -
                          nb_subdomain_grid_pts};
      bool is_not_ghost_right{
          std::all_of(right_check.begin(), right_check.end(),
                      [](const auto & elem) { return elem < 0; })};

      if (is_not_ghost_left && is_not_ghost_right) {
        auto && global_coords{(subdomain_locations + local_coords) %
                              nb_domain_grid_pts};
        field_map[id] << get_ref_value(global_coords);
      } else {
        field_map[id] << -1;
      }
    }

    // Communicate ghosts - this should work even with zero grid points on some
    // processes
    cart_decomp.communicate_ghosts(field_name);

    // Verify ghost values are correct
    for (auto && pixel_id_coords : pixels.enumerate()) {
      auto && id(std::get<0>(pixel_id_coords));
      auto && local_coords{std::get<1>(pixel_id_coords)};
      auto && global_coords{(subdomain_locations + local_coords) %
                            nb_domain_grid_pts};
      BOOST_CHECK_EQUAL(field_map[id].coeffRef(0, 0),
                        get_ref_value(global_coords));
    }
  }

  BOOST_AUTO_TEST_SUITE_END();
}  // namespace muGrid
