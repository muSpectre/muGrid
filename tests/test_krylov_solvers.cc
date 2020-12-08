/**
 * @file   test_krylov_solvers.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   30 Aug 2020
 *
 * @brief  tests for the iterative krylov solvers
 *
 * Copyright © 2020 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include "tests.hh"
#include "libmugrid/test_goodies.hh"

#include "solver/krylov_solver_cg.hh"
#include "solver/krylov_solver_pcg.hh"

#include <libmugrid/ccoord_operations.hh>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <boost/mpl/list.hpp>

namespace muSpectre {

  struct LinearProblemFixturePerfect {
    constexpr static Index_t ProblemSize{31};

    LinearProblemFixturePerfect()
        : matrix{std::make_shared<DenseEigenAdaptor>(ProblemSize)},
          pre_conditioner{std::make_shared<DenseEigenAdaptor>(ProblemSize)},
          identity{std::make_shared<DenseEigenAdaptor>(
              Eigen::MatrixXd::Identity(ProblemSize, ProblemSize))} {
      // 1-d finite element matrix with constant stiffness
      Real k{5.};
      auto & mat{matrix->get_matrix()};
      for (Index_t i{0}; i < ProblemSize; ++i) {
        mat(i, i) = 2 * k;
        mat(muGrid::CcoordOps::modulo(i - 1, ProblemSize), i) = -k;
        mat(muGrid::CcoordOps::modulo(i + 1, ProblemSize), i) = -k;
      }
      this->pre_conditioner->get_matrix() =
          mat.completeOrthogonalDecomposition().pseudoInverse();
      this->rhs = mat * this->solution;
    }

    constexpr static Index_t get_nb_iter() { return ProblemSize / 2; }
    std::shared_ptr<DenseEigenAdaptor> matrix;
    std::shared_ptr<DenseEigenAdaptor> pre_conditioner;
    std::shared_ptr<DenseEigenAdaptor> identity;
    Eigen::VectorXd rhs{};
    static const Eigen::VectorXd solution;
  };
  constexpr Index_t LinearProblemFixturePerfect::ProblemSize;
  const Eigen::VectorXd LinearProblemFixturePerfect::solution{
      [](const Eigen::VectorXd mat) -> Eigen::VectorXd {
        return (mat.array() - mat.mean()).matrix();
      }(Eigen::VectorXd::Random(LinearProblemFixturePerfect::ProblemSize))};

  struct LinearProblemFixtureDirty : public LinearProblemFixturePerfect {
    using Parent = LinearProblemFixturePerfect;
    LinearProblemFixtureDirty() : Parent{} {
      Eigen::VectorXd deltas{Eigen::VectorXd::Random(Parent::ProblemSize)};
      deltas.array() -= deltas.mean();
      auto & mat{this->matrix->get_matrix()};
      for (Index_t i{0}; i < Parent::ProblemSize; ++i) {
        mat(muGrid::CcoordOps::modulo(i - 1, ProblemSize), i) += deltas(i);
        mat(i, i) -= deltas(i);

        mat(muGrid::CcoordOps::modulo(i + 1, ProblemSize), i) +=
            deltas(muGrid::CcoordOps::modulo(i + 1, ProblemSize));
        mat(i, i) -= deltas(muGrid::CcoordOps::modulo(i + 1, ProblemSize));
      }
      this->rhs = mat * this->solution;
    }

    constexpr static Index_t get_nb_iter() { return ProblemSize - 1; }
  };

  using LinearProblems =
      boost::mpl::list<LinearProblemFixturePerfect, LinearProblemFixtureDirty>;

  BOOST_AUTO_TEST_SUITE(krylov_solvers);

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(cg_test, Fix, LinearProblems, Fix) {
    KrylovSolverCG cg(this->matrix, tol, Fix::ProblemSize, Verbosity::Silent);

    auto && sol{cg.solve(this->rhs)};

    Real error{muGrid::testGoodies::rel_error(sol, this->solution)};

    BOOST_CHECK_LE(error, tol);

    if (not(error <= tol)) {
      std::cout << "nb_iter = " << cg.get_counter() << std::endl;
      std::cout << "matrix:" << std::endl
                << this->matrix->get_matrix() << std::endl
                << std::endl;
      std::cout << "rhs:" << this->rhs.transpose() << std::endl;
      std::cout << "found solution:" << sol.transpose() << std::endl;
      std::cout << "correct sol:" << this->solution.transpose() << std::endl;
    }
    BOOST_CHECK_EQUAL(cg.get_counter(), Fix::get_nb_iter());
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(pcg_test_identity, Fix, LinearProblems,
                                   Fix) {
    KrylovSolverPCG pcg(this->matrix, this->identity, tol, Fix::ProblemSize,
                        Verbosity::Silent);
    auto && pcg_sol{pcg.solve(this->rhs)};

    KrylovSolverCG cg(this->matrix, tol, Fix::ProblemSize, Verbosity::Silent);
    auto && cg_sol{cg.solve(this->rhs)};

    Real error{muGrid::testGoodies::rel_error(pcg_sol, this->solution)};

    BOOST_CHECK_LE(error, tol);

    if (not(error <= tol)) {
      std::cout << "nb_iter = " << pcg.get_counter() << std::endl;
      std::cout << "matrix:" << std::endl
                << this->matrix->get_matrix() << std::endl
                << std::endl;
      std::cout << "identity:" << std::endl
                << this->identity->get_matrix() << std::endl
                << std::endl;
      std::cout << "rhs:" << this->rhs.transpose() << std::endl;
      std::cout << "found solution:" << pcg_sol.transpose() << std::endl;
      std::cout << "correct sol:" << this->solution.transpose() << std::endl;
    }
    error = muGrid::testGoodies::rel_error(pcg_sol, cg_sol);
    BOOST_CHECK_LE(error, tol);

    BOOST_CHECK_EQUAL(pcg.get_counter(), Fix::get_nb_iter());
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE(pcg_test_perfect_pseudoinverse,
                          LinearProblemFixturePerfect) {
    using Fix = LinearProblemFixturePerfect;
    KrylovSolverPCG pcg(this->matrix, this->pre_conditioner, tol,
                        2 * Fix::ProblemSize, Verbosity::Silent);
    auto && pcg_sol{pcg.solve(this->rhs)};

    Real error{muGrid::testGoodies::rel_error(pcg_sol, this->solution)};

    BOOST_CHECK_LE(error, tol);

    if (not(error <= tol)) {
      std::cout << "nb_iter = " << pcg.get_counter() << std::endl;
      std::cout << "matrix:" << std::endl
                << this->matrix->get_matrix() << std::endl
                << std::endl;
      std::cout << "identity:" << std::endl
                << this->identity->get_matrix() << std::endl
                << std::endl;
      std::cout << "rhs:" << this->rhs.transpose() << std::endl;
      std::cout << "found solution:" << pcg_sol.transpose() << std::endl;
      std::cout << "correct sol:" << this->solution.transpose() << std::endl;
    }

    // the perfect preconditioner should converge immediately
    BOOST_CHECK_EQUAL(pcg.get_counter(), 1);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE(pcg_test_dirtry_pseudoinverse,
                          LinearProblemFixtureDirty) {
    using Fix = LinearProblemFixtureDirty;
    constexpr Real DirtyTol{1e-6};
    KrylovSolverPCG pcg(this->matrix, this->pre_conditioner, DirtyTol,
                        2 * Fix::ProblemSize, Verbosity::Silent);
    auto && pcg_sol{pcg.solve(this->rhs)};

    KrylovSolverCG cg(this->matrix, DirtyTol, Fix::ProblemSize,
                      Verbosity::Silent);
    cg.solve(this->rhs);

    Real error{muGrid::testGoodies::rel_error(pcg_sol, this->solution)};

    BOOST_CHECK_LE(error, 10 * DirtyTol);

    if (not(error <= 10 * DirtyTol)) {
      std::cout << "nb_iter = " << pcg.get_counter() << std::endl;
      std::cout << "matrix:" << std::endl
                << this->matrix->get_matrix() << std::endl
                << std::endl;
      std::cout << "identity:" << std::endl
                << this->identity->get_matrix() << std::endl
                << std::endl;
      std::cout << "rhs:" << this->rhs.transpose() << std::endl;
      std::cout << "found solution:" << pcg_sol.transpose() << std::endl;
      std::cout << "correct sol:" << this->solution.transpose() << std::endl;
    }

    // heuristic: preconditioned solver should be much faster than regular
    BOOST_CHECK_LE(pcg.get_counter(), cg.get_counter() / 4);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre
