// example: construct a quadratic program from data
// the QP below is the first quadratic program example in the user manual
#include <doctest.h>

#include <prism/cgal/QP.hpp>
#include <prism/common.hpp>
#include <prism/osqp/osqp_normal.hpp>

TEST_CASE("first qp") {
  RowMatd N(12, 3);
  N << -0.9486833415780438, 5.0279891799382285e-06, 0.3162276353942456,
      -0.999991169948381, 6.681749257755522e-08, 0.004202383283774774,
      -0.9486750614923991, -8.799672176083576e-06, 0.31625247449618255,
      -0.9486750614923991, -8.799672176075144e-06, 0.31625247449618255,
      -0.9486934595912638, -9.103923427002401e-05, 0.3161972666558282,
      2.4836718473564433e-05, 0.9999999994251878, -2.3081635029728415e-05,
      -0.9999804455872972, -0.00625352337133846, -4.3456585218043606e-05, -0.0,
      1.0, 0.0, 0.0, 1.0, 0.0, -0.0, 1.0, 0.0, 1.0, 0.0, -0.0, 1.0, 0.0, 0.0;
  std::vector<int> nb(12);
  for (int i = 0; i < 12; i++) nb[i] = i;
  auto x = prism::cgal::qp_normal(N, nb);
  CAPTURE(x);
  REQUIRE(x.norm() == 0);
}

TEST_CASE("compare qp") {
  RowMatd N(6, 3);
  std::vector<int> nb;
  for (int i = 0; i < N.rows(); i++) nb.push_back(i);
  for (int i = 0; i < 10; i++) {
    N.setRandom();

    auto x = prism::cgal::qp_normal(N, nb);
    auto x2 = prism::osqp_normal(N, nb);
    // auto x2 = prism::qp_normal_igl(N, nb);
    CAPTURE(x);
    CAPTURE(x2);
    REQUIRE_EQ((x).norm(), doctest::Approx(x2.norm()));
    if (x.norm() == 0) continue;
    REQUIRE(x.norm() == doctest::Approx(1.));
    for (int i = 0; i < N.rows(); i++) {
      REQUIRE_GT(N.row(i).dot(x), 0);
    }
  }
}

TEST_CASE("osqp") {}