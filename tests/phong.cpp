#include "prism/common.hpp"
#include "prism/phong/projection.hpp"
#include <Eigen/Core>
#include <array>
#include <doctest.h>
#include <igl/upsample.h>
#include <vector>

TEST_CASE("sanity-phong") {
  // sanity check of the projection within the standard one.
  using namespace Eigen;
  Vec3d t0 = STANDARD_PRISM[0];
  Vec3d t1 = STANDARD_PRISM[1];
  Vec3d t2 = STANDARD_PRISM[2];
  Vec3d t3 = STANDARD_PRISM[3];
  Vec3d t4 = STANDARD_PRISM[4];
  Vec3d t5 = STANDARD_PRISM[5];
  std::array<Vec3d, 6> base_top{t0, t1, t2, t3, t4, t5};

  Vec3d point = (t0 + 2 * t1 + 3 * t2 + t3 + 2 * t4 + 3 * t5) / 12;

  SUBCASE("Not Symmetric") {
    using doctest::Approx;
    for (auto boolcase : {true, false}) {
      std::array<double, 3> tuple;
      REQUIRE(prism::phong::phong_projection(base_top, point, boolcase, tuple));

      CAPTURE(tuple);
      auto [a, b, t] = tuple;
      REQUIRE(a == Approx(1 / 3.));
      REQUIRE(b == Approx(1 / 2.));
      REQUIRE(t == Approx(1 / 2.));
    }
  }
}

TEST_CASE("phong-numeric") {
  // some numeric example for phong
  using namespace Eigen;
  Eigen::MatrixXd N(3, 3), P(3, 3);
  Eigen::RowVector3d point(0.1604241749682691, 0.555453476270909,
                           0.04448819585023656);
  P << 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 2.0, 0.0;
  N << 0.5554048211476403, -0.5249175599301754, 0.8242785326613685,
      0.9314983960859995, 0.9452022278097867, 0.4534492474173122,
      0.21808492552255587, 0.5510530292096933, 0.6416133447590692;
  N = N + P;

  std::array<Vec3d, 6> base_top{P.row(0), P.row(1), P.row(2),
                                N.row(0), N.row(1), N.row(2)};
  std::array<double, 3> tuple;
  REQUIRE(prism::phong::phong_projection(base_top, point, true, tuple));

  auto [a, b, t] = tuple;
  Eigen::RowVector3d bary(a, b, 1 - a - b);
  INFO("Note the following is not carefully verified, just try to keep the "
       "same result as Oct 24");
  CHECK(a == doctest::Approx(0.0304147639));
  CHECK(b == doctest::Approx(0.2501706989));
  CHECK(t == doctest::Approx(0.0782576654));
}

// #include <unsupported/Eigen/MPRealSupport>
// TEST_CASE("multiprec") {
//   using namespace mpfr;
//   using namespace Eigen;
//     // set precision to 256 bits (double has only 53 bits)
//   mpreal::set_default_prec(256);
//   // Declare matrix and vector types with multi-precision scalar type
//   typedef Matrix<mpreal,Dynamic,Dynamic>  MatrixXmp;
//   typedef Matrix<mpreal,Dynamic,1>        VectorXmp;
 
//   MatrixXmp A = MatrixXmp::Random(100,100);
//   VectorXmp b = VectorXmp::Random(100);
 
//   // Solve Ax=b using LU
//   VectorXmp x = A.lu().solve(b);
//   std::cout << "relative error: " << (A*x - b).norm() / b.norm() << std::endl;

// }