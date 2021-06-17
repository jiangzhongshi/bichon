#include <prism/common.hpp>
#include <doctest.h>

#include <igl/read_triangle_mesh.h>
#include <igl/volume.h>
#include <spdlog/spdlog.h>
#include "prism/PrismCage.hpp"
#include "prism/geogram/AABB.hpp"
#include "prism/local_operations/remesh_pass.hpp"

#include "test_common.hpp"

#include <prism/energy/prism_quality.hpp>
constexpr auto total_energy = [](auto& V, auto& F) {
  std::set<int> low_quality_vertices;
  double total_quality = 0;
  double max_quality = 0;
  for (auto [v0, v1, v2] : F) {
    auto q = prism::energy::triangle_quality({V[v0], V[v1], V[v2]});
    total_quality += q;
    max_quality = std::max(max_quality, q);
  }

  spdlog::info("Total Q {} number {}, divide {}, max {}", total_quality, F.size(),
               total_quality / F.size(), max_quality);
  if (max_quality < 10){spdlog::info("SUCCESS"); exit(0);}
  return max_quality;
};
