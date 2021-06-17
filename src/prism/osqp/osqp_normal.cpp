#include "osqp_normal.hpp"
namespace osqp {
#include <osqp.h>
};
#include <vector>

Vec3d prism::osqp_normal(const RowMatd &N, const std::vector<int> &nb){
  // Load problem data
  using osqp::c_float, osqp::c_int, osqp::OSQPWorkspace, osqp::OSQPSettings;
  using osqp::OSQPSolution, osqp::OSQPTimer, osqp::OSQPData;
  c_float P_x[3] = {1.0, 1.0, 1.0};
  c_int P_nnz = 3;
  c_int P_i[3] = {
      0,
      1,
      2,
  };
  c_int P_p[4] = {
      0,
      1,
      2,
      3,
  };

  c_float q[3] = {0.0, 0.0, 0.};

  c_int n = 3;
  c_int m = nb.size();
  c_int A_nnz = m * n;
  std::vector<c_float> A_x(m * n);
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) A_x[j * m + i] = N(nb[i], j);
  std::vector<c_float> l(m, 1.);
  std::vector<c_float> u(m, std::numeric_limits<c_float>::infinity());
  std::vector<c_int> A_i(m * n), A_p(n + 1);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < m; j++) {
      A_i[i * m + j] = j;
    }
  for (int i = 0; i <= n; i++) A_p[i] = i * m;

  // Exitflag
  c_int exitflag = 0;
  Vec3d res(0, 0, 0);
  {
    // Workspace structures
    OSQPWorkspace *work;
    OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
    OSQPData *data = (OSQPData *)c_malloc(sizeof(OSQPData));

    // Populate data
    if (data) {
      data->n = n;
      data->m = m;
      data->P = ::osqp::csc_matrix(data->n, data->n, P_nnz, P_x, P_i, P_p);
      data->q = q;
      data->A = ::osqp::csc_matrix(data->m, data->n, A_nnz, A_x.data(), A_i.data(),
                           A_p.data());
      data->l = l.data();
      data->u = u.data();
    }

    // Define solver settings as default
    if (settings) ::osqp::osqp_set_default_settings(settings);
    settings->verbose = false;

    // Setup workspace
    exitflag = ::osqp::osqp_setup(&work, data, settings);

    // Solve Problem
    ::osqp::osqp_solve(work);
    if (work->info->status_val == 1) {
      auto sol = work->solution->x;
      res = Vec3d(sol[0], sol[1], sol[2]);
    }
    // Clean workspace
    ::osqp::osqp_cleanup(work);
    if (data) {
      if (data->A) c_free(data->A);
      if (data->P) c_free(data->P);
      c_free(data);
    }
    if (settings) c_free(settings);
  }
  return res.normalized();
};