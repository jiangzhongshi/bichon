#ifndef PRISM_LOCAL_REMESH_WITH_FEATURE_HPP
#define PRISM_LOCAL_REMESH_WITH_FEATURE_HPP
#include "remesh_pass.hpp"

namespace prism::local {
int feature_collapse_pass(PrismCage &, RemeshOptions &);
int feature_slide_pass(PrismCage &, RemeshOptions &);
int feature_split_pass(PrismCage &, RemeshOptions &);
int zig_collapse_pass(PrismCage &pc, RemeshOptions &option);
int zig_slide_pass(PrismCage &pc, RemeshOptions &option);
int zig_split_pass(PrismCage &pc, RemeshOptions &option);
int zig_comb_pass(PrismCage &pc, RemeshOptions &option);
}  // namespace prism::local

#endif