// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2017-2019 Zhongshi Jiang <zhongshi@cims.nyu.edu>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include "triangle_tuple.h"

namespace igl {
namespace triangle_tuple {
template <typename Derived>
IGL_INLINE size_t element_access(const Eigen::MatrixBase<Derived> &mat,
                                 size_t i, size_t j) {
  return mat(i, j);
}
template <typename Derived>
IGL_INLINE size_t element_access(const std::vector<Derived> &mat, size_t i,
                                 size_t j) {
  return mat[i][j];
}
}  // namespace triangle_tuple
template <typename DerivedF, typename DerivedFF, typename DerivedFFi>
IGL_INLINE void triangle_tuple_switch_vert(int &fi, int &ei, bool &along,
                                           const DerivedF &F,
                                           const DerivedFF &FF,
                                           const DerivedFFi &FFi) {
  along = !along;
};

template <typename DerivedF, typename DerivedFF, typename DerivedFFi>
IGL_INLINE void triangle_tuple_switch_edge(int &fi, int &ei, bool &along,
                                           const DerivedF &F,
                                           const DerivedFF &FF,
                                           const DerivedFFi &FFi) {
  ei = (ei + ((!along) ? 1 : 2)) % 3;
  triangle_tuple_switch_vert(fi, ei, along, F, FF, FFi);
};

template <typename DerivedF, typename DerivedFF, typename DerivedFFi>
IGL_INLINE void triangle_tuple_switch_face(int &fi, int &ei, bool &along,
                                           const DerivedF &F,
                                           const DerivedFF &FF,
                                           const DerivedFFi &FFi) {
  if (triangle_tuple_is_on_boundary(fi, ei, along, F, FF, FFi)) return;
  int fin = triangle_tuple::element_access(FF, fi, ei);
  int ein = triangle_tuple::element_access(FFi, fi, ei);

  fi = fin;
  ei = ein;
  triangle_tuple_switch_vert(fi, ei, along, F, FF, FFi);
};

template <typename DerivedF, typename DerivedFF, typename DerivedFFi>
IGL_INLINE int triangle_tuple_get_vert(const int &fi, const int &ei,
                                       const bool &along, const DerivedF &F,
                                       const DerivedFF &FF,
                                       const DerivedFFi &FFi) {
  assert(fi >= 0);
  assert(ei >= 0);
  assert(ei <= 2);

  // legacy edge indexing
  return triangle_tuple::element_access(F, fi, along ? ei : (ei + 1) % 3);
};

template <typename DerivedF, typename DerivedFF, typename DerivedFFi>
IGL_INLINE int triangle_tuple_get_edge(const int &fi, const int &ei,
                                       const bool &along, const DerivedF &F,
                                       const DerivedFF &FF,
                                       const DerivedFFi &FFi) {
  return ei;
};

template <typename DerivedF, typename DerivedFF, typename DerivedFFi>
IGL_INLINE int triangle_tuple_get_face(const int &fi, const int &ei,
                                       const bool &along, const DerivedF &F,
                                       const DerivedFF &FF,
                                       const DerivedFFi &FFi) {
  return fi;
};

template <typename DerivedF, typename DerivedFF, typename DerivedFFi>
IGL_INLINE bool triangle_tuple_next_in_one_ring(int &fi, int &ei, bool &along,
                                                const DerivedF &F,
                                                const DerivedFF &FF,
                                                const DerivedFFi &FFi) {
  if (triangle_tuple_is_on_boundary(fi, ei, along, F, FF, FFi)) {
    do {
      triangle_tuple_switch_face(fi, ei, along, F, FF, FFi);
      triangle_tuple_switch_edge(fi, ei, along, F, FF, FFi);
    } while (!triangle_tuple_is_on_boundary(fi, ei, along, F, FF, FFi));
    triangle_tuple_switch_edge(fi, ei, along, F, FF, FFi);
    return false;
  } else {
    triangle_tuple_switch_face(fi, ei, along, F, FF, FFi);
    triangle_tuple_switch_edge(fi, ei, along, F, FF, FFi);
    return true;
  }
};

template <typename DerivedF, typename DerivedFF, typename DerivedFFi>
IGL_INLINE bool triangle_tuple_is_on_boundary(const int &fi, const int &ei,
                                              const bool &along,
                                              const DerivedF &F,
                                              const DerivedFF &FF,
                                              const DerivedFFi &FFi) {
  return triangle_tuple::element_access(FF, fi, ei) == -1;
};

IGL_INLINE bool triangle_tuples_equal(const int &f1, const int &e1,
                                      const bool &a1, const int &f2,
                                      const int &e2, const bool &a2) {
  return f1 == f2 && e1 == e2 && a1 == a2;
};
}  // namespace igl

#ifdef IGL_STATIC_LIBRARY
// Explicit template instantiation
template bool igl::triangle_tuple_next_in_one_ring<
    std::vector<std::array<int, 3ul>, std::allocator<std::array<int, 3ul> > >,
    std::vector<std::array<int, 3ul>, std::allocator<std::array<int, 3ul> > >,
    std::vector<std::array<int, 3ul>, std::allocator<std::array<int, 3ul> > > >(
    int &, int &, bool &,
    std::vector<std::array<int, 3ul>,
                std::allocator<std::array<int, 3ul> > > const &,
    std::vector<std::array<int, 3ul>,
                std::allocator<std::array<int, 3ul> > > const &,
    std::vector<std::array<int, 3ul>,
                std::allocator<std::array<int, 3ul> > > const &);
#endif
