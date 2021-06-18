#ifndef CUMIN_CURVE_COMMON_HPP
#define CUMIN_CURVE_COMMON_HPP

#include <Eigen/Core>
#include <map>
#include <utility>

#include "prism/common.hpp"

// template <size_t r, size_t c> using ArrXX = std::array<std::array<int, c>,
// r>; template <size_t i> using Arri = std::array<int, i>;

template <typename T>
struct MatLexComp {
  bool operator()(const T &a, const T &b) const {
    return std::lexicographical_compare(a.data(), a.data() + a.size(), b.data(),
                                        b.data() + b.size());
  }
};

// Generic Map with a lex comparator that works for Eigen array.
template <typename T, typename V>
using MatLexMap = std::map<T, V, MatLexComp<T>>;
using MatLexCompi = MatLexComp<Eigen::VectorXi>;
// Map Codec to some indices
using CodecMap = std::map<Eigen::VectorXi, int, MatLexCompi>;

// sorted(face[cod]) with triangle case.
inline Eigen::VectorXi sort_slice(const Vec3i &face,
                                  const Eigen::VectorXi &cod) {
  Eigen::VectorXi face_cod = cod;
  for (int i = 0; i < cod.size(); i++) {
    face_cod[i] = face[cod[i]];
  }
  std::sort(face_cod.data(), face_cod.data() + face_cod.size());

  return face_cod;
};

// a sorting based global dof ordering
inline auto global_entry_map(const std::vector<Vec3i> &F,
                             const RowMati &codec) {
  auto n = codec.rows(), e = codec.cols();
  CodecMap entries;  // map codec to index of cp.
  std::vector<std::pair<int, int>>
      index_entries;  // which face, and whose node is in the list.
  for (auto i = 0; i < F.size(); i++)
    for (int c = 0; c < n; c++) {
      auto cnt = index_entries.size();
      auto it = entries.try_emplace(sort_slice(F[i], codec.row(c)), cnt);
      if (it.second)  // insertion happened
        index_entries.emplace_back(i, c);
    }
  return std::tuple(index_entries, entries);
};

template <int N, int order>
constexpr auto expand_codec(const std::array<int,N> &cod) {
  std::array<int,order> r{};
  int k=0;
  for (auto i = 0; i < cod.size(); i++) {
    for (auto j = 0; j < cod[i]; j++) {
      r[k++] = i; // repeat each i by cod[i] times.
    }
  }
  return r;
}

inline auto codecs_gen(int order, int num_var) {
  // num_var = 2 for triangle 3 for tetra
  if (num_var == 0) return std::vector<std::vector<int>>({{order}});
  std::vector<std::vector<int>> l;
  for (int i = 0; i < order + 1; i++) {
    auto r = codecs_gen(order - i, num_var - 1);
    for (auto t : r) {
      t.push_back(i);
      l.emplace_back(t);
    }
  }
  auto to_lex = [](auto &a) {
    int sqnorm = 0;
    for (auto i : a) sqnorm += i * i;
    std::vector<int> l(a.rbegin(), a.rend());
    l.insert(l.begin(), -sqnorm);
    return l;
  };
  std::sort(l.begin(), l.end(), [&to_lex](auto &c, auto &d) {
    auto cl = to_lex(c);
    auto dl = to_lex(d);
    return std::lexicographical_compare(cl.begin(), cl.end(), dl.begin(),
                                        dl.end());
  });
  return l;
};

inline auto codec_bc2id(const RowMati& codec_bc){
    auto order = codec_bc(0,0);
    assert (order == codec_bc.row(0).sum());
    auto codec_i = RowMati(codec_bc.rows(), order);
    for (auto i=0; i<codec_bc.rows(); i++) {
      auto c = 0;
      for (auto j=0; j<codec_bc.cols(); j++) {
          for (auto k=0; k<codec_bc(i,j); k++) {
            codec_i(i,c++) = j;
          }
      }
    }
    return codec_i;
  };

inline auto codecs_gen_id(int order, int num_var) {
  // id based codec
  // first row is (0,0,0,0)
  auto cod_bc = codecs_gen(order, num_var);
  RowMati cod;
  vec2eigen(cod_bc, cod);
  auto codec_i = codec_bc2id(cod);
  return codec_i;
}
#endif