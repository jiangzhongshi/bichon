#!/usr/bin/env python
# coding: utf-8



import h5py
import sys
from curve import fem_generator

import igl
import numpy as np
import itertools
from pathlib import Path

Path('../python/curve/data').mkdir(parents=True, exist_ok=True)

codec_to_n = lambda co: [k for i,j in enumerate(co) for k in [i]*j]

level=3
usV, usF = igl.upsample(
    np.array([[0, 0.], [1, 0], [0, 1]]), np.array([[0, 1, 2]]), level)
u, v = usV[:, :1], usV[:, 1:]



for order in range(1,5):
    tri_o = fem_generator.basis_info(order=order, nsd=2, derivative=True)
    basis = tri_o['basis']
    tri_o1 = fem_generator.basis_info(order=order+1, nsd=2, derivative=False)
    cod_o1 = tri_o1['codec']

    values_at_nodes = np.hstack(basis(u, v)).T
    sample_deri = np.array([[f(*p) for p in usV] for f in tri_o['basis_d']])
    _, cod_o4_u, cod_o4_v = np.array(tri_o['codec']).T/(order)
    _, cod_o5_u, cod_o5_v = np.array(cod_o1).T/(order+1)
    assert(cod_o5_u.max() == 1)
    with h5py.File(f'../python/curve/data/tri_o{order}_lv{level}.h5','w') as f:
        f['bern'] = values_at_nodes
        f['bern2elevlag'] = np.vstack(basis(cod_o5_u,cod_o5_v))
        f['bern2lag'] = np.vstack(basis(cod_o4_u,cod_o4_v))
        f['deri_u'] = sample_deri[0]
        f['deri_v'] = sample_deri[1]


N=5
sample_pts = np.asarray(list(filter(lambda x: sum(x) == N, itertools.product(range(N+1), repeat=4))))[:,1:]/N


for order in range(1,10):
    tet_o = fem_generator.basis_info(order=order, nsd=3, derivative=False)
    bern_from_lag = tet_o['l2b']
    with h5py.File(f'../python/curve/data/tetra_o{order}_l2b.h5','w') as f:
        f['l2b'] = bern_from_lag

for order in range(6):
    tet_o = fem_generator.basis_info(order=order+1, nsd=3, derivative=True)
    bern_from_lag = tet_o['l2b']
    with h5py.File(f'../python/curve/data/p{order+1}_quniform{N}_dxyz.h5','w') as f:
        f['dxyz'] = np.array([[f(*p) for p in sample_pts] for f in tet_o['basis_d']])@bern_from_lag






