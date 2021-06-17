#!/usr/bin/env python
import torch as th
import numpy as np
import gmsh
import sys
import meshio
import pdb
import os
folder = os.path.dirname(os.path.abspath(__file__))


def mips_p4(filename, rule):
    '''
    nodes35: (num_tets, 35, 3)
    '''
    m = meshio.read(filename)
    V, T = m.points, m.cells[0].data
    with np.load(os.path.join(folder, f'data/p4_q{rule}_dxyz.npz')) as npl:
        dxyz, weights, pts = map(
            lambda x: th.from_numpy(npl[x]), ['dxyz', 'weights', 'points'])
    print('Total Shape', T.shape)
    quadpoints = dxyz.shape[1]
    split_num = len(T)//5000 + 1
    for T0 in np.array_split(T, split_num):
        print('>> current shape', T0.shape)
        nodes35 = th.from_numpy(V[T0])
        jacs = (dxyz@(nodes35.unsqueeze(1))
                ).transpose(1, 2).reshape(-1, 3, 3)
        dets = th.sum(jacs[:, :, 0] *
                      th.cross(jacs[:, :, 1], jacs[:, :, 2]), dim=-1)
        frob2 = th.sum(jacs.reshape(-1, 9)**2, dim=-1)
        mipses = (frob2/dets**(2/3)).reshape(len(nodes35), -1)
        print(f'dets {dets.min()}, {dets.max()}')
        if (dets.min() < 0):
            dets = dets.reshape(len(nodes35), quadpoints)
            print(f'flip at', np.unravel_index(dets.argmin(), dets.shape))
            continue
        print(f'mipses {mipses.min()}, {mipses.max()}')
        print((mipses@weights).mean())


def gmsh_check(filename: str):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.open(filename)
    gmsh.plugin.setNumber('AnalyseMeshQuality', 'Recompute', 1)
    gmsh.plugin.setNumber('AnalyseMeshQuality', 'JacobianDeterminant', 1)
    gmsh.plugin.run('AnalyseMeshQuality')
    gmsh.finalize()


def gmsh_optimize(filename: str):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.open(filename)
    m = gmsh.model.mesh
    m.optimize("HighOrder")


if __name__ == '__main__':
    import fire
    fire.Fire()
