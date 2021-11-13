#!/usr/bin/env python
import meshio
import h5py
import numpy as np

def test_convert():
    import curve.fem_generator as feta

    gmsh_c = feta.basis_info(order=2, nsd=3, force_codec='tetra10')['codec']

    auto_c = feta.basis_info(order=2, nsd=3)['codec']

    np.lexsort(np.array(auto_c).T)[
        feta.invert_permutation(
        np.lexsort(np.array(gmsh_c).T))]

    pts = np.eye(4)[:,1:]

    codec = np.array([list(map(int,x)) for x in '00,11,22,33,01,12,02,03,13,23'.split(',')])

    meshio.write_points_cells('temp.msh', 
                              points=pts[codec].mean(axis=1), 
                              cells=[('tetra10',np.arange(10).reshape(1,-1))])

def convert_cutet(file1, file2):
    # a handy conversion for the default output mesh (p4).
    with h5py.File(file1, 'r') as fp:
        lagr, p4T = fp['lagr'][()], fp['cells'][()]
    num = p4T.shape[1]
    if num == 10: # P2
        reorder = np.array([0, 1, 2, 3, 4, 6, 5, 7, 8, 9])
    elif num == 35:
        reorder = np.array([0,  1,  2,  3,  4, 16,  5,  7, 18,  9,  8, 17,  6, 13, 19, 10, 15,
                        21, 12, 14, 20, 11, 22, 24, 23, 25, 26, 31, 27, 32, 29, 33, 28, 30, 34])
    elif num == 20:
        reorder = np.array([ 0,  1,  2,  3,  4,  5,  7,  9,  8,  6, 13, 10, 15, 12, 14, 11, 16,
       17, 18, 19])
    else:
        assert False, "only hard-coded P3 or P4 for now."
    meshio.write(file2,
                 meshio.Mesh(points=lagr, cells=[(f'tetra{num}', p4T[:, reorder])]))


if __name__ == '__main__':
    import sys
    convert_cutet(sys.argv[1], sys.argv[2])
