import meshio
import h5py
import vtk
import numpy as np
import pyvista as pv
import numpy_indexed as npi
from vtk_node_ordering import vtk_node_ordering
from curve.fem_generator import tuple_gen

def convert_cutet(file1, file2):
    # a handy conversion for the default output mesh (p4).
    with h5py.File(file1, 'r') as fp:
        lagr, p4T = fp['lagr'][()], fp['cells'][()]
    if p4T.shape[1] == 35:
        reorder = np.array([0,  1,  2,  3,  4, 16,  5,  7, 18,  9,  8, 17,  6, 13, 19, 10, 15,
                        21, 12, 14, 20, 11, 22, 24, 23, 25, 26, 31, 27, 32, 29, 33, 28, 30, 34])
    elif p4T.shape[1] == 20:
        reorder = np.array([ 0,  1,  2,  3,  4,  5,  7,  9,  8,  6, 13, 10, 15, 12, 14, 11, 16,
       17, 18, 19])
    else:
        assert False, "only hard-coded P3 or P4 for now."
    meshio.gmsh.write(file2,
                 meshio.Mesh(points=lagr, cells=[('tetra35', p4T[:, reorder])]))

def codec_to_points(codec):

    codec_op = np.eye(4)[codec].mean(axis=1)
    points = codec_op@np.vstack([np.zeros(3),np.eye(3)])
    return points

def codec_to_n(co): return [k for i, j in enumerate(co) for k in [i]*j]

def vtk_reordering(order, precision=5):
    vtk_points = vtk_node_ordering(order)
    codec = np.array(tuple_gen(order=order, var_n=3))

    auto_cod_n = np.array([codec_to_n(c) for c in codec])
    auto_points = codec_to_points(auto_cod_n)
    reorder = npi.indices(np.round(auto_points, precision),
                          np.round(vtk_points, precision), axis=0)
    return reorder

def convert_vtk(file1, file2):
    with h5py.File(file1, 'r') as fp:
        lagr, p4T = fp['lagr'][()], fp['cells'][()]

    n_points_per_cell = p4T.shape[1]
    n_cells = p4T.shape[0]

    shape_order = {10:2, 20:3, 35:4, 56:5}
    order = shape_order[n_points_per_cell]
    reorder = vtk_reordering(order)

    p4T = p4T[:,reorder]

    cell_type = np.array([vtk.VTK_LAGRANGE_TETRAHEDRON]*n_cells)
    cells = np.hstack( [np.ones((n_cells,1), dtype=np.int32)*n_points_per_cell, p4T])
    grid = pv.UnstructuredGrid(cells, cell_type, lagr)
    grid.save(file2)


if __name__ == '__main__':
    import sys, os
    file_extension = os.path.splitext(sys.argv[2])[-1]
    if file_extension==".msh":
        convert_cutet(sys.argv[1],sys.argv[2])
    elif file_extension==".vtu":
        convert_vtk(sys.argv[1],sys.argv[2])
    else:
        print(f"file type {file_extension} not supported. Use gmsh (.msh) for P4 or .vtu for arbitrary order")

