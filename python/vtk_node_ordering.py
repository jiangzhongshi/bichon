import numpy as np

# from https://github.com/ju-kreber/paraview-scripts

def np_array(ordering):
    """Wrapper for np.array to simplify common modifications"""
    return np.array(ordering, dtype=np.float64)

def n_verts_between(n, frm, to):
    """Places `n` vertices on the edge between `frm` and `to`"""
    if n <= 0:
        return np.ndarray((0, 3)) # empty
    edge_verts = np.stack((
        np.linspace(frm[0], to[0], num=n+1, endpoint=False, axis=0),   # n+1 since start is included, remove later
        np.linspace(frm[1], to[1], num=n+1, endpoint=False, axis=0),
        np.linspace(frm[2], to[2], num=n+1, endpoint=False, axis=0),
        ), axis=1)
    return edge_verts[1:] # remove start point

def number_triangle(corner_verts, order, skip=False):
    """Outputs the list of coordinates of a right-angled triangle of arbitrary order in the right ordering"""
    if order < 0:
        return np.ndarray((0, 3)) # empty
    if order == 0: # single point, for recursion
        assert np.isclose(corner_verts[0], corner_verts[1]).all() and np.isclose(corner_verts[0], corner_verts[2]).all() # all corners must be same point
        return np.array([corner_verts[0]])

    # first: corner vertices
    coords = np_array(corner_verts) if not skip else np.ndarray((0, 3)) # empty if skip
    if order == 1:
        return coords
    # second: edges
    num_verts_on_edge = order - 1
    edges = [(0,1), (1,2), (2,0)]
    for frm, to in edges:
        coords = np.concatenate([coords, n_verts_between(num_verts_on_edge, corner_verts[frm], corner_verts[to])], axis=0) if not skip else coords # do nothing if skip
    if order == 2:
        return coords
    # third: face, use recursion
    e_x = (corner_verts[1] - corner_verts[0]) / order
    e_y = (corner_verts[2] - corner_verts[0]) / order
    inc = np.array([e_x + e_y, -2*e_x + e_y, e_x -2*e_y]) # adjust corner vertices for recursion
    return np.concatenate([coords, number_triangle(np.array(corner_verts) + inc, order - 3, skip=False)], axis=0) # recursive call, decrease order



def number_tetrahedron(corner_verts, order):
    """Outputs the list of coordinates of a right-angled tetrahedron of arbitrary order in the right ordering"""
    if order < 0:
        return np.ndarray((0, 3)) # empty
    if order == 0: # single point
        assert np.isclose(corner_verts[1], corner_verts[0]).all() and np.isclose(corner_verts[2], corner_verts[0]).all() and np.isclose(corner_verts[3], corner_verts[0]).all() # all corners must be same point
        return np.array([corner_verts[0]])

    # first: corner vertices
    coords = np_array(corner_verts)
    if order == 1:
        return coords
    # second: edges
    num_verts_on_edge = order - 1
    edges = [(0,1), (1,2), (2,0), (0,3), (1,3), (2,3)]
    for frm, to in edges:
        coords = np.concatenate([coords, n_verts_between(num_verts_on_edge, corner_verts[frm], corner_verts[to])], axis=0)
    if order == 2:
        return coords
    # third: faces, use triangle numbering method
    faces = [(0,1,3), (2,3,1), (0,3,2), (0,2,1)]  # x-z, top, y-z, x-y (CCW)  TODO: not as in documentation, beware of future changes!!
    for v_x, v_y, v_z in faces:
        coords = np.concatenate([coords, number_triangle([corner_verts[v_x], corner_verts[v_y], corner_verts[v_z]], order, skip=True)], axis=0) # use number_triangle to number face, but skip corners and edges
    if order == 3:
        return coords
    # fourth: volume, use recursion
    e_x = (corner_verts[1] - corner_verts[0]) / order
    e_y = (corner_verts[2] - corner_verts[0]) / order
    e_z = (corner_verts[3] - corner_verts[0]) / order
    inc = np.array([e_x + e_y + e_z, -3*e_x + e_y + e_z, e_x -3*e_y + e_z, e_x + e_y -3*e_z]) # adjust corner vertices for recursion
    return np.concatenate([coords, number_tetrahedron(np.array(corner_verts) + inc, order - 4)], axis=0) # recursive call, decrease order

def vtk_node_ordering(order):
    return number_tetrahedron(np_array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]]), order)
