#!/usr/bin/env python

import scipy
import tqdm
import torch
import numpy as np
import numba
from util import param_util, torch_utils, timer_utils, utils
from sksparse.cholmod import cholesky_AAt

# [x^0, x^1, x^2, x^3]
poly_coef = [[
    np.array([0, 0, 0, 1]) / 6,
    np.array([4, -12, 12, -3]) / 6,
    np.array([-44, 60, -24, 3]) / 6,
    np.array([64, -48, 12, -1]) / 6
], [
    np.array([0, 0, 0, 1]) / 6,
    np.array([9, -27, 27, -7]) / 12,
    np.array([-135, 189, -81, 11]) / 12
], [
    np.array([0, 0, 0, 1]) / 4,
    np.array([8, -24, 24, -7]) / 4
], [
    np.array([0, 0, 0, 1])
]] + [[  # reversed part
    np.array([1, -3, 3, -1]),
], [
    np.array([0, 12, -18, 7]) / 4,
    np.array([8, -12, 6, -1]) / 4
], [
    np.array([0, 0, 18, -11]) / 12,
    np.array([-18, 54, -36, 7]) / 12,
    np.array([27, -27, 9, -1]) / 6
]]


def poly_coef_derivative(coef):
    """
    Take derivative of poly coefficients
    degree(coef) is len(j)-1
    """
    return [[np.arange(1, len(j)) * j[1:] for j in i] for i in coef]


poly_coef_d1 = poly_coef_derivative(poly_coef)
poly_coef_d2 = poly_coef_derivative(poly_coef_d1)


def flatten(l): return np.asarray([j for i in l for j in i])


poly_coefs = (list(map(flatten, [poly_coef, poly_coef_d1, poly_coef_d2])))

multi_segment_to_coefid_cs = np.cumsum([0, 4, 3, 2, 1, 1, 2])
"""
This is a specific table construction. Each row stores the polynomial segments to use
Left: 0 point for x, evaluated as f(x-Left)
Multi: knot multi type for the basis, range -3, -2,-1, 0 (normal)
Segment: from left to right, which of the segment is used here.
b_id: id of the basis, corresponding to control coefficent access
in the end, an additional last row is added for the purpose of evaluating on the last ending knot.
"""
@numba.jit(nopython=True)
def multi_segment_to_coefid(multi, segment):
    return multi_segment_to_coefid_cs[multi] + segment

@numba.jit(nopython=True)
def left_multi_to_c(left, multi):
    return multi + 3 if left == 0 else left + 3

@numba.jit(nopython=True)
def base_from_i_j(length, i,j):
    if i==length: # add last row to avoid numerical isssue
        i -= 1
    segment, multi, left = 2 - j, 0, i + j - 2
    if i + j < 2:
        multi = (i + j) - 2
        segment = i
        left = 0
    elif i + j >= length - 2:
        multi = (i + j) - length + 2
    b_id = left_multi_to_c(left, multi)
    c_id = multi_segment_to_coefid(multi, segment)
    return left, c_id, b_id

@numba.jit(nopython=True)
def table_1d(length=10):
    indices = [[base_from_i_j(length, i,j) for j in range(-1,3)] for i in range(length+1)]
    return indices

@numba.jit(nopython=True)
def table_constructor(length, x):
    return [[base_from_i_j(length, i, j) for j in range(-1, 3)]for i in x]


class BSplineSurface:
    def __init__(self, start, resolution, width, coef=None):
        self.start = np.asarray(start)
        self.scale = 1 / np.asarray(resolution)
        self.table = [np.asarray(table_1d(w)) for w in width]
        self.width = width # number of intervals
        # Note to myself, this coef is control vertices value, not the same as coefs of basis functions.
        # Maybe better be renamed to self.control TODO
        self.coef = coef
        self.cache_factor = None
        self.TH, self.device = False, 'cpu'

    @staticmethod
    def _bspev_and_c_vec(x, table, poly_coef):
        # timer_utils.timer()
        if type(table) is int:
            xi = np.floor(x).astype(np.int64)
            unique_xi, idx = np.unique(xi, return_inverse=True)
            temp_table = np.asarray(table_constructor(table-1, unique_xi))
            tg = temp_table[idx]
            # timer_utils.timer('Make Table')
            # timer_utils.timer()
        else:
            tg = table[np.floor(x).astype(np.int64)]
        degree = poly_coef.shape[1]

        b = np.sum(poly_coef[tg[:, :, 1]] *
                  np.power((x[:, None] - tg[:, :, 0])[:, :, None],
                   np.arange(degree)),
                      axis=2)
        i = tg[:, :, 2]
        # timer_utils.timer('Bspline ev')
        return b, i # there may be many zeros here to prune

    @staticmethod
    def _global_basis_row_vec(x, tables, scale, du=0, dv=0):
        bu, iu = BSplineSurface._bspev_and_c_vec(x[:,0], tables[0], poly_coefs[du])
        bv, iv = BSplineSurface._bspev_and_c_vec(x[:,1], tables[1], poly_coefs[dv])
        outer = np.einsum('bi,bo->bio', bu * (scale[0]**du), bv * (scale[1]**dv)).flatten()
        if type(tables[0]) is int:
            dim1 = tables[1] + 2
        else:
            dim1 = len(tables[1]) + 2
        cols = (np.expand_dims(iu, 2) * dim1 + np.expand_dims(iv, 1)).flatten()
        rows = np.arange(iu.shape[0])[:, None].repeat(iu.shape[1]*iv.shape[1], axis=1).flatten()
        return rows, cols, outer

    @staticmethod
    def _global_basis_d2_row_vec(x, tables, scale):
        row0, col0, data0 = BSplineSurface._global_basis_row_vec(x, tables, scale, 0, 2)
        row1, col1, data1 = BSplineSurface._global_basis_row_vec(x, tables, scale, 2, 0)
        row1 += row0.max()
        return (np.concatenate([row0, row1]),
                np.concatenate([col0, col1]),
                np.concatenate([data0, data1]))


    @staticmethod
    def _global_basis_hessian_row_vec(x, tables, scale): # regularization in [Forsey and Wong 1998]
        row0, col0, data0 = BSplineSurface._global_basis_row_vec(x, tables, scale, 0, 2)
        row1, col1, data1 = BSplineSurface._global_basis_row_vec(x, tables, scale, 2, 0)
        row2, col2, data2 = BSplineSurface._global_basis_row_vec(x, tables, scale, 1, 1)
        row1 += row0.max()+1
        row2 += row1.max()+1
        return (np.concatenate([row0, row1, row2]),
                np.concatenate([col0, col1, col2]),
                np.concatenate([data0, data1, data2*2]))


    def ev(self, x, du=0, dv=0):
        x = self.transform(x)
        bu, iu = BSplineSurface._bspev_and_c_vec(x[:,0], self.table[0], poly_coefs[du])
        bv, iv = BSplineSurface._bspev_and_c_vec(x[:,1], self.table[1], poly_coefs[dv])

        coef_iuv = [c[(np.expand_dims(iu,2), np.expand_dims(iv,1))] for c in self.coef]
        outer = np.einsum('bi,bo->bio', bu* (self.scale[0]**(du)), bv * (self.scale[1]**(dv)))
        return np.hstack([np.sum(c*outer, axis=(1,2)).reshape(-1,1) for c in coef_iuv])

    def interpolate(self, X, f, regularizer=None, cur_ev=None):
        if self.cache_factor is None:
            X = self.transform(X)
            if regularizer is not None:
                regularizer = self.transform(regularizer)
            else:
                width = [w+1 for w in self.width]
                def add_half(l):
                    return [0.5] + list(range(l)) #+ [l-0.5]
                num_reg = 4
                regularizer = [[i,j] for i in add_half(num_reg*width[0]) for j in add_half(num_reg*width[1])]
                regularizer = np.array(regularizer)/num_reg
            row0, col0, data0 = self._global_basis_row_vec(X, self.table, self.scale)
            row1, col1, data1 = self._global_basis_hessian_row_vec(regularizer, self.table, self.scale)
            row1 += row0.max()+1
            reg_scale = 1e-3
            data1 *= reg_scale
            row0, col0, data0 = (np.concatenate([row0, row1]),
                                 np.concatenate([col0, col1]),
                                 np.concatenate([data0, data1]))
            A = scipy.sparse.csr_matrix((data0, (row0, col0)),
                                         shape=(row0.max()+1,
                                                 (self.width[0]+3)*(self.width[1]+3)))

            print(A.shape)

            factor = cholesky_AAt(A.T, beta=1e-10)
            self_cache_factor = factor
            self_cache_At = A.T

        reg = regularizer
        if cur_ev is None:
            reg_vec = np.zeros((reg.shape[0]*3,3))
        else:
            reg_vec =  np.vstack([cur_ev(reg/self.width, dv=2), 
                                cur_ev(reg/self.width,du=2),
                                2*cur_ev(reg/self.width, du=1, dv=1)])
        f2 = reg_scale * reg_vec
        f = np.vstack([f, f2])[:A.shape[0]]
        coef = self_cache_factor(A.T@f)
        res = (A @ coef - f)
        print('Residual', np.linalg.norm(res,axis=0))
        # print('Residual', np.linalg.norm(A.T@A @ coef - A.T@f, axis=0))
        self.coef = [c.reshape(self.width[0]+3, self.width[1]+3) for c in coef.T]
        if self.TH:
            self.coef = [torch.from_numpy(c).to(self.device) for c in self.coef]
        return res[:X.shape[0]]

    def init_TH(self, cuda):
        """
        prepare for PyTorch
        """
        if self.TH:
            return
        self.TH = True
        self.coef = [torch.from_numpy(c) for c in self.coef]
        self.start, self.scale = map(torch.from_numpy, [self.start.astype(np.float64), self.scale])
        self.table = [torch.from_numpy(t) for t in self.table]
        self.ev = self.ev_TH

        if cuda:
            self.start, self.scale = self.start.cuda(), self.scale.cuda()
            self.table = [t.cuda() for t in self.table]
            self.coef = [c.cuda() for c in self.coef]
            self.device = 'cuda'

    def untransform(self, y):
        return self.start + y / self.scale
    def transform(self, x):
        return (x - self.start) * self.scale

    @staticmethod
    def _bspev_and_c_TH_vec(x, table, poly_coef = poly_coefs[0]):
        # if x > len(table) or x < 0: return 0, 0
        tg = table[torch.floor(x).to(torch.long)]
        degree = poly_coef.shape[1]
        poly_coef = poly_coef.to(x.device)

        b = torch.sum(poly_coef[tg[:, :, 1]] *
                      (x[:, None] - tg[:, :, 0].
                       to(torch.float64))[:, :, None].
                      pow(torch.arange(degree,
                                       dtype=torch.float64).to(x.device)),
                      dim=2)
        i = tg[:, :, 2]
        return b, i

    def ev_TH(self, X, du=0, dv=0):
        X = self.transform(X)
        in_range = (X[:,0] < len(self.table[0])) * (X[:,1] < len(self.table[1])) * (X[:,0] >= 0) * (X[:,1] >=0)
        result = torch.zeros(X.size(0), len(self.coef), device=self.device, dtype=X.dtype)
        X = X[in_range]
        
        bu, iu = self._bspev_and_c_TH_vec(X[:, 0], self.table[0], poly_coef=torch.from_numpy(poly_coefs[du]))
        bv, iv = self._bspev_and_c_TH_vec(X[:, 1], self.table[1], poly_coef=torch.from_numpy(poly_coefs[dv]))
        c = [c[iu.unsqueeze(1), iv.unsqueeze(2)] for c in self.coef]
        bu = bu*self.scale[0].pow(du)
        bv = bv.unsqueeze(1)*self.scale[1].pow(dv)
        e = [(bu * torch.bmm(bv, cc).squeeze(1)).sum(dim=1) for cc in c]
        result[in_range] = torch.stack(e, dim=1)
        return result



class SparseBSplineSurface:
    def __init__(self, start, resolution, width, coef=None):
        self.start = np.asarray(start)
        self.scale = 1 / np.asarray(resolution)
        # Note to myself, this coef is control vertices value, not the same as coefs of basis functions.
        # Maybe better be renamed to self.control TODO
        self.width = width
        assert width[0] == width[1], "Some of out-of-range detector relies on square"
        self.dim = [w+3 for w in width]
        self.table = [w+1 for w in width]
        self.coef = coef
        self.cache_factor = None
        self.TH, self.device = False, 'cpu'

    def _get_cell(self, X): # internal, after transformed
        Xi = np.floor(X).astype(np.int64)
        dim0 = self.width[1] + 1 # have the final one 
        cell = Xi[:,0]*dim0 + Xi[:,1]
        return cell

    @staticmethod
    def get_2d_neighbors(X, ran, dim):
        points = np.asarray([x+[i,j] for i in ran for j in ran for x in X])
        points[points<0] = 0
        points[points>=dim] = dim
        points = np.unique(points, axis=0)
        return points

    def remap_row_col_data(self, row,col,data):
        col = self.sparse_col(col)
        remove = np.where(col == -1)
        row, col, data = np.delete(row, remove), np.delete(col,remove), np.delete(data,remove)
        return row,col,data

    def interpolate(self, X, f, cur_ev, regularizer=None, init=None, reg_scale=1e-3):
        timer_utils.timer()
        if self.cache_factor is None:
            X = self.transform(X)
            Xi = np.unique(np.floor(X).astype(np.int64),axis=0)
            dim = (self.table[0]+2), (self.table[1]+2)

            # active_corners = SparseBSplineSurface.get_2d_neighbors(Xi, range(2), self.width[0]) + 3
            # col0 = active_corners[:,0]*dim[0]+active_corners[:,1]
            _,col0, _ = BSplineSurface._global_basis_row_vec(Xi, self.table, self.scale)
            self.unique_col = np.unique(col0)
            unique_map = {i:v for v,i in enumerate(self.unique_col)}
            self.sparse_col = np.vectorize(lambda k:unique_map.get(k,-1))
            zero_points = SparseBSplineSurface.get_2d_neighbors(Xi, range(-4,5), self.width[0])
            self.in_cell_range = np.unique(self._get_cell(np.floor(zero_points)).astype(np.int64))

            timer_utils.timer('Range')
            timer_utils.timer()
            init_points, new_values = init
            init_points = self.transform(init_points)
            in_range = np.isin(self._get_cell(init_points).ravel(), self.in_cell_range)
            init_points = init_points[in_range]
            assert len(init_points)!=0
            new_values = new_values[in_range]
            f = new_values
            
            timer_utils.timer('Val')
            timer_utils.timer()
            total_points = init_points
            row0, col0, data0 = BSplineSurface._global_basis_row_vec(total_points, self.table, self.scale)
            row0, col0, data0 = self.remap_row_col_data(row0, col0, data0)

            timer_utils.timer('Data')

            reg = SparseBSplineSurface.get_2d_neighbors(Xi, np.linspace(-4,4,18), self.width[0])
            row1, col1, data1 = BSplineSurface._global_basis_hessian_row_vec(reg, self.table, self.scale)
            row1, col1, data1 = self.remap_row_col_data(row1, col1, data1)
            row1 += row0.max()+1
            data1 *= reg_scale
            reg_vec =  np.vstack([cur_ev(reg/self.width, dv=2), 
                                  cur_ev(reg/self.width,du=2),
                                  2*cur_ev(reg/self.width, du=1, dv=1)])
            f2 = -reg_scale * reg_vec
            row0, col0, data0 = np.concatenate([row0, row1]), np.concatenate([col0,col1]), np.concatenate([data0,data1])
            # f2 = f

            timer_utils.timer()
            A = scipy.sparse.csr_matrix((data0, (row0, col0)))
            A.eliminate_zeros()
            timer_utils.timer('Matrix')

            factor = cholesky_AAt(A.T, beta=1e-10)
            self_cache_factor = factor
            self_cache_At = A.T


        f = np.vstack([f, f2])[:A.shape[0]]
        coef = self_cache_factor(self_cache_At@f)
        timer_utils.timer('Solve')
        res = (A @ coef - f)
        print('Residual', np.linalg.norm(res,axis=0))
        dim = [t+2 for t in self.table]
        self.coef = [np.concatenate((c,[0])) for c in coef.T]
        if self.TH:
            self.coef = [torch.from_numpy(c).to(self.device) for c in self.coef]
        return in_range, res[:total_points.shape[0]]

    def untransform(self, y):
        return self.start + y / self.scale
    def transform(self, x):
        return (x - self.start) * self.scale

    def ev_TH(self, X, du=0, dv=0):
        X = self.transform(X)
        in_range = (X[:,0] < len(self.table[0])) * (X[:,1] < len(self.table[1])) * (X[:,0] >= 0) * (X[:,1] >=0)
        result = torch.zeros(X.size(0), len(self.coef), device=self.device, dtype=X.dtype)
        X = X[in_range]
        
        bu, iu = self._bspev_and_c_TH_vec(X[:, 0], self.table[0], poly_coef=torch.from_numpy(poly_coefs[du]))
        bv, iv = self._bspev_and_c_TH_vec(X[:, 1], self.table[1], poly_coef=torch.from_numpy(poly_coefs[dv]))

        dim1 = self.tables[1]+2
        ic = iu.unsqueeze(1)*dim1 + iv.unsqueeze(2) # Nx4x4
        raise NotImplementedError

        # c = [c[iu.unsqueeze(1), iv.unsqueeze(2)] for c in self.coef]
        bu = bu*self.scale[0].pow(du)
        bv = bv.unsqueeze(1)*self.scale[1].pow(dv)
        e = [(bu * torch.bmm(bv, cc).squeeze(1)).sum(dim=1) for cc in c]
        result[in_range] = torch.stack(e, dim=1)
        return result


    def ev(self, x, du=0, dv=0):
        dim1 = self.table[1] +2
        x = self.transform(x)

        result = np.zeros((x.shape[0], len(self.coef)))
        # test range, maybe duplicated computation?
        xi = np.floor(x).astype(np.int64)
        in_range = np.isin(self._get_cell(xi).ravel(), self.in_cell_range)
        x = x[in_range]
        if len(x) == 0:
            return result
        bu, iu = BSplineSurface._bspev_and_c_vec(x[:,0], self.table[0], poly_coefs[du])
        bv, iv = BSplineSurface._bspev_and_c_vec(x[:,1], self.table[1], poly_coefs[dv])
        ic = (np.expand_dims(iu, 2) * dim1 + np.expand_dims(iv, 1))
        uic, inv = np.unique(ic, return_inverse=True)
        iuv_mapped = self.sparse_col(uic)[inv].reshape(-1,4,4)
        coef_iuv = [c[iuv_mapped] for c in self.coef]
        outer = np.einsum('bi,bo->bio', bu* (self.scale[0]**(du)), bv * (self.scale[1]**(dv)))
        bu *= self.scale[0]**(du)
        bv *= self.scale[1]**(dv)
        
        result[in_range] = np.hstack([np.einsum('bij,bjk,bk->bi', np.expand_dims(bu,1), c, bv) for c in coef_iuv])
        return result
    
    def decompose(self):
        self.eval_patches = []
        # accelearate by spliitting
        spline_dim = [w+3 for w in self.width]
        xi, yi = np.unravel_index(self.unique_col, spline_dim) 
        corner = xi.min(), yi.min()
        width = xi.max() - xi.min() + 1, yi.max() - yi.min() + 1
        import ipdb;ipdb.set_trace()
        x, y = xi - xi.min(), yi - yi.min()
        x*width[0]*y
        return



def mesh_coord(num):
    x = np.linspace(0, 1, num=num, endpoint=True)
    y = np.linspace(0, 1, num=num, endpoint=True)
    x, y = np.meshgrid(x, y)
    return np.vstack([x.ravel(), y.ravel()]).transpose()


def interp_surface():
    import plotly.offline as py
    import plotly.graph_objs as go

    ns =  np.load('nullspace.npy')
    for i in range(8):
        cbs = BSplineSurface(start=[0, 0], resolution=[0.1, 0.1], width=[10, 10], coef=[ns[:,i].reshape(13,13)])

        X = np.asarray([[i/10, j/10] for i in np.linspace(5, 7, 5) for j in np.linspace(5, 7, 5)])
        Z = np.array([i+j for i,j in X])

        R = np.asarray([[i/10, j/10] for i in np.linspace(0, 10, 20) for j in np.linspace(0, 10, 20)])

        # cbs.interpolate(X, Z, R)

        num =200
        xx = np.linspace(0, 10, num=num, endpoint=True)/10
        yy = xx.copy()
        zz = np.array([[cbs.ev((x,y)).ravel() for x in xx] for y in yy]).reshape(num,num)
        py.plot([go.Scatter3d(x=X[:, 0], y=X[:, 1], z=Z.ravel(), mode='markers')
                    , go.Surface(x=xx,y=yy, z=zz)])

import quadpy
def fit(uv_fit, V, F, size, surf, filename=None):
    timer_utils.timer()
    X = np.asarray([[i, j] for i in np.linspace(0, 1, size * 2) for j in np.linspace(0, 1, size * 2)])
    fid, bc = utils.embree_project_face_bary(uv_fit, F,
                                             source=X, normals=None)
    invalid = np.where(fid == -1)[0]
    fid, bc, X = np.delete(fid, invalid), np.delete(bc, invalid, axis=0), np.delete(X, invalid, axis=0)
    Z = np.einsum('bi, bij->bj', bc, V[F[fid]])

    for _ in range(1):
        uv_fit, _ = utils.upsample(uv_fit, F)
        V, _ = utils.upsample(V,F)
    print('Upsampled to', V.shape)
    X = np.vstack([X, uv_fit, uv_fit[F].mean(axis=1)])
    Z = np.vstack([Z, V, V[F].mean(axis=1)])
    timer_utils.timer('tree')

    timer_utils.timer()
    surf.interpolate(X, Z)
    timer_utils.timer('interpolate')
    if filename is not None:
        surf.serialize(filename)
    return surf

if __name__ == '__main__':
    interp_surface()
    pass
