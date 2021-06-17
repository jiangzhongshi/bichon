#!/usr/bin/env python

import scipy
import tqdm
import torch
import numpy as np
import numba
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
from util import param_util, torch_utils, timer_utils, utils
from sksparse.cholmod import cholesky_AAt

# [x^0, x^1, x^2]
poly_coef = [[
    np.array([0,0,1])/2,
    np.array([-3,6,-2])/2,
    np.array([9,-6,1])/2
]]

def poly_coef_derivative(coef):
    """
    Take derivative of poly coefficients
    degree(coef) is len(j)-1
    """
    return [[np.arange(1, len(j)) * j[1:] for j in i] for i in coef]


poly_coef_d1 = poly_coef_derivative(poly_coef)


def flatten(l): return np.asarray([j for i in l for j in i])

poly_coefs = (list(map(flatten, [poly_coef, poly_coef_d1])))

"""
This is a specific table construction. Each row stores the polynomial segments to use
# Left: 0 point for x, evaluated as f(x-Left). Always b_id -3
b_id: id of the basis, corresponding to control coefficent access.
Segment: from left to right, which of the segment is used here.
in the end, an additional last row is added for the purpose of evaluating on the last ending knot.
"""

def base_from_i(length, i):
    if i==length: # add last row to avoid numerical isssue
        i -= 1
    return [(i-j+2, j) for j in range(3)] # cp_id, segment


def table_constructor(length, x):
    return [base_from_i(length, i) for i in x]


class BSplineSurface:
    def __init__(self, start, resolution, width, coef=None):
        self.start = np.asarray(start)
        self.scale = 1 / np.asarray(resolution)
        self.width = width # number of intervals
        # Note to myself, this coef is control vertices value, not the same as coefs of basis functions.
        # Maybe better be renamed to self.control TODO
        self.coef = coef
        self.cache_factor = None
        self.TH, self.device = False, 'cpu'

    @staticmethod
    def _bspev_and_c_vec(x, width, poly_coef):
        degree = poly_coef.shape[1]

        xi = np.floor(np.clip(x,0,width-0.1)).astype(np.int64)
        left = xi[:,None] - np.arange(3)[None,:]
        cid = np.tile(np.arange(3), (xi.shape[0],1))

        b = np.sum(poly_coef[cid] *
                  np.power((x[:, None] - left)[:, :, None],
                            np.arange(degree)),
                      axis=2)
        i = left + 2
        return b, i # there may be many zeros here to prune

    @staticmethod
    def _global_basis_row_vec(x, width, scale, du=0, dv=0):
        bu, iu = BSplineSurface._bspev_and_c_vec(x[:,0], width[0], poly_coefs[du])
        bv, iv = BSplineSurface._bspev_and_c_vec(x[:,1], width[1], poly_coefs[dv])
        outer = np.einsum('bi,bo->bio', bu * (scale[0]**du), bv * (scale[1]**dv)).flatten()

        dim1 = (width[1]) + 2 # dim of controls, due to np rowmajor
        cols = (np.expand_dims(iu, 2) * dim1 + np.expand_dims(iv, 1)).flatten()
        rows = np.arange(iu.shape[0])[:, None].repeat(iu.shape[1]*iv.shape[1], axis=1).flatten()
        return rows, cols, outer

    @staticmethod
    def _global_basis_grad_row_vec(x, width, scale): # regularization in [Forsey and Wong 1998]
        row0, col0, data0 = BSplineSurface._global_basis_row_vec(x, width, scale, 0, 1)
        row1, col1, data1 = BSplineSurface._global_basis_row_vec(x, width, scale, 1, 0)
        row1 += row0.max()+1
        return (np.concatenate([row0, row1 ]),
                np.concatenate([col0, col1]),
                np.concatenate([data0,data1]))

    def ev(self, x, du=0, dv=0):
        x = self.transform(x)
        bu, iu = BSplineSurface._bspev_and_c_vec(x[:,0], self.width[0], poly_coefs[du])
        bv, iv = BSplineSurface._bspev_and_c_vec(x[:,1], self.width[1], poly_coefs[dv])

        coef_iuv = [c[(np.expand_dims(iu,2), np.expand_dims(iv,1))] for c in self.coef]
        bu *= self.scale[0]**(du)
        bv *= self.scale[1]**(dv)
        return np.hstack([np.einsum('bij,bjk,bk->bi', np.expand_dims(bu,1), c, bv) for c in coef_iuv])

    def interpolate(self, X, f, regularize=True, cur_ev=None):
        X = self.transform(X)
        width = [w+1 for w in self.width]
        def add_half(l):
            return [0.5] + list(range(l)) #+ [l-0.5]
        num_reg = 4
        row0, col0, data0 = self._global_basis_row_vec(X, self.width, self.scale)
        if regularize:
            regularizer = [[i,j] for i in add_half(num_reg*width[0]) for j in add_half(num_reg*width[1])]
            regularizer = np.array(regularizer)/num_reg

            row1, col1, data1 = self._global_basis_grad_row_vec(regularizer, self.width, self.scale)
            row1 += row0.max()+1
            reg_scale = 1e-3
            data1 *= reg_scale
            row0, col0, data0 = (np.concatenate([row0, row1]),
                                    np.concatenate([col0, col1]),
                                    np.concatenate([data0, data1]))
        A = scipy.sparse.csr_matrix((data0, (row0, col0)),
                                        shape=(row0.max()+1,
                                                (self.width[0]+2)*(self.width[1]+2)))

        print(A.shape)

        factor = cholesky_AAt(A.T)#, beta=1e-10)
        self_cache_factor = factor
        self_cache_At = A.T

        if regularize:
            reg = regularizer
            if cur_ev is None:
                reg_vec = np.zeros((reg.shape[0]*3,3))
            f2 = reg_scale * reg_vec
            f = np.vstack([f, f2])[:A.shape[0]]
        coef = self_cache_factor(A.T@f)
        res = (A @ coef - f)
        print('Residual', np.linalg.norm(res,axis=0))
        self.coef = [c.reshape(self.width[0]+2, self.width[1]+2) for c in coef.T]
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
        self.ev = self.ev_TH

        if cuda:
            self.start, self.scale = self.start.cuda(), self.scale.cuda()
            self.coef = [c.cuda() for c in self.coef]
            self.device = 'cuda'

    def untransform(self, y):
        return self.start + y / self.scale
    def transform(self, x):
        return (x - self.start) * self.scale



class SparseBSplineSurface:
    def __init__(self, start, resolution, width, coef=None):
        self.start = np.asarray(start)
        self.scale = 1 / np.asarray(resolution)
        # Note to myself, this coef is control vertices value, not the same as coefs of basis functions.
        # Maybe better be renamed to self.control TODO
        self.width = width
        assert width[0] == width[1], "Some of out-of-range detector relies on square"
        self.dim = [w+2 for w in width]
        self.coef = coef
        self.cache_factor = None
        self.TH, self.device = False, 'cpu'

    def _get_cell(self, X): # internal, after transformed
        Xi = np.floor(X).astype(np.int64)
        dim0 = self.width[1] + 1 # have the final one 
        col = Xi[:,0]*dim0 + Xi[:,1]
        return col

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

    def interpolate(self, X_offend, cur_ev, regularize=True, init=None, reg_scale=1e-3, active_cp=None):
        timer_utils.timer()
        X_offend = self.transform(X_offend)
        Xi, cnts = np.unique(np.round(X_offend).astype(np.int64),axis=0, return_counts=True)

        if active_cp is None:
            _, col_offend,_ = BSplineSurface._global_basis_row_vec(X_offend, self.width,self.scale)
            col0, cnt = np.unique(col_offend, return_counts=True)
            self.unique_col =col0
        else:
            active_cp = active_cp + 1
            self.unique_col = active_cp[:,0]*self.dim[1] + active_cp[:,1]

        init_points, new_values = init
        init_points = self.transform(init_points)
        range_points = SparseBSplineSurface.get_2d_neighbors(Xi, range(-4,5), self.width[0])
        self.in_cell_range = np.unique(self._get_cell(np.floor(range_points)).astype(np.int64))
        in_range = np.isin(self._get_cell(init_points).ravel(), self.in_cell_range)
        total_points = init_points[in_range]
        assert len(init_points)!=0
        f = new_values[in_range]

        row0, col_total, data0 = BSplineSurface._global_basis_row_vec(total_points, self.width, self.scale)
        
        unique_map = {i:v for v,i in enumerate(self.unique_col)}
        self.sparse_col = np.vectorize(lambda k:unique_map.get(k,-1))

        row0, col0, data0 = self.remap_row_col_data(row0, col_total, data0)

        timer_utils.timer('Data')
        timer_utils.timer()
        if regularize:
            reg = SparseBSplineSurface.get_2d_neighbors(Xi, np.linspace(-4,4,9), self.width[0])
            row1, col1, data1 = BSplineSurface._global_basis_hessian_row_vec(reg, self.width, self.scale)
            row1, col1, data1 = self.remap_row_col_data(row1, col1, data1)
            timer_utils.timer('Regularize')
            timer_utils.timer()
            row1 += row0.max()+1
            data1 *= reg_scale
            reg_vec =  np.vstack([cur_ev(reg/self.width, dv=2), 
                                cur_ev(reg/self.width,du=2),
                                np.sqrt(2)*cur_ev(reg/self.width, du=1, dv=1)])
            f2 = -reg_scale * reg_vec
            row0, col0, data0 = np.concatenate([row0, row1]), np.concatenate([col0,col1]), np.concatenate([data0,data1])
        else:
            f2 = f
        timer_utils.timer('ev')
        timer_utils.timer()
        A = scipy.sparse.csr_matrix((data0, (row0, col0)))
        A.eliminate_zeros()
        timer_utils.timer('Matrix')
        print(A.shape)
        factor = cholesky_AAt(A.T, beta=1e-14)
        self_cache_factor = factor
        self_cache_At = A.T


        f = np.vstack([f, f2])[:A.shape[0]]
        coef = self_cache_factor(self_cache_At@f)
        timer_utils.timer('Solve')
        res = (A @ coef - f)
        print('Residual', np.linalg.norm(res,axis=0))
        if active_cp is not None:
            small_coefs = np.where(np.linalg.norm(coef,axis=1)<1e-3)
            coef = np.delete(coef, small_coefs, axis=0)
            self.unique_col = np.delete(self.unique_col, small_coefs)
            unique_map = {i:v for v,i in enumerate(self.unique_col)}
            self.sparse_col = np.vectorize(lambda k:unique_map.get(k,-1))

        self.coef = [np.concatenate((c,[0])) for c in coef.T]
        if self.TH:
            self.coef = [torch.from_numpy(c).to(self.device) for c in self.coef]
        return in_range, res[:total_points.shape[0]]

    def untransform(self, y):
        return self.start + y / self.scale
    def transform(self, x):
        return (x - self.start) * self.scale

    def ev(self, x, du=0, dv=0):
        dim1 = self.dim[1]
        x = self.transform(x)

        result = np.zeros((x.shape[0], len(self.coef)))
        # test range, maybe duplicated computation?
        xi = np.floor(x).astype(np.int64)
        in_range = np.isin(self._get_cell(xi).ravel(), self.in_cell_range)
        x = x[in_range]
        if len(x) == 0:
            return result
        bu, iu = BSplineSurface._bspev_and_c_vec(x[:,0], self.width[0], poly_coefs[du])
        bv, iv = BSplineSurface._bspev_and_c_vec(x[:,1], self.width[1], poly_coefs[dv])
        ic = (np.expand_dims(iu, 2) * dim1 + np.expand_dims(iv, 1))
        uic, inv = np.unique(ic, return_inverse=True)
        iuv_mapped = self.sparse_col(uic)[inv].reshape(-1,4,4)
        coef_iuv = [c[iuv_mapped] for c in self.coef]
        outer = np.einsum('bi,bo->bio', bu* (self.scale[0]**(du)), bv * (self.scale[1]**(dv)))
        bu *= self.scale[0]**(du)
        bv *= self.scale[1]**(dv)
        
        result[in_range] = np.hstack([np.einsum('bij,bjk,bk->bi', np.expand_dims(bu,1), c, bv) for c in coef_iuv])
        return result

def mesh_coord(num):
    x = np.linspace(0, 1, num=num, endpoint=True)
    y = np.linspace(0, 1, num=num, endpoint=True)
    x, y = np.meshgrid(x, y)
    return np.vstack([x.ravel(), y.ravel()]).transpose()


# import quadpy
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

from util.igl_common import *
import scipy.interpolate
if __name__ == '__main__':
    V, F = Xd(), Xi()
    UV,VN, TF, FN = Xd(),Xd(),Xi(),Xi()
    igl.readOBJ('/home/zhongshi/Workspace/Scaffold-Map/camel_b.obj', V, UV, VN, F, TF, FN)

    V,F,uv_fit = e2p(V),e2p(F),e2p(UV)
    V /= V.max()*4
        # V, uv_fit = npl['V'], npl['uv_opt']
    uv_fit -= uv_fit.min(axis=0)
    uv_fit /= uv_fit.max()
        # V = np.hstack([uv_fit, (uv_fit[:,:1]**2)])

        # uv_fit *=[1,2]
        # disp(uv_fit*[1,2],F)
        # uv_fit, F = utils.triangulate_boundary(uv_fit, F, flag='qa0.01Q')
        # V = np.hstack([uv_fit, (uv_fit[:,:1]**2)])
    # for _ in range(2):
    #     uv_fit, _ = utils.upsample(uv_fit, F)
    #     V, F = utils.upsample(V,F)
    disp(uv_fit,F)
    size = 2**7

    cbs = BSplineSurface(start=[0, 0],
                        resolution=[1 / size, 1 / size],
                        width=[size, size], coef=None)
    cbs.interpolate(uv_fit,V)
    # disp(cbs.ev(uv_fit), F, UV = uv_fit*5)
    uv_vis, F_vis = utils.triangulate_boundary(uv_fit, F, flag='qa0.00001Q')
    X,Z,D = uv_fit,V,cbs.ev(uv_vis)
    disp(cbs.ev(uv_vis), F_vis, UV = uv_vis*size/2)
    disp_fork(np.hstack([uv_vis,D[:,2:]]),F_vis, UV=uv_vis*size/2)  
    pass
