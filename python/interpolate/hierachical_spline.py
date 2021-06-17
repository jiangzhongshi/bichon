import scipy
import tqdm
import torch
import numpy as np
import sys, os
sys.path.append(os.path.dirname(__file__) + '/../')
from util import utils, timer_utils
from util.igl_common import *

# First level, manual.

tol = 1e-3


# def disp(V,F,**kwargs):
#     np.savez(timer_utils.time_string()+'disp.npz', V=V,F=F, **kwargs)
#     input('Saved, continue?')

def to_torch(x):
    return torch.from_numpy(x)

def sampler(uv, V, F, X, current_f=None):
    fid, bc = utils.embree_project_face_bary(uv, F, source=X, normals=None)
    invalid = np.where(fid == -1)[0]
    fid, bc, X = np.delete(fid, invalid), np.delete(bc, invalid, axis=0), np.delete(X, invalid, axis=0)
    Z = np.einsum('bi, bij->bj', bc, V[F[fid]])
    if current_f is not None:
        Z = Z - current_f(X)
    return X, Z

def cp_parent_to_child(parent, width, scale):
    # control points, -1 to w+1
    assert scale==2
    points = np.vstack([scale*parent + [i,j] for i in range(-scale, scale+1) for j in range(-scale, scale+1)])
    points = np.unique(points,axis=0)
    new_width = width*scale
    remove = np.where(np.logical_or(points<-1, points>new_width+1))[0]
    return np.delete(points,remove, axis=0)


def extend_3d(x, val):
    return np.hstack([x, val*np.ones((x.shape[0],1))])


def hill_visualize():
    with np.load('data/camelb_square.npz') as npl:
        F = npl['F']
        V, uv_fit = npl['V'], npl['uv_opt']
        uv_fit -= uv_fit.min(axis=0)
        uv_fit /= uv_fit.max()
        # X, _ = utils.triangulate_boundary(uv_fit, F, flag='qa0.001Q')
        # V = np.hstack([uv_fit, np.abs(np.cos(uv_fit[:,:1]*8))/4])

    # X,Z = sampler(uv_fit, V,F, X)
    # disp(uv_fit,F,P=X)  
    X,Z = uv_fit,V
    ratio = 1
    size = 2**6
    uv_vis, F_vis = utils.triangulate_boundary(uv_fit, F, flag='qa0.000001Q')
    print(uv_vis.shape)
    while ratio > 0.2:
        cbs = simple_cubic_spline.BSplineSurface(start=[0, 0],
                                    resolution=[1 / size, 1 / size],
                                    width=[size, size], coef=None)

        res = cbs.interpolate(X, Z, regularize=True)[:X.shape[0]]
        D = cbs.ev(X)
        print(np.linalg.norm(D[:,2] - X[:,0]**2))
        residual = np.linalg.norm(res, axis=1)
        print(residual.max())
        disp(cbs.ev(uv_vis),F_vis, UV=size/2*uv_vis)

        X_offend = X[residual>tol]
        print('Offend', X_offend.shape)
        Xi_offend = (cbs.transform(X_offend) * 2).astype(np.int64) # for the subdivided
        # subcell_offend = np.unique(np.floor(Xi_offend).astype(np.int64),axis=0)
        subcell_offend = np.unique([x+[i,j] for i in np.linspace(-4,4,9) for j in np.linspace(-4,4,9) for x in Xi_offend],axis=0)
        subcell_offend[subcell_offend<0] = 0; subcell_offend[subcell_offend > size+2] = size+2;
        subcell_offend = np.unique(subcell_offend,axis=0)
        ratio = len(subcell_offend) / (2*(size+3))**2
        print(size, ratio)
        size *= 2 # if not enough sparsity, not worth hiearchy, but rerun with larger size
        disp(cbs.ev(uv_vis),F_vis)#, P = V, UV=size/2*uv_vis)
        
        # temp visual code.
        Dv = cbs.ev(uv_vis)
        disp_fork(np.hstack([uv_vis,Dv[:,1:2]]),F_vis, P = np.hstack([X,Z[:,1:2]]), UV=uv_vis*size/2)
        disp_fork(np.hstack([uv_fit,Z[:,1:2]]),F, P = np.hstack([X,Z[:,1:2]]), UV= uv_fit*size/2) 

import autogen
def compute_energy(Jx, Jyinv, A):
    x0, x1, x2, x3 = Jx.T #[:, 0], J[:, 1], J[:, 2], J[:, 3]
    y0, y1, y3 = Jyinv.T #[:, 0], Jyinv[:, 1], Jyinv[:, 2], Jyinv[:, 3]
    y2 = 0

    # Note here y2 should always be zero.
    # J = Yi . X.t(), note that this expression already handles the transpose of X
    sym_e = autogen.Energy['ssd'](x0, x1, x2, x3, y0, y1, y2, y3)    
    energy = sym_e
    return energy  # / A.sum()

def main():
    with np.load('data/camelb_square.npz') as npl:
        F = npl['F']
        V, uv_fit = npl['V'], npl['uv_opt']
        uv_fit -= uv_fit.min(axis=0)
        uv_fit /= uv_fit.max()
        # X, _ = utils.triangulate_boundary(uv_fit, F, flag='qa0.001Q')
        # V = np.hstack([uv_fit, np.abs(np.cos(uv_fit[:,:1]*8))/4])
    for _ in range(0):
        uv_fit, _ = utils.upsample(uv_fit, F)
        V, F = utils.upsample(V,F)
    # disp(V,F)

    # X,Z = sampler(uv_fit, V,F, X)
    # disp(uv_fit,F,P=X)  
    X,Z = uv_fit,V
    ratio = 1
    size = 2**6
    uv_vis, F_vis = utils.triangulate_boundary(uv_fit, F, flag='qa0.000001Q')
    print(uv_vis.shape)
    while ratio > 0.2:
        cbs = simple_cubic_spline.BSplineSurface(start=[0, 0],
                                    resolution=[1 / size, 1 / size],
                                    width=[size, size], coef=None)

        res = cbs.interpolate(X, Z, regularize=True)[:X.shape[0]]
        D = cbs.ev(X)
        print(np.linalg.norm(D[:,2] - X[:,0]**2))
        residual = np.linalg.norm(res, axis=1)
        print(residual.max())
        disp(cbs.ev(uv_vis),F_vis, UV=size/2*uv_vis)

        X_offend = X[residual>tol]
        print('Offend', X_offend.shape)
        Xi_offend = (cbs.transform(X_offend) * 2).astype(np.int64) # for the subdivided
        # subcell_offend = np.unique(np.floor(Xi_offend).astype(np.int64),axis=0)
        subcell_offend = np.unique([x+[i,j] for i in np.linspace(-4,4,9) for j in np.linspace(-4,4,9) for x in Xi_offend],axis=0)
        subcell_offend[subcell_offend<0] = 0; subcell_offend[subcell_offend > size+2] = size+2;
        subcell_offend = np.unique(subcell_offend,axis=0)
        ratio = len(subcell_offend) / (2*(size+3))**2
        print(size, ratio)
        size *= 2 # if not enough sparsity, not worth hiearchy, but rerun with larger size
        disp(cbs.ev(uv_vis),F_vis)#, P = V, UV=size/2*uv_vis)
        break

    current_patches = [cbs]

    level = 1
    active_cp = np.array([[i,j] for i in range(size//2+3) for j in range(size//2+3)]) - 1
    while len(X_offend) > 0:
        def current_ev(x, du=0, dv=0):
            return sum(c.ev(x, du, dv) for c in current_patches)
        timer_utils.timer()
        residual = Z-current_ev(X)
        timer_utils.timer('Residual')

        cbs_sparse = cubic_spline.SparseBSplineSurface(start=[0, 0], resolution=[1 / size, 1 / size], width=[size, size], coef=None)
        reg_weight = 1e-5#/10**(level)
        active_cp = cp_parent_to_child(active_cp,width=size/2, scale=2)
        disp(uv_fit, F, UV=uv_fit*size/2, P=np.vstack([extend_3d(active_cp/size, 0), extend_3d(X_offend,1)]))
        c_points, c_res = cbs_sparse.interpolate(X, regularize=True, init=(X,residual), cur_ev=current_ev, reg_scale=reg_weight, active_cp=active_cp)

        active_cp = np.vstack(np.unravel_index(cbs_sparse.unique_col, cbs_sparse.dim)).T -1

        c_residual = np.linalg.norm(c_res,axis=1)
        print('Points residual',np.linalg.norm(c_res,axis=0))
        I_offend = np.where(c_points)[0][np.where(c_residual>tol)[0]]
        X_offend = X[I_offend]
        if len(X_offend) >0:
            subcell = len(np.unique(np.floor(cbs_sparse.transform(X_offend)*2),axis=0))
            ratio = (subcell) / size**2
        else:
            ratio = 0
        print(f'Offending:{len(X_offend)}, size: {size}, ratio: {ratio}, level {level}')
        size *= 2
        child_error = np.linalg.norm(sum(c.ev(uv_fit) for c in current_patches) - V,axis=1)

        current_patches.append(cbs_sparse)
        level += 1
        D = [c.ev(uv_vis) for c in current_patches]
        vw = disp(sum(D), F_vis, UV=size/4*uv_vis, P = Z, return_vw=True);vw.data().point_size/=10; vw.launch()
        # disp(np.hstack([uv_vis, sum(D)[:,2:]]),F_vis, UV=size/4*uv_vis, P = np.hstack([X, Z[:,2:]]))
        # disp(sum(c.ev(uv_vis) for c in current_patches), F_vis, UV=size/4*uv_vis)#,P=Z[I_offend])

    
    child_error = np.linalg.norm(sum(c.ev(uv_fit) for c in current_patches) - V,axis=1)
    disp(sum(c.ev(uv_vis) for c in current_patches), F_vis)
   
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('function', default='sparse')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    from interpolate import simple_quadratic_spline
    cubic_spline = simple_cubic_spline
    hill_visualize()
