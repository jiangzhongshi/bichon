import numpy as np
import quad.quad_curve_utils as qr
import tqdm

def eval_bc(verts, faces, bc_i, denom:int):
    vf = (verts[faces[bc_i[:,0]]])
    return np.einsum('sed,se->sd', vf, bc_i[:,1:])/denom

def bezier_fit_matrix(order : int, level : int) -> np.ndarray:
    std_x, std_y = qr.quad_tuple_gen(level).T
    bsv = qr.tp_sample_value(std_x/level, std_y/level, order=order)
    bsv = bsv.reshape(len(bsv),-1)
    return bsv

def bezier_check_validity(mB,mT,F, quads, q2t, trim_types, quad_cp, order, progress_bar = True):
    all_b, all_t = qr.top_and_bottom_sample(mB,mT, F, quads, q2t, trim_types, level=1)
    v4, f4 = qr.split_square(1)

    tup = feta.tuple_gen(order = order + 1, var_n=2) # elevated one order for quartic tetrahedra
    grids = np.einsum('fed,Ee->fEd', v4[f4], np.asarray(tup))
    
    grid_ids = np.ravel_multi_index(grids.reshape(-1,2).T, dims = (order + 2, 
                                                                   order + 2)).reshape(len(f4), -1)
    valid_quad = np.ones(len(quad_cp), dtype=bool)
    
    A13 = bezier_fit_matrix(order, order+1)
    if progress_bar: 
        pbar = tqdm.tqdm(quad_cp, desc='Bezier Quads Checking validity')
    else:
        pbar = quad_cp
    for q,qcp in enumerate(pbar):
        lagr = A13@qcp
        for t,g in zip(f4, grid_ids):
            if not (prism.elevated_positive_check(all_b[q][t], all_t[q][t], 
                                                  lagr[g], True)):
                valid_quad[q] = False
                break
    return valid_quad