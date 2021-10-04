#!/usr/bin/env python
# coding: utf-8
import fem_tabulator as feta
import h5py
import quad_curve_utils as qr
import quad_utils
import numpy as np
import igl
import sys
sys.path.append('/home/zhongshi/Workspace/bichon/python/debug')
import prism
import os
import scipy
import bezier as qb
import tqdm
from occ_step import cp_write_to_step
import occ_step

def valid_pairing(faces, score, sharp_markers, valid_combine_func):
    tt, tti = igl.triangle_triangle_adjacency(faces)
    occupied = -np.ones(len(faces),dtype=int)
    qid = 0
    pairs = []
    queue = []
    for fi in range(len(faces)):
        for e in range(3):
            if sharp_markers[fi,e]:
                continue
            queue.append((score(fi,e), fi,e))
    queue = sorted(queue)
    for _, fi,e in queue:
        if occupied[fi] >= 0:
            continue
        fo = tt[fi,e]
        
        if fo < 0:
            continue
        if occupied[fo] >= 0:
            continue

        if not valid_combine_func(fi, fo): # combine fi with fo.
            continue

        occupied[fi] = fo
        occupied[fo] = fi
        # q = list(faces[fi])
        # q.insert(e+1, faces[fo][tti[fi,e]-1])
        # pairs.append(q)
        qid += 1
    return occupied, pairs


def main(input_file, output_file = None, order =3, level=6, post_check=False):
    with h5py.File(input_file, 'r') as f:
        V,F,refV,refF,inpV,mB,mT = map(lambda x:f[x][()], ('mV','mF','ref.V','ref.F','inpV','mbase','mtop'))

    ## Bezier fitting
    A = scipy.sparse.coo_matrix(qb.bezier_fit_matrix(order, level)).tocsr()
    query = qr.query
    query.aabb, query.F, query.mB, query.mT, query.inpV, query.refF = prism.AABB(refV, refF), F, mB, mT, inpV, refF

    def valid_combine_func(fi, fo) -> bool:
        # First fit
        quad, trims = qr.combine_tris(F[fi], F[fo])
        tbc0 = np.array(qr.sample_for_quad_trim(trims[0], trims[1], level),
                        dtype=int)
        tbc0[:, 0] = np.asarray([fi, fo])[tbc0[:, 0]]
        sample_vals = query(tbc0, denom=level)
        local_cp = qr.quadratic_minimize(A, sample_vals)
        # Second, check
        v = qb.bezier_check_validity(mB, mT, F[[fi,fo]], quad.reshape(-1,4), np.array([[0,1]]), 
                                 trims, np.array([local_cp]), order,
                                 valid_check = prism.elevated_positive_check,
                                progress_bar=False)

        return v
    siblings, _ = valid_pairing(F, score = lambda f,e: -np.linalg.norm(V[F[f,e]] - V[F[f,(e+1)%3]]), 
                                                sharp_markers=quad_utils.edge_dots(V,F) < np.cos(np.pi/4),
                                                valid_combine_func=valid_combine_func)
    print('empty siblings', np.count_nonzero(siblings == -1), '/', len(siblings))
    t2q, q2t,trim_types, quads = qr.quad_trim_assign(siblings, F)


    quad_cp, samples = qr.quad_fit(V, F, quads, q2t, trim_types, level, order, A, query, None)

    if post_check:
        valids = bezier_check_validity(mB, mT, F,quads, q2t, trim_types, quad_cp, 3)

        print('valids', np.count_nonzero(valids), '/', len(valids))
        # adjust quads, quads_cp, q2t, t2q based on validity
        siblings[q2t[valids==False].flatten()] = -1
        quads = quads[valids]
        quad_cp = quad_cp[valids]
        q2t = q2t[valids]
        t2q = np.ones_like(t2q) * -1
        for q, (t0,t1) in enumerate(q2t):
            t2q[t0] = q
            t2q[t1] = q
    
    new_v, known_cp, newquads = qr.solo_cc_split(V, F, siblings, t2q, quads, quad_cp, order, subd=None)
    cc_cp = qr.constrained_cc_fit(V, F, siblings, newquads, known_cp, level, order, A, query)
    if output_file is None:
        output_file = f'/home/zhongshi/ntopo/ntopmodels/fit/{os.path.basename(input_file)}'
    _, stripe0 = quad_utils.group_quads(quads)
    _, stripe1 = quad_utils.group_quads(np.array(newquads))
    print(f'Quad Counts: {len(quads)} + {len(newquads)}')
    print(f'Stripe Counts: {len(stripe0)} + {len(stripe1)}')
    offset = lambda x: (x[0] + len(quads), x[1])
    occ_step.stripe_writer(output_file + '.stp',
                 stripe0 + [list(map(offset, stripe)) for stripe in stripe1],
                 np.vstack([quad_cp,cc_cp]))

    np.savez(output_file + '.npz', quad_cp = quad_cp, cc_cp = cc_cp, quads=quads, newquads=newquads,
                stripe0=stripe0, stripe1=stripe1)

def test_stripe():
    with np.load('temp.npz') as npl:
        quads, quad_cp = npl['quads'], npl['quad_cp']

    # quads -- quad_cp
    stripe_paint, all_stripes = quad_utils.group_quads(quads)
    print('num of stripes', len(all_stripes))
    stripe_writer(all_stripes, 'quad_cp, stripe.stp')

if __name__ == '__main__':
    import fire
    fire.Fire(main)
