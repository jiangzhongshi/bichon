import numpy as np
from collections import defaultdict
import scipy
import igl


def catmul_clark_split(v,f):
    verts = list(v)
    tt, tti = igl.triangle_triangle_adjacency(f)
    edge_id = - np.ones_like(tt)
    avail_id = len(v)
    for fi, _ in enumerate(tt):
        for ei in range(3):
            if edge_id[fi,ei] == -1:
                edge_id[fi,ei] = avail_id
                edge_id[tt[fi,ei],tti[fi,ei]] = avail_id
                v0, v1 = f[fi,ei], f[fi,(ei+1)%3]
                verts.append((v[v0]+v[v1])/2)
                avail_id += 1

    bc = igl.barycenter(v,f)
    quads = []
    for fi, [v0,v1,v2] in enumerate(f):
        pid = fi + avail_id
        for ei in range(3):
            v0, e0, e2 = f[fi,ei], edge_id[fi,ei], edge_id[fi,(ei+2)%3]
            quads += [[v0,e0,pid,e2]]
    return np.asarray(verts + list(bc)), np.asarray(quads)


def q2e(f):
    return np.array([f[:,i] for i in [[0, 1],[1,2],[2,3],[3,0]]]).reshape(-1,2)


def greedy_pairing(f, score, sharp_markers):
    tt, tti = igl.triangle_triangle_adjacency(f)
    occupied = -np.ones(len(f),dtype=int)
    qid = 0
    pairs = []
    queue = []
    for fi in range(len(f)):
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
        occupied[fi] = fo
        occupied[fo] = fi
        q = list(f[fi])
        q.insert(e+1, f[fo][tti[fi,e]-1])
        pairs.append(q)
        qid += 1
    return occupied, pairs


def crawling(hybrid):
    def set_conn(v0,v1,x):
        if v0<v1:
            connect[(v0,v1)][0] = x
        else:
            connect[(v1,v0)][1] = x
    def get_conn(v0,v1):
        return connect[(v0,v1)][0] if v0<v1 else connect[(v1,v0)][1]

    # BFS
    def bfs_find_tri(t:int):
        visited = set()
        visited.add(t)
        path = [(t,None)]
        bfs_queue = [path]
        
        while len(bfs_queue) > 0:
            if (len(bfs_queue)) > len(hybrid):
                print('Wrong!InfLoop Warn')
                assert False

            p = bfs_queue[0]
            bfs_queue.pop(0)
            t = p[-1][0]
            ne = len(hybrid[t])
            
            for i in range(ne):
                e = hybrid[t][(i+1)%ne], hybrid[t][i]
                fo = get_conn(*e) # oppo
                if fo is None or fo in visited:
                    continue
                visited.add(fo)
                bfs_queue.append(p + [(fo,e)])
#                 print(fo)
#                 print(fo, hybrid[fo])
                if len(hybrid[fo]) == 3:
                    return bfs_queue[-1] 
        return None

    def merge_tri_quad(tri, edge, quad): # this is a backward path
        q, e0 = quad
        v0,v1 = edge
        qv = list(hybrid[q].copy())
#         print('In', tri, edge, qv)
        x = list(set(tri)-set(edge))[0] # opposite vertex
        qv.insert(qv.index(v0), x)
#         print('Pent',qv, e0)
        
        if len(qv) == 4:
            assert e0 is None
            return None, None, qv
        newtri = [qv[qv.index(e0[0])-1], e0[0], e0[1]]
        qv.remove(e0[0])
        return newtri, e0, qv

    connect = defaultdict(lambda:[None,None])
    cnt_tri = 0
    for fi, f in enumerate(hybrid):
        ne = len(f)
        if ne == 3:
            cnt_tri += 1
        for i in range(ne):
            set_conn(f[i], f[(i+1)%ne],fi)

    for i in range(cnt_tri//2 + 1):
        bachelor = np.where(np.array([len(h) for h in hybrid]) == 3)[0]

        if len(bachelor) == 0:
            break
        path = bfs_find_tri(bachelor[0])

        tid, edge = path[-1]
        tri = hybrid[tid]
        new_quads = []
        pid = len(path) - 2
        while tri is not None:
            tri, edge, new_q = merge_tri_quad(tri, edge, path[pid])
            pid -= 1
            new_quads.append(new_q)

        for p,_ in path:
            f = hybrid[p]
            ne = len(f)
            for i in range(ne):
                set_conn(f[i], f[(i+1)%ne], None)
            hybrid[p] = []
        for f in new_quads:
            ne = len(f)
            for i in range(ne):
                set_conn(f[i], f[(i+1)%ne], len(hybrid))
            hybrid.append(f)
    return hybrid
    
def split_edges_between_odd_components(f, cp):
    tt,tti = igl.triangle_triangle_adjacency(f)
    assert tt.min() >=0
    arr = defaultdict(lambda:set())
    for i in range(len(f)):
        r0 = cp[i]
        for j in range(3):
            r1 = cp[tt[i,j]]
            if r0 != r1:
                v0, v1 = f[i,j], f[i,j-2]
                arr[(min(r0,r1), max(r0,r1))].add((min(v0,v1), max(v0,v1)))
    ijk = np.array([(k[0],k[1],1) for k, l in arr.items()],dtype=int)
    n_comp = cp.max()+1
    comp_adj = scipy.sparse.coo_matrix(arg1=(ijk[:,2], ijk[:,:2].T), shape=(n_comp,n_comp))

    odd_comps = np.where([np.count_nonzero(cp==i)%2 for i in range(max(cp) + 1)])[0]
    assert len(odd_comps)%2 == 0
    
    length, predec = scipy.sparse.csgraph.shortest_path(comp_adj, directed=False, indices=odd_comps, return_predecessors=True)

    def trace_path(prec, i,j):
        assert prec[i] <0
        path = [j]
        while True:
            p = prec[path[-1]]
            if p < 0:
                break
            path.append(p)
        return path

    odd_pairing = []

    
    visited = np.zeros(len(odd_comps))
    for i,c in enumerate(odd_comps):
        if visited[i] == 1:
            continue
        length[i][c] = np.inf
        length[i][odd_comps[visited==1]] = np.inf
        j = length[i][odd_comps].argmin()
        visited[i] = visited[j] = 1
        path = trace_path(predec[i], c, odd_comps[j])
        odd_pairing.append((c, odd_comps[j], path))

    split_pairs = []
    for i,j, path in odd_pairing:
        split_pairs += [tuple(sorted(i)) for i in zip(path[:-1], path[1:])]
    return [next(iter(arr[s])) for s in split_pairs]


def edge_dots(V, F):
    FN  = igl.per_face_normals(V,F, np.ones(3))
    tt, tti = igl.triangle_triangle_adjacency(F)
    return np.einsum('fad,fd->fa',FN[tt],FN)

# BFS traversal
def group_quads(quads):
    def quad_connectivity(quads):
        connect = defaultdict(lambda:[None,None])
        cnt_tri = 0
        def set_conn(v0,v1,x):
            if v0<v1:
                connect[(v0,v1)][0] = x
            else:
                connect[(v1,v0)][1] = x
        
        for fi, f in enumerate(quads):
            ne = len(f)
            for i in range(ne):
                set_conn(f[i], f[(i+1)%ne],fi)
        return connect   
    def get_conn(connect, v0,v1):
        return connect[(v0,v1)][0] if v0<v1 else connect[(v1,v0)][1]
    conn = quad_connectivity(quads)
    
    stripe_paint = - np.ones(len(quads), dtype=int)
    next_color = 0
    all_stripes = []
    while stripe_paint.min() < 0:
        f = np.where(stripe_paint < 0) [0][0]
        stripe_paint[f] = next_color
                
        for e in range(4):
            v0, v1 = quads[f][[e,(e+1)%4]]
            f_oppo = conn[(v0,v1)][1] if v0<v1 else conn[(v1,v0)][0]
            if f_oppo is not None and stripe_paint[f_oppo] == -1:
                break
        else:
            next_color += 1
            all_stripes.append([(f,0)])
            continue

        stripe = [(f,(e+3)%4)]
        vertset = set(list(quads[f]))
        for _ in range(10):
            v0, v1 = quads[f][[e,(e+1)%4]]
            f_oppo = conn[(v0,v1)][1] if v0<v1 else conn[(v1,v0)][0]
            if f_oppo is None or stripe_paint[f_oppo] >= 0:
                break
            vertset = vertset.union(list(quads[f_oppo]))
            if len(vertset) != 2*(len(stripe)+2):
                break  # not topology stripe.
            stripe_paint[f_oppo] = next_color
            v0id = list(quads[f_oppo]).index(v0)
            e_next = (v0id + 1) %4
            f,e = f_oppo, e_next
            stripe.append((f_oppo, v0id))
        next_color += 1
        all_stripes.append(stripe)
    return stripe_paint, all_stripes