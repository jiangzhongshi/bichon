import igl
import meshplot as mp
import h5py
import numpy as np
import itertools
import scipy

def h5deser(filename):
    with h5py.File(filename,'r') as f:
        F, V, base, top = f['mF'][()], f['mV'][()], f['mbase'][()], f['mtop'][()]
        refV, refF = f['ref.V'][()],f['ref.F'][()]
        grouped_tracker = np.split(f['track_flat'][()].astype(np.int32).flatten(), np.cumsum(f['track_size'][()]).astype(np.int))
    return refV, refF, base, top, V,F, grouped_tracker[:-1]

def mips2d(V,f):
    e1 = V[f[1]] - V[f[0]]
    e2 = V[f[2]] - V[f[0]]
    e1_len = np.linalg.norm(e1)
    e2_x = np.dot(e1,e2) / e1_len
    e2_y = np.linalg.norm(e2 - e2_x * e1 / e1_len)
    tri = np.array([[e1_len, e2_x], [0, e2_y]])
    ref = np.array([[1,0.5],[0, np.sqrt(3)/2]])
    jac = tri@np.linalg.inv(ref)
    invf2 = np.sum(np.linalg.inv(jac)**2)
    frob2 = np.sum(jac**2)
    return frob2*invf2

def centered(V):
    V -= V.min(axis=0)
    V /= V.max()
    return V

def stack_V_F(Vl,Fl):
    v_cnt = 0
    for v,f in zip(Vl,Fl):
        f[:] += v_cnt
        v_cnt += v.shape[0]
    return np.vstack(Vl), np.vstack(Fl)

def smoother(L, b, V):
    mask = np.zeros(L.shape[0])
    mask[b] = 1
    Lii = L[mask==0][:,mask==0]
    Lib = L[mask==0][:,mask==1]
    V = V.copy()
    print(np.linalg.norm(L@V))
    
    V[mask==0] = scipy.sparse.linalg.spsolve(-Lii, Lib@V[mask==1])
    print(np.linalg.norm(L@V))
    return V


def doo_sabin(V,F, eps=0.05):
    dsF = []
    v_num = V.shape[0]
    f_num = F.shape[0]
    he_num = F.shape[0]*3
    # total v_num + he_num (==3*f_num) + 3*f_num
    dsV = np.zeros((v_num + he_num + 3*f_num,3))
    dsV[:v_num] = V
    FF, FFi = igl.triangle_triangle_adjacency(F)
    for i in range(F.shape[0]):
        v0,v1,v2 = F[i]
        f0, f1, f2 = v_num + he_num + 3*i + np.arange(3)
        e0 = v_num + 3*i+np.arange(3)
        e1 = np.zeros(3)
        for e in range(3):
            f_oppo, e_oppo = FF[i,e], FFi[i,e]
            e1[e] = v_num + 3*f_oppo + e_oppo
        newF = np.array([[v0, e0[0], f0],
                        [v0, f0, e1[2]],
                        [e0[0],e1[0],f1],
                        [e0[0],f1,f0],
                        [v1,f1,e1[0]],
                        [v1, e0[1],f1],
                        [e0[1],f2,f1],
                        [e0[1],e1[1],f2],
                        [e1[1],v2,f2],
                         [f2,v2,e0[2]],
                         [f2,e0[2],f0],
                         [f0,e0[2],e1[2]],
                        [f0,f1,f2]])
        dsF.append(newF)
        fx = [f0,f1,f2]
        for e in range(3):
            dsV[e0[e]] = (1-eps)*V[F[i,e]] + eps * V[F[i,(e+1)%3]]
            dsV[fx[e]] = (1-eps)*V[F[i,e]] + eps/2 * V[F[i,(e+1)%3]] + eps/2*V[F[i,(e+2)%3]]
    return dsV, np.array(dsF)


def triangle_quality(p0,p1,p2):
    e1, e2 = p1 - p0, p2 - p0
    e1_len = np.linalg.norm(e1)
    e2_x = e1.dot(e2) / e1_len
    e2_y = np.linalg.norm(e2 - e2_x * e1 / e1_len)
    tri = np.array([[e1_len, e2_x],[0, e2_y]])
    ref = np.array([[1, 0.5],[0,  np.sqrt(3) / 2]])
    jac = tri@np.linalg.inv(ref)
    frob2 = (jac**2).sum()
    det = np.linalg.det(jac)
    return frob2/det

def msubplot(v_list,f_list,shape ,**sh):
    plt = None
    for i,(v,f) in enumerate(zip(v_list,f_list)):
        vw = mp.Viewer(sh)
        vw.add_mesh(v,f,shading=sh)
        plt = mp.Subplot(plt, vw, s=[shape[0],shape[1],i])
    return plt


def str_to_array(s):
    return np.array(list(map(float,s.split())))

def f2e(F):
    return np.vstack([F[:,[0,2]],F[:,[0,1]],F[:,[1,2]]])

def t2e(T):
    return np.vstack([T[:,e] for e in (itertools.combinations(range(T.shape[1]),2))])

def h5reader(file, *names):
    with h5py.File(file,'r') as fp:
        if len(names) == 0:
            return str(fp.keys())
        a = list(map(lambda x:fp[x][()], names))
    return a

def obj_writer():
    with open('temp.obj','w+') as fp:
        fp.writelines(['v ' + ' '.join(map(str,list(v)))+'\n' for v in SV])
        fp.writelines(['vt ' + ' '.join(map(str,list(v)))+'\n' for v in UV])
        fp.writelines([f'f {f[0]}/{tf[0]} {f[1]}/{tf[1]}  {f[2]}/{tf[2]}\n' for f,tf in zip(SF+1,TF+1)])


def ply_writer(filename, uV,uF,uC):
    vertex=np.array([tuple(list(v)+ list(c)) for v,c in zip(uV,uC)],
                dtype=[('x', 'f4'), ('y', 'f4'),('z','f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    face = np.array([tuple([f]) for f in uF],dtype=[('vertex_indices', 'i4', (3,))])
    from plyfile import PlyData, PlyElement
    vel = PlyElement.describe(vertex, 'vertex')
    fel = PlyElement.describe(face, 'face')

    PlyData([vel,fel]).write(filename)

def scale(x):
    scale.a = x.min(axis=0)
    y = x-scale.a
    scale.b = y.max()
    y = y/scale.b
    return y
use_scale=lambda x:(x-scale.a)/scale.b


def twelve_tetra_table():
    V = np.array([[0,0,0],
                  [1,0,0],
                  [0,1,0],
                  [0,0,1],
                  [1,0,1],
                  [0,1,1]
                  ])
    twelve = []
    for tet in set(itertools.combinations(range(6),4)):
        tet = np.array(tet)
        vol = np.linalg.det(V[tet[1:]]-V[tet[0]])
        if vol == 0.0:
            continue
        if vol < 0:
            tet[2:] = tet[-1], tet[-2]
        twelve.append(tet)
    return np.array(twelve)


def corner_to_cube(c0, c1):
    mi, ma = np.minimum(c0,c1), np.maximum(c0,c1)
    pts = np.array([[mi[i] if tup[i] else ma[i] for i in range(3)]
                    for tup in itertools.product([False,True],repeat=3)])
    tris = np.array([[0,1,2],[1,2,3],[4,5,6],[6,7,5],
                    [0,2,4],[2,6,4],[3,7,5],[3,1,5]])
    return pts, tris
        
def shrink(tetV,tetT, alpha):
    VT = tetV[tetT]
    mean = VT.mean(axis=1,keepdims=True)
    return (VT - mean)*alpha + mean
    # mp.plot(shrink(V,T,0.7).reshape(-1,3),np.arange(4*len(T)).reshape(-1,4))
    
    
def tetmesh_from_shell(base, top, F):
    tetra_splits = (np.array([0, 3, 4, 5, 1, 4, 2, 0, 2, 5, 0, 4]).reshape(-1, 4),
                    np.array([0, 3, 4, 5, 1, 4, 5, 0, 2, 5, 0, 1]).reshape(-1, 4))
    vnum = len(base)
    T = []
    for f in F:
        tet_c = tetra_splits[0] if f[1] > f[2] else tetra_splits[1]
        T.append((tet_c // 3)*vnum + f[tet_c % 3])
    return np.vstack([base, top]), np.vstack(T)

def control_points_duplicate_consistent():
    VF,NI = igl.vertex_triangle_adjacency(mF, len(mV))
    VFs = np.split(VF,NI[1:-1])
    for v, nbF in enumerate(VFs):
        vals = [cp[f, np.argmax(mF[f] == v)] for f in nbF]
        if sum((v-vals[0]).sum()>0 for v in vals) > 0:
            print(v)

def highorder_sv(cp,level=3, order=3):
    from curve import fem_generator
    def local_upsample(level:int):
        usV, usF = igl.upsample(np.eye(3)[:,1:], np.arange(3)[None,:], level)
        bnd0 = igl.boundary_loop(usF)
        usE = np.vstack([bnd0, np.roll(bnd0, -1)]).T
        return usV, usF, usE
    usV,usF,usE = local_upsample(level=level)
    if order is None:
        size_to_order = lambda x: int((np.sqrt(x*8+1) - 1)/2) - 1
        order = size_to_order(cp.shape[1])
    bas_val = fem_generator.bernstein_evaluator(usV[:,0],usV[:,1],usV[:,0]*0,
                                                fem_generator.tuple_gen(order=order,var_n=2)).T
    sv = (bas_val@cp)
    return sv, np.vstack([usF+i*len(usV) for i in range(len(sv))]), np.vstack([usE+i*len(usV) for i in range(len(sv))])                  
        
def tw_mips3d(V,T):
    from numpy import sqrt
    Jacs = V[T[:,1:]]-V[T[:,:1]]
    Jacs = np.linalg.inv(np.array([[1,0,0],[0.5,sqrt(3)/2.,0],[0.5,sqrt(3)/6.,sqrt(2)/sqrt(3)]]))@Jacs
    frob = np.sum(Jacs**2,axis=1).sum(axis=1)
    dets = np.linalg.det(Jacs)
    return frob/dets**(2/3)

def mark_feature(V,F,deg):
    FN = igl.per_face_normals(V,F,np.ones(3))
    TT,TTi = igl.triangle_triangle_adjacency(F)
    di_angles = (FN[TT]*FN[:,None,:]).sum(axis=2)
    E =np.array([(F[f,e], F[f,e-2]) for f,e in zip(*np.where(di_angles <np.cos(np.deg2rad(180-deg))))])
    return E


def write_obj_lines(filename, V,E):
    with open(filename,'w') as fp:
        fp.write('\n'.join([f'v {v[0]} {v[1]} {v[2]}' for v in V] + 
                           [f'l {e[0]} {e[1]}' for e in E+1]))
        
def convert_tri10():
    from curve import fem_generator
    gmshcod = (fem_generator.codecs()['tri10'][2]*3).astype(np.int)
    autocod = fem_generator.tuple_gen(order=3,var_n=2)
    reorder = np.lexsort(
                np.array(autocod).T)[fem_generator.invert_permutation(np.lexsort(gmshcod.T))]
    np.all(np.array(autocod)[reorder] == gmshcod)

    tri10info = fem_generator.basis_info(order=3,nsd=2)

    def codec_to_n(co): return [k for i, j in enumerate(co) for k in [i]*j]

    auto_cod_n = np.array([codec_to_n(c) for c in autocod])

    uniq_tup, tup_ind, tup_inv = np.unique(np.sort(mF[:,auto_cod_n].reshape(-1,3)), axis=0,return_index=True, return_inverse=True)

    m = meshio.Mesh(points=(tri10info['b2l']@scale_cp).reshape(-1,3)[tup_ind], cells = [('triangle10', 
                                                  tup_inv[np.arange(len(cp)*10).reshape(-1,10)][:,reorder]
                                                                      )])
    

def tet_highorder_sv(cp,level=3, order=3):
    dim = 3
    def local_upsample(level:int):
        tet = igl.boundary_facets(np.array([0, 1, 2, 3]).reshape(1,4))
        vv = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        usV, usF = igl.upsample(vv, tet, level)
        FN = igl.per_face_normals(usV,usF,np.ones(3))
        TT,TTi = igl.triangle_triangle_adjacency(usF)
        di_angles = (FN[TT]*FN[:,None,:]).sum(axis=2)
        usE =np.array([(usF[f,e], usF[f,e-2]) for f,e in zip(*np.where(di_angles < 0.5))])
        return usV, usF, usE
    usV,usF,usE = local_upsample(level=level)
    info = fem_generator.basis_info(order, dim, force_codec='tetra20')
    l2b = info['l2b']
    bas_val = fem_generator.bernstein_evaluator(usV[:,0],usV[:,1],usV[:,2], info['codec']).T
    sv = (bas_val@(l2b@cp))
    return sv, np.vstack([usF+i*len(usV) for i in range(len(sv))]), np.vstack([usE+i*len(usV) for i in range(len(sv))])

  
 def convert_p4(file1, file2):
    import meshio
    # a handy conversion for the default output mesh (p4).
    with h5py.File(file1,'r') as fp:
        lagr, p4T  = fp['lagr'][()], fp['cells'][()]
        reorder = np.array([ 0,  1,  2,  3,  4, 16,  5,  7, 18,  9,  8, 17,  6, 13, 19, 10, 15,
               21, 12, 14, 20, 11, 22, 24, 23, 25, 26, 31, 27, 32, 29, 33, 28, 30, 34])
        meshio.write(file2, meshio.Mesh(points=lagr, cells=[('tetra35', p4T[:,reorder])]))