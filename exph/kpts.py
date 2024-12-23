import numpy as np
from scipy.spatial import cKDTree
from exph_precision import numpy_float


def make_kpositive(klist, tol=1e-6):
    ## brings all kpoints in [0,1)
    kpos = klist - np.floor(klist)
    return (kpos + tol) % 1


def get_kgrid(klist, tol=1e-5):
    kpos = make_kpositive(klist)
    kgrid = 1 - np.max(kpos, axis=0)
    assert len(kgrid[kgrid < tol]) == 0
    kgrid = np.rint(1 / kgrid).astype(int)
    assert len(klist) == np.prod(kgrid)
    return kgrid


def generate_kgrid(kgrid):
    kx = np.arange(kgrid[0]) / kgrid[0]
    ky = np.arange(kgrid[1]) / kgrid[1]
    kz = np.arange(kgrid[2]) / kgrid[2]
    kpts_tmp = np.zeros((kgrid[0], kgrid[1], kgrid[2], 3), dtype=numpy_float)
    kpts_tmp[..., 0], kpts_tmp[..., 1], kpts_tmp[...,
                                                 2] = np.meshgrid(kx,
                                                                  ky,
                                                                  kz,
                                                                  indexing='ij')
    return kpts_tmp.reshape(-1, 3)


#tree = spatial.KDTree(klist)
def find_kpt(tree, kpt_search, tol=1e-5):
    kpt_search = make_kpositive(kpt_search)
    dist, idx = tree.query(kpt_search, workers=1)
    if len(dist[dist > tol]) != 0:
        print("Kpoint not found")
        exit()
    return idx


def build_ktree(kpts):
    tree = make_kpositive(kpts)
    return cKDTree(tree, boxsize=[1, 1, 1])


def find_kindx(kpt_search, tree):
    ## find the indices of elements of kpt_search in kpt_list
    return find_kpt(tree, kpt_search)
