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


def find_kpatch(kpts, kcentre, kdist, lat_vecs):
    """
    find set of kpoints around the kcentre with in kdist

    Parameters
    ----------
    kpts : kpoints in crystal coordinates (nk,3)
    kcentre : kpoint centre in crystal coordinates (3)
    kdist : distance around kcentre to be considered in atomic units
            i.e 1/bohr.
    lat_vecs: lattice vectors. ith lattice vector is ai = a[:,i]
    Returns
    -------
    int array
        Indices of kpoints in kpts array which satify the given condition i.e
        | k - kcentre + G0| <= kdist, where G0 is reciprocal lattice vector to bring to BZ
    """
    #
    blat = 2 * np.pi * np.linalg.inv(lat_vecs)
    kdiff = kpts - kcentre[None, :]
    kdiff = kdiff - np.floor(kdiff)
    #
    tmp_arr = np.array([-3, -2, -1, 0, 1, 2, 3])
    nG0 = len(tmp_arr)
    G0 = np.zeros((nG0, nG0, nG0, 3))
    G0[..., 0], G0[..., 1], G0[..., 2] = np.meshgrid(tmp_arr,
                                                     tmp_arr,
                                                     tmp_arr,
                                                     indexing='ij')
    G0 = G0.reshape(-1, 3)
    kdiff = kdiff[:, None, :] - G0[None, :, :]
    kdiff = kdiff.reshape(-1, 3) @ blat
    kdiff = np.linalg.norm(kdiff, axis=-1).reshape(len(kpts), -1)
    kdiff = np.min(kdiff, axis=-1)
    return np.where(kdiff <= kdist)[0]
