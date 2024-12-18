import numpy as np
from scipy.spatial import cKDTree

def make_kpositive(klist, tol=1e-6):
    ## brings all kpoints in [0,1)
    kpos = klist-np.floor(klist)
    return (kpos+tol)%1

#tree = spatial.KDTree(klist)
def find_kpt(tree, kpt_search, tol=1e-5):
    kpt_search = make_kpositive(kpt_search)
    dist, idx = tree.query(kpt_search, workers=1)
    if len(dist[dist>tol]) !=0:
        print("Kpoint not found")
        exit();
    return idx

def build_ktree(kpts):
    tree = make_kpositive(kpts)
    return cKDTree(tree,boxsize=[1,1,1])

def find_kindx(kpt_search, tree):
    ## find the indices of elements of kpt_search in kpt_list
    return find_kpt(tree, kpt_search)

