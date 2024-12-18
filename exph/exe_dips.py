## compute exe_dipoles for phonon emission
import numpy as np
from exph_precision import *


def exe_dipoles(ele_dipoles, exe_wfc_gamma, kmap, symm_mats, time_rev):
    ## exe_dipoles are for photon emission
    ## ele_dipoles are in iBZ (k,c,v,pol)
    nv = exe_wfc_gamma.shape[-1]
    rot_mats = symm_mats[kmap[:, 1], ...]
    dip_expanded = np.einsum('kij,kcvj->kcvi', rot_mats, ele_dipoles[kmap[:, 0],
                                                                     ...])
    time_rev_s = (kmap[:, 1] >= symm_mats.shape[0] / (int(time_rev) + 1))
    dip_expanded[time_rev_s] = dip_expanded[time_rev_s].conj()
    return np.einsum('nkcv,kcvi->in',
                     exe_wfc_gamma.conj(),
                     dip_expanded,
                     optimize=True).conj().astype(numpy_Cmplx)  ##(pol,nexe)
