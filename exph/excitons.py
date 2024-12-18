import numpy as np
from scipy.spatial import cKDTree
from kpts import *
import torch as pytorch
import os
from exph_precision import *


## function to rotate exciton wavefunction
def rotate_exc_wfc(wfc, symm_mat_red, kpoints, ktree, exe_qpt, dmats, time_rev):
    ## kpoints are kpts in full BZ (crystal coordinates)
    ## qpt is momentum of exciton of unrotated ex-wfc
    ## Dmats are rep mats (nkpts, Rk_band, k_band,)
    ## symm_mat_red : symm_matrix in reduced coordinates
    rot_wfc = pytorch.zeros(wfc.shape, dtype=pytor_Cmplx)
    nk, nc, nv = wfc.shape[1:]
    ## compute the indices of Rk and Rk-q
    Rkpts = np.einsum('ik,nk->ni', symm_mat_red, kpoints, optimize=True)
    k_minus_q = kpoints - exe_qpt[None, :]
    idx_Rk = find_kindx(Rkpts, ktree)
    idx_k_minus_q = find_kindx(k_minus_q, ktree)
    ## rotate the wfc
    Dcc = pytorch.from_numpy(dmats)[:, nv:, nv:]
    Dvv = pytorch.from_numpy(dmats)[idx_k_minus_q, :nv, :nv].conj()
    wfc_tmp = pytorch.from_numpy(wfc)
    if time_rev:
        rot_wfc[:, idx_Rk, ...] = pytorch.einsum('nkcv,kic,kjv->nkij',
                                                 wfc_tmp.conj(), Dcc, Dvv)
    else:
        rot_wfc[:, idx_Rk, ...] = pytorch.einsum('nkcv,kic,kjv->nkij', wfc_tmp,
                                                 Dcc, Dvv)
    return rot_wfc.numpy()


def ex_ph_mat(wfc_k_q, wfc_k, elph_mat, qpt_exe, qpt_ph, kpts, ktree):
    ## compute exction phonon matrix element
    ## < Q+q|dv|Q> where Q (qpt_exe) is exciton momentum
    ## q (qpt_ph) is phonon momentum
    ## elph_mat for qpt_ph point (nmodes,nk, final_band_PH_abs, initial_band)
    # outpur : modes, initial_state, final_state
    nmodes = elph_mat.shape[0]
    n_exe_states, nk, nc, nv = wfc_k_q.shape
    #
    idx_k_minus_Q_minus_q = find_kindx(kpts - qpt_ph[None, :] -
                                       qpt_exe[None, :], ktree)  # k-Q-q
    idx_k_minus_q = find_kindx(kpts - qpt_ph[None, :], ktree)  # k-q
    #
    gcc = pytorch.from_numpy(elph_mat)[:, idx_k_minus_q, nv:, nv:]
    gvv = pytorch.from_numpy(elph_mat)[:, idx_k_minus_Q_minus_q, :nv, :nv]
    #
    # make arrays C_CONTIGUOUS to reduce repeated cache misses
    wfc_k_electron = pytorch.from_numpy(wfc_k)[:, idx_k_minus_q,
                                               ...].contiguous()
    wfc_k_hole = pytorch.from_numpy(wfc_k)
    wfc_kq_conj = pytorch.from_numpy(wfc_k_q).reshape(n_exe_states, -1).conj()
    #
    ex_ph_mat = pytorch.zeros((nmodes, n_exe_states, n_exe_states),
                              dtype=pytor_Cmplx)
    ## compute the ex-ph mats
    for imode in range(nmodes):
        ## 1) Compute electron contribution # (k-q,k-q)
        tmp_wfc = pytorch.einsum('nkiv,kci->nkcv', wfc_k_electron, gcc[imode])
        ### 2) Compute Hole contribution #(k,k-q-Q) and subtract to (1)
        tmp_wfc -= pytorch.einsum('nkci,kiv->nkcv', wfc_k_hole, gvv[imode])
        tmp_wfc = tmp_wfc.reshape(n_exe_states, -1)
        pytorch.matmul(tmp_wfc, wfc_kq_conj.T, out=ex_ph_mat[imode])
    ## return ex-ph mat
    return ex_ph_mat.numpy()
