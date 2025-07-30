## Compute resonant Raman intensities
## Sven Reichardt, etal Sci. Adv.6,eabb5915(2020)
import numpy as np
import math
#from scipy.spatial import KDTree
from numba_kdtree import KDTree


###### // One phonon Raman ##############
def compute_Raman_oneph_exc(ome_light,
                            ph_freq,
                            ex_ene,
                            ex_dip,
                            ex_ph,
                            nkpts,
                            CellVol,
                            broading=0.1,
                            npol=3,
                            ph_fre_th=5):
    ## resonant Raman with excitonic effects (stokes Raman
    ## intensities are computed i.e phonon emission)
    ## We need exciton dipoles for light emission (<0|v|S>)
    ## where v is velocity operator (3, nexc_states)
    ## and exciton phonon matrix elements for phonon absorption <S',0|dV|S,0>
    ## except ph_fre_th, ome_light and broading, all are in Hartree
    ## ph_fre_th is phonon freqency threshould. Raman tensor for phonons with freq below ph_fre_th
    ## will be set to 0. The unit of ph_fre_th is cm-1
    ## Raman (nmodes, npol_in(3), npol_out(3))
    nmode, nbnd_i, nbnd_f = ex_ph.shape
    ex_dip_absorp = ex_dip[:npol, :].conj(
    )  ## exciton dipoles for light absoption
    #
    broading_Ha = broading / 27.211 / 2.0
    ome_light_Ha = ome_light / 27.211
    #
    BS_energies = ex_ene - 1j * broading_Ha
    ph_fre_th = ph_fre_th * 0.12398 / 1000 / 27.211

    dipS_res = ex_dip_absorp / ((ome_light_Ha - BS_energies)[None, :])
    dipS_ares = ex_dip_absorp.conj() / ((ome_light_Ha + BS_energies)[None, :])

    Ram_ten = np.zeros((nmode, 3, 3), dtype=dipS_res.dtype)

    ram_fac = 1.0 / nkpts / math.sqrt(CellVol)
    # Now compute raman tensor for each mode
    for i in range(nmode):
        if (abs(ph_freq[i]) <= ph_fre_th):
            Ram_ten[i] = 0
        else:
            dipSp_res = ex_dip_absorp.conj() / (
                (ome_light_Ha - BS_energies - ph_freq[i])[None, :])
            dipSp_ares = ex_dip_absorp / (
                (ome_light_Ha + BS_energies - ph_freq[i])[None, :])
            # 1) Compute the resonant term
            Ram_ten[i, :npol, :npol] = np.conj(
                dipS_res.conj() @ ex_ph[i].T @ dipSp_res.T.conj())
            # 2) anti resonant
            Ram_ten[i, :npol, :npol] += dipS_ares @ ex_ph[i].T @ dipSp_ares.T
            ## scale to get raman tensor
            Ram_ten[i] *= (
                math.sqrt(math.fabs(ome_light_Ha - ph_freq[i]) / ome_light_Ha) *
                ram_fac)
    return Ram_ten


def compute_Raman_oneph_ip(ome_light,
                           ph_freq,
                           Qp_ene,
                           elec_dip,
                           gkkp,
                           CellVol,
                           broading=0.1,
                           npol=3,
                           ph_fre_th=5):
    ## resonant Raman at independent particle level (stokes Raman
    ## intensities are computed i.e phonon emission)
    ## We need electronic dipoles for light emission (3,k,c,v)
    ## i.e stored in ndb.dipoles
    ## where v is velocity operator
    ## Qp_ene (nk,nbnds), nbds = v +c, QP energies (in Ha)
    ## and gkkp are electron-phonon matrix elements for phonon absorption
    ## (nmodes, nk, final_band_PH_abs, initial_band) (in Ha). Note that
    ## el-ph must be normalized with sqrt(2*omega_ph)
    ## except ph_fre_th, ome_light and broading, all are in Hartree
    ## ph_fre_th is phonon freqency threshould. Raman tensor for phonons with freq below ph_fre_th
    ## will be set to 0. The unit of ph_fre_th is cm-1
    ##
    ## Raman (nmodes, npol_in(3), npol_out(3))
    nk, nc, nv = elec_dip.shape[1:]
    assert (nc + nv == Qp_ene.shape[1]
           ), "Sum of valence and conduction bands not equal to total bands."
    assert (nk == Qp_ene.shape[0]), "Number of kpoints not Compatible dipoles."
    assert (gkkp.shape[1] == Qp_ene.shape[0]
           ), "Number of kpoints not Compatible with gkkp"
    assert (gkkp.shape[2] == Qp_ene.shape[1]
           ), "Number of bands not Compatible with gkkp"
    #
    nmode = gkkp.shape[0]
    #
    elec_dip_absorp = elec_dip[:npol,
                               ...].conj()  ## dipoles for light absoption
    #
    broading_Ha = broading / 27.211 / 2.0
    ome_light_Ha = ome_light / 27.211
    #
    delta_energies = Qp_ene[:, nv:, None] - Qp_ene[:,
                                                   None, :nv] - 1j * broading_Ha
    ph_fre_th = ph_fre_th * 0.12398 / 1000 / 27.211
    #
    dipS_res = elec_dip_absorp / ((ome_light_Ha - delta_energies)[None, ...])
    dipS_res = dipS_res.reshape(3, -1)
    dipS_ares = elec_dip_absorp.conj() / (
        (ome_light_Ha + delta_energies)[None, ...])
    dipS_ares = dipS_ares.reshape(3, -1)
    #
    gcc = gkkp[:, :, nv:, nv:]
    gvv = gkkp[:, :, :nv, :nv]
    #
    Ram_ten = np.zeros((nmode, 3, 3), dtype=dipS_res.dtype)
    ram_fac = 1.0 / nk / math.sqrt(CellVol)
    # Now compute raman tensor for each mode
    for i in range(nmode):
        if (abs(ph_freq[i]) <= ph_fre_th):
            Ram_ten[i] = 0
        else:
            gcc_tmp = gcc[i][None, ...]
            gvv_tmp = gvv[i][None, ...]
            dipSp_res = elec_dip_absorp.conj() / (
                (ome_light_Ha - delta_energies - ph_freq[i])[None, ...])
            dipSp_ares = elec_dip_absorp / (
                (ome_light_Ha + delta_energies - ph_freq[i])[None, ...])
            # 1) Compute the resonant term
            tmp_tensor = gcc_tmp.conj() @ dipSp_res - dipSp_res @ gvv_tmp.conj()
            Ram_ten[i, :npol, :npol] = dipS_res @ tmp_tensor.reshape(npol, -1).T
            # 2) anti resonant
            tmp_tensor = gcc_tmp @ dipSp_ares - dipSp_ares @ gvv_tmp
            Ram_ten[i, :npol, :npol] += dipS_ares @ tmp_tensor.reshape(
                npol, -1).T
            ## scale to get raman tensor
            Ram_ten[i] *= (
                math.sqrt(math.fabs(ome_light_Ha - ph_freq[i]) / ome_light_Ha) *
                ram_fac)
    return Ram_ten


#
#
###### // Two phonon Raman ##############
#
def compute_Raman_twoph_iq2(ome_light,
                           ph_freq,
                           Qp_ene,
                           elec_dip,
                           gkkp,
                           kpts,
                           qpts,
                           CellVol,
                           broading=0.1,
                           npol=3,
                           ktree=None,
                            out_freq=False
                           ):
    """
    Computes the resonant Raman tensor for four distinct two-phonon processes at the
    independent particle level.

    The function calculates contributions from 16 scattering pathways in total.

    Processes calculated:
    1.  Anti-Stokes (two-phonon absorption, AA)
    2.  Stokes (two-phonon emission, EE)
    3.  Absorb-Emit (absorption of phonon lambda, emission of phonon l, AE)
    4.  Emit-Absorb (emission of phonon lambda, absorption of phonon l, EA)

    Args:
        ome_light (float): Incident light energy in eV.
        ph_freq (np.ndarray): Phonon frequencies (nq, n_modes) in Hartree.
        Qp_ene (np.ndarray): Quasiparticle energies (nk, nbnds) in Hartree.
        elec_dip (np.ndarray): Dipoles for light emission <c|v|v> (npol, nk, nc, nv) in Hartree.
        gkkp (np.ndarray): e-ph matrix elements for phonon absorption <k+q|dV|k>
                             (nq, n_modes, nk, nbands, nbands) in Hartree.
        kpts (np.ndarray): K-points in crystal coordinates.
        qpts (np.ndarray): Q-points in crystal coordinates.
        CellVol (float): Cell volume in atomic units.
        broading (float): Broadening parameter in eV.
        npol (int): Number of polarizations.
        ktree (KDTree, optional): Pre-computed KDTree for k-points.

    Returns:
        np.ndarray: The two-phonon Raman tensor with shape
                    (4, nq, n_modes, n_modes, npol, npol).
                    The first dimension corresponds to the process:
                    0: Anti-Stokes (AA)
                    1: Stokes (EE)
                    2: Absorb-Emit (AE)
                    3: Emit-Absorb (EA)
    """
    nk, nc, nv = elec_dip.shape[1:]
    nbnds = nc + nv
    assert (nbnds == Qp_ene.shape[1]), "Band number mismatch."
    assert (nk == Qp_ene.shape[0]), "K-point number mismatch."
    assert (gkkp.shape[2:4] == (nk, nbnds)), "gkkp dimensions are incompatible."

    nq, n_modes = gkkp.shape[:2]
    assert (nq == len(qpts) and nq == len(ph_freq)), "Q-point number mismatch."

    elec_dip_absorp = elec_dip[:npol, ...].conj()

    broading_Ha = broading / 27.211 / 2.0
    ome_light_Ha = ome_light / 27.211

    delta_energies = ome_light_Ha - Qp_ene[:, nv:,
                                           None] + Qp_ene[:, None, :
                                                          nv] + 1j * broading_Ha
    dipS_res = elec_dip_absorp / delta_energies[None, :]

    tol = 1e-6
    if ktree is None:
        kpos = (kpts + tol) % 1
        ktree = KDTree(kpos, boxsize=[1, 1, 1])

    twoph_raman_ten = np.zeros((4, nq, n_modes, n_modes, 3, 3),
                               dtype=gkkp.dtype)
    out_freq_2ph = []
    if out_freq : out_freq_2ph = np.zeros((4,nq, n_modes,n_modes))
    for iq in range(nq):
        iqpt = qpts[iq]
        #
        # Find indices for -q, k+q, k-q
        dist = qpts + iqpt[None]
        dist = dist - np.rint(dist)
        dist = np.linalg.norm(dist, axis=-1)
        minus_iq_idx = np.argmin(dist)
        dist = dist[minus_iq_idx]
        assert dist < 1e-5, "-q not found"
        #
        kplusq = (kpts + iqpt[None, :] + tol) % 1
        dist_kpq, idx_kplusq = ktree.query(kplusq, workers=-1)
        assert np.max(dist_kpq) < 1e-5, "k+q not found"
        #
        kminusq = (kpts - iqpt[None, :] + tol) % 1
        dist_kmq, idx_kminusq = ktree.query(kminusq, workers=-1)
        assert np.max(dist_kmq) < 1e-5, "k-q not found"
        # Ec - Ev
        delta_energies_kpq = delta_energies[idx_kplusq]
        delta_energies_kmq = delta_energies[idx_kminusq]
        delta_energies_kqc_kv = ome_light_Ha - Qp_ene[
            idx_kplusq, nv:, None] + Qp_ene[:, None, :nv] + 1j * broading_Ha
        delta_energies_kc_kmqv = ome_light_Ha - Qp_ene[:, nv:, None] + Qp_ene[
            idx_kminusq, None, :nv] + 1j * broading_Ha
        # gkkps for (k,q), (k,-q), (k+q,k), (k-q,k)
        g_q = gkkp[iq]
        g_mq = gkkp[minus_iq_idx]
        #
        # Make C conitgious to minimize Cache misses
        gcc_k_q = g_q[:, :, nv:, nv:]
        gvv_k_q =g_q[:, :, :nv, :nv]
        gcc_kpq_mq =g_mq[:, idx_kplusq, nv:, nv:]
        gvv_kpq_mq =g_mq[:, idx_kplusq, :nv, :nv]
        gcc_k_mq =g_mq[:, :, nv:, nv:]
        gvv_k_mq =g_mq[:, :, :nv, :nv]
        gcc_kmq_q =g_q[:, idx_kminusq, nv:, nv:]
        gvv_kmq_q =g_q[:, idx_kminusq, :nv, :nv]
        # phonon freqs
        ph_freq_q = ph_freq[iq]
        ph_freq_mq = ph_freq[minus_iq_idx]
        #
        for il in range(n_modes):  # First phonon
            for jl in range(n_modes):  # Second phonon
                #
                # ==============================================================================
                # Process 0: ANTI-STOKES (Absorb/Absorb)
                # Phonon (l, -q) and Phonon (lambda, q) are absorbed.
                # ==============================================================================
                #
                # Einsum are retained only for understanding what is going is not
                # optimzed.
                #
                ph_sum_aa = ph_freq_mq[jl] + ph_freq_q[il]
                ram_fac = np.sqrt(
                    np.abs(ome_light_Ha + ph_sum_aa) / ome_light_Ha)
                #
                # M1 (E-E)
                G1 = ram_fac * elec_dip_absorp.conj() / (
                    ph_sum_aa + delta_energies)[None, ...]
                G2 = 1.0 / (delta_energies_kqc_kv + ph_freq_q[il])
                # tmp = np.einsum('ykCv,kCc,kcv,kcp,xkpv->xy',
                #                 G1,
                #                 gcc_kpq_mq[jl],
                #                 G2,
                #                 gcc_k_q[il],
                #                 dipS_res,
                #                 optimize=True)
                tmp1 = ((gcc_kpq_mq[jl].transpose(0,2,1)@G1)*G2).reshape(npol,-1).T
                tmp2 = (gcc_k_q[il]@dipS_res).reshape(npol,-1)
                twoph_raman_ten[0, iq, il, jl, :npol, :npol] += tmp2@tmp1
                #
                # M2 (H-H)
                G1 = ram_fac * elec_dip_absorp.conj() / (
                    ph_sum_aa + delta_energies)[None, ...]
                G2 = 1.0 / (delta_energies_kc_kmqv + ph_freq_q[il])
                # tmp = np.einsum('ykcV,kvV,kcv,kpv,xkcp->xy',
                #                 G1,
                #                 gvv_k_mq[jl],
                #                 G2,
                #                 gvv_kmq_q[il],
                #                 dipS_res,
                #                 optimize=True)
                tmp1 = ((G1@gvv_k_mq[jl].transpose(0,2,1))*G2).reshape(npol,-1).T
                tmp2 = (dipS_res@gvv_kmq_q[il]).reshape(npol,-1)
                twoph_raman_ten[0, iq, il, jl, :npol, :npol] += tmp2@tmp1
                #
                # M3 (E-H)
                G1 = ram_fac * elec_dip_absorp[:, idx_kplusq].conj() / (
                    ph_sum_aa + delta_energies_kpq)[None, ...]
                G2 = 1.0 / (delta_energies_kqc_kv + ph_freq_q[il])
                # tmp = np.einsum('ykCV,kvV,kCv,kCc,xkcv->xy',
                #                 G1,
                #                 gvv_kpq_mq[jl],
                #                 G2,
                #                 gcc_k_q[il],
                #                 dipS_res,
                #                 optimize=True)
                tmp1 = (G1@gvv_kpq_mq[jl].transpose(0,2,1))*G2
                tmp2 = (gcc_k_q[il].transpose(0,2,1)@tmp1).reshape(npol,-1).T
                twoph_raman_ten[0, iq, il, jl, :npol, :npol] -= dipS_res.reshape(npol,-1)@tmp2
                #
                # M4 (H-E)
                G1 = ram_fac * elec_dip_absorp[:, idx_kminusq].conj() / (
                    ph_sum_aa + delta_energies_kmq)[None, ...]
                G2 = 1.0 / (delta_energies_kc_kmqv + ph_freq_q[il])
                # tmp = np.einsum('ykCV,kCc,kcV,kvV,xkcv->xy',
                #                 G1,
                #                 gcc_k_mq[jl],
                #                 G2,
                #                 gvv_kmq_q[il],
                #                 dipS_res,
                #                 optimize=True)
                tmp1 = (gcc_k_mq[jl].transpose(0,2,1)@G1)*G2
                tmp2 = (tmp1@gvv_kmq_q[il].transpose(0,2,1)).reshape(npol,-1).T
                twoph_raman_ten[0, iq, il, jl, :npol, :npol] -= dipS_res.reshape(npol,-1)@tmp2
                #
                # ==============================================================================
                # Process 1: STOKES (Emit/Emit, EE)
                # Phonon (l, q) and Phonon (lambda, -q) are emitted.
                # ==============================================================================
                ph_sum_ee = -ph_freq_q[jl] - ph_freq_mq[il]
                ram_fac = np.sqrt(
                    np.abs(ome_light_Ha + ph_sum_ee) / ome_light_Ha)
                #
                # M1 (E-E)
                G1 = ram_fac * elec_dip_absorp.conj() / (
                    ph_sum_ee + delta_energies)[None, ...]
                G2 = 1.0 / (delta_energies_kqc_kv - ph_freq_mq[il])
                # Note: (g^l)* is emission of l, (g^lambda)* is emission of lambda
                # tmp = np.einsum('ykCv,kcC,kcv,kpc,xkpv->xy',
                #                 G1,
                #                 gcc_k_q[jl].conj(),
                #                 G2,
                #                 gcc_kpq_mq[il].conj(),
                #                 dipS_res,
                #                 optimize=True)
                tmp1 = ((gcc_k_q[jl].conj()@G1)*G2).reshape(npol,-1).T
                tmp2 = (gcc_kpq_mq[il].transpose(0,2,1).conj()@dipS_res).reshape(npol,-1)
                twoph_raman_ten[1, iq, il, jl, :npol, :npol] += tmp2@tmp1
                #
                # M2 (H-H)
                G1 = ram_fac * elec_dip_absorp.conj() / (
                    ph_sum_ee + delta_energies)[None, ...]
                G2 = 1.0 / (delta_energies_kc_kmqv - ph_freq_mq[il])
                # tmp = np.einsum('ykcV,kVv,kcv,kvp,xkcp->xy',
                #                 G1,
                #                 gvv_kmq_q[jl].conj(),
                #                 G2,
                #                 gvv_k_mq[il].conj(),
                #                 dipS_res,
                #                 optimize=True)
                tmp1 = ((G1@gvv_kmq_q[jl].conj())*G2).reshape(npol,-1).T
                tmp2 = (dipS_res@gvv_k_mq[il].transpose(0,2,1).conj()).reshape(npol,-1)
                twoph_raman_ten[1, iq, il, jl, :npol, :npol] += tmp2@tmp1
                #
                # M3 (E-H)
                G1 = ram_fac * elec_dip_absorp[:, idx_kplusq].conj() / (
                    ph_sum_ee + delta_energies_kpq)[None, ...]
                G2 = 1.0 / (delta_energies_kqc_kv - ph_freq_mq[il])
                # tmp = np.einsum('ykCV,kVv,kCv,kcC,xkcv->xy',
                #                 G1,
                #                 gvv_k_q[jl].conj(),
                #                 G2,
                #                 gcc_kpq_mq[il].conj(),
                #                 dipS_res,
                #                 optimize=True)
                tmp1 = (G1@gvv_k_q[jl].conj())*G2
                tmp2 = (gcc_kpq_mq[il].conj()@tmp1).reshape(npol,-1).T
                twoph_raman_ten[1, iq, il, jl, :npol, :npol] -= dipS_res.reshape(npol,-1)@tmp2
                #
                # M4 (H-E)
                G1 = ram_fac * elec_dip_absorp[:, idx_kminusq].conj() / (
                    ph_sum_ee + delta_energies_kmq)[None, ...]
                G2 = 1.0 / (delta_energies_kc_kmqv - ph_freq_mq[il])
                # tmp = np.einsum('ykCV,kcC,kcV,kVv,xkcv->xy',
                #                 G1,
                #                 gcc_kmq_q[jl].conj(),
                #                 G2,
                #                 gvv_k_mq[il].conj(),
                #                 dipS_res,
                #                 optimize=True)
                tmp1 = (gcc_kmq_q[jl].conj()@G1)*G2
                tmp2 = (tmp1@gvv_k_mq[il].conj()).reshape(npol,-1).T
                twoph_raman_ten[1, iq, il, jl, :npol, :npol] -= dipS_res.reshape(npol,-1)@tmp2
                #
                # ==============================================================================
                # Process 2: ABSORB/EMIT (AE)
                # Phonon (lambda, q) is absorbed, Phonon (l, q) is emitted.
                # ==============================================================================
                ph_sum_ae = ph_freq_q[il] - ph_freq_q[jl]
                ram_fac = np.sqrt(
                    np.abs(ome_light_Ha + ph_sum_ae) / ome_light_Ha)
                #
                # M1 (E-E)
                G1 = ram_fac * elec_dip_absorp.conj() / (
                    ph_sum_ae + delta_energies)[None, ...]
                G2 = 1.0 / (delta_energies_kqc_kv + ph_freq_q[il])
                # tmp = np.einsum('ykCv,kcC,kcv,kcp,xkpv->xy',
                #                 G1,
                #                 gcc_k_q[jl].conj(),
                #                 G2,
                #                 gcc_k_q[il],
                #                 dipS_res,
                #                 optimize=True)
                tmp1 = ((gcc_k_q[jl].conj()@G1)*G2).reshape(npol,-1).T
                tmp2 = (gcc_k_q[il]@dipS_res).reshape(npol,-1)
                twoph_raman_ten[2, iq, il, jl, :npol, :npol] += tmp2@tmp1
                #
                # M2 (H-H)
                G1 = ram_fac * elec_dip_absorp.conj() / (
                    ph_sum_ae + delta_energies)[None, ...]
                G2 = 1.0 / (delta_energies_kc_kmqv + ph_freq_q[il])
                # tmp = np.einsum('ykcV,kVv,kcv,kpv,xkcp->xy',
                #                 G1,
                #                 gvv_kmq_q[jl].conj(),
                #                 G2,
                #                 gvv_kmq_q[il],
                #                 dipS_res,
                #                 optimize=True)
                tmp1 = ((G1@gvv_kmq_q[jl].conj())*G2).reshape(npol,-1).T
                tmp2 = (dipS_res@gvv_kmq_q[il]).reshape(npol,-1)
                twoph_raman_ten[2, iq, il, jl, :npol, :npol] += tmp2@tmp1
                #
                # M3 (E-H)
                G1 = ram_fac * elec_dip_absorp[:, idx_kplusq].conj() / (
                    ph_sum_ae + delta_energies_kpq)[None, ...]
                G2 = 1.0 / (delta_energies_kqc_kv + ph_freq_q[il])
                # tmp = np.einsum('ykCV,kVv,kCv,kCc,xkcv->xy',
                #                 G1,
                #                 gvv_k_q[jl].conj(),
                #                 G2,
                #                 gcc_k_q[il],
                #                 dipS_res,
                #                 optimize=True)
                tmp1 = (G1@gvv_k_q[jl].conj())*G2
                tmp2 = (gcc_k_q[il].transpose(0,2,1)@tmp1).reshape(npol,-1).T
                twoph_raman_ten[2, iq, il, jl, :npol, :npol] -= dipS_res.reshape(npol,-1)@tmp2
                #
                # M4 (H-E)
                G1 = ram_fac * elec_dip_absorp[:, idx_kminusq].conj() / (
                    ph_sum_ae + delta_energies_kmq)[None, ...]
                G2 = 1.0 / (delta_energies_kc_kmqv + ph_freq_q[il])
                # tmp = np.einsum('ykCV,kcC,kcV,kvV,xkcv->xy',
                #                 G1,
                #                 gcc_kmq_q[jl].conj(),
                #                 G2,
                #                 gvv_kmq_q[il],
                #                 dipS_res,
                #                 optimize=True)
                tmp1 = (gcc_kmq_q[jl].conj()@G1)*G2
                tmp2 = (tmp1@gvv_kmq_q[il].transpose(0,2,1)).reshape(npol,-1).T
                twoph_raman_ten[2, iq, il, jl, :npol, :npol] -= dipS_res.reshape(npol,-1)@tmp2
                #
                # ==============================================================================
                # Process 3: EMIT/ABSORB (EA)
                # Phonon (lambda, -q) is emitted, Phonon (l, -q) is absorbed.
                # ==============================================================================
                ph_sum_ea = ph_freq_mq[jl] - ph_freq_mq[il]
                ram_fac = np.sqrt(
                    np.abs(ome_light_Ha + ph_sum_ea) / ome_light_Ha)
                #
                # M1 (E-E)
                G1 = ram_fac * elec_dip_absorp.conj() / (
                    ph_sum_ea + delta_energies)[None, ...]
                G2 = 1.0 / (delta_energies_kqc_kv - ph_freq_mq[il])
                # tmp = np.einsum('ykCv,kCc,kcv,kpc,xkpv->xy',
                #                 G1,
                #                 gcc_kpq_mq[jl],
                #                 G2,
                #                 gcc_kpq_mq[il].conj(),
                #                 dipS_res,
                #                 optimize=True)
                tmp1 = ((gcc_kpq_mq[jl].transpose(0,2,1)@G1)*G2).reshape(npol,-1).T
                tmp2 = (gcc_kpq_mq[il].transpose(0,2,1).conj()@dipS_res).reshape(npol,-1)
                twoph_raman_ten[3, iq, il, jl, :npol, :npol] += tmp2@tmp1
                #
                # M2 (H-H)
                G1 = ram_fac * elec_dip_absorp.conj() / (
                    ph_sum_ea + delta_energies)[None, ...]
                G2 = 1.0 / (delta_energies_kc_kmqv - ph_freq_mq[il])
                # tmp = np.einsum('ykcV,kvV,kcv,kvp,xkcp->xy',
                #                 G1,
                #                 gvv_k_mq[jl],
                #                 G2,
                #                 gvv_k_mq[il].conj(),
                #                 dipS_res,
                #                 optimize=True)
                tmp1 = ((G1@gvv_k_mq[jl].transpose(0,2,1))*G2).reshape(npol,-1).T
                tmp2 = (dipS_res@gvv_k_mq[il].transpose(0,2,1).conj()).reshape(npol,-1)
                twoph_raman_ten[3, iq, il, jl, :npol, :npol] += tmp2@tmp1
                #
                # M3 (E-H)
                G1 = ram_fac * elec_dip_absorp[:, idx_kplusq].conj() / (
                    ph_sum_ea + delta_energies_kpq)[None, ...]
                G2 = 1.0 / (delta_energies_kqc_kv - ph_freq_mq[il])
                # tmp = np.einsum('ykCV,kvV,kCv,kcC,xkcv->xy',
                #                 G1,
                #                 gvv_kpq_mq[jl],
                #                 G2,
                #                 gcc_kpq_mq[il].conj(),
                #                 dipS_res,
                #                 optimize=True)
                tmp1 = (G1@gvv_kpq_mq[jl].transpose(0,2,1))*G2
                tmp2 = (gcc_kpq_mq[il].conj()@tmp1).reshape(npol,-1).T
                twoph_raman_ten[3, iq, il, jl, :npol, :npol] -= dipS_res.reshape(npol,-1)@tmp2
                #
                # M4 (H-E)
                G1 = ram_fac * elec_dip_absorp[:, idx_kminusq].conj() / (
                    ph_sum_ea + delta_energies_kmq)[None, ...]
                G2 = 1.0 / (delta_energies_kc_kmqv - ph_freq_mq[il])
                # tmp = np.einsum('ykCV,kCc,kcV,kVv,xkcv->xy',
                #                 G1,
                #                 gcc_k_mq[jl],
                #                 G2,
                #                 gvv_k_mq[il].conj(),
                #                 dipS_res,
                #                 optimize=True)
                tmp1 = (gcc_k_mq[jl].transpose(0,2,1)@G1)*G2
                tmp2 = (tmp1@gvv_k_mq[il].conj()).reshape(npol,-1).T
                twoph_raman_ten[3, iq, il, jl, :npol, :npol] -= dipS_res.reshape(npol,-1)@tmp2
                if out_freq:
                    out_freq_2ph[0,iq,il,jl] = ph_sum_aa
                    out_freq_2ph[1,iq,il,jl] = ph_sum_ee
                    out_freq_2ph[2,iq,il,jl] = ph_sum_ae
                    out_freq_2ph[3,iq,il,jl] = ph_sum_ea

    #
    # muliply with prefactors.
    norm_factor = 1.0 / nk / math.sqrt(CellVol) / np.sqrt(nq)
    twoph_raman_ten *= norm_factor
    #
    if out_freq : return out_freq_2ph, twoph_raman_ten
    else : return twoph_raman_ten


import numpy as np
import numba as nb
import math
# Assuming you have the numba-kdtree library
# from numba_kdtree import KDTree

@nb.njit(parallel=True, cache=True, nogil=True)
def _compute_Raman_twoph_iq_numba(
    ome_light_Ha,
    ph_freq,
    Qp_ene,
    elec_dip_absorp,
    gkkp,
    kpts,
    qpts,
    broading_Ha,
    ktree,
    dipS_res
):
    """
    Numba-jitted core function for computing the two-phonon Raman tensor.
    This function is designed for performance and is not intended for direct use.
    """
    # Get dimensions from input arrays
    nk, nbnds = Qp_ene.shape
    npol, _, nc, nv = elec_dip_absorp.shape
    nq, n_modes = gkkp.shape[:2]

    # Initialize output arrays
    twoph_raman_ten = np.zeros((4, nq, n_modes, n_modes, npol, npol), dtype=gkkp.dtype)
    out_freq_2ph = np.zeros((4, nq, n_modes, n_modes), dtype=np.float64)

    # Pre-calculate once outside the loop
    delta_energies = ome_light_Ha - Qp_ene[:, nv:, None] + Qp_ene[:, None, :nv] + 1j * broading_Ha

    # Parallelize the main loop over q-points
    for iq in nb.prange(nq):
        iqpt = qpts[iq]
        tol = 1e-6

        # Find index for -q
        dist = qpts + iqpt[None, :]
        dist = dist - np.rint(dist)
        dist_norm = np.sqrt(np.sum(dist**2, axis=1))
        minus_iq_idx = np.argmin(dist_norm)

        kplusq = kpts + iqpt[None, :]
        diff = kpts[:,None,:]-kplusq[None,:,:]
        diff = diff - np.rint(diff)
        diff = np.sqrt(diff[:,:,0]**2 + diff[:,:,1]**2 + diff[:,:,2]**2)
        idx_kplusq = np.argmin(diff,axis=1)
        # Find indices for k+q and k-q using the KDTree
        # kplusq = (kpts + iqpt[None, :] + tol) % 1.0
        # dist, idx_kplusq = ktree.query(kplusq)

        kminusq = kpts - iqpt[None, :]
        diff = kpts[:,None,:]-kminusq[None,:,:]
        diff = diff - np.rint(diff)
        diff = np.sqrt(diff[:,:,0]**2 + diff[:,:,1]**2 + diff[:,:,2]**2)
        idx_kminusq = np.argmin(diff,axis=1)
        # kminusq = (kpts - iqpt[None, :] + tol) % 1.0
        # dist, idx_kminusq = ktree.query(kminusq)

        # Gather indexed arrays. Numba handles this advanced indexing efficiently.
        delta_energies_kpq = delta_energies[idx_kplusq]
        delta_energies_kmq = delta_energies[idx_kminusq]

        delta_energies_kqc_kv = ome_light_Ha - Qp_ene[idx_kplusq, nv:, None] + Qp_ene[:, None, :nv] + 1j * broading_Ha
        delta_energies_kc_kmqv = ome_light_Ha - Qp_ene[:, nv:, None] + Qp_ene[idx_kminusq, None, :nv] + 1j * broading_Ha

        # Sliced e-ph matrix elements for q and -q
        g_q = gkkp[iq]
        g_mq = gkkp[minus_iq_idx]

        gcc_k_q = g_q[:, :, nv:, nv:]
        gvv_k_q = g_q[:, :, :nv, :nv]
        gcc_kpq_mq = g_mq[:, idx_kplusq, nv:, nv:]
        gvv_kpq_mq = g_mq[:, idx_kplusq, :nv, :nv]
        gcc_k_mq = g_mq[:, :, nv:, nv:]
        gvv_k_mq = g_mq[:, :, :nv, :nv]
        gcc_kmq_q = g_q[:, idx_kminusq, nv:, nv:]
        gvv_kmq_q = g_q[:, idx_kminusq, :nv, :nv]

        # Sliced dipole elements
        elec_dip_absorp_kpq = elec_dip_absorp[:, idx_kplusq]
        elec_dip_absorp_kmq = elec_dip_absorp[:, idx_kminusq]

        ph_freq_q = ph_freq[iq]
        ph_freq_mq = ph_freq[minus_iq_idx]

        for il in range(n_modes):
            for jl in range(n_modes):
                # PROCESS 0: ANTI-STOKES (AA)
                ph_sum_aa = ph_freq_mq[jl] + ph_freq_q[il]
                ram_fac = np.sqrt(np.abs(ome_light_Ha + ph_sum_aa) / ome_light_Ha)

                # M1
                G1 = ram_fac * elec_dip_absorp.conj() / (ph_sum_aa + delta_energies)
                G2 = 1.0 / (delta_energies_kqc_kv + ph_freq_q[il])
                tmp1 = (np.matmul(gcc_kpq_mq[jl].transpose(0, 2, 1), G1) * G2).reshape(npol, -1).T
                tmp2 = np.matmul(gcc_k_q[il], dipS_res).reshape(npol, -1)
                twoph_raman_ten[0, iq, il, jl, :, :] += tmp2 @ tmp1

                # M2
                G1 = ram_fac * elec_dip_absorp.conj() / (ph_sum_aa + delta_energies)
                G2 = 1.0 / (delta_energies_kc_kmqv + ph_freq_q[il])
                tmp1 = (np.matmul(G1, gvv_k_mq[jl].transpose(0, 2, 1)) * G2).reshape(npol, -1).T
                tmp2 = np.matmul(dipS_res, gvv_kmq_q[il]).reshape(npol, -1)
                twoph_raman_ten[0, iq, il, jl, :, :] += tmp2 @ tmp1

                # M3
                G1 = ram_fac * elec_dip_absorp_kpq.conj() / (ph_sum_aa + delta_energies_kpq)
                G2 = 1.0 / (delta_energies_kqc_kv + ph_freq_q[il])
                tmp1 = np.matmul(G1, gvv_kpq_mq[jl].transpose(0, 2, 1)) * G2
                tmp2 = np.matmul(gcc_k_q[il].transpose(0, 2, 1), tmp1).reshape(npol, -1).T
                twoph_raman_ten[0, iq, il, jl, :, :] -= dipS_res.reshape(npol, -1) @ tmp2

                # M4
                G1 = ram_fac * elec_dip_absorp_kmq.conj() / (ph_sum_aa + delta_energies_kmq)
                G2 = 1.0 / (delta_energies_kc_kmqv + ph_freq_q[il])
                tmp1 = np.matmul(gcc_k_mq[jl].transpose(0, 2, 1), G1) * G2
                tmp2 = np.matmul(tmp1, gvv_kmq_q[il].transpose(0, 2, 1)).reshape(npol, -1).T
                twoph_raman_ten[0, iq, il, jl, :, :] -= dipS_res.reshape(npol, -1) @ tmp2

                # PROCESS 1: STOKES (EE)
                ph_sum_ee = -ph_freq_q[jl] - ph_freq_mq[il]
                ram_fac = np.sqrt(np.abs(ome_light_Ha + ph_sum_ee) / ome_light_Ha)

                # M1
                G1 = ram_fac * elec_dip_absorp.conj() / (ph_sum_ee + delta_energies)
                G2 = 1.0 / (delta_energies_kqc_kv - ph_freq_mq[il])
                tmp1 = (np.matmul(gcc_k_q[jl].conj(), G1) * G2).reshape(npol, -1).T
                tmp2 = np.matmul(gcc_kpq_mq[il].transpose(0, 2, 1).conj(), dipS_res).reshape(npol, -1)
                twoph_raman_ten[1, iq, il, jl, :, :] += tmp2 @ tmp1

                # M2
                G1 = ram_fac * elec_dip_absorp.conj() / (ph_sum_ee + delta_energies)
                G2 = 1.0 / (delta_energies_kc_kmqv - ph_freq_mq[il])
                tmp1 = (np.matmul(G1, gvv_kmq_q[jl].conj()) * G2).reshape(npol, -1).T
                tmp2 = np.matmul(dipS_res, gvv_k_mq[il].transpose(0, 2, 1).conj()).reshape(npol, -1)
                twoph_raman_ten[1, iq, il, jl, :, :] += tmp2 @ tmp1

                # M3
                G1 = ram_fac * elec_dip_absorp_kpq.conj() / (ph_sum_ee + delta_energies_kpq)
                G2 = 1.0 / (delta_energies_kqc_kv - ph_freq_mq[il])
                tmp1 = np.matmul(G1, gvv_k_q[jl].conj()) * G2
                tmp2 = np.matmul(gcc_kpq_mq[il].conj(), tmp1).reshape(npol, -1).T
                twoph_raman_ten[1, iq, il, jl, :, :] -= dipS_res.reshape(npol, -1) @ tmp2

                # M4
                G1 = ram_fac * elec_dip_absorp_kmq.conj() / (ph_sum_ee + delta_energies_kmq)
                G2 = 1.0 / (delta_energies_kc_kmqv - ph_freq_mq[il])
                tmp1 = np.matmul(gcc_kmq_q[jl].conj(), G1) * G2
                tmp2 = np.matmul(tmp1, gvv_k_mq[il].conj()).reshape(npol, -1).T
                twoph_raman_ten[1, iq, il, jl, :, :] -= dipS_res.reshape(npol, -1) @ tmp2

                # PROCESS 2: ABSORB/EMIT (AE)
                ph_sum_ae = ph_freq_q[il] - ph_freq_q[jl]
                ram_fac = np.sqrt(np.abs(ome_light_Ha + ph_sum_ae) / ome_light_Ha)

                # M1
                G1 = ram_fac * elec_dip_absorp.conj() / (ph_sum_ae + delta_energies)
                G2 = 1.0 / (delta_energies_kqc_kv + ph_freq_q[il])
                tmp1 = (np.matmul(gcc_k_q[jl].conj(), G1) * G2).reshape(npol, -1).T
                tmp2 = np.matmul(gcc_k_q[il], dipS_res).reshape(npol, -1)
                twoph_raman_ten[2, iq, il, jl, :, :] += tmp2 @ tmp1

                # M2
                G1 = ram_fac * elec_dip_absorp.conj() / (ph_sum_ae + delta_energies)
                G2 = 1.0 / (delta_energies_kc_kmqv + ph_freq_q[il])
                tmp1 = (np.matmul(G1, gvv_kmq_q[jl].conj()) * G2).reshape(npol, -1).T
                tmp2 = np.matmul(dipS_res, gvv_kmq_q[il]).reshape(npol, -1)
                twoph_raman_ten[2, iq, il, jl, :, :] += tmp2 @ tmp1

                # M3
                G1 = ram_fac * elec_dip_absorp_kpq.conj() / (ph_sum_ae + delta_energies_kpq)
                G2 = 1.0 / (delta_energies_kqc_kv + ph_freq_q[il])
                tmp1 = np.matmul(G1, gvv_k_q[jl].conj()) * G2
                tmp2 = np.matmul(gcc_k_q[il].transpose(0, 2, 1), tmp1).reshape(npol, -1).T
                twoph_raman_ten[2, iq, il, jl, :, :] -= dipS_res.reshape(npol, -1) @ tmp2

                # M4
                G1 = ram_fac * elec_dip_absorp_kmq.conj() / (ph_sum_ae + delta_energies_kmq)
                G2 = 1.0 / (delta_energies_kc_kmqv + ph_freq_q[il])
                tmp1 = np.matmul(gcc_kmq_q[jl].conj(), G1) * G2
                tmp2 = np.matmul(tmp1, gvv_kmq_q[il].transpose(0, 2, 1)).reshape(npol, -1).T
                twoph_raman_ten[2, iq, il, jl, :, :] -= dipS_res.reshape(npol, -1) @ tmp2

                # PROCESS 3: EMIT/ABSORB (EA)
                ph_sum_ea = ph_freq_mq[jl] - ph_freq_mq[il]
                ram_fac = np.sqrt(np.abs(ome_light_Ha + ph_sum_ea) / ome_light_Ha)

                # M1
                G1 = ram_fac * elec_dip_absorp.conj() / (ph_sum_ea + delta_energies)
                G2 = 1.0 / (delta_energies_kqc_kv - ph_freq_mq[il])
                tmp1 = (np.matmul(gcc_kpq_mq[jl].transpose(0, 2, 1), G1) * G2).reshape(npol, -1).T
                tmp2 = np.matmul(gcc_kpq_mq[il].transpose(0, 2, 1).conj(), dipS_res).reshape(npol, -1)
                twoph_raman_ten[3, iq, il, jl, :, :] += tmp2 @ tmp1

                # M2
                G1 = ram_fac * elec_dip_absorp.conj() / (ph_sum_ea + delta_energies)
                G2 = 1.0 / (delta_energies_kc_kmqv - ph_freq_mq[il])
                tmp1 = (np.matmul(G1, gvv_k_mq[jl].transpose(0, 2, 1)) * G2).reshape(npol, -1).T
                tmp2 = np.matmul(dipS_res, gvv_k_mq[il].transpose(0, 2, 1).conj()).reshape(npol, -1)
                twoph_raman_ten[3, iq, il, jl, :, :] += tmp2 @ tmp1

                # M3
                G1 = ram_fac * elec_dip_absorp_kpq.conj() / (ph_sum_ea + delta_energies_kpq)
                G2 = 1.0 / (delta_energies_kqc_kv - ph_freq_mq[il])
                tmp1 = np.matmul(G1, gvv_kpq_mq[jl].transpose(0, 2, 1)) * G2
                tmp2 = np.matmul(gcc_kpq_mq[il].conj(), tmp1).reshape(npol, -1).T
                twoph_raman_ten[3, iq, il, jl, :, :] -= dipS_res.reshape(npol, -1) @ tmp2

                # M4
                G1 = ram_fac * elec_dip_absorp_kmq.conj() / (ph_sum_ea + delta_energies_kmq)
                G2 = 1.0 / (delta_energies_kc_kmqv - ph_freq_mq[il])
                tmp1 = np.matmul(gcc_k_mq[jl].transpose(0, 2, 1), G1) * G2
                tmp2 = np.matmul(tmp1, gvv_k_mq[il].conj()).reshape(npol, -1).T
                twoph_raman_ten[3, iq, il, jl, :, :] -= dipS_res.reshape(npol, -1) @ tmp2

                # Store frequencies
                out_freq_2ph[0, iq, il, jl] = ph_sum_aa
                out_freq_2ph[1, iq, il, jl] = ph_sum_ee
                out_freq_2ph[2, iq, il, jl] = ph_sum_ae
                out_freq_2ph[3, iq, il, jl] = ph_sum_ea

    return twoph_raman_ten, out_freq_2ph


def compute_Raman_twoph_iq(
    ome_light,
    ph_freq,
    Qp_ene,
    elec_dip,
    gkkp,
    kpts,
    qpts,
    CellVol,
    broading=0.1,
    npol=3,
    ktree=None,
    out_freq=False
):
    """
    Computes the resonant Raman tensor for four distinct two-phonon processes at the
    independent particle level, accelerated with Numba.

    Args:
        (Same as the original function)

    Returns:
        np.ndarray or tuple:
            If out_freq is False: The two-phonon Raman tensor (4, nq, n_modes, n_modes, npol, npol).
            If out_freq is True: A tuple containing (out_freq_2ph, twoph_raman_ten).
    """
    nk, nc, nv = elec_dip.shape[1:]
    nbnds = nc + nv
    nq = len(qpts)

    # --- Input Validation ---
    if nbnds != Qp_ene.shape[1]:
        raise ValueError("Band number mismatch between Qp_ene and elec_dip.")
    if nk != Qp_ene.shape[0]:
        raise ValueError("K-point number mismatch between Qp_ene and elec_dip.")
    if gkkp.shape[2:4] != (nk, nbnds):
        raise ValueError("gkkp dimensions are incompatible.")
    if not (nq == len(qpts) and nq == len(ph_freq)):
        raise ValueError("Q-point number mismatch.")

    # --- Pre-computation ---
    elec_dip_absorp = elec_dip[:npol, ...].conj()
    broading_Ha = broading / 27.211 / 2.0
    ome_light_Ha = ome_light / 27.211

    delta_energies = ome_light_Ha - Qp_ene[:, nv:, None] + Qp_ene[:, None, :nv] + 1j * broading_Ha
    dipS_res = elec_dip_absorp / delta_energies[None, :]

    # Handle KDTree creation outside the jitted function
    if ktree is None:
        # Lazy import if needed
        #from numba_kdtree import KDTree
        tol = 1e-6
        kpos = (kpts + tol) % 1.0
        ktree = KDTree(kpos)

    # --- Call the JIT-compiled Core Function ---
    twoph_raman_ten, out_freq_2ph = _compute_Raman_twoph_iq_numba(
        ome_light_Ha, ph_freq, Qp_ene, elec_dip_absorp, gkkp, kpts, qpts,
        broading_Ha, ktree, dipS_res
    )

    # --- Final Normalization ---
    norm_factor = 1.0 / nk / np.sqrt(CellVol) / np.sqrt(nq)
    twoph_raman_ten *= norm_factor

    # --- Return based on user request ---
    if out_freq:
        return out_freq_2ph, twoph_raman_ten
    else:
        return twoph_raman_ten
#
#
######################## TESTING ##############################
######################## TESTING ##############################
######################## TESTING ##############################
######################## TESTING ##############################
######################## TESTING ##############################
#### tests..
def test_compute_Raman_oneph_exc():
    # Test input parameters
    ome_light = 2.0  # in eV
    ph_freq = np.array([100.0, 200.0,
                        50.0])  # in cm-1 (one will be below threshold)
    ex_ene = np.array([1.8, 2.2])  # exciton energies in Hartree
    ex_dip = np.array([  # exciton dipoles (3 components for 2 excitons)
        [1.0 + 0.1j, 0.2 + 0.0j], [0.0 + 0.3j, 0.4 + 0.2j],
        [0.5 + 0.0j, 0.6 + 0.1j]
    ])
    ex_ph = np.arange(3 * 2 * 2) * 1.9 + 1j * np.arange(3 * 2 * 2) * 0.2
    ex_ph = ex_ph.reshape(3, 2, 2)
    nkpts = 10
    CellVol = 100.0
    broadening = 0.1  # default value
    npol = 3  # default value
    ph_fre_th = 60.0  # threshold in cm-1 (50.0 cm-1 mode should be filtered out)
    result = compute_Raman_oneph_exc(ome_light, ph_freq, ex_ene, ex_dip, ex_ph,
                                     nkpts, CellVol, broadening, npol,
                                     ph_fre_th).reshape(-1)
    ref_res = np.array([
        8.7455621e-05 - 0.00072732j, 1.4466088e-04 + 0.00279391j,
        2.1724285e-04 - 0.00030287j, 5.7119269e-05 - 0.00316499j,
        6.6436754e-05 - 0.00164784j, 8.2043938e-05 - 0.00568365j,
        1.1587636e-04 - 0.00145294j, 1.2217960e-04 + 0.00336734j,
        2.0013885e-04 - 0.00097748j, 1.0483168e-03 - 0.00385753j,
        5.2506319e-04 + 0.0127307j, 1.0985762e-03 - 0.0030669j, 2.8223562e-04 -
        0.01480559j, 2.9438769e-04 - 0.00248879j, 2.9685081e-04 - 0.01548769j,
        8.7379210e-04 - 0.00378421j, 4.2504439e-04 + 0.01157149j,
        9.0361061e-04 - 0.00303779j, 2.8493311e-04 - 0.01441317j,
        -2.6725902e-04 + 0.04701047j, -1.4930009e-04 - 0.01184826j,
        -1.1350836e-04 - 0.05479187j, -1.9741847e-04 - 0.00763448j,
        -2.6681539e-04 - 0.05396536j, -6.6271023e-05 - 0.01309499j,
        -4.0244934e-04 + 0.04156112j, -4.5003719e-04 - 0.01077925j
    ])
    return np.abs(result - ref_res).max() < 1e-6


def test_compute_Raman_oneph_ip():
    # Test parameters
    ome_light = 2.0  # in eV
    ph_freq = np.array([50.0, 100.0,
                        200.0])  # in cm-1 (first mode below threshold)
    broading = 0.1
    npol = 3
    ph_fre_th = 60.0  # cm-1 threshold (50 cm-1 mode should be filtered)
    nk = 2  # number of k-points
    nv = 2  # number of valence bands
    nc = 2  # number of conduction bands
    CellVol = 100.0

    # Create test data with appropriate shapes
    # QP energies (nk, nbands) where nbands = nv + nc
    Qp_ene = 1 + 0.5 * np.arange(nk * (nv + nc)).reshape(nk, nv + nc)

    # Electronic dipoles (3, nk, nc, nv)
    elec_dip = np.linspace(0, 1,
                           num=npol * nk * nc * nv).reshape(npol, nk, nc, nv)
    elec_dip = elec_dip * 10.981 + 1j * elec_dip * 1.890

    # Electron-phonon matrix elements (nmodes, nk, nbands, nbands)
    gkkp = np.linspace(4, 10,
                       num=len(ph_freq) * nk * (nv + nc) * (nv + nc)).reshape(
                           len(ph_freq), nk, nv + nc, nv + nc)
    gkkp = gkkp * 1.98756 + 1j * 0.92384 * gkkp

    # Call the function
    result = compute_Raman_oneph_ip(ome_light=ome_light,
                                    ph_freq=ph_freq,
                                    Qp_ene=Qp_ene,
                                    elec_dip=elec_dip,
                                    gkkp=gkkp,
                                    CellVol=CellVol,
                                    broading=broading,
                                    npol=npol,
                                    ph_fre_th=ph_fre_th)
    result = result.reshape(-1)
    ref_res = np.array([
        0.87344817 - 2.97095471j, 1.52861965 - 6.4944297j, 2.18379112 -
        10.0179047j, 2.81285261 - 10.02515497j, 4.77328589 - 21.54322432j,
        6.73371918 - 33.06129366j, 4.75225705 - 17.07935523j,
        8.01795214 - 36.59201893j, 11.28364723 - 56.10468262j,
        0.79414948 - 2.33852229j, 1.28720643 - 4.83054824j,
        1.78026338 - 7.32257418j, 2.70392822 - 8.39249515j,
        4.24140655 - 16.5385661j, 5.77888489 - 24.68463704j,
        4.61370695 - 14.44646801j, 7.19560667 - 28.24658396j,
        9.7775064 - 42.04669991j, 0.67517089 - 1.82149895j,
        1.02840615 - 3.58368937j, 1.3816414 - 5.34587979j,
        2.38392375 - 6.85606281j, 3.51857799 - 12.61695823j,
        4.65323223 - 18.37785365j, 4.09267661 - 11.89062666j,
        6.00874983 - 21.65022709j, 7.92482305 - 31.40982752j
    ])
    return np.abs(result - ref_res).max() < 1e-6


def test_compute_raman_twoph():
    # --- 1. Define Test Parameters ---
    nk = 4  # Number of k-points
    nq = 2  # Number of q-points
    nv = 2  # Number of valence bands
    nc = 2  # Number of conduction bands
    nbnds = nv + nc
    n_modes = 3  # Number of phonon modes
    npol = 3  # Number of polarizations
    ome_light = 2.5  # Incident light energy in eV
    broading = 0.1  # Broadening in eV
    CellVol = 100.0  # Cell volume in a.u.

    Qp_ene = np.linspace(-5, 5, nk * nbnds).reshape(
        nk, nbnds) / 27.211  # In Hartree
    ph_freq = np.linspace(10, 50, nq * n_modes).reshape(
        nq, n_modes) / 219474.63  # cm-1 to Ha

    dip_real = np.arange(1, npol * nk * nc * nv + 1).reshape(npol, nk, nc, nv)
    dip_imag = np.arange(npol * nk * nc * nv + 1,
                         2 * npol * nk * nc * nv + 1).reshape(npol, nk, nc, nv)
    elec_dip = (dip_real + 1j * dip_imag) * 1e-3

    g_real = np.arange(1, nq * n_modes * nk * nbnds * nbnds + 1).reshape(
        nq, n_modes, nk, nbnds, nbnds)
    g_imag = np.arange(nq * n_modes * nk * nbnds * nbnds + 1,
                       2 * nq * n_modes * nk * nbnds * nbnds + 1).reshape(
                           nq, n_modes, nk, nbnds, nbnds)
    gkkp = (g_real + 1j * g_imag) * 1e-4

    kpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0],
                     [0.5, 0.5, 0.0]])
    qpts = np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.0]])
    k_map = np.roll(kpts, -1, axis=0)
    q_map = np.array([qpts[1], qpts[0]])
    #
    result = compute_Raman_twoph_iq(ome_light=ome_light,
                                           ph_freq=ph_freq,
                                           Qp_ene=Qp_ene,
                                           elec_dip=elec_dip,
                                           gkkp=gkkp,
                                           kpts=kpts,
                                           qpts=qpts,
                                           CellVol=CellVol,
                                           broading=broading,
                                           npol=npol,
                                           ktree=None)
    result = result.reshape(4,-1)
    rand_pick1 = np.array([ 127, 150, 116, 68, 67, 81, 74, 155, 98, 115, 84, 131, 40, 140, 110, 46 ])
    rand_pick2 = np.array([ 138, 57, 59, 74, 146, 155, 40, 149, 157, 113, 120, 148, 40, 24, 85, 142 ])
    rand_pick3 = np.array([ 47, 53, 40, 154, 94, 22, 127, 121, 40, 100, 12, 29, 24, 21, 116, 99 ])
    rand_pick4 = np.array([ 97, 82, 141, 109, 156, 131, 20, 5, 11, 93, 46, 141, 131, 2, 143, 0 ])
    #
    ref_res = np.array([
        [0.01192986-0.03019767j, 0.02978467-0.02869628j, 0.03233621-0.04474664j,
         0.00989246-0.00941383j, 0.00859875-0.00660231j, 0.01107977-0.0134547j,
         0.0062875-0.00994134j, 0.01038468-0.04452347j, 0.03249347-0.04408665j,
         0.02903444-0.03264489j, 0.01741703-0.01513064j, 0.02323789-0.04918927j,
         0.00795517-0.00481848j, 0.02260384-0.04321622j, 0.01148852-0.03062815j,
         0.00566335-0.00570336j],
        [0.02151343+0.02055804j, 0.00672177+0.00179631j,
         0.011762-0.00009826j, 0.01222663+0.00094123j,
         0.03954748+0.01592407j, 0.04352852+0.0198752j,
         0.00934234+0.00156097j, 0.04821254+0.02968255j,
         0.03858684+0.03229266j, 0.04016525+0.01972921j,
         0.0215035+0.0213518j, 0.0356333+0.02719178j,
         0.00934234+0.00156097j, 0.00800414+0.00380962j,
         0.02745707+0.01531404j, 0.03825257+0.03191573j],
        [-0.00831116+0.00310714j, -0.01487569+0.0010705j, -0.00783106+0.00143035j,
         -0.03291897+0.01215251j, -0.0324311+0.00416147j, -0.00790955-0.00069783j,
         -0.02988729+0.00951316j, -0.03637175+0.00682201j, -0.00783106+0.00143035j,
         -0.02685074+0.00686733j, -0.00507707-0.00083575j, -0.0052129+0.00380744j,
         -0.00707233-0.00283462j, -0.00581241-0.00164083j, -0.05042094+0.01252742j,
         -0.02097336+0.00165968j],
        [-0.04679024+0.00660723j, -0.02344677+0.00876329j, -0.03920278-0.00641103j,
         -0.02685453+0.00859138j, -0.03927349+0.00089543j, -0.05594197+0.01907144j,
         -0.00811318+0.00611459j, -0.00962123+0.00292443j, -0.00741523+0.00473135j,
         -0.02806981+0.00200005j, -0.00854872+0.00401467j, -0.03920278-0.00641103j,
         -0.05594197+0.01907144j, -0.00671469+0.00334149j, -0.0664776+0.00702562j,
         -0.00441558+0.00086929j]
    ])
    max1 = np.max(np.abs(result[0][rand_pick1]-ref_res[0]))
    max2 = np.max(np.abs(result[1][rand_pick2]-ref_res[1]))
    max3 = np.max(np.abs(result[2][rand_pick3]-ref_res[2]))
    max4 = np.max(np.abs(result[3][rand_pick4]-ref_res[3]))
    max_final = max([max1,max2,max3,max4])
    return max_final < 1e-6


if __name__ == "__main__":
    print(test_compute_Raman_oneph_exc())
    print(test_compute_Raman_oneph_ip())
    print(test_compute_raman_twoph())
