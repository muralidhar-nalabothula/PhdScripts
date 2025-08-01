## Compute resonant Raman intensities
## Sven Reichardt, etal Sci. Adv.6,eabb5915(2020)
import numpy as np
import math
from scipy.spatial import KDTree


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
def compute_Raman_twoph_iq(ome_light,
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
                           out_freq=False,
                           contrib="all"
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
        out_freq (bool): If True, also returns the phonon frequency shifts.
        contrib (str): Specifies which contributions to compute.
                       "all": Compute all terms (M1, M2, M3, M4) (default).
                       "ee": Compute only electron-electron (M1) terms.
                       "hh": Compute only hole-hole (M2) terms.
                       "eh": Compute only electron-hole (M3) terms.
                       "he": Compute only hole-electron (M4) terms.

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

    # Process contrib argument
    contrib_lower = contrib.lower()
    valid_contribs = ["all", "ee", "hh", "eh", "he"]
    if contrib_lower not in valid_contribs:
        raise ValueError(f"Invalid value for 'contrib'. "
                         f"Must be one of {valid_contribs}, but got '{contrib}'.")

    elec_dip_absorp = elec_dip[:npol, ...].conj()

    broading_Ha = broading / 27.211 / 2.0
    ome_light_Ha = ome_light / 27.211

    delta_energies = ome_light_Ha - Qp_ene[:, nv:,
                                           None] + Qp_ene[:, None, :
                                                           nv] + 1j * broading_Ha
    dipS_res = elec_dip_absorp / delta_energies[None, :]

    tol = 1e-6
    if ktree is None:
        kpos = kpts-np.floor(kpts)
        kpos = (kpos + tol) % 1
        ktree = KDTree(kpos, boxsize=[1, 1, 1])

    twoph_raman_ten = np.zeros((4, nq, n_modes, n_modes, 3, 3),
                               dtype=gkkp.dtype)
    out_freq_2ph = []
    if out_freq : out_freq_2ph = np.zeros((4,nq, n_modes,n_modes))

    for iq in range(nq):
        iqpt = qpts[iq]
        
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
        
        for il in range(n_modes):  # First phonon
            for jl in range(n_modes):  # Second phonon
                
                # ==============================================================================
                # Process 0: ANTI-STOKES (Absorb/Absorb)
                # ==============================================================================
                ph_sum_aa = ph_freq_mq[jl] + ph_freq_q[il]
                ram_fac = np.sqrt(
                    np.abs(ome_light_Ha + ph_sum_aa) / ome_light_Ha)
                
                if contrib_lower in ["all", "ee"]:
                    # M1 (E-E)
                    G1 = ram_fac * elec_dip_absorp.conj() / (
                        ph_sum_aa + delta_energies)[None, ...]
                    G2 = 1.0 / (delta_energies_kqc_kv + ph_freq_q[il])
                    tmp1 = ((gcc_kpq_mq[jl].transpose(0,2,1)@G1)*G2).reshape(npol,-1).T
                    tmp2 = (gcc_k_q[il]@dipS_res).reshape(npol,-1)
                    twoph_raman_ten[0, iq, il, jl, :npol, :npol] += tmp2@tmp1
                
                if contrib_lower in ["all", "hh"]:
                    # M2 (H-H)
                    G1 = ram_fac * elec_dip_absorp.conj() / (
                        ph_sum_aa + delta_energies)[None, ...]
                    G2 = 1.0 / (delta_energies_kc_kmqv + ph_freq_q[il])
                    tmp1 = ((G1@gvv_k_mq[jl].transpose(0,2,1))*G2).reshape(npol,-1).T
                    tmp2 = (dipS_res@gvv_kmq_q[il]).reshape(npol,-1)
                    twoph_raman_ten[0, iq, il, jl, :npol, :npol] += tmp2@tmp1
                
                if contrib_lower in ["all", "eh"]:
                    # M3 (E-H)
                    G1 = ram_fac * elec_dip_absorp[:, idx_kplusq].conj() / (
                        ph_sum_aa + delta_energies_kpq)[None, ...]
                    G2 = 1.0 / (delta_energies_kqc_kv + ph_freq_q[il])
                    tmp1 = (G1@gvv_kpq_mq[jl].transpose(0,2,1))*G2
                    tmp2 = (gcc_k_q[il].transpose(0,2,1)@tmp1).reshape(npol,-1).T
                    twoph_raman_ten[0, iq, il, jl, :npol, :npol] -= dipS_res.reshape(npol,-1)@tmp2
                
                if contrib_lower in ["all", "he"]:
                    # M4 (H-E)
                    G1 = ram_fac * elec_dip_absorp[:, idx_kminusq].conj() / (
                        ph_sum_aa + delta_energies_kmq)[None, ...]
                    G2 = 1.0 / (delta_energies_kc_kmqv + ph_freq_q[il])
                    tmp1 = (gcc_k_mq[jl].transpose(0,2,1)@G1)*G2
                    tmp2 = (tmp1@gvv_kmq_q[il].transpose(0,2,1)).reshape(npol,-1).T
                    twoph_raman_ten[0, iq, il, jl, :npol, :npol] -= dipS_res.reshape(npol,-1)@tmp2
                
                # ==============================================================================
                # Process 1: STOKES (Emit/Emit, EE)
                # ==============================================================================
                ph_sum_ee = -ph_freq_q[jl] - ph_freq_mq[il]
                ram_fac = np.sqrt(
                    np.abs(ome_light_Ha + ph_sum_ee) / ome_light_Ha)

                if contrib_lower in ["all", "ee"]:
                    # M1 (E-E)
                    G1 = ram_fac * elec_dip_absorp.conj() / (
                        ph_sum_ee + delta_energies)[None, ...]
                    G2 = 1.0 / (delta_energies_kqc_kv - ph_freq_mq[il])
                    tmp1 = ((gcc_k_q[jl].conj()@G1)*G2).reshape(npol,-1).T
                    tmp2 = (gcc_kpq_mq[il].transpose(0,2,1).conj()@dipS_res).reshape(npol,-1)
                    twoph_raman_ten[1, iq, il, jl, :npol, :npol] += tmp2@tmp1

                if contrib_lower in ["all", "hh"]:
                    # M2 (H-H)
                    G1 = ram_fac * elec_dip_absorp.conj() / (
                        ph_sum_ee + delta_energies)[None, ...]
                    G2 = 1.0 / (delta_energies_kc_kmqv - ph_freq_mq[il])
                    tmp1 = ((G1@gvv_kmq_q[jl].conj())*G2).reshape(npol,-1).T
                    tmp2 = (dipS_res@gvv_k_mq[il].transpose(0,2,1).conj()).reshape(npol,-1)
                    twoph_raman_ten[1, iq, il, jl, :npol, :npol] += tmp2@tmp1

                if contrib_lower in ["all", "eh"]:
                    # M3 (E-H)
                    G1 = ram_fac * elec_dip_absorp[:, idx_kplusq].conj() / (
                        ph_sum_ee + delta_energies_kpq)[None, ...]
                    G2 = 1.0 / (delta_energies_kqc_kv - ph_freq_mq[il])
                    tmp1 = (G1@gvv_k_q[jl].conj())*G2
                    tmp2 = (gcc_kpq_mq[il].conj()@tmp1).reshape(npol,-1).T
                    twoph_raman_ten[1, iq, il, jl, :npol, :npol] -= dipS_res.reshape(npol,-1)@tmp2

                if contrib_lower in ["all", "he"]:
                    # M4 (H-E)
                    G1 = ram_fac * elec_dip_absorp[:, idx_kminusq].conj() / (
                        ph_sum_ee + delta_energies_kmq)[None, ...]
                    G2 = 1.0 / (delta_energies_kc_kmqv - ph_freq_mq[il])
                    tmp1 = (gcc_kmq_q[jl].conj()@G1)*G2
                    tmp2 = (tmp1@gvv_k_mq[il].conj()).reshape(npol,-1).T
                    twoph_raman_ten[1, iq, il, jl, :npol, :npol] -= dipS_res.reshape(npol,-1)@tmp2

                # ==============================================================================
                # Process 2a: ABSORB/EMIT (AE)
                # ==============================================================================
                ph_sum_ae = ph_freq_q[il] - ph_freq_q[jl]
                ram_fac = np.sqrt(
                    np.abs(ome_light_Ha + ph_sum_ae) / ome_light_Ha)

                if contrib_lower in ["all", "ee"]:
                    # M1 (E-E)
                    G1 = ram_fac * elec_dip_absorp.conj() / (
                        ph_sum_ae + delta_energies)[None, ...]
                    G2 = 1.0 / (delta_energies_kqc_kv + ph_freq_q[il])
                    tmp1 = ((gcc_k_q[jl].conj()@G1)*G2).reshape(npol,-1).T
                    tmp2 = (gcc_k_q[il]@dipS_res).reshape(npol,-1)
                    twoph_raman_ten[2, iq, il, jl, :npol, :npol] += tmp2@tmp1
                
                if contrib_lower in ["all", "hh"]:
                    # M2 (H-H)
                    G1 = ram_fac * elec_dip_absorp.conj() / (
                        ph_sum_ae + delta_energies)[None, ...]
                    G2 = 1.0 / (delta_energies_kc_kmqv + ph_freq_q[il])
                    tmp1 = ((G1@gvv_kmq_q[jl].conj())*G2).reshape(npol,-1).T
                    tmp2 = (dipS_res@gvv_kmq_q[il]).reshape(npol,-1)
                    twoph_raman_ten[2, iq, il, jl, :npol, :npol] += tmp2@tmp1

                if contrib_lower in ["all", "eh"]:
                    # M3 (E-H)
                    G1 = ram_fac * elec_dip_absorp[:, idx_kplusq].conj() / (
                        ph_sum_ae + delta_energies_kpq)[None, ...]
                    G2 = 1.0 / (delta_energies_kqc_kv + ph_freq_q[il])
                    tmp1 = (G1@gvv_k_q[jl].conj())*G2
                    tmp2 = (gcc_k_q[il].transpose(0,2,1)@tmp1).reshape(npol,-1).T
                    twoph_raman_ten[2, iq, il, jl, :npol, :npol] -= dipS_res.reshape(npol,-1)@tmp2

                if contrib_lower in ["all", "he"]:
                    # M4 (H-E)
                    G1 = ram_fac * elec_dip_absorp[:, idx_kminusq].conj() / (
                        ph_sum_ae + delta_energies_kmq)[None, ...]
                    G2 = 1.0 / (delta_energies_kc_kmqv + ph_freq_q[il])
                    tmp1 = (gcc_kmq_q[jl].conj()@G1)*G2
                    tmp2 = (tmp1@gvv_kmq_q[il].transpose(0,2,1)).reshape(npol,-1).T
                    twoph_raman_ten[2, iq, il, jl, :npol, :npol] -= dipS_res.reshape(npol,-1)@tmp2

                # ==============================================================================
                # Process 2b: EMIT/ABSORB (EA)
                # ==============================================================================
                ph_sum_ea = ph_freq_mq[jl] - ph_freq_mq[il]
                ram_fac = np.sqrt(
                    np.abs(ome_light_Ha + ph_sum_ea) / ome_light_Ha)

                if contrib_lower in ["all", "ee"]:
                    # M1 (E-E)
                    G1 = ram_fac * elec_dip_absorp.conj() / (
                        ph_sum_ea + delta_energies)[None, ...]
                    G2 = 1.0 / (delta_energies_kqc_kv - ph_freq_mq[il])
                    tmp1 = ((gcc_kpq_mq[jl].transpose(0,2,1)@G1)*G2).reshape(npol,-1).T
                    tmp2 = (gcc_kpq_mq[il].transpose(0,2,1).conj()@dipS_res).reshape(npol,-1)
                    twoph_raman_ten[3, iq, il, jl, :npol, :npol] += tmp2@tmp1

                if contrib_lower in ["all", "hh"]:
                    # M2 (H-H)
                    G1 = ram_fac * elec_dip_absorp.conj() / (
                        ph_sum_ea + delta_energies)[None, ...]
                    G2 = 1.0 / (delta_energies_kc_kmqv - ph_freq_mq[il])
                    tmp1 = ((G1@gvv_k_mq[jl].transpose(0,2,1))*G2).reshape(npol,-1).T
                    tmp2 = (dipS_res@gvv_k_mq[il].transpose(0,2,1).conj()).reshape(npol,-1)
                    twoph_raman_ten[3, iq, il, jl, :npol, :npol] += tmp2@tmp1

                if contrib_lower in ["all", "eh"]:
                    # M3 (E-H)
                    G1 = ram_fac * elec_dip_absorp[:, idx_kplusq].conj() / (
                        ph_sum_ea + delta_energies_kpq)[None, ...]
                    G2 = 1.0 / (delta_energies_kqc_kv - ph_freq_mq[il])
                    tmp1 = (G1@gvv_kpq_mq[jl].transpose(0,2,1))*G2
                    tmp2 = (gcc_kpq_mq[il].conj()@tmp1).reshape(npol,-1).T
                    twoph_raman_ten[3, iq, il, jl, :npol, :npol] -= dipS_res.reshape(npol,-1)@tmp2

                if contrib_lower in ["all", "he"]:
                    # M4 (H-E)
                    G1 = ram_fac * elec_dip_absorp[:, idx_kminusq].conj() / (
                        ph_sum_ea + delta_energies_kmq)[None, ...]
                    G2 = 1.0 / (delta_energies_kc_kmqv - ph_freq_mq[il])
                    tmp1 = (gcc_k_mq[jl].transpose(0,2,1)@G1)*G2
                    tmp2 = (tmp1@gvv_k_mq[il].conj()).reshape(npol,-1).T
                    twoph_raman_ten[3, iq, il, jl, :npol, :npol] -= dipS_res.reshape(npol,-1)@tmp2

                if out_freq:
                    out_freq_2ph[0,iq,il,jl] = ph_sum_aa
                    out_freq_2ph[1,iq,il,jl] = ph_sum_ee
                    out_freq_2ph[2,iq,il,jl] = ph_sum_ae
                    out_freq_2ph[3,iq,il,jl] = ph_sum_ea

    # Multiply with prefactors.
    norm_factor = 1.0 / nk / math.sqrt(CellVol) / np.sqrt(nq)
    twoph_raman_ten *= norm_factor
    
    # Now add q and -q terms
    qpos = qpts-np.floor(qpts)
    qpos = (qpos + tol) % 1
    qtree = KDTree(qpos, boxsize=[1, 1, 1])
    _,mq_idxs = qtree.query(-qpts,k=1)
    #
    tmp = twoph_raman_ten[0] + twoph_raman_ten[0][mq_idxs].transpose(0,2,1,3,4)
    twoph_raman_ten[0] = tmp
    #
    tmp = twoph_raman_ten[1] + twoph_raman_ten[1][mq_idxs].transpose(0,2,1,3,4)
    twoph_raman_ten[1] = tmp
    #
    twoph_raman_ten[2] += twoph_raman_ten[3][mq_idxs].transpose(0,2,1,3,4)
    twoph_raman_ten[3] = twoph_raman_ten[2][mq_idxs].transpose(0,2,1,3,4)
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

    kpts = np.array([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.5, 0.0],
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
    rand_pick4 = np.array([ 47, 53, 40, 154, 94, 22, 127, 121, 40, 100, 12, 29, 24, 21, 116, 99 ])[::-1]
    #
    ref_res = np.array([
        [0.02370519-0.0606795j, 0.0595814-0.05695087j, 0.06482968-0.08883329j,
         0.01997198-0.01840819j, 0.01732709-0.01284454j, 0.02215954-0.0269094j,
         0.01257501-0.01988268j, 0.02076935-0.08904694j, 0.06482968-0.08883329j,
         0.05815613-0.06474335j, 0.03483406-0.03026129j, 0.04629167-0.09888207j,
         0.01591035-0.00963696j, 0.04557751-0.08544062j, 0.02319235-0.06093612j,
         0.01117465-0.01165123j],
        [0.04298482+0.04175829j, 0.01337865+0.00415358j, 0.02373522+0.00074705j,
         0.02445325+0.00188247j, 0.07926921+0.03226187j, 0.08705704+0.0397504j,
         0.01868467+0.00312194j, 0.09658909+0.05992155j, 0.07717367+0.06458532j,
         0.08048352+0.04000037j, 0.043007+0.04270359j, 0.0713391+0.05482515j,
         0.01868467+0.00312194j, 0.01613142+0.00695626j, 0.05491413+0.03062807j,
         0.07659879+0.06490992j],
        [-0.01933132+0.00756446j, -0.03490316+0.00345063j, -0.01851361+0.00354942j,
         -0.06968648+0.02585299j, -0.06954453+0.00950634j, -0.01866642-0.00057332j,
         -0.06341267+0.02058042j, -0.07771536+0.0148068j, -0.01851361+0.00354942j,
         -0.0571203+0.01528793j, -0.01206034-0.00137282j, -0.01262813+0.00853879j,
         -0.01697254-0.00540643j, -0.01382215-0.002895j, -0.10852985+0.02673477j,
         -0.04462379+0.00411623j],
        [-0.04278976+0.01233946j, -0.11108469+0.01530806j, -0.01358739+0.0032732j,
         -0.01742857+0.00232366j, -0.01434377+0.00508203j, -0.01194858+0.00171198j,
         -0.05294519+0.02556946j, -0.01851361+0.00354942j, -0.07771536+0.0148068j,
         -0.06130214+0.02570927j, -0.0170354+0.0073942j, -0.06795021+0.0166856j,
         -0.06968648+0.02585299j, -0.01851361+0.00354942j, -0.03357904+0.00975868j,
         -0.0176001+0.01101092j]
    ])
    max1 = np.max(np.abs(result[0][rand_pick1]-ref_res[0]))
    max2 = np.max(np.abs(result[1][rand_pick2]-ref_res[1]))
    max3 = np.max(np.abs(result[2][rand_pick3]-ref_res[2]))
    max_final = max([max1,max2,max3])
    return max_final < 1e-6


if __name__ == "__main__":
    print(test_compute_Raman_oneph_exc())
    print(test_compute_Raman_oneph_ip())
    print(test_compute_raman_twoph())
