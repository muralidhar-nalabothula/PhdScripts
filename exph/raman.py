## Compute resonant Raman intensities
## Sven Reichardt, etal Sci. Adv.6,eabb5915(2020)
import numpy as np
import math
from scipy.spatial import KDTree
from numba import njit, prange
from kpts import build_ktree, find_kpt


###### // One phonon Raman ##############
@njit(cache=True, nogil=True, parallel=True)
def compute_Raman_oneph_exc_numba(ome_light_arr, ph_freq, BS_energies,
                                  ex_dip_absorp, ex_ph, ram_fac, ph_fre_th):

    N_ome = len(ome_light_arr)
    nmode, N_exc, _ = ex_ph.shape
    npol = ex_dip_absorp.shape[0]

    Ram_ten = np.zeros((N_ome, nmode, 3, 3), dtype=ex_dip_absorp.dtype)
    # Initialize the output tensor dynamically matching the complex type of the dipoles

    dipS_ares_base = np.conj(ex_dip_absorp)
    dipSp_res_base = np.conj(ex_dip_absorp)
    # Pre-conjugate these arrays outside the loops to save redundant operations

    for i_ome in prange(N_ome):
        ome_light_Ha = ome_light_arr[i_ome]

        dipS_res = ex_dip_absorp / (ome_light_Ha - BS_energies)
        dipS_res_conj = np.conj(dipS_res)
        dipS_ares = dipS_ares_base / (ome_light_Ha + BS_energies)
        for i in range(nmode):
            freq = ph_freq[i]

            if np.abs(freq) <= ph_fre_th:
                continue
                # Skip modes below the frequency threshold (tensor remains initialized as 0)

            dipSp_res = dipSp_res_base / (ome_light_Ha - BS_energies - freq)
            dipSp_ares = ex_dip_absorp / (ome_light_Ha + BS_energies - freq)

            ex_ph_T = ex_ph[i].T
            # Hoist the transpose of the exciton-phonon matrix

            # 1) Compute the resonant term
            dipSp_res_T_conj = np.conj(dipSp_res.T)
            term1 = dipS_res_conj @ ex_ph_T @ dipSp_res_T_conj
            term1 = np.conj(term1)

            # 2) Compute the anti-resonant term
            dipSp_ares_T = dipSp_ares.T
            term2 = dipS_ares @ ex_ph_T @ dipSp_ares_T

            # Scale to get the final Raman tensor
            scale = np.sqrt(
                np.abs(ome_light_Ha - freq) / ome_light_Ha) * ram_fac
            Ram_ten[i_ome, i, :npol, :npol] = (term1 + term2) * scale

    return Ram_ten


def compute_Raman_oneph_exc(ome_light,
                            ph_freq,
                            ex_ene,
                            ex_dip,
                            ex_ph,
                            nkpts,
                            CellVol,
                            broading=0.1,
                            npol=3,
                            ph_fre_th=5,
                            precision='s'):
    is_scalar_ome = np.isscalar(ome_light) or np.ndim(ome_light) == 0
    ome_light_arr = np.atleast_1d(ome_light)
    # Safely convert incoming light to an array to handle both single values and grids

    prec = str(precision).strip().lower()
    if prec in ['s', 'single']:
        f_type = np.float32
        c_type = np.complex64
    elif prec in ['d', 'double']:
        f_type = np.float64
        c_type = np.complex128
    else:
        raise ValueError(
            "Unknown precision: use 's' for single or 'd' for double.")

    # 2. Convert physics units
    ome_light_Ha = ome_light_arr / 27.211
    broading_Ha = broading / 27.211 / 2.0
    ph_fre_th_Ha = ph_fre_th * 0.12398 / 1000 / 27.211
    ram_fac = 1.0 / nkpts / math.sqrt(CellVol)

    # 3. Precompute arrays outside Numba for cleaner internal logic
    ex_dip_absorp = np.conj(ex_dip[:npol, :])
    BS_energies = ex_ene - 1j * broading_Ha

    ome_light_c = np.ascontiguousarray(ome_light_Ha, dtype=f_type)
    ph_freq_c = np.ascontiguousarray(ph_freq, dtype=f_type)
    BS_energies_c = np.ascontiguousarray(BS_energies, dtype=c_type)
    ex_dip_absorp_c = np.ascontiguousarray(ex_dip_absorp, dtype=c_type)
    ex_ph_c = np.ascontiguousarray(ex_ph, dtype=c_type)

    Ram_ten = compute_Raman_oneph_exc_numba(ome_light_c, ph_freq_c,
                                            BS_energies_c, ex_dip_absorp_c,
                                            ex_ph_c, ram_fac, ph_fre_th_Ha)

    if is_scalar_ome:
        return Ram_ten[0]
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
# Cannot use numba due to kdtrees
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
                           contrib="all"):
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
                    (3, nq, n_modes, n_modes, npol, npol).
                    The first dimension corresponds to the process:
                    0: Anti-Stokes (AA)
                    1: Stokes (EE)
                    2: Absorb-Emit (AE) + Emit-Absorb (EA)
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
        raise ValueError(
            f"Invalid value for 'contrib'. "
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
        kpos = kpts - np.floor(kpts)
        kpos = (kpos + tol) % 1
        ktree = KDTree(kpos, boxsize=[1, 1, 1])

    twoph_raman_ten = np.zeros((4, nq, n_modes, n_modes, 3, 3),
                               dtype=gkkp.dtype)
    out_freq_2ph = []
    if out_freq:
        out_freq_2ph = np.zeros((4, nq, n_modes, n_modes))

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
        gvv_k_q = g_q[:, :, :nv, :nv]
        gcc_kpq_mq = g_mq[:, idx_kplusq, nv:, nv:]
        gvv_kpq_mq = g_mq[:, idx_kplusq, :nv, :nv]
        gcc_k_mq = g_mq[:, :, nv:, nv:]
        gvv_k_mq = g_mq[:, :, :nv, :nv]
        gcc_kmq_q = g_q[:, idx_kminusq, nv:, nv:]
        gvv_kmq_q = g_q[:, idx_kminusq, :nv, :nv]

        # phonon freqs
        ph_freq_q = ph_freq[iq]
        ph_freq_mq = ph_freq[minus_iq_idx]

        for il in range(n_modes):  # First phonon
            for jl in range(n_modes):  # Second phonon

                # ==============================================================================
                # Process 0: ANTI-STOKES (Absorb/Absorb)
                # ==============================================================================
                #
                ph_sum_aa = ph_freq_mq[jl] + ph_freq_q[il]
                ram_fac = np.sqrt(
                    np.abs(ome_light_Ha + ph_sum_aa) / ome_light_Ha)

                if contrib_lower in ["all", "ee"]:
                    # M1 (E-E)
                    G1 = ram_fac * elec_dip_absorp.conj() / (
                        ph_sum_aa + delta_energies)[None, ...]
                    G2 = 1.0 / (delta_energies_kqc_kv + ph_freq_q[il])
                    tmp1 = ((gcc_kpq_mq[jl].transpose(0, 2, 1) @ G1) *
                            G2).reshape(npol, -1).T
                    tmp2 = (gcc_k_q[il] @ dipS_res).reshape(npol, -1)
                    twoph_raman_ten[0, iq, il, jl, :npol, :npol] += tmp2 @ tmp1

                if contrib_lower in ["all", "hh"]:
                    # M2 (H-H)
                    G1 = ram_fac * elec_dip_absorp.conj() / (
                        ph_sum_aa + delta_energies)[None, ...]
                    G2 = 1.0 / (delta_energies_kc_kmqv + ph_freq_q[il])
                    tmp1 = ((G1 @ gvv_k_mq[jl].transpose(0, 2, 1)) *
                            G2).reshape(npol, -1).T
                    tmp2 = (dipS_res @ gvv_kmq_q[il]).reshape(npol, -1)
                    twoph_raman_ten[0, iq, il, jl, :npol, :npol] += tmp2 @ tmp1

                if contrib_lower in ["all", "eh"]:
                    # M3 (E-H)
                    G1 = ram_fac * elec_dip_absorp[:, idx_kplusq].conj() / (
                        ph_sum_aa + delta_energies_kpq)[None, ...]
                    G2 = 1.0 / (delta_energies_kqc_kv + ph_freq_q[il])
                    tmp1 = (G1 @ gvv_kpq_mq[jl].transpose(0, 2, 1)) * G2
                    tmp2 = (gcc_k_q[il].transpose(0, 2, 1) @ tmp1).reshape(
                        npol, -1).T
                    twoph_raman_ten[0, iq, il,
                                    jl, :npol, :npol] -= dipS_res.reshape(
                                        npol, -1) @ tmp2

                if contrib_lower in ["all", "he"]:
                    # M4 (H-E)
                    G1 = ram_fac * elec_dip_absorp[:, idx_kminusq].conj() / (
                        ph_sum_aa + delta_energies_kmq)[None, ...]
                    G2 = 1.0 / (delta_energies_kc_kmqv + ph_freq_q[il])
                    tmp1 = (gcc_k_mq[jl].transpose(0, 2, 1) @ G1) * G2
                    tmp2 = (tmp1 @ gvv_kmq_q[il].transpose(0, 2, 1)).reshape(
                        npol, -1).T
                    twoph_raman_ten[0, iq, il,
                                    jl, :npol, :npol] -= dipS_res.reshape(
                                        npol, -1) @ tmp2

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
                    tmp1 = ((gcc_k_q[jl].conj() @ G1) * G2).reshape(npol, -1).T
                    tmp2 = (gcc_kpq_mq[il].transpose(0, 2, 1).conj()
                            @ dipS_res).reshape(npol, -1)
                    twoph_raman_ten[1, iq, il, jl, :npol, :npol] += tmp2 @ tmp1

                if contrib_lower in ["all", "hh"]:
                    # M2 (H-H)
                    G1 = ram_fac * elec_dip_absorp.conj() / (
                        ph_sum_ee + delta_energies)[None, ...]
                    G2 = 1.0 / (delta_energies_kc_kmqv - ph_freq_mq[il])
                    tmp1 = ((G1 @ gvv_kmq_q[jl].conj()) * G2).reshape(npol,
                                                                      -1).T
                    tmp2 = (dipS_res @ gvv_k_mq[il].transpose(
                        0, 2, 1).conj()).reshape(npol, -1)
                    twoph_raman_ten[1, iq, il, jl, :npol, :npol] += tmp2 @ tmp1

                if contrib_lower in ["all", "eh"]:
                    # M3 (E-H)
                    G1 = ram_fac * elec_dip_absorp[:, idx_kplusq].conj() / (
                        ph_sum_ee + delta_energies_kpq)[None, ...]
                    G2 = 1.0 / (delta_energies_kqc_kv - ph_freq_mq[il])
                    tmp1 = (G1 @ gvv_k_q[jl].conj()) * G2
                    tmp2 = (gcc_kpq_mq[il].conj() @ tmp1).reshape(npol, -1).T
                    twoph_raman_ten[1, iq, il,
                                    jl, :npol, :npol] -= dipS_res.reshape(
                                        npol, -1) @ tmp2

                if contrib_lower in ["all", "he"]:
                    # M4 (H-E)
                    G1 = ram_fac * elec_dip_absorp[:, idx_kminusq].conj() / (
                        ph_sum_ee + delta_energies_kmq)[None, ...]
                    G2 = 1.0 / (delta_energies_kc_kmqv - ph_freq_mq[il])
                    tmp1 = (gcc_kmq_q[jl].conj() @ G1) * G2
                    tmp2 = (tmp1 @ gvv_k_mq[il].conj()).reshape(npol, -1).T
                    twoph_raman_ten[1, iq, il,
                                    jl, :npol, :npol] -= dipS_res.reshape(
                                        npol, -1) @ tmp2

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
                    tmp1 = ((gcc_k_q[jl].conj() @ G1) * G2).reshape(npol, -1).T
                    tmp2 = (gcc_k_q[il] @ dipS_res).reshape(npol, -1)
                    twoph_raman_ten[2, iq, il, jl, :npol, :npol] += tmp2 @ tmp1

                if contrib_lower in ["all", "hh"]:
                    # M2 (H-H)
                    G1 = ram_fac * elec_dip_absorp.conj() / (
                        ph_sum_ae + delta_energies)[None, ...]
                    G2 = 1.0 / (delta_energies_kc_kmqv + ph_freq_q[il])
                    tmp1 = ((G1 @ gvv_kmq_q[jl].conj()) * G2).reshape(npol,
                                                                      -1).T
                    tmp2 = (dipS_res @ gvv_kmq_q[il]).reshape(npol, -1)
                    twoph_raman_ten[2, iq, il, jl, :npol, :npol] += tmp2 @ tmp1

                if contrib_lower in ["all", "eh"]:
                    # M3 (E-H)
                    G1 = ram_fac * elec_dip_absorp[:, idx_kplusq].conj() / (
                        ph_sum_ae + delta_energies_kpq)[None, ...]
                    G2 = 1.0 / (delta_energies_kqc_kv + ph_freq_q[il])
                    tmp1 = (G1 @ gvv_k_q[jl].conj()) * G2
                    tmp2 = (gcc_k_q[il].transpose(0, 2, 1) @ tmp1).reshape(
                        npol, -1).T
                    twoph_raman_ten[2, iq, il,
                                    jl, :npol, :npol] -= dipS_res.reshape(
                                        npol, -1) @ tmp2

                if contrib_lower in ["all", "he"]:
                    # M4 (H-E)
                    G1 = ram_fac * elec_dip_absorp[:, idx_kminusq].conj() / (
                        ph_sum_ae + delta_energies_kmq)[None, ...]
                    G2 = 1.0 / (delta_energies_kc_kmqv + ph_freq_q[il])
                    tmp1 = (gcc_kmq_q[jl].conj() @ G1) * G2
                    tmp2 = (tmp1 @ gvv_kmq_q[il].transpose(0, 2, 1)).reshape(
                        npol, -1).T
                    twoph_raman_ten[2, iq, il,
                                    jl, :npol, :npol] -= dipS_res.reshape(
                                        npol, -1) @ tmp2

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
                    tmp1 = ((gcc_kpq_mq[jl].transpose(0, 2, 1) @ G1) *
                            G2).reshape(npol, -1).T
                    tmp2 = (gcc_kpq_mq[il].transpose(0, 2, 1).conj()
                            @ dipS_res).reshape(npol, -1)
                    twoph_raman_ten[3, iq, il, jl, :npol, :npol] += tmp2 @ tmp1

                if contrib_lower in ["all", "hh"]:
                    # M2 (H-H)
                    G1 = ram_fac * elec_dip_absorp.conj() / (
                        ph_sum_ea + delta_energies)[None, ...]
                    G2 = 1.0 / (delta_energies_kc_kmqv - ph_freq_mq[il])
                    tmp1 = ((G1 @ gvv_k_mq[jl].transpose(0, 2, 1)) *
                            G2).reshape(npol, -1).T
                    tmp2 = (dipS_res @ gvv_k_mq[il].transpose(
                        0, 2, 1).conj()).reshape(npol, -1)
                    twoph_raman_ten[3, iq, il, jl, :npol, :npol] += tmp2 @ tmp1

                if contrib_lower in ["all", "eh"]:
                    # M3 (E-H)
                    G1 = ram_fac * elec_dip_absorp[:, idx_kplusq].conj() / (
                        ph_sum_ea + delta_energies_kpq)[None, ...]
                    G2 = 1.0 / (delta_energies_kqc_kv - ph_freq_mq[il])
                    tmp1 = (G1 @ gvv_kpq_mq[jl].transpose(0, 2, 1)) * G2
                    tmp2 = (gcc_kpq_mq[il].conj() @ tmp1).reshape(npol, -1).T
                    twoph_raman_ten[3, iq, il,
                                    jl, :npol, :npol] -= dipS_res.reshape(
                                        npol, -1) @ tmp2

                if contrib_lower in ["all", "he"]:
                    # M4 (H-E)
                    G1 = ram_fac * elec_dip_absorp[:, idx_kminusq].conj() / (
                        ph_sum_ea + delta_energies_kmq)[None, ...]
                    G2 = 1.0 / (delta_energies_kc_kmqv - ph_freq_mq[il])
                    tmp1 = (gcc_k_mq[jl].transpose(0, 2, 1) @ G1) * G2
                    tmp2 = (tmp1 @ gvv_k_mq[il].conj()).reshape(npol, -1).T
                    twoph_raman_ten[3, iq, il,
                                    jl, :npol, :npol] -= dipS_res.reshape(
                                        npol, -1) @ tmp2

                if out_freq:
                    out_freq_2ph[0, iq, il, jl] = ph_sum_aa
                    out_freq_2ph[1, iq, il, jl] = ph_sum_ee
                    out_freq_2ph[2, iq, il, jl] = ph_sum_ae
                    out_freq_2ph[3, iq, il, jl] = ph_sum_ea

    # Multiply with prefactors.
    norm_factor = 1.0 / nk / math.sqrt(CellVol) / np.sqrt(nq)
    twoph_raman_ten *= norm_factor

    # Now add q and -q terms
    qpos = qpts - np.floor(qpts)
    qpos = (qpos + tol) % 1
    qtree = KDTree(qpos, boxsize=[1, 1, 1])
    _, mq_idxs = qtree.query(-qpts, k=1)
    #
    # We compute only 4 terms, add the other four terms (q,l,m) + (-q, m, l)
    # as we sum over -q and q, we need to multiple by 1/sqrt(2), so to avoid
    # double counting when computing the Intensity by suming all q's (which include -q too)
    idx_fac_mul = (mq_idxs != np.arange(len(qpts), dtype=int))
    #
    tmp = twoph_raman_ten[0] + twoph_raman_ten[0][mq_idxs].transpose(
        0, 2, 1, 3, 4)
    tmp[idx_fac_mul] *= 1.0 / np.sqrt(2)
    twoph_raman_ten[0] = tmp
    #
    tmp = twoph_raman_ten[1] + twoph_raman_ten[1][mq_idxs].transpose(
        0, 2, 1, 3, 4)
    tmp[idx_fac_mul] *= 1.0 / np.sqrt(2)
    twoph_raman_ten[1] = tmp
    #
    # We donot need to Absorp-emit as it is does not have double counting (q and -q )
    # are distinct terms
    twoph_raman_ten[2] += twoph_raman_ten[3][mq_idxs].transpose(0, 2, 1, 3, 4)
    #
    if out_freq:
        return out_freq_2ph[:3], twoph_raman_ten[:3]
    else:
        return twoph_raman_ten[:3]


#
#
# 2ph raman with excitonic effects


@njit(cache=True, nogil=True)
def two_ph_raman_exc_numba_kernel(iq, N_ome, nmode, npol, N_q, N_mq,
                                  ome_light_arr, ph_freq_q, ph_freq_mq,
                                  ex_ene_0, ex_ene_q, ex_ene_mq, ex_dip, g_0_q,
                                  g_0_mq, g_q_mq, g_mq_q, gamma_c_typed,
                                  scale_bz, scale_one, zero_val, dip_conj):

    M_tensor_iq = np.zeros((N_ome, 3, nmode, nmode, npol, npol),
                           dtype=ex_dip.dtype)

    B_0_q_T = np.zeros((nmode, npol, N_q), dtype=ex_dip.dtype)
    B_0_mq_T = np.zeros((nmode, npol, N_mq), dtype=ex_dip.dtype)
    B_conj_mq_q = np.zeros((nmode, npol, N_mq), dtype=ex_dip.dtype)
    B_conj_q_mq = np.zeros((nmode, npol, N_q), dtype=ex_dip.dtype)

    for i_ome in range(N_ome):
        ome_light_typed = ome_light_arr[i_ome]

        D3 = ome_light_typed - ex_ene_0 + gamma_c_typed
        W3 = ex_dip / D3

        for i_mode in range(nmode):
            B_0_q_T[i_mode] = W3 @ g_0_q[iq, i_mode]
            B_0_mq_T[i_mode] = W3 @ g_0_mq[iq, i_mode]
            B_conj_mq_q[i_mode] = W3 @ np.conj(g_mq_q[iq, i_mode]).T
            B_conj_q_mq[i_mode] = W3 @ np.conj(g_q_mq[iq, i_mode]).T

        for il in range(nmode):
            omega_q_l = ph_freq_q[iq, il]
            omega_mq_l = ph_freq_mq[iq, il]

            G_0_q_l = g_0_q[iq, il]
            G_0_mq_l = g_0_mq[iq, il]
            G_q_mq_l = g_q_mq[iq, il]
            G_mq_q_l = g_mq_q[iq, il]

            for ilambda in range(nmode):
                omega_q_lam = ph_freq_q[iq, ilambda]
                omega_mq_lam = ph_freq_mq[iq, ilambda]

                G_0_q_lam = g_0_q[iq, ilambda]
                G_0_mq_lam = g_0_mq[iq, ilambda]
                G_q_mq_lam = g_q_mq[iq, ilambda]
                G_mq_q_lam = g_mq_q[iq, ilambda]

                # ==========================================
                # Process 0: Anti-Stokes (Two-Phonon Absorb)
                # ==========================================
                O1 = omega_mq_l
                O2 = omega_q_lam
                E_shift = O1 + O2
                ome_out = ome_light_typed + E_shift

                if ome_out > zero_val:
                    prefac = scale_bz * np.sqrt(ome_out / ome_light_typed)

                    D1 = ome_light_typed - ex_ene_0 + E_shift + gamma_c_typed
                    W1 = dip_conj / D1

                    A1 = W1 @ G_q_mq_l.T
                    B1 = B_0_q_T[ilambda]
                    D2 = ome_light_typed - ex_ene_q[iq, :] + O2 + gamma_c_typed
                    M1 = B1 @ (A1 / D2[None, :]).T

                    A2 = W1 @ G_mq_q_lam.T
                    B2 = B_0_mq_T[il]
                    D5 = ome_light_typed - ex_ene_mq[iq, :] + O1 + gamma_c_typed
                    M2 = B2 @ (A2 / D5[None, :]).T

                    M_tensor_iq[i_ome, 0, il,
                                ilambda, :, :] = prefac * (M1 + M2)

                # ==========================================
                # Process 1: Mixed (Absorb lambda, Emit l)
                # ==========================================
                O1 = -omega_q_l
                O2 = omega_q_lam
                E_shift = O1 + O2
                ome_out = ome_light_typed + E_shift

                if ome_out > zero_val:
                    prefac = scale_one * np.sqrt(ome_out / ome_light_typed)

                    D1 = ome_light_typed - ex_ene_0 + E_shift + gamma_c_typed
                    W1 = dip_conj / D1

                    A1 = W1 @ np.conj(G_0_q_l)
                    B1 = B_0_q_T[ilambda]
                    D2 = ome_light_typed - ex_ene_q[iq, :] + O2 + gamma_c_typed
                    M1 = B1 @ (A1 / D2[None, :]).T

                    A2 = W1 @ G_mq_q_lam.T
                    B2 = B_conj_mq_q[il]
                    D5 = ome_light_typed - ex_ene_mq[iq, :] + O1 + gamma_c_typed
                    M2 = B2 @ (A2 / D5[None, :]).T

                    M_tensor_iq[i_ome, 1, il,
                                ilambda, :, :] = prefac * (M1 + M2)

                # ==========================================
                # Process 2: Stokes (Two-Phonon Emit)
                # ==========================================
                O1 = -omega_q_l
                O2 = -omega_mq_lam
                E_shift = O1 + O2
                ome_out = ome_light_typed + E_shift

                if ome_out > zero_val:
                    prefac = scale_bz * np.sqrt(ome_out / ome_light_typed)

                    D1 = ome_light_typed - ex_ene_0 + E_shift + gamma_c_typed
                    W1 = dip_conj / D1

                    A1 = W1 @ np.conj(G_0_q_l)
                    B1 = B_conj_q_mq[ilambda]
                    D2 = ome_light_typed - ex_ene_q[iq, :] + O2 + gamma_c_typed
                    M1 = B1 @ (A1 / D2[None, :]).T

                    A2 = W1 @ np.conj(G_0_mq_lam)
                    B2 = B_conj_mq_q[il]
                    D5 = ome_light_typed - ex_ene_mq[iq, :] + O1 + gamma_c_typed
                    M2 = B2 @ (A2 / D5[None, :]).T

                    M_tensor_iq[i_ome, 2, il,
                                ilambda, :, :] = prefac * (M1 + M2)

    return M_tensor_iq


@njit(cache=True, nogil=True, parallel=True)
def compute_two_ph_raman_exc_numba(ome_light_arr, ph_freq_q, ph_freq_mq,
                                   ex_ene_0, ex_ene_q, ex_ene_mq, ex_dip, g_0_q,
                                   g_0_mq, g_q_mq, g_mq_q, gamma):
    """!
    @brief Computes the two-phonon double resonant Raman scattering tensor with excitonic effects.

    @param ome_light_arr Array of incoming light frequencies. Shape: (N_ome,)
    @param ph_freq_q Phonon frequencies at wavevector q. Shape: (nq, nmode)
    @param ph_freq_mq Phonon frequencies at wavevector -q. Shape: (nq, nmode)
    @param ex_ene_0 Exciton energies at q=0. Shape: (nstates_at_0,)
    @param ex_ene_q Exciton energies at wavevector q. Shape: (nq, nstates_at_Q)
    @param ex_ene_mq Exciton energies at wavevector -q. Shape: (nq, nstates_at_mq)
    @param ex_dip Exciton optical dipole matrix elements for photon absorption. Shape: (npol, nstates_at_0)
    @param g_0_q Exciton-phonon coupling matrices from q=0 to q. Shape: (nq, nmode, nstates_at_0, nstates_at_Q)
    @param g_0_mq Exciton-phonon coupling matrices from q=0 to -q. Shape: (nq, nmode, nstates_at_0, nstates_at_mq)
    @param g_q_mq Exciton-phonon coupling matrices from q to -q. Shape: (nq, nmode, nstates_at_Q, nstates_at_0)
    @param g_mq_q Exciton-phonon coupling matrices from -q to q. Shape: (nq, nmode, nstates_at_mq, nstates_at_0)
    @param gamma Phenomenological broadening factor.

    @return The computed Raman scattering tensor for Anti-Stokes (index 0),
    Mixed (index 1), and Stokes (index 2) processes. Shape: (N_ome, nq, 3, nmode, nmode, npol, npol)
    """
    N_ome = len(ome_light_arr)
    Nqpts, nmode = ph_freq_q.shape
    npol, N_exc = ex_dip.shape

    N_q = ex_ene_q.shape[1]
    N_mq = ex_ene_mq.shape[1]

    M_tensor = np.zeros((N_ome, Nqpts, 3, nmode, nmode, npol, npol),
                        dtype=ex_dip.dtype)

    dip_conj = np.conj(ex_dip)

    tmp_c = np.zeros_like(ex_dip[0:1, 0])
    tmp_f = np.zeros_like(ph_freq_q[0:1, 0])

    tmp_c[0] = 1j * gamma
    gamma_c_typed = tmp_c[0]

    tmp_f[0] = 1.0 / np.sqrt(2.0)
    scale_bz = 1.0  #tmp_f[0] # we add the prefactor outside

    tmp_f[0] = 1.0
    scale_one = tmp_f[0]

    tmp_f[0] = 0.0
    zero_val = tmp_f[0]

    for iq in prange(Nqpts):
        M_tensor[:, iq, :, :, :, :, :] = two_ph_raman_exc_numba_kernel(
            iq, N_ome, nmode, npol, N_q, N_mq, ome_light_arr, ph_freq_q,
            ph_freq_mq, ex_ene_0, ex_ene_q, ex_ene_mq, ex_dip, g_0_q, g_0_mq,
            g_q_mq, g_mq_q, gamma_c_typed, scale_bz, scale_one, zero_val,
            dip_conj)

    return M_tensor


def compute_two_ph_raman_exc(ome_light,
                             qpoints_crys,
                             ph_freq_q,
                             ex_ene_0,
                             ex_ene_q,
                             ex_dip,
                             g_0_q,
                             g_q_mq,
                             gamma=0.01,
                             precision='s'):
    #// gamma nad ome_light are in eV
    gamma = gamma / 27.2111
    ome_light = ome_light / 27.21111
    qtree = build_ktree(qpoints_crys)
    idx_mq = find_kpt(qtree, -qpoints_crys)
    ph_freq_mq = ph_freq_q[idx_mq]
    ex_ene_mq = ex_ene_q[idx_mq]
    g_0_mq = g_0_q[idx_mq]
    g_mq_q = g_q_mq[idx_mq]
    is_scalar_ome = np.isscalar(ome_light) or np.ndim(ome_light) == 0
    ome_light_arr = np.atleast_1d(ome_light)
    prec = str(precision).strip().lower()
    if prec in ['s', 'single']:
        f_type = np.float32
        c_type = np.complex64
    elif prec in ['d', 'double']:
        f_type = np.float64
        c_type = np.complex128
    else:
        raise ValueError(
            "Unknown precision: use 's' for single or 'd' for double.")
    ome_light_c = np.ascontiguousarray(ome_light_arr, dtype=f_type)
    ph_freq_q_c = np.ascontiguousarray(ph_freq_q, dtype=f_type)
    ph_freq_mq_c = np.ascontiguousarray(ph_freq_mq, dtype=f_type)
    ex_ene_0_c = np.ascontiguousarray(ex_ene_0, dtype=f_type)
    ex_ene_q_c = np.ascontiguousarray(ex_ene_q, dtype=f_type)
    ex_ene_mq_c = np.ascontiguousarray(ex_ene_mq, dtype=f_type)
    ex_dip_c = np.ascontiguousarray(ex_dip, dtype=c_type)
    g_0_q_c = np.ascontiguousarray(g_0_q, dtype=c_type)
    g_0_mq_c = np.ascontiguousarray(g_0_mq, dtype=c_type)
    g_q_mq_c = np.ascontiguousarray(g_q_mq, dtype=c_type)
    g_mq_q_c = np.ascontiguousarray(g_mq_q, dtype=c_type)
    #
    M_tensor = compute_two_ph_raman_exc_numba(ome_light_c, ph_freq_q_c,
                                              ph_freq_mq_c, ex_ene_0_c,
                                              ex_ene_q_c, ex_ene_mq_c, ex_dip_c,
                                              g_0_q_c, g_0_mq_c, g_q_mq_c,
                                              g_mq_q_c, gamma)
    #
    # compute prefactors for stokes and anti-stokes to avoid double counting.
    prefactor = np.where(idx_mq == np.arange(len(idx_mq), dtype=int), 1.0,
                         1 / np.sqrt(2))
    M_tensor[:, :, 0, ...] *= prefactor[None, :, None, None, None, None]
    M_tensor[:, :, 2, ...] *= prefactor[None, :, None, None, None, None]
    #
    Nqpts, nmode = ph_freq_q_c.shape
    raman_shift_freq = np.zeros((Nqpts, 3, nmode, nmode), dtype=f_type)
    # Process 0: Anti-Stokes (+omega_mq_l + omega_q_lam)
    raman_shift_freq[:, 0, :, :] = ph_freq_mq_c[:, :,
                                                None] + ph_freq_q_c[:, None, :]
    # Process 1: Mixed (-omega_q_l + omega_q_lam)
    raman_shift_freq[:,
                     1, :, :] = -ph_freq_q_c[:, :, None] + ph_freq_q_c[:,
                                                                       None, :]
    # Process 2: Stokes (-omega_q_l - omega_mq_lam)
    raman_shift_freq[:,
                     2, :, :] = -ph_freq_q_c[:, :, None] - ph_freq_mq_c[:,
                                                                        None, :]

    if is_scalar_ome:
        return raman_shift_freq, M_tensor[0]
    return raman_shift_freq, M_tensor


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
    result = result.reshape(3, -1)
    rand_pick1 = np.array([
        127, 150, 116, 68, 67, 81, 74, 155, 98, 115, 84, 131, 40, 140, 110, 46
    ])
    rand_pick2 = np.array([
        138, 57, 59, 74, 146, 155, 40, 149, 157, 113, 120, 148, 40, 24, 85, 142
    ])
    rand_pick3 = np.array(
        [47, 53, 40, 154, 94, 22, 127, 121, 40, 100, 12, 29, 24, 21, 116, 99])
    #
    array1 = np.array([
        0.02370519 - 0.0606795j, 0.0595814 - 0.05695087j,
        0.06482968 - 0.08883329j, 0.01997198 - 0.01840819j,
        0.01732709 - 0.01284454j, 0.02215954 - 0.0269094j,
        0.01257501 - 0.01988268j, 0.02076935 - 0.08904694j,
        0.06482968 - 0.08883329j, 0.05815613 - 0.06474335j,
        0.03483406 - 0.03026129j, 0.04629167 - 0.09888207j,
        0.01591035 - 0.00963696j, 0.04557751 - 0.08544062j,
        0.02319235 - 0.06093612j, 0.01117465 - 0.01165123j
    ])

    array2 = np.array([
        0.04298482 + 0.04175829j, 0.01337865 + 0.00415358j,
        0.02373522 + 0.00074705j, 0.02445325 + 0.00188247j,
        0.07926921 + 0.03226187j, 0.08705704 + 0.0397504j,
        0.01868467 + 0.00312194j, 0.09658909 + 0.05992155j,
        0.07717367 + 0.06458532j, 0.08048352 + 0.04000037j,
        0.043007 + 0.04270359j, 0.0713391 + 0.05482515j,
        0.01868467 + 0.00312194j, 0.01613142 + 0.00695626j,
        0.05491413 + 0.03062807j, 0.07659879 + 0.06490992j
    ])

    array3 = np.array([
        -0.01933132 + 0.00756446j, -0.03490316 + 0.00345063j,
        -0.01851361 + 0.00354942j, -0.06968648 + 0.02585299j,
        -0.06954453 + 0.00950634j, -0.01866642 - 0.00057332j,
        -0.06341267 + 0.02058042j, -0.07771536 + 0.0148068j,
        -0.01851361 + 0.00354942j, -0.0571203 + 0.01528793j,
        -0.01206034 - 0.00137282j, -0.01262813 + 0.00853879j,
        -0.01697254 - 0.00540643j, -0.01382215 - 0.002895j,
        -0.10852985 + 0.02673477j, -0.04462379 + 0.00411623j
    ])

    ref_res = np.vstack([array1, array2, array3])

    max1 = np.max(np.abs(result[0][rand_pick1] - ref_res[0]))
    max2 = np.max(np.abs(result[1][rand_pick2] - ref_res[1]))
    max3 = np.max(np.abs(result[2][rand_pick3] - ref_res[2]))
    max_final = max([max1, max2, max3])
    return max_final < 1e-6


def test_compute_two_ph_raman_exc():
    qpoints_crys = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    # Define q-points in crystal coordinates

    Nqpts = len(qpoints_crys)
    # Number of q-points, matches qpoints_crys length

    nmode = 2
    # Number of phonon modes

    npol = 3
    # Number of polarizations

    N_exc0 = 3
    # Number of exciton states at q=0

    N_exc_q = 2
    # Number of exciton states at q

    ome_light = 2.5
    # Incident light frequency

    gamma = 0.1
    # Broadening factor

    ph_freq_q = np.linspace(0.01, 0.05, Nqpts * nmode).reshape(Nqpts, nmode)
    # Phonon frequencies at wavevector q

    ex_ene_0 = np.linspace(1.5, 2.5, N_exc0)
    # Exciton energies at q=0

    ex_ene_q = np.linspace(1.6, 2.6, Nqpts * N_exc_q).reshape(Nqpts, N_exc_q)
    # Exciton energies at wavevector q

    ex_dip_real = np.linspace(0.1, 1.0, npol * N_exc0).reshape(npol, N_exc0)
    ex_dip_imag = np.linspace(-0.5, 0.5, npol * N_exc0).reshape(npol, N_exc0)
    ex_dip = ex_dip_real + 1j * ex_dip_imag
    # Exciton optical dipole matrix elements

    g_0_q_real = np.linspace(0.01, 0.1,
                             Nqpts * nmode * N_exc0 * N_exc_q).reshape(
                                 Nqpts, nmode, N_exc0, N_exc_q)
    g_0_q_imag = np.linspace(-0.05, 0.05,
                             Nqpts * nmode * N_exc0 * N_exc_q).reshape(
                                 Nqpts, nmode, N_exc0, N_exc_q)
    g_0_q = g_0_q_real + 1j * g_0_q_imag
    # Exciton-phonon coupling from q=0 to q

    g_q_mq_real = np.linspace(0.015, 0.15,
                              Nqpts * nmode * N_exc_q * N_exc0).reshape(
                                  Nqpts, nmode, N_exc_q, N_exc0)
    g_q_mq_imag = np.linspace(-0.02, 0.08,
                              Nqpts * nmode * N_exc_q * N_exc0).reshape(
                                  Nqpts, nmode, N_exc_q, N_exc0)
    g_q_mq = g_q_mq_real + 1j * g_q_mq_imag
    # Exciton-phonon coupling from q to -q

    _, result = compute_two_ph_raman_exc(ome_light,
                                         qpoints_crys,
                                         ph_freq_q,
                                         ex_ene_0,
                                         ex_ene_q,
                                         ex_dip,
                                         g_0_q,
                                         g_q_mq,
                                         gamma=gamma,
                                         precision='d')

    #ref = result[:,:,0,1,:,:].reshape(-1)
    ref = np.array([
        -0.29916172 - 5.89001823e-02j, -0.37578530 + 3.45437057e-01j,
        -0.45240889 + 7.49774297e-01j, -0.22735277 - 3.93005322e-01j,
        -0.75553781 - 8.24170594e-02j, -1.28372285 + 2.28171203e-01j,
        -0.15554382 - 7.27110461e-01j, -1.13529032 - 5.10271176e-01j,
        -2.11503682 - 2.93431891e-01j, -0.14591269 - 2.17357012e-01j,
        -0.46073524 - 2.13548532e-02j, -0.77555779 + 1.74647305e-01j,
        0.08904460 - 3.85049455e-01j, -0.45134335 - 5.30109977e-01j,
        -0.99173130 - 6.75170498e-01j, 0.32400188 - 5.52741899e-01j,
        -0.44195147 - 1.03886510e+00j, -1.20790481 - 1.52498830e+00j,
        -0.18332960 - 6.60575291e-02j, -0.32485200 + 1.93673982e-01j,
        -0.46637440 + 4.53405494e-01j, -0.11870946 - 2.69229427e-01j,
        -0.54999018 - 1.53565493e-01j, -0.98127090 - 3.79015586e-02j,
        -0.05408931 - 4.72401325e-01j, -0.77512836 - 5.00804968e-01j,
        -1.49616740 - 5.29208611e-01j, -4.33882694 - 1.10430477e+00j,
        -5.54010435 + 4.72331254e+00j, -6.74138176 + 1.05509299e+01j,
        -3.19817579 - 5.94453050e+00j, -10.89498276 - 1.33583264e+00j,
        -18.59178972 + 3.27286523e+00j, -2.05752465 - 1.07847562e+01j,
        -16.24986116 - 7.39497781e+00j, -30.44219768 - 4.00519941e+00j,
        -1.60739485 + 3.39431231e+00j, 3.06923746 + 5.72931115e+00j,
        7.74586976 + 8.06430999e+00j, -5.37419667 + 1.70210282e+00j,
        -3.16163037 + 9.22924730e+00j, -0.94906406 + 1.67563918e+01j,
        -9.14099850 + 9.89333717e-03j, -9.39249819 + 1.27291834e+01j,
        -9.64399789 + 2.54484735e+01j, 1.34018927 + 3.20160755e-01j,
        2.20693176 - 1.68409220e+00j, 3.07367425 - 3.68834516e+00j,
        1.02077464 + 1.79351765e+00j, 4.09984780 + 6.91269283e-01j,
        7.17892097 - 4.10979080e-01j, 0.70136002 + 3.26687454e+00j,
        5.99276385 + 3.06663077e+00j, 11.28416768 + 2.86638700e+00j
    ],
                   dtype=np.complex128)

    if np.abs(result[:, :, 0, 1, :, :].reshape(-1) - ref).max() > 1e-6:
        return False
    #
    expected_shape = (Nqpts, 3, nmode, nmode, npol, npol)
    # The expected shape is (Nqpts, processes, modes, modes, pol, pol) for scalar ome_light

    assert result.shape == expected_shape, f"Shape mismatch: {result.shape} != {expected_shape}"
    # Validate output shape against expected structure

    assert np.all(np.isfinite(result)), "Result contains NaN or Inf values"
    # Validate output does not contain infinite or undefined values

    assert np.any(result), "Result tensor is completely zero"
    # Validate that computational loop processed properly to yield non-zero numbers

    return True


if __name__ == "__main__":
    print(test_compute_Raman_oneph_exc())
    print(test_compute_Raman_oneph_ip())
    print(test_compute_raman_twoph())
    print(test_compute_two_ph_raman_exc())
