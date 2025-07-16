## Compute resonant Raman intensities
## Sven Reichardt, etal Sci. Adv.6,eabb5915(2020)
import numpy as np
import math


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
#
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


if __name__ == "__main__":
    print(test_compute_Raman_oneph_exc())
    print(test_compute_Raman_oneph_ip())
