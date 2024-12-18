## COmpute phonon assisted luminescence base on
# M.Zanfrognini etal, PRL.131.206902 (2023)
import numpy as np
import warnings
from numba import njit, prange
import os
from exph_precision import *

warnings.filterwarnings('ignore')


@njit(cache=True, nogil=True, parallel=True)
def compute_luminescence(ome_light,
                         ph_freq,
                         ex_ene,
                         exe_low_energy,
                         ex_dip,
                         ex_ph,
                         temp=20,
                         broading=0.00124,
                         npol=3):
    ## We need exciton dipoles for light emission (<0|r|S>)
    ## and exciton phonon matrix elements for phonon absorption <S',Q|dV_Q|S,0>
    ## energy of the lowest energy energy exe_low_energy
    Nqpts, nmode, nbnd_i, nbnd_f = ex_ph.shape
    broading = numpy_float(broading / 27.211 / 2)
    ome_light_Ha = numpy_float(ome_light / 27.211)
    KbT = numpy_float(3.1726919127302026e-06 * temp)  ## Ha
    bolt_man_fac = -(ex_ene - exe_low_energy) / KbT
    bolt_man_fac = np.exp(bolt_man_fac)  ##(iq,nexe)
    sum_out = 0.0
    for iq in prange(Nqpts):
        for iv in range(nmode):
            ome_fac = ome_light_Ha * (ome_light_Ha + 2 * ph_freq[iq, iv])**2
            bose_ph_fac = 1 + (1 / (np.exp(ph_freq[iq, iv] / KbT) - 1))
            E_f_omega = ex_ene[iq, :] - ph_freq[iq, iv]
            Tmu = np.zeros((npol, nbnd_f), dtype=numpy_Cmplx)  # D*G
            ## compute scattering matrix
            for ipol in range(npol):
                for ii in range(nbnd_i):
                    Tmu[ipol,:] = Tmu[ipol,:] + np.conj(ex_ph[iq,iv,ii,:]) * ex_dip[ipol,ii] \
                        /(ex_ene[0,ii] - E_f_omega + numpy_Cmplx(1j*broading))
            ## abs and sum over initial states and pols
            Gamma_mu = bose_ph_fac * np.sum(np.abs(Tmu)**2,axis=0) * ome_fac * bolt_man_fac[iq,:] \
                        /E_f_omega/((ome_light_Ha-E_f_omega)**2 + broading**2)
            sum_out = sum_out + np.sum(Gamma_mu)
    return sum_out * broading / np.pi / Nqpts
