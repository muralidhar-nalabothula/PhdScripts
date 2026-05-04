import numpy as np
from numba import njit, prange
import os
from exph_precision import *

# Y. chan  et.al  Nano Lett. 2023, 23, 3971−3977
@njit(cache=True, nogil=True, parallel=True)
def compute_exph_lifetimes(ph_freq, ex_ene, ex_ph, temp=20, broading=0.002):
    Nqpts, nmode, nbnd_i, nbnd_f = ex_ph.shape
    # All of them must be in atomic units, except broading
    broading_Ha = numpy_float(broading / 27.211 / 2.0)
    # Convert broadening to Hartrees
    KbT = numpy_float(3.1726919127302026e-06 * temp)
    # Convert Temperature to KbT in Hartrees
    Gamma_all = np.zeros((Nqpts, nbnd_i), dtype=numpy_float)
    # 2D Array to accumulate scattering rates per q-point safely inside prange
    for iq in prange(Nqpts):
        for iv in range(nmode):
            omega = ph_freq[iq, iv]
            # Get the frequency for the current phonon mode and q-point
            exp_ph_bose = np.exp(abs(omega) / KbT)
            # Compute the exponential term for the Bose-Einstein distribution
            N_q = 0.0
            # Initialize the phonon occupation factor
            if exp_ph_bose > 1.0:
                N_q = 1.0 / (exp_ph_bose - 1.0)
                # Calculate N_q safely to avoid division by zero
            for n in range(nbnd_i):
                E_initial = ex_ene[0, n]
                # Initial exciton energy E_nQ (assuming Q=0 is the first index)
                for m in range(nbnd_f):
                    E_final = ex_ene[iq, m]
                    # Final exciton energy E'_m(Q+q)
                    g2 = np.abs(ex_ph[iq, iv, n, m])**2
                    # Modulus squared of the exciton-phonon coupling matrix element
                    diff_emission = E_initial - E_final - omega
                    # Energy difference for the phonon emission term
                    delta_em = 2.0 * broading_Ha / (diff_emission**2 + broading_Ha**2)
                    # Lorentzian approximation of the delta function for emission (includes the 2*pi prefactor)
                    diff_absorption = E_initial - E_final + omega
                    # Energy difference for the phonon absorption term
                    delta_abs = 2.0 * broading_Ha / (diff_absorption**2 + broading_Ha**2)
                    # Lorentzian approximation of the delta function for absorption (includes the 2*pi prefactor)
                    Gamma_all[iq, n] += g2 * ((N_q + 1.0) * delta_em + N_q * delta_abs)
                    # Accumulate the rate for this specific q-point and initial state n
    Gamma = np.zeros(nbnd_i, dtype=numpy_float)
    # Initialize the final scattering rate array for all initial states
    for iq in range(Nqpts):
        for n in range(nbnd_i):
            Gamma[n] += Gamma_all[iq, n]
            # Sum the contributions from all q-points
            
    return Gamma / Nqpts
    # Returns both the scattering rates (in atomic units) and the relaxation times
