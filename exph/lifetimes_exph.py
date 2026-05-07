import numpy as np
from numba import njit, prange
import os
from exph_precision import *
import itertools
from scipy.interpolate import LinearNDInterpolator


# H.Y Chen et al PhysRevLett.125.107401
# Y. chan  et.al  Nano Lett. 2023, 23, 3971−3977
@njit(cache=True, nogil=True, parallel=True)
def compute_exph_lifetimes_kernel(ph_freq,
                                  ex_ene,
                                  ex_ph_abs_square,
                                  temp=20,
                                  broading=0.002):
    Nqpts, nmode, nbnd_i, nbnd_f = ex_ph_abs_square.shape
    # All of them must be in atomic units, except broading
    broading_Ha = numpy_float(broading / 27.211 / 2.0)
    # Convert broadening to Hartrees
    KbT = numpy_float(3.1726919127302026e-06 * temp)
    # Convert Temperature to KbT in Hartrees
    Gamma_all = np.zeros((Nqpts, nbnd_i), dtype=numpy_float)
    # 2D Array to accumulate scattering rates per q-point safely inside prange
    Emin = np.min(ex_ene)
    for iq in prange(Nqpts):
        for iv in range(nmode):
            omega = abs(ph_freq[iq, iv])
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
                    bolt_man_fac = -(E_final - Emin) / KbT
                    bolt_man_fac = np.exp(bolt_man_fac)  ##(iq,nexe)
                    # Final exciton energy E'_m(Q+q)
                    g2 = ex_ph_abs_square[iq, iv, n, m]
                    # Modulus squared of the exciton-phonon coupling matrix element
                    diff_emission = E_initial - E_final - omega
                    # Energy difference for the phonon emission term
                    delta_em = 2.0 * broading_Ha / (diff_emission**2 +
                                                    broading_Ha**2)
                    # Lorentzian approximation of the delta function for emission (includes the 2*pi prefactor)
                    diff_absorption = E_initial - E_final + omega
                    # Energy difference for the phonon absorption term
                    delta_abs = 2.0 * broading_Ha / (diff_absorption**2 +
                                                     broading_Ha**2)
                    # Lorentzian approximation of the delta function for absorption (includes the 2*pi prefactor)
                    Gamma_all[iq, n] += g2 * (
                        (N_q + 1.0 + bolt_man_fac) * delta_em +
                        (N_q - bolt_man_fac) * delta_abs)
                    # Accumulate the rate for this specific q-point and initial state n
    Gamma = np.zeros(nbnd_i, dtype=numpy_float)
    # Initialize the final scattering rate array for all initial states
    for iq in range(Nqpts):
        for n in range(nbnd_i):
            Gamma[n] += Gamma_all[iq, n]
            # Sum the contributions from all q-points

    # Returns both the scattering rates (in atomic units) and the relaxation times
    return Gamma / Nqpts


def periodic_expand_grid(q_red, data):
    shifts = np.array(list(itertools.product([-1, 0, 1], repeat=3)))
    q_expanded = np.vstack([q_red + shift for shift in shifts])
    tile_shape = (27,) + (1,) * (data.ndim - 1)
    data_expanded = np.tile(data, tile_shape)
    return q_expanded, data_expanded


def interpolate_and_compute_lifetimes(ph_freq_c,
                                      ex_ene_c,
                                      exph_c,
                                      a_lattice=None,
                                      q_coarse_red=None,
                                      fine_dims=None,
                                      temp=20,
                                      broading=0.002):
    if a_lattice is None or fine_dims is None or q_coarse_red is None:
        # Fallback if no grids are provided
        print("Computing lifetimes")
        return compute_exph_lifetimes_kernel(ph_freq_c,
                                             ex_ene_c,
                                             np.abs(exph_c)**2,
                                             temp=temp,
                                             broading=broading)
    print("Performing interpolation ...")
    B = 2 * np.pi * np.linalg.inv(a_lattice).T
    q_coarse_red = q_coarse_red % 1.0
    q_exp_red, ph_exp = periodic_expand_grid(q_coarse_red, ph_freq_c)
    _, ex_exp = periodic_expand_grid(q_coarse_red, ex_ene_c)
    _, g2_exp = periodic_expand_grid(q_coarse_red, np.abs(exph_c)**2)
    q_exp_cart = q_exp_red @ B.T
    interp_ph = LinearNDInterpolator(q_exp_cart, ph_exp)
    interp_ex = LinearNDInterpolator(q_exp_cart, ex_exp)
    interp_g2 = LinearNDInterpolator(q_exp_cart, g2_exp)
    n1, n2, n3 = fine_dims
    grid_1d = [np.linspace(0, 1, n, endpoint=False) for n in (n1, n2, n3)]
    q1, q2, q3 = np.meshgrid(*grid_1d, indexing='ij')
    q_fine_red = np.column_stack((q1.ravel(), q2.ravel(), q3.ravel()))
    q_fine_cart = q_fine_red @ B.T
    ph_freq_fine = interp_ph(q_fine_cart)
    ex_ene_fine = interp_ex(q_fine_cart)
    g2_fine = interp_g2(q_fine_cart)
    print("Computing lifetimes ...")
    Gamma = compute_exph_lifetimes_kernel(ph_freq_fine,
                                          ex_ene_fine,
                                          g2_fine,
                                          temp=temp,
                                          broading=broading)
    return Gamma
