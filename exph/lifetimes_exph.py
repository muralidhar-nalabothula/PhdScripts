import numpy as np
from numba import njit, prange
import os
from exph_precision import *
import itertools
from scipy.interpolate import LinearNDInterpolator
from tqdm import tqdm


# H.Y Chen et al PhysRevLett.125.107401
# Y. chan  et.al  Nano Lett. 2023, 23, 3971−3977
@njit(cache=True, nogil=True, parallel=True)
def compute_exph_lifetimes_kernel(ph_freq,
                                  ex_ene,
                                  ex_ph_abs_square,
                                  E_initial_bands,
                                  Emin,
                                  temp=20,
                                  broading=0.002):
    Nqpts, nmode, nbnd_i, nbnd_f = ex_ph_abs_square.shape
    broading_Ha = numpy_float(broading / 27.211 / 2.0)
    KbT = numpy_float(3.1726919127302026e-06 * temp)
    Gamma_all = np.zeros((Nqpts, nbnd_i), dtype=numpy_float)
    for iq in prange(Nqpts):
        for iv in range(nmode):
            omega = abs(ph_freq[iq, iv])
            exp_ph_bose = np.exp(abs(omega) / KbT)
            N_q = 0.0
            if exp_ph_bose > 1.0:
                N_q = 1.0 / (exp_ph_bose - 1.0)
            #
            for n in range(nbnd_i):
                E_initial = E_initial_bands[n]
                # Use the globally evaluated initial band energies (Q=0)
                #
                for m in range(nbnd_f):
                    E_final = ex_ene[iq, m]
                    bolt_man_fac = -(E_final - Emin) / KbT
                    bolt_man_fac = np.exp(bolt_man_fac)  ##(iq,nexe)
                    #
                    g2 = ex_ph_abs_square[iq, iv, n, m]
                    diff_emission = E_initial - E_final - omega
                    delta_em = 2.0 * broading_Ha / (diff_emission**2 +
                                                    broading_Ha**2)
                    #
                    diff_absorption = E_initial - E_final + omega
                    delta_abs = 2.0 * broading_Ha / (diff_absorption**2 +
                                                     broading_Ha**2)
                    #
                    Gamma_all[iq, n] += g2 * (
                        (N_q + 1.0 + bolt_man_fac) * delta_em +
                        (N_q - bolt_man_fac) * delta_abs)

    Gamma = np.zeros(nbnd_i, dtype=numpy_float)
    for iq in range(Nqpts):
        for n in range(nbnd_i):
            Gamma[n] += Gamma_all[iq, n]

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
                                      broading=0.002,
                                      batch_size=1024):
    if a_lattice is None or fine_dims is None or q_coarse_red is None:
        print("Computing lifetimes")
        # Ensure we pass the global parameters for the fallback as well
        E_initial_bands = ex_ene_c[0, :exph_c.shape[2]]
        Emin = np.min(ex_ene_c)
        return compute_exph_lifetimes_kernel(ph_freq_c,
                                             ex_ene_c,
                                             np.abs(exph_c)**2,
                                             E_initial_bands,
                                             Emin,
                                             temp=temp,
                                             broading=broading)

    print("Performing interpolation setup ...")
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

    # 1. Pre-calculate small arrays wholly to avoid batch-boundary issues
    print("Interpolating energies and frequencies ...")
    ph_freq_fine = interp_ph(q_fine_cart)
    ex_ene_fine = interp_ex(q_fine_cart)

    # 2. Extract global energy parameters needed by the kernel
    nbnd_i = exph_c.shape[2]
    E_initial_bands = ex_ene_fine[
        0, :nbnd_i]  # Assuming Q=0 is the first index from meshgrid
    Emin = np.min(ex_ene_fine)

    total_qpts = q_fine_cart.shape[0]
    total_Gamma = np.zeros(nbnd_i, dtype=numpy_float)

    # Adjust batch size if it's larger than the total number of fine q-points
    batch_size = min(batch_size, total_qpts)
    #
    print(
        f"Interpolating couplings and computing lifetimes in batches of {batch_size}..."
    )
    for i in tqdm(range(0, total_qpts, batch_size), desc="Processing Batches"):
        q_batch = q_fine_cart[i:i + batch_size]
        ph_batch = ph_freq_fine[i:i + batch_size]
        ex_batch = ex_ene_fine[i:i + batch_size]
        g2_batch = interp_g2(q_batch)
        gamma_batch_avg = compute_exph_lifetimes_kernel(ph_batch,
                                                        ex_batch,
                                                        g2_batch,
                                                        E_initial_bands,
                                                        Emin,
                                                        temp=temp,
                                                        broading=broading)

        # The kernel returns the average over the batch.
        # We multiply by the current batch size to accumulate the true sum.
        current_batch_size = q_batch.shape[0]
        total_Gamma += gamma_batch_avg * current_batch_size
    return total_Gamma / total_qpts
