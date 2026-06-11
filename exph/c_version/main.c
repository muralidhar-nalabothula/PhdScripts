#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "netcdf_utils.h"
#include "raman_kernels.h"
#include "read_nc_data.h"
#include "symmetry.h"

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printf("Usage: %s <config.ini>\n", argv[0]);
        return 1;
    }

    struct Config conf;
    if (parse_config(argv[1], &conf) != 0)
    {
        printf("Error reading config %s\n", argv[1]);
        return 1;
    }

    printf("Config loaded, opening NetCDF files...\n");

    // NetCDF IDs
    int ncid_save_ns, ncid_elph, ncid_dip;
    char path[1024];

    sprintf(path, "%s/ns.db1", conf.SAVE_dir);
    if (nc_open(path, NC_NOWRITE, &ncid_save_ns) != NC_NOERR)
    {
        return 1;
    }

    if (nc_open(conf.elph_file, NC_NOWRITE, &ncid_elph) != NC_NOERR)
    {
        return 1;
    }
    if (nc_open(conf.dipole_file, NC_NOWRITE, &ncid_dip) != NC_NOERR)
    {
        return 1;
    }

    // --- 1. Read Lattice and Symmetry (SAVE) ---
    real_t* lat_vecs;
    size_t lat_size;
    read_nc_real(ncid_save_ns, "LATTICE_VECTORS", &lat_vecs, &lat_size);
    real_t CellVol = fabs(
        lat_vecs[0] * (lat_vecs[4] * lat_vecs[8] - lat_vecs[5] * lat_vecs[7]) -
        lat_vecs[1] * (lat_vecs[3] * lat_vecs[8] - lat_vecs[5] * lat_vecs[6]) +
        lat_vecs[2] * (lat_vecs[3] * lat_vecs[7] - lat_vecs[4] * lat_vecs[6]));

    real_t* symm_mats;
    size_t symm_size;
    read_nc_real(ncid_save_ns, "SYMMETRY", &symm_mats, &symm_size);
    int_type nsym = symm_size / 9;

    real_t* dims;
    size_t dims_size;
    read_nc_real(ncid_save_ns, "DIMENSIONS", &dims, &dims_size);
    int_type time_rev = (int_type)rint(dims[9]);
    int_type nelec = (int_type)rint(dims[14]);
    int_type nspinor = (int_type)rint(dims[11]);
    int_type nval = (nspinor == 2) ? nelec : nelec / 2;

    real_t* qp_ene_full;
    size_t qp_size;
    read_nc_real(ncid_save_ns, "EIGENVALUES", &qp_ene_full, &qp_size);

    // --- 2. Read Phonon/Elph Data ---
    real_t* kpts;
    size_t nkpts_size;
    read_nc_real(ncid_elph, "kpoints", &kpts, &nkpts_size);
    int_type nkpts_full = nkpts_size / 3;

    real_t* ph_freq;
    size_t ph_freq_size;
    read_nc_real(ncid_elph, "FREQ", &ph_freq, &ph_freq_size);

    real_t* qpts;
    size_t nqpts_size;
    read_nc_real(ncid_elph, "qpoints", &qpts, &nqpts_size);
    int_type nqpts = nqpts_size / 3;

    int_type nmodes = ph_freq_size / nqpts;

    int_type* kmap;
    size_t kmap_size;
    read_nc_int(ncid_elph, "kmap", &kmap, &kmap_size);
    int_type nk_ibz = 0;
    for (int_type i = 0; i < nkpts_full; i++)
    {
        if (kmap[i * 2 + 0] >= nk_ibz)
        {
            nk_ibz = kmap[i * 2 + 0] + 1;
        }
    }

    // --- 3. Slicing logic ---
    int_type min_bnd = conf.bands[0];
    int_type max_bnd = conf.bands[1];
    int_type nbnds = max_bnd - min_bnd + 1;

    int_type nvmin = 1;  // Hardcoded based on PARS dump
    int_type start_bnd_idx = min_bnd - nvmin;

    int_type nvalance_bnds = nval - min_bnd + 1;
    int_type nc = nbnds - nvalance_bnds;
    int_type nv = nvalance_bnds;

    int_type nband_full_elph = 20;  // 1 to 20
    int_type v_start = start_bnd_idx;
    int_type c_start = nval;  // val_bnd_idx

    complex_t* dipoles_ibz;
    read_dipoles(ncid_dip, &dipoles_ibz, nk_ibz, 8, 12, nc, nv, v_start,
                 c_start, 3);

    complex_t* dipoles_full =
        malloc(3 * nkpts_full * nc * nv * sizeof(complex_t));
    expand_dipoles(dipoles_full, dipoles_ibz, kmap, symm_mats, nkpts_full,
                   nk_ibz, nc, nv, 3, nsym, time_rev);

    complex_t* elph_mat = NULL;
    int_type elph_start_bnd = min_bnd - 1;  // 1-based to 0-based
    int_type elph_end_bnd = max_bnd;        // exclusive

    // QP Energies
    real_t* Qp_ene = malloc(nkpts_full * nbnds * sizeof(real_t));
    for (int_type k = 0; k < nkpts_full; k++)
    {
        int_type ibz_k = kmap[k * 2 + 0];
        for (int_type b = 0; b < nbnds; b++)
        {
            Qp_ene[k * nbnds + b] =
                qp_ene_full[ibz_k * nband_full_elph + elph_start_bnd + b];
        }
    }

    // --- 4. Call Raman ---
    int ncid_out;
    create_nc_file("Raman_tensors.nc", &ncid_out);

    if (conf.one_ph)
    {
        printf("Computing 1-Phonon Raman...\n");
        int_type n_omega_1ph = (int_type)conf.omega_one_ph_range[2];
        complex_t* all_Ram_ten =
            malloc(n_omega_1ph * nmodes * 3 * 3 * sizeof(complex_t));

        real_t* omega_1ph = malloc(n_omega_1ph * sizeof(real_t));
        for (int_type w = 0; w < n_omega_1ph; w++)
        {
            omega_1ph[w] =
                conf.omega_one_ph_range[0] +
                w * (conf.omega_one_ph_range[1] - conf.omega_one_ph_range[0]) /
                    (n_omega_1ph - 1);
        }

        // Read elph for iq = 0
        read_elph(ncid_elph, &elph_mat, 0, nqpts, nkpts_full, nmodes,
                  nband_full_elph, elph_start_bnd, elph_end_bnd);

        for (int_type m = 0; m < nmodes; m++)
        {
            real_t freq = fabs(ph_freq[0 * nmodes + m]) * 0.5;
            real_t sqrt_EPh = 0;
            if (freq > 1e-10)
            {
                sqrt_EPh = 1.0 / sqrt(2 * freq) * pow(0.5, 1.5);
            }
            for (int_type k = 0; k < nkpts_full; k++)
            {
                for (int_type b1 = 0; b1 < nbnds; b1++)
                {
                    for (int_type b2 = 0; b2 < nbnds; b2++)
                    {
                        int_type idx =
                            ((m * nkpts_full + k) * nbnds + b2) * nbnds + b1;
                        elph_mat[idx] *= sqrt_EPh;
                    }
                }
            }
        }

        real_t* ph_freq_iq0 = malloc(nmodes * sizeof(real_t));
        for (int_type m = 0; m < nmodes; m++)
        {
            ph_freq_iq0[m] = fabs(ph_freq[m]) * 0.5;
        }

        for (int_type w = 0; w < n_omega_1ph; w++)
        {
            compute_Raman_oneph_ip(
                &all_Ram_ten[w * nmodes * 3 * 3], omega_1ph[w], ph_freq_iq0,
                Qp_ene, dipoles_full, elph_mat, CellVol, conf.broading, 3, 5.0,
                nkpts_full, nc, nv, nmodes);
        }

        // Save to NetCDF
        write_nc_real_1d(ncid_out, "1ph_light_omega_eV", "n_omega_1ph",
                         omega_1ph, n_omega_1ph);

        real_t* ph_freq_cm = malloc(nmodes * sizeof(real_t));
        for (int m = 0; m < nmodes; m++)
        {
            ph_freq_cm[m] = ph_freq_iq0[m] * 219474.63;
        }
        write_nc_real_1d(ncid_out, "1ph_ph_freq_cm-1", "nmodes", ph_freq_cm,
                         nmodes);

        const char* dims_1ph[] = {"n_omega_1ph", "nmodes", "pol1", "pol2"};
        size_t sizes_1ph[] = {(size_t)n_omega_1ph, (size_t)nmodes, 3, 3};
        write_nc_complex_nd(ncid_out, "1ph_Raman_tensor", 4, dims_1ph,
                            sizes_1ph, all_Ram_ten);

        free(ph_freq_iq0);
        free(ph_freq_cm);
        free(omega_1ph);
        free(all_Ram_ten);
        free(elph_mat);
        elph_mat = NULL;
    }

    if (conf.two_ph)
    {
        printf("Computing 2-Phonon Raman...\n");
        // Output tensor shape: (n_omega_2ph, 4, nq, nmode, nmode, npol, npol)
        // For now, C code only has one omega.
        int_type n_omega_2ph = 1;
        complex_t* all_Ram_ten_twoph =
            malloc(n_omega_2ph * 4 * nqpts * nmodes * nmodes * 3 * 3 *
                   sizeof(complex_t));

        complex_t* tmp_ten =
            malloc(4 * nqpts * nmodes * nmodes * 3 * 3 * sizeof(complex_t));
        memset(tmp_ten, 0,
               4 * nqpts * nmodes * nmodes * 3 * 3 * sizeof(complex_t));

        // Frequencies expected in (nq, nmode)
        real_t* ph_freq_in = malloc(nqpts * nmodes * sizeof(real_t));
        for (int_type iq = 0; iq < nqpts; iq++)
        {
            for (int_type m = 0; m < nmodes; m++)
            {
                ph_freq_in[iq * nmodes + m] =
                    fabs(ph_freq[iq * nmodes + m]) * 0.5;
            }
        }

        // Precompute minus_iq_idx for all iq
        int_type* minus_iq_arr = malloc(nqpts * sizeof(int_type));
        for (int_type iq = 0; iq < nqpts; iq++)
        {
            double qx = -qpts[iq * 3 + 0];
            double qy = -qpts[iq * 3 + 1];
            double qz = -qpts[iq * 3 + 2];
            int_type best_idx = -1;
            double min_dist = 1e9;
            for (int_type j = 0; j < nqpts; j++)
            {
                double dx = qpts[j * 3 + 0] - qx;
                double dy = qpts[j * 3 + 1] - qy;
                double dz = qpts[j * 3 + 2] - qz;
                dx = dx - floor(dx + 0.5);
                dy = dy - floor(dy + 0.5);
                dz = dz - floor(dz + 0.5);
                double d = dx * dx + dy * dy + dz * dz;
                if (d < min_dist)
                {
                    min_dist = d;
                    best_idx = j;
                }
            }
            minus_iq_arr[iq] = best_idx;
        }

        real_t omega_2ph[] = {1.0};  // Default in code

        for (int_type w = 0; w < n_omega_2ph; w++)
        {
            double ome_light = omega_2ph[w];

            for (int_type iq = 0; iq < nqpts; iq++)
            {
                int_type minus_iq = minus_iq_arr[iq];

                complex_t* gkkp_iq = NULL;
                complex_t* gkkp_miq = NULL;

                {
                    read_elph(ncid_elph, &gkkp_iq, iq, nqpts, nkpts_full,
                              nmodes, nband_full_elph, elph_start_bnd,
                              elph_end_bnd);
                    read_elph(ncid_elph, &gkkp_miq, minus_iq, nqpts, nkpts_full,
                              nmodes, nband_full_elph, elph_start_bnd,
                              elph_end_bnd);
                }

                // Scale gkkp_iq
                for (int_type m = 0; m < nmodes; m++)
                {
                    real_t freq = ph_freq_in[iq * nmodes + m];
                    real_t sqrt_EPh =
                        (freq > 1e-10) ? (1.0 / sqrt(2 * freq) * pow(0.5, 1.5))
                                       : 0;
                    for (int_type k = 0; k < nkpts_full; k++)
                    {
                        for (int_type b1 = 0; b1 < nbnds; b1++)
                        {
                            for (int_type b2 = 0; b2 < nbnds; b2++)
                            {
                                int_type idx =
                                    ((m * nkpts_full + k) * nbnds + b2) *
                                        nbnds +
                                    b1;
                                gkkp_iq[idx] *= sqrt_EPh;
                            }
                        }
                    }
                }

                // Scale gkkp_miq
                for (int_type m = 0; m < nmodes; m++)
                {
                    real_t freq = ph_freq_in[minus_iq * nmodes + m];
                    real_t sqrt_EPh =
                        (freq > 1e-10) ? (1.0 / sqrt(2 * freq) * pow(0.5, 1.5))
                                       : 0;
                    for (int_type k = 0; k < nkpts_full; k++)
                    {
                        for (int_type b1 = 0; b1 < nbnds; b1++)
                        {
                            for (int_type b2 = 0; b2 < nbnds; b2++)
                            {
                                int_type idx =
                                    ((m * nkpts_full + k) * nbnds + b2) *
                                        nbnds +
                                    b1;
                                gkkp_miq[idx] *= sqrt_EPh;
                            }
                        }
                    }
                }

                compute_Raman_twoph_iq(
                    tmp_ten, ome_light, ph_freq_in, Qp_ene, dipoles_full,
                    gkkp_iq, gkkp_miq, kpts, qpts, iq, minus_iq, CellVol,
                    conf.broading, 3, nkpts_full, nc, nv, nmodes, nqpts);

                free(gkkp_iq);
                free(gkkp_miq);
            }  // end parallel for iq

            // Now add q and -q terms
            complex_t* current_Ram_ten =
                &all_Ram_ten_twoph[w * 4 * nqpts * nmodes * nmodes * 3 * 3];
            for (int_type iq = 0; iq < nqpts; iq++)
            {
                int_type minus_iq_idx = minus_iq_arr[iq];

                for (int_type il = 0; il < nmodes; il++)
                {
                    for (int_type jl = 0; jl < nmodes; jl++)
                    {
                        for (int_type pol1 = 0; pol1 < 3; pol1++)
                        {
                            for (int_type pol2 = 0; pol2 < 3; pol2++)
                            {
                                int_type idx0 =
                                    ((((0 * nqpts + iq) * nmodes + il) *
                                          nmodes +
                                      jl) *
                                         3 +
                                     pol1) *
                                        3 +
                                    pol2;
                                int_type mq_idx0 =
                                    ((((0 * nqpts + minus_iq_idx) * nmodes +
                                       jl) *
                                          nmodes +
                                      il) *
                                         3 +
                                     pol1) *
                                        3 +
                                    pol2;
                                current_Ram_ten[idx0] =
                                    (tmp_ten[idx0] + tmp_ten[mq_idx0]) *
                                    (1.0 / sqrt(2.0));

                                int_type idx1 =
                                    ((((1 * nqpts + iq) * nmodes + il) *
                                          nmodes +
                                      jl) *
                                         3 +
                                     pol1) *
                                        3 +
                                    pol2;
                                int_type mq_idx1 =
                                    ((((1 * nqpts + minus_iq_idx) * nmodes +
                                       jl) *
                                          nmodes +
                                      il) *
                                         3 +
                                     pol1) *
                                        3 +
                                    pol2;
                                current_Ram_ten[idx1] =
                                    (tmp_ten[idx1] + tmp_ten[mq_idx1]) *
                                    (1.0 / sqrt(2.0));

                                int_type idx2 =
                                    ((((2 * nqpts + iq) * nmodes + il) *
                                          nmodes +
                                      jl) *
                                         3 +
                                     pol1) *
                                        3 +
                                    pol2;
                                int_type mq_idx3 =
                                    ((((3 * nqpts + minus_iq_idx) * nmodes +
                                       jl) *
                                          nmodes +
                                      il) *
                                         3 +
                                     pol1) *
                                        3 +
                                    pol2;
                                current_Ram_ten[idx2] =
                                    (tmp_ten[idx2] + tmp_ten[mq_idx3]);
                            }
                        }
                    }
                }
            }
        }

        // Save to NetCDF
        write_nc_real_1d(ncid_out, "2ph_light_omega_eV", "n_omega_2ph",
                         omega_2ph, n_omega_2ph);

        const char* dims_2ph[] = {"n_omega_2ph", "n_processes", "nqpts",
                                  "nmodes",      "nmodes",      "pol1",
                                  "pol2"};
        size_t sizes_2ph[] = {(size_t)n_omega_2ph,
                              4,
                              (size_t)nqpts,
                              (size_t)nmodes,
                              (size_t)nmodes,
                              3,
                              3};
        write_nc_complex_nd(ncid_out, "2ph_Raman_tensor", 7, dims_2ph,
                            sizes_2ph, all_Ram_ten_twoph);

        free(all_Ram_ten_twoph);
        free(ph_freq_in);
        free(tmp_ten);
        free(minus_iq_arr);
    }

    nc_close(ncid_out);

    nc_close(ncid_save_ns);
    nc_close(ncid_elph);
    nc_close(ncid_dip);

    free(lat_vecs);
    free(symm_mats);
    free(dims);
    free(qp_ene_full);
    free(kpts);
    free(ph_freq);
    free(qpts);
    free(kmap);
    free(dipoles_ibz);
    free(dipoles_full);
    free(Qp_ene);

    printf("Program finished successfully.\n");
    return 0;
}
