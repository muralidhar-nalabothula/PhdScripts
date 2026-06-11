#include "read_nc_data.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "netcdf_utils.h"

int read_dipoles(int ncid, complex_t** dip_out, int_type nk_ibz,
                 int_type nv_full, int_type nc_full, int_type nc, int_type nv,
                 int_type v_start, int_type c_start, int_type npol)
{
    complex_t* full_dip;
    size_t full_size;
    if (read_nc_complex(ncid, "DIP_v", &full_dip, &full_size) != 0)
    {
        return -1;
    }

    // allocate requested slice: (nk_ibz, nc, nv, npol)
    *dip_out = malloc(nk_ibz * nc * nv * npol * sizeof(complex_t));

    for (int_type k = 0; k < nk_ibz; k++)
    {
        for (int_type c = 0; c < nc; c++)
        {
            for (int_type v = 0; v < nv; v++)
            {
                for (int_type p = 0; p < npol; p++)
                {
                    // python: dips_sliced = dips_data[:, v_start_bnd:,
                    // :c_end_bnd, ...] Python shape: [nk_ibz, nv_full, nc_full,
                    // npol] Here v is indexing from 0 to nv-1 mapping to
                    // v_start to v_start+nv-1 c is indexing from 0 to nc-1
                    // mapping to 0 to c_end_bnd-1 (which is 0 to nc-1)

                    int_type v_in = v_start + v;
                    int_type c_in = c;  // Note Python does :c_end_bnd, which
                                        // starts from 0 of the nc_full array

                    int_type idx_full =
                        ((k * nv_full + v_in) * nc_full + c_in) * npol + p;
                    complex_t val = full_dip[idx_full];

                    // Python does .conj()
                    val = conj(val);

                    // Python transposes to: (k, c, v, pol)
                    int_type idx_out = ((k * nc + c) * nv + v) * npol + p;
                    (*dip_out)[idx_out] = val;
                }
            }
        }
    }

    free(full_dip);
    return 0;
}

int read_elph(int ncid, complex_t** elph_out, int_type iq, int_type nq, int_type nk,
              int_type nmode, int_type nband_full, int_type start_bnd_idx,
              int_type end_bnd)
{
    // elph shape: (nq, nk, nmode, 1, nband_full, nband_full, 2)
    int varid;
    if (nc_inq_varid(ncid, "elph_mat", &varid) != NC_NOERR)
    {
        return -1;
    }

    size_t start[7] = {iq, 0, 0, 0, 0, 0, 0};
    size_t count[7] = {1, nk, nmode, 1, nband_full, nband_full, 2};

    size_t raw_size = 1;
    for (int i = 0; i < 7; i++) raw_size *= count[i];

    real_t* raw_data = malloc(raw_size * sizeof(real_t));
    
    if (sizeof(real_t) == sizeof(double)) {
        if (nc_get_vara_double(ncid, varid, start, count, (double*)raw_data) != NC_NOERR) {
            free(raw_data);
            return -1;
        }
    } else {
        if (nc_get_vara_float(ncid, varid, start, count, (float*)raw_data) != NC_NOERR) {
            free(raw_data);
            return -1;
        }
    }

    size_t cmplx_size = raw_size / 2;
    complex_t* full_elph = malloc(cmplx_size * sizeof(complex_t));
    for (size_t i = 0; i < cmplx_size; i++)
    {
        full_elph[i] = raw_data[2 * i] + I * raw_data[2 * i + 1];
    }
    free(raw_data);

    int_type nbnds = end_bnd - start_bnd_idx;
    // target shape: (nmode, nk, nbnds, nbnds)
    if (*elph_out == NULL) {
        *elph_out =
            malloc(nmode * nk * nbnds * nbnds * sizeof(complex_t));
    }

    for (int_type m = 0; m < nmode; m++)
    {
        for (int_type k = 0; k < nk; k++)
        {
            for (int_type b1 = 0; b1 < nbnds; b1++)
            {
                for (int_type b2 = 0; b2 < nbnds; b2++)
                {
                    // full_elph shape: (1, nk, nmode, 1, nband_full, nband_full)
                    int_type full_b1 = start_bnd_idx + b1;
                    int_type full_b2 = start_bnd_idx + b2;

                    int_type idx_full =
                        (((k * nmode + m) * 1 + 0) *
                             nband_full +
                         full_b1) *
                            nband_full +
                        full_b2;
                    complex_t val = full_elph[idx_full];

                    // Output python shape: (nmodes, nk, final_band, initial_band)
                    // So in C, we map output as: [m, k, b2, b1]
                    int_type idx_out =
                        ((m * nk + k) * nbnds + b2) * nbnds +
                        b1;
                    (*elph_out)[idx_out] = val;
                }
            }
        }
    }

    free(full_elph);
    return 0;
}
