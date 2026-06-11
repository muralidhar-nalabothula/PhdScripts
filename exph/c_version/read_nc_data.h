#ifndef READ_NC_DATA_H
#define READ_NC_DATA_H

#include "types.h"

// Read specific slices from netCDF.

// elph shape: (nq, nk, nmode, 1, nband_full, nband_full, 2)
// dipoles shape: (1, nk_ibz, nval_full, ncond_full, npol, 2)
// QP energies: (1, nk_ibz, nband_full)

int read_dipoles(int ncid, complex_t** dip_out, int_type nk_ibz,
                 int_type nc_full, int_type nv_full, int_type nc, int_type nv,
                 int_type v_start, int_type c_start, int_type npol);

int read_elph(int ncid, complex_t** elph_out, int_type iq, int_type nq, int_type nk,
              int_type nmode, int_type nband_full, int_type start_bnd_idx,
              int_type end_bnd);

#endif
