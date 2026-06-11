#ifndef SYMMETRY_H
#define SYMMETRY_H

#include "types.h"

// Expand dipoles from IBZ to full BZ using symmetry matrices.
// Outputs dipoles in shape (npol, nk_full, nc, nv)
// ele_dipoles_ibz is (nk_ibz, nc, nv, npol)
// kmap is (nk_full, 2)
// symm_mats is (nsym, 3, 3)
void expand_dipoles(complex_t* dip_out,       // (npol, nk_full, nc, nv)
                    const complex_t* dip_in,  // (nk_ibz, nc, nv, npol)
                    const int_type* kmap,     // (nk_full, 2)
                    const real_t* symm_mats,  // (nsym, 3, 3)
                    int_type nk_full, int_type nk_ibz, int_type nc, int_type nv,
                    int_type npol, int_type nsym, int_type time_rev);

#endif
