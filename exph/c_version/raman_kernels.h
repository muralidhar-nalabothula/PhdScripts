#ifndef RAMAN_KERNELS_H
#define RAMAN_KERNELS_H

#include "types.h"

// compute_Raman_oneph_ip
// Ram_ten: (nmode, 3, 3)
void compute_Raman_oneph_ip(complex_t* Ram_ten,  // output
                            double ome_light,
                            const real_t* ph_freq,      // (nmode)
                            const real_t* Qp_ene,       // (nk, nc+nv)
                            const complex_t* elec_dip,  // (npol, nk, nc, nv)
                            const complex_t* gkkp,  // (nmode, nk, nc+nv, nc+nv)
                            double CellVol, double broading, int_type npol,
                            double ph_fre_th, int_type nk, int_type nc,
                            int_type nv, int_type nmode);

// compute_Raman_twoph_iq
void compute_Raman_twoph_iq(
    complex_t* twoph_raman_ten,  // output
    double ome_light,
    const real_t* ph_freq,      // (nq, nmode)
    const real_t* Qp_ene,       // (nk, nc+nv)
    const complex_t* elec_dip,  // (npol, nk, nc, nv)
    const complex_t* gkkp_iq,   // (nmode, nk, nc+nv, nc+nv)
    const complex_t* gkkp_miq,  // (nmode, nk, nc+nv, nc+nv)
    const real_t* kpts,         // (nk, 3)
    const real_t* qpts,         // (nq, 3)
    int_type iq, int_type minus_iq_idx, double CellVol, double broading,
    int_type npol, int_type nk, int_type nc, int_type nv, int_type nmode,
    int_type nq);

#endif
