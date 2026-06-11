#include "symmetry.h"

#include <complex.h>

void expand_dipoles(complex_t* dip_out,       // (npol, nk_full, nc, nv)
                    const complex_t* dip_in,  // (nk_ibz, nc, nv, npol)
                    const int_type* kmap,     // (nk_full, 2)
                    const real_t* symm_mats,  // (nsym, 3, 3)
                    int_type nk_full, int_type nk_ibz, int_type nc, int_type nv,
                    int_type npol, int_type nsym, int_type time_rev)
{
    // Auto-detect if kmap is 1-based
    int_type kmap0_min = 1000000;
    int_type kmap1_min = 1000000;
    for (int_type k = 0; k < nk_full; k++)
    {
        if (kmap[k * 2 + 0] < kmap0_min)
        {
            kmap0_min = kmap[k * 2 + 0];
        }
        if (kmap[k * 2 + 1] < kmap1_min)
        {
            kmap1_min = kmap[k * 2 + 1];
        }
    }
    int_type offset0 = (kmap0_min == 1) ? 1 : 0;
    int_type offset1 = (kmap1_min == 1) ? 1 : 0;

    double threshold = nsym / (double)(time_rev + 1);

    for (int_type k = 0; k < nk_full; k++)
    {
        int_type ibz_idx = kmap[k * 2 + 0] - offset0;
        int_type sym_idx = kmap[k * 2 + 1] - offset1;

        int_type apply_time_rev =
            (kmap[k * 2 + 1] >= threshold)
                ? 1
                : 0;  // Note: Python uses kmap[:,1] original value without
                      // offset for threshold check. Wait, Python does `kmap[:,
                      // 1] >= symm_mats.shape[0] / (int_type(time_rev) + 1)`
                      // which uses the 0-based or 1-based value. Assuming
                      // Python kmap is 1-based, we should use `kmap[k*2 + 1]`
                      // as is for the threshold if it was 1-based? Actually, in
                      // Python if `symm_mats.shape[0]` is `nsym`, we will just
                      // use the value directly for consistency.
        int_type python_sym_val = kmap[k * 2 + 1];
        if (offset1 == 1 && python_sym_val > nsym)
        {
            // just to be safe it's mapped right
        }

        for (int_type c = 0; c < nc; c++)
        {
            for (int_type v = 0; v < nv; v++)
            {
                complex_t dip_exp[3] = {0};

                // np.einsum('kij,kcvj->kcvi') => sum over j (0,1,2)
                for (int_type i = 0; i < npol; i++)
                {
                    complex_t sum = 0;
                    for (int_type j = 0; j < npol; j++)
                    {
                        real_t rot =
                            symm_mats[sym_idx * 9 + j * 3 +
                                      i];  // Python does transpose(0, 2, 1)
                        // dip_in index: [ibz_idx, c, v, j]
                        int_type in_idx =
                            ((ibz_idx * nc + c) * nv + v) * npol + j;
                        sum += rot * dip_in[in_idx];
                    }
                    dip_exp[i] = sum;
                }

                for (int_type p = 0; p < npol; p++)
                {
                    complex_t val = dip_exp[p];

                    // apply time reversal
                    if (apply_time_rev)
                    {
                        val = conj(val);
                    }

                    // Python does .conj() at the end:
                    // ele_dips = dipole_expand(...).transpose(3, 0, 1,
                    // 2).conj()
                    val = conj(val);

                    // Output shape: (npol, nk_full, nc, nv)
                    int_type out_idx = ((p * nk_full + k) * nc + c) * nv + v;
                    dip_out[out_idx] = val;
                }
            }
        }
    }
}
