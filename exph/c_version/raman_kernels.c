#include "raman_kernels.h"

#include <math.h>
#ifdef WITH_OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <string.h>

void compute_Raman_oneph_ip(complex_t* Ram_ten,  // (nmode, 3, 3)
                            double ome_light,
                            const real_t* ph_freq,      // (nmode)
                            const real_t* Qp_ene,       // (nk, nc+nv)
                            const complex_t* elec_dip,  // (npol, nk, nc, nv)
                            const complex_t* gkkp,  // (nmode, nk, nc+nv, nc+nv)
                            double CellVol, double broading, int_type npol,
                            double ph_fre_th, int_type nk, int_type nc,
                            int_type nv, int_type nmode)
{
    double broading_Ha = broading / 27.211 / 2.0;
    double ome_light_Ha = ome_light / 27.211;
    double ph_fre_th_Ha = ph_fre_th * 0.12398 / 1000.0 / 27.211;
    double ram_fac = 1.0 / nk / sqrt(CellVol);
    int_type nbnds = nc + nv;

    memset(Ram_ten, 0, nmode * 3 * 3 * sizeof(complex_t));

#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
    for (int_type i = 0; i < nmode; i++)
    {
        if (fabs(ph_freq[i]) <= ph_fre_th_Ha)
        {
            continue;
        }

        complex_t tensor_res[3][3] = {0};
        complex_t tensor_ares[3][3] = {0};

        for (int_type k = 0; k < nk; k++)
        {
            for (int_type c = 0; c < nc; c++)
            {
                for (int_type v = 0; v < nv; v++)
                {
                    double delta_E =
                        Qp_ene[k * nbnds + nv + c] - Qp_ene[k * nbnds + v];
                    complex_t delta_energies = delta_E - I * broading_Ha;

                    complex_t dipS_res[3];
                    complex_t dipS_ares[3];

                    for (int_type p1 = 0; p1 < npol; p1++)
                    {
                        complex_t dip_absorp =
                            conj(elec_dip[((p1 * nk + k) * nc + c) * nv + v]);
                        dipS_res[p1] =
                            dip_absorp / (ome_light_Ha - delta_energies);
                        dipS_ares[p1] =
                            conj(dip_absorp) / (ome_light_Ha + delta_energies);
                    }

                    for (int_type p2 = 0; p2 < npol; p2++)
                    {
                        complex_t tmp_res = 0;
                        complex_t tmp_ares = 0;

                        // gcc @ dip
                        for (int_type cp = 0; cp < nc; cp++)
                        {
                            // gcc[i, k, c, cp] = gkkp[i, k, nv+c, nv+cp]
                            complex_t gcc_val =
                                gkkp[((i * nk + k) * nbnds + nv + c) * nbnds +
                                     nv + cp];

                            double delta_E_cp = Qp_ene[k * nbnds + nv + cp] -
                                                Qp_ene[k * nbnds + v];
                            complex_t delta_energies_cp =
                                delta_E_cp - I * broading_Ha;

                            complex_t dip_absorp_cp = conj(
                                elec_dip[((p2 * nk + k) * nc + cp) * nv + v]);
                            complex_t dipSp_res_cp =
                                conj(dip_absorp_cp) /
                                (ome_light_Ha - delta_energies_cp - ph_freq[i]);
                            complex_t dipSp_ares_cp =
                                dip_absorp_cp /
                                (ome_light_Ha + delta_energies_cp - ph_freq[i]);

                            tmp_res += conj(gcc_val) * dipSp_res_cp;
                            tmp_ares += gcc_val * dipSp_ares_cp;
                        }

                        // dip @ gvv
                        for (int_type vp = 0; vp < nv; vp++)
                        {
                            // gvv[i, k, vp, v] = gkkp[i, k, vp, v]
                            complex_t gvv_val =
                                gkkp[((i * nk + k) * nbnds + vp) * nbnds + v];

                            double delta_E_vp = Qp_ene[k * nbnds + nv + c] -
                                                Qp_ene[k * nbnds + vp];
                            complex_t delta_energies_vp =
                                delta_E_vp - I * broading_Ha;

                            complex_t dip_absorp_vp = conj(
                                elec_dip[((p2 * nk + k) * nc + c) * nv + vp]);
                            complex_t dipSp_res_vp =
                                conj(dip_absorp_vp) /
                                (ome_light_Ha - delta_energies_vp - ph_freq[i]);
                            complex_t dipSp_ares_vp =
                                dip_absorp_vp /
                                (ome_light_Ha + delta_energies_vp - ph_freq[i]);

                            tmp_res -= dipSp_res_vp * conj(gvv_val);
                            tmp_ares -= dipSp_ares_vp * gvv_val;
                        }

                        for (int_type p1 = 0; p1 < npol; p1++)
                        {
                            tensor_res[p1][p2] += dipS_res[p1] * tmp_res;
                            tensor_ares[p1][p2] += dipS_ares[p1] * tmp_ares;
                        }
                    }
                }
            }
        }

        double scale =
            sqrt(fabs(ome_light_Ha - ph_freq[i]) / ome_light_Ha) * ram_fac;
        for (int_type p1 = 0; p1 < npol; p1++)
        {
            for (int_type p2 = 0; p2 < npol; p2++)
            {
                Ram_ten[(i * 3 + p1) * 3 + p2] =
                    (tensor_res[p1][p2] + tensor_ares[p1][p2]) * scale;
            }
        }
    }
}
