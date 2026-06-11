#include <math.h>
#ifdef WITH_OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raman_kernels.h"

static inline double mod1(double x)
{
    double res = x - floor(x);
    if (res < 0)
    {
        res += 1.0;
    }
    return res;
}

static int_type find_nearest(const real_t* pts, int_type n, double qx,
                             double qy, double qz)
{
    int_type best_idx = -1;
    double min_dist = 1e9;
    for (int_type i = 0; i < n; i++)
    {
        double dx = mod1(pts[i * 3 + 0] - qx + 0.5) - 0.5;
        double dy = mod1(pts[i * 3 + 1] - qy + 0.5) - 0.5;
        double dz = mod1(pts[i * 3 + 2] - qz + 0.5) - 0.5;
        double d = dx * dx + dy * dy + dz * dz;
        if (d < min_dist)
        {
            min_dist = d;
            best_idx = i;
        }
    }
    return best_idx;
}

void compute_Raman_twoph_iq(complex_t* twoph_raman_ten, double ome_light,
                            const real_t* ph_freq, const real_t* Qp_ene,
                            const complex_t* elec_dip, const complex_t* gkkp_iq,
                            const complex_t* gkkp_miq, const real_t* kpts,
                            const real_t* qpts, int_type iq,
                            int_type minus_iq_idx, double CellVol,
                            double broading, int_type npol, int_type nk,
                            int_type nc, int_type nv, int_type nmode,
                            int_type nq)
{
    double broading_Ha = broading / 27.211 / 2.0;
    double ome_light_Ha = ome_light / 27.211;
    int_type nbnds = nc + nv;

    complex_t* tmp_ten = twoph_raman_ten;

    {
        double qx = qpts[iq * 3 + 0];
        double qy = qpts[iq * 3 + 1];
        double qz = qpts[iq * 3 + 2];

        int_type* idx_kplusq = malloc(nk * sizeof(int_type));
        int_type* idx_kminusq = malloc(nk * sizeof(int_type));

        for (int_type k = 0; k < nk; k++)
        {
            idx_kplusq[k] =
                find_nearest(kpts, nk, kpts[k * 3 + 0] + qx,
                             kpts[k * 3 + 1] + qy, kpts[k * 3 + 2] + qz);
            idx_kminusq[k] =
                find_nearest(kpts, nk, kpts[k * 3 + 0] - qx,
                             kpts[k * 3 + 1] - qy, kpts[k * 3 + 2] - qz);
        }

#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
        for (int_type il = 0; il < nmode; il++)
        {
            for (int_type jl = 0; jl < nmode; jl++)
            {
                double ph_sum_aa = ph_freq[minus_iq_idx * nmode + jl] +
                                   ph_freq[iq * nmode + il];
                double ram_fac_aa =
                    sqrt(fabs(ome_light_Ha + ph_sum_aa) / ome_light_Ha);

                double ph_sum_ee = -ph_freq[iq * nmode + jl] -
                                   ph_freq[minus_iq_idx * nmode + il];
                double ram_fac_ee =
                    sqrt(fabs(ome_light_Ha + ph_sum_ee) / ome_light_Ha);

                double ph_sum_ae =
                    ph_freq[iq * nmode + il] - ph_freq[iq * nmode + jl];
                double ram_fac_ae =
                    sqrt(fabs(ome_light_Ha + ph_sum_ae) / ome_light_Ha);

                double ph_sum_ea = ph_freq[minus_iq_idx * nmode + jl] -
                                   ph_freq[minus_iq_idx * nmode + il];
                double ram_fac_ea =
                    sqrt(fabs(ome_light_Ha + ph_sum_ea) / ome_light_Ha);

                for (int_type pol1 = 0; pol1 < npol; pol1++)
                {
                    for (int_type pol2 = 0; pol2 < npol; pol2++)
                    {
                        complex_t sum_aa = 0, sum_ee = 0, sum_ae = 0,
                                  sum_ea = 0;

                        for (int_type k = 0; k < nk; k++)
                        {
                            int_type kpq = idx_kplusq[k];
                            int_type kmq = idx_kminusq[k];

                            for (int_type c = 0; c < nc; c++)
                            {
                                for (int_type v = 0; v < nv; v++)
                                {
                                    complex_t dipS_res =
                                        conj(elec_dip[((pol2 * nk + k) * nc +
                                                       c) *
                                                          nv +
                                                      v]) /
                                        (ome_light_Ha -
                                         (Qp_ene[k * nbnds + nv + c] -
                                          Qp_ene[k * nbnds + v]) +
                                         I * broading_Ha);

                                    // Process 0: AA
                                    // M1
                                    complex_t m1_tmp1_aa = 0, m1_tmp2_aa = 0;
                                    for (int_type cp = 0; cp < nc; cp++)
                                    {
                                        complex_t G1 =
                                            ram_fac_aa *
                                            elec_dip[((pol1 * nk + k) * nc +
                                                      cp) *
                                                         nv +
                                                     v] /
                                            (ph_sum_aa +
                                             (ome_light_Ha -
                                              (Qp_ene[k * nbnds + nv + cp] -
                                               Qp_ene[k * nbnds + v]) +
                                              I * broading_Ha));
                                        complex_t gcc1 =
                                            gkkp_miq[((jl * nk + kpq) * nbnds +
                                                      nv + cp) *
                                                         nbnds +
                                                     nv + c];
                                        m1_tmp1_aa += gcc1 * G1;

                                        complex_t gcc2 =
                                            gkkp_iq[((il * nk + k) * nbnds +
                                                     nv + c) *
                                                        nbnds +
                                                    nv + cp];
                                        complex_t dS =
                                            conj(
                                                elec_dip[((pol2 * nk + k) * nc +
                                                          cp) *
                                                             nv +
                                                         v]) /
                                            (ome_light_Ha -
                                             (Qp_ene[k * nbnds + nv + cp] -
                                              Qp_ene[k * nbnds + v]) +
                                             I * broading_Ha);
                                        m1_tmp2_aa += gcc2 * dS;
                                    }
                                    complex_t D_kqc_kv_il =
                                        1.0 / (ome_light_Ha -
                                               (Qp_ene[kpq * nbnds + nv + c] -
                                                Qp_ene[k * nbnds + v]) +
                                               I * broading_Ha +
                                               ph_freq[iq * nmode + il]);
                                    sum_aa +=
                                        m1_tmp2_aa * m1_tmp1_aa * D_kqc_kv_il;

                                    // M2
                                    complex_t m2_tmp1_aa = 0, m2_tmp2_aa = 0;
                                    for (int_type vp = 0; vp < nv; vp++)
                                    {
                                        complex_t G1 =
                                            ram_fac_aa *
                                            elec_dip[((pol1 * nk + k) * nc +
                                                      c) *
                                                         nv +
                                                     vp] /
                                            (ph_sum_aa +
                                             (ome_light_Ha -
                                              (Qp_ene[k * nbnds + nv + c] -
                                               Qp_ene[k * nbnds + vp]) +
                                              I * broading_Ha));
                                        complex_t gvv1 =
                                            gkkp_miq[((jl * nk + k) * nbnds +
                                                      v) *
                                                         nbnds +
                                                     vp];
                                        m2_tmp1_aa += G1 * gvv1;

                                        complex_t dS =
                                            conj(
                                                elec_dip[((pol2 * nk + k) * nc +
                                                          c) *
                                                             nv +
                                                         vp]) /
                                            (ome_light_Ha -
                                             (Qp_ene[k * nbnds + nv + c] -
                                              Qp_ene[k * nbnds + vp]) +
                                             I * broading_Ha);
                                        complex_t gvv2 =
                                            gkkp_iq[((il * nk + kmq) * nbnds +
                                                     vp) *
                                                        nbnds +
                                                    v];
                                        m2_tmp2_aa += dS * gvv2;
                                    }
                                    complex_t D_kc_kmqv_il =
                                        1.0 / (ome_light_Ha -
                                               (Qp_ene[k * nbnds + nv + c] -
                                                Qp_ene[kmq * nbnds + v]) +
                                               I * broading_Ha +
                                               ph_freq[iq * nmode + il]);
                                    sum_aa +=
                                        m2_tmp2_aa * m2_tmp1_aa * D_kc_kmqv_il;

                                    // M3
                                    complex_t m3_tmp2_aa = 0;
                                    for (int_type cp = 0; cp < nc; cp++)
                                    {
                                        complex_t m3_tmp1_aa_cp = 0;
                                        for (int_type vp = 0; vp < nv; vp++)
                                        {
                                            complex_t G1 =
                                                ram_fac_aa *
                                                elec_dip[((pol1 * nk + kpq) *
                                                              nc +
                                                          cp) *
                                                             nv +
                                                         vp] /
                                                (ph_sum_aa +
                                                 (ome_light_Ha -
                                                  (Qp_ene[kpq * nbnds + nv +
                                                          cp] -
                                                   Qp_ene[kpq * nbnds + vp]) +
                                                  I * broading_Ha));
                                            complex_t gvv1 = gkkp_miq
                                                [((jl * nk + kpq) * nbnds + v) *
                                                     nbnds +
                                                 vp];
                                            m3_tmp1_aa_cp += G1 * gvv1;
                                        }
                                        complex_t D_kqc_kv_il_cp =
                                            1.0 /
                                            (ome_light_Ha -
                                             (Qp_ene[kpq * nbnds + nv + cp] -
                                              Qp_ene[k * nbnds + v]) +
                                             I * broading_Ha +
                                             ph_freq[iq * nmode + il]);
                                        m3_tmp1_aa_cp *= D_kqc_kv_il_cp;
                                        complex_t gcc =
                                            gkkp_iq[((il * nk + k) * nbnds +
                                                     nv + cp) *
                                                        nbnds +
                                                    nv + c];
                                        m3_tmp2_aa += gcc * m3_tmp1_aa_cp;
                                    }
                                    sum_aa -= dipS_res * m3_tmp2_aa;

                                    // M4
                                    complex_t m4_tmp2_aa = 0;
                                    for (int_type vp = 0; vp < nv; vp++)
                                    {
                                        complex_t m4_tmp1_aa_vp = 0;
                                        for (int_type cp = 0; cp < nc; cp++)
                                        {
                                            complex_t G1 =
                                                ram_fac_aa *
                                                elec_dip[((pol1 * nk + kmq) *
                                                              nc +
                                                          cp) *
                                                             nv +
                                                         vp] /
                                                (ph_sum_aa +
                                                 (ome_light_Ha -
                                                  (Qp_ene[kmq * nbnds + nv +
                                                          cp] -
                                                   Qp_ene[kmq * nbnds + vp]) +
                                                  I * broading_Ha));
                                            complex_t gcc1 =
                                                gkkp_miq[((jl * nk + k) *
                                                              nbnds +
                                                          nv + cp) *
                                                             nbnds +
                                                         nv + c];
                                            m4_tmp1_aa_vp += gcc1 * G1;
                                        }
                                        complex_t D_kc_kmqv_il_vp =
                                            1.0 / (ome_light_Ha -
                                                   (Qp_ene[k * nbnds + nv + c] -
                                                    Qp_ene[kmq * nbnds + vp]) +
                                                   I * broading_Ha +
                                                   ph_freq[iq * nmode + il]);
                                        m4_tmp1_aa_vp *= D_kc_kmqv_il_vp;
                                        complex_t gvv =
                                            gkkp_iq[((il * nk + kmq) * nbnds +
                                                     v) *
                                                        nbnds +
                                                    vp];
                                        m4_tmp2_aa += m4_tmp1_aa_vp * gvv;
                                    }
                                    sum_aa -= dipS_res * m4_tmp2_aa;

                                    // Process 1: EE
                                    // M1 (E-E)
                                    complex_t m1_tmp1_ee = 0, m1_tmp2_ee = 0;
                                    for (int_type cp = 0; cp < nc; cp++)
                                    {
                                        complex_t G1 =
                                            ram_fac_ee *
                                            elec_dip[((pol1 * nk + k) * nc +
                                                      cp) *
                                                         nv +
                                                     v] /
                                            (ph_sum_ee +
                                             (ome_light_Ha -
                                              (Qp_ene[k * nbnds + nv + cp] -
                                               Qp_ene[k * nbnds + v]) +
                                              I * broading_Ha));
                                        complex_t gcc1 = conj(
                                            gkkp_iq[((jl * nk + k) * nbnds +
                                                     nv + c) *
                                                        nbnds +
                                                    nv + cp]);
                                        m1_tmp1_ee += gcc1 * G1;

                                        complex_t dS =
                                            conj(
                                                elec_dip[((pol2 * nk + k) * nc +
                                                          cp) *
                                                             nv +
                                                         v]) /
                                            (ome_light_Ha -
                                             (Qp_ene[k * nbnds + nv + cp] -
                                              Qp_ene[k * nbnds + v]) +
                                             I * broading_Ha);
                                        complex_t gcc2 = conj(
                                            gkkp_miq[((il * nk + kpq) * nbnds +
                                                      nv + cp) *
                                                         nbnds +
                                                     nv + c]);
                                        m1_tmp2_ee += gcc2 * dS;
                                    }
                                    complex_t D_kqc_kv_minus_il =
                                        1.0 /
                                        (ome_light_Ha -
                                         (Qp_ene[kpq * nbnds + nv + c] -
                                          Qp_ene[k * nbnds + v]) +
                                         I * broading_Ha -
                                         ph_freq[minus_iq_idx * nmode + il]);
                                    sum_ee += m1_tmp2_ee * m1_tmp1_ee *
                                              D_kqc_kv_minus_il;

                                    // M2 (H-H)
                                    complex_t m2_tmp1_ee = 0, m2_tmp2_ee = 0;
                                    for (int_type vp = 0; vp < nv; vp++)
                                    {
                                        complex_t G1 =
                                            ram_fac_ee *
                                            elec_dip[((pol1 * nk + k) * nc +
                                                      c) *
                                                         nv +
                                                     vp] /
                                            (ph_sum_ee +
                                             (ome_light_Ha -
                                              (Qp_ene[k * nbnds + nv + c] -
                                               Qp_ene[k * nbnds + vp]) +
                                              I * broading_Ha));
                                        complex_t gvv1 = conj(
                                            gkkp_iq[((jl * nk + kmq) * nbnds +
                                                     vp) *
                                                        nbnds +
                                                    v]);
                                        m2_tmp1_ee += G1 * gvv1;

                                        complex_t dS =
                                            conj(
                                                elec_dip[((pol2 * nk + k) * nc +
                                                          c) *
                                                             nv +
                                                         vp]) /
                                            (ome_light_Ha -
                                             (Qp_ene[k * nbnds + nv + c] -
                                              Qp_ene[k * nbnds + vp]) +
                                             I * broading_Ha);
                                        complex_t gvv2 = conj(
                                            gkkp_miq[((il * nk + k) * nbnds +
                                                      v) *
                                                         nbnds +
                                                     vp]);
                                        m2_tmp2_ee += dS * gvv2;
                                    }
                                    complex_t D_kc_kmqv_minus_il =
                                        1.0 /
                                        (ome_light_Ha -
                                         (Qp_ene[k * nbnds + nv + c] -
                                          Qp_ene[kmq * nbnds + v]) +
                                         I * broading_Ha -
                                         ph_freq[minus_iq_idx * nmode + il]);
                                    sum_ee += m2_tmp2_ee * m2_tmp1_ee *
                                              D_kc_kmqv_minus_il;

                                    // M3 (E-H)
                                    complex_t m3_tmp2_ee = 0;
                                    for (int_type cp = 0; cp < nc; cp++)
                                    {
                                        complex_t m3_tmp1_ee_cp = 0;
                                        for (int_type vp = 0; vp < nv; vp++)
                                        {
                                            complex_t G1 =
                                                ram_fac_ee *
                                                elec_dip[((pol1 * nk + kpq) *
                                                              nc +
                                                          cp) *
                                                             nv +
                                                         vp] /
                                                (ph_sum_ee +
                                                 (ome_light_Ha -
                                                  (Qp_ene[kpq * nbnds + nv +
                                                          cp] -
                                                   Qp_ene[kpq * nbnds + vp]) +
                                                  I * broading_Ha));
                                            complex_t gvv1 = conj(
                                                gkkp_iq[((jl * nk + k) * nbnds +
                                                         vp) *
                                                            nbnds +
                                                        v]);
                                            m3_tmp1_ee_cp += G1 * gvv1;
                                        }
                                        complex_t D_kqc_kv_minus_il_cp =
                                            1.0 /
                                            (ome_light_Ha -
                                             (Qp_ene[kpq * nbnds + nv + cp] -
                                              Qp_ene[k * nbnds + v]) +
                                             I * broading_Ha -
                                             ph_freq[minus_iq_idx * nmode +
                                                     il]);
                                        m3_tmp1_ee_cp *= D_kqc_kv_minus_il_cp;
                                        complex_t gcc = conj(
                                            gkkp_miq[((il * nk + kpq) * nbnds +
                                                      nv + c) *
                                                         nbnds +
                                                     nv + cp]);
                                        m3_tmp2_ee += gcc * m3_tmp1_ee_cp;
                                    }
                                    sum_ee -= dipS_res * m3_tmp2_ee;

                                    // M4 (H-E)
                                    complex_t m4_tmp2_ee = 0;
                                    for (int_type vp = 0; vp < nv; vp++)
                                    {
                                        complex_t m4_tmp1_ee_vp = 0;
                                        for (int_type cp = 0; cp < nc; cp++)
                                        {
                                            complex_t G1 =
                                                ram_fac_ee *
                                                elec_dip[((pol1 * nk + kmq) *
                                                              nc +
                                                          cp) *
                                                             nv +
                                                         vp] /
                                                (ph_sum_ee +
                                                 (ome_light_Ha -
                                                  (Qp_ene[kmq * nbnds + nv +
                                                          cp] -
                                                   Qp_ene[kmq * nbnds + vp]) +
                                                  I * broading_Ha));
                                            complex_t gcc1 =
                                                conj(gkkp_iq[((jl * nk + kmq) *
                                                                  nbnds +
                                                              nv + c) *
                                                                 nbnds +
                                                             nv + cp]);
                                            m4_tmp1_ee_vp += gcc1 * G1;
                                        }
                                        complex_t D_kc_kmqv_minus_il_vp =
                                            1.0 /
                                            (ome_light_Ha -
                                             (Qp_ene[k * nbnds + nv + c] -
                                              Qp_ene[kmq * nbnds + vp]) +
                                             I * broading_Ha -
                                             ph_freq[minus_iq_idx * nmode +
                                                     il]);
                                        m4_tmp1_ee_vp *= D_kc_kmqv_minus_il_vp;
                                        complex_t gvv = conj(
                                            gkkp_miq[((il * nk + k) * nbnds +
                                                      vp) *
                                                         nbnds +
                                                     v]);
                                        m4_tmp2_ee += m4_tmp1_ee_vp * gvv;
                                    }
                                    sum_ee -= dipS_res * m4_tmp2_ee;

                                    // Process 2: AE
                                    // M1
                                    complex_t m1_tmp1_ae = 0, m1_tmp2_ae = 0;
                                    for (int_type cp = 0; cp < nc; cp++)
                                    {
                                        complex_t G1 =
                                            ram_fac_ae *
                                            elec_dip[((pol1 * nk + k) * nc +
                                                      cp) *
                                                         nv +
                                                     v] /
                                            (ph_sum_ae +
                                             (ome_light_Ha -
                                              (Qp_ene[k * nbnds + nv + cp] -
                                               Qp_ene[k * nbnds + v]) +
                                              I * broading_Ha));
                                        complex_t gcc1 = conj(
                                            gkkp_iq[((jl * nk + k) * nbnds +
                                                     nv + c) *
                                                        nbnds +
                                                    nv + cp]);
                                        m1_tmp1_ae += gcc1 * G1;

                                        complex_t dS =
                                            conj(
                                                elec_dip[((pol2 * nk + k) * nc +
                                                          cp) *
                                                             nv +
                                                         v]) /
                                            (ome_light_Ha -
                                             (Qp_ene[k * nbnds + nv + cp] -
                                              Qp_ene[k * nbnds + v]) +
                                             I * broading_Ha);
                                        complex_t gcc2 =
                                            gkkp_iq[((il * nk + k) * nbnds +
                                                     nv + c) *
                                                        nbnds +
                                                    nv + cp];
                                        m1_tmp2_ae += gcc2 * dS;
                                    }
                                    sum_ae +=
                                        m1_tmp2_ae * m1_tmp1_ae * D_kqc_kv_il;

                                    // M2
                                    complex_t m2_tmp1_ae = 0, m2_tmp2_ae = 0;
                                    for (int_type vp = 0; vp < nv; vp++)
                                    {
                                        complex_t G1 =
                                            ram_fac_ae *
                                            elec_dip[((pol1 * nk + k) * nc +
                                                      c) *
                                                         nv +
                                                     vp] /
                                            (ph_sum_ae +
                                             (ome_light_Ha -
                                              (Qp_ene[k * nbnds + nv + c] -
                                               Qp_ene[k * nbnds + vp]) +
                                              I * broading_Ha));
                                        complex_t gvv1 = conj(
                                            gkkp_iq[((jl * nk + kmq) * nbnds +
                                                     vp) *
                                                        nbnds +
                                                    v]);
                                        m2_tmp1_ae += G1 * gvv1;

                                        complex_t dS =
                                            conj(
                                                elec_dip[((pol2 * nk + k) * nc +
                                                          c) *
                                                             nv +
                                                         vp]) /
                                            (ome_light_Ha -
                                             (Qp_ene[k * nbnds + nv + c] -
                                              Qp_ene[k * nbnds + vp]) +
                                             I * broading_Ha);
                                        complex_t gvv2 =
                                            gkkp_iq[((il * nk + kmq) * nbnds +
                                                     vp) *
                                                        nbnds +
                                                    v];
                                        m2_tmp2_ae += dS * gvv2;
                                    }
                                    sum_ae +=
                                        m2_tmp2_ae * m2_tmp1_ae * D_kc_kmqv_il;

                                    // M3
                                    complex_t m3_tmp2_ae = 0;
                                    for (int_type cp = 0; cp < nc; cp++)
                                    {
                                        complex_t m3_tmp1_ae_cp = 0;
                                        for (int_type vp = 0; vp < nv; vp++)
                                        {
                                            complex_t G1 =
                                                ram_fac_ae *
                                                elec_dip[((pol1 * nk + kpq) *
                                                              nc +
                                                          cp) *
                                                             nv +
                                                         vp] /
                                                (ph_sum_ae +
                                                 (ome_light_Ha -
                                                  (Qp_ene[kpq * nbnds + nv +
                                                          cp] -
                                                   Qp_ene[kpq * nbnds + vp]) +
                                                  I * broading_Ha));
                                            complex_t gvv1 = conj(
                                                gkkp_iq[((jl * nk + k) * nbnds +
                                                         vp) *
                                                            nbnds +
                                                        v]);
                                            m3_tmp1_ae_cp += G1 * gvv1;
                                        }
                                        complex_t D_kqc_kv_il_cp =
                                            1.0 /
                                            (ome_light_Ha -
                                             (Qp_ene[kpq * nbnds + nv + cp] -
                                              Qp_ene[k * nbnds + v]) +
                                             I * broading_Ha +
                                             ph_freq[iq * nmode + il]);
                                        m3_tmp1_ae_cp *= D_kqc_kv_il_cp;
                                        complex_t gcc =
                                            gkkp_iq[((il * nk + k) * nbnds +
                                                     nv + cp) *
                                                        nbnds +
                                                    nv + c];
                                        m3_tmp2_ae += gcc * m3_tmp1_ae_cp;
                                    }
                                    sum_ae -= dipS_res * m3_tmp2_ae;

                                    // M4
                                    complex_t m4_tmp2_ae = 0;
                                    for (int_type vp = 0; vp < nv; vp++)
                                    {
                                        complex_t m4_tmp1_ae_vp = 0;
                                        for (int_type cp = 0; cp < nc; cp++)
                                        {
                                            complex_t G1 =
                                                ram_fac_ae *
                                                elec_dip[((pol1 * nk + kmq) *
                                                              nc +
                                                          cp) *
                                                             nv +
                                                         vp] /
                                                (ph_sum_ae +
                                                 (ome_light_Ha -
                                                  (Qp_ene[kmq * nbnds + nv +
                                                          cp] -
                                                   Qp_ene[kmq * nbnds + vp]) +
                                                  I * broading_Ha));
                                            complex_t gcc1 =
                                                conj(gkkp_iq[((jl * nk + kmq) *
                                                                  nbnds +
                                                              nv + c) *
                                                                 nbnds +
                                                             nv + cp]);
                                            m4_tmp1_ae_vp += gcc1 * G1;
                                        }
                                        complex_t D_kc_kmqv_il_vp =
                                            1.0 / (ome_light_Ha -
                                                   (Qp_ene[k * nbnds + nv + c] -
                                                    Qp_ene[kmq * nbnds + vp]) +
                                                   I * broading_Ha +
                                                   ph_freq[iq * nmode + il]);
                                        m4_tmp1_ae_vp *= D_kc_kmqv_il_vp;
                                        complex_t gvv =
                                            gkkp_iq[((il * nk + kmq) * nbnds +
                                                     v) *
                                                        nbnds +
                                                    vp];
                                        m4_tmp2_ae += m4_tmp1_ae_vp * gvv;
                                    }
                                    sum_ae -= dipS_res * m4_tmp2_ae;

                                    // Process 3: EA
                                    // M1
                                    complex_t m1_tmp1_ea = 0, m1_tmp2_ea = 0;
                                    for (int_type cp = 0; cp < nc; cp++)
                                    {
                                        complex_t G1 =
                                            ram_fac_ea *
                                            elec_dip[((pol1 * nk + k) * nc +
                                                      cp) *
                                                         nv +
                                                     v] /
                                            (ph_sum_ea +
                                             (ome_light_Ha -
                                              (Qp_ene[k * nbnds + nv + cp] -
                                               Qp_ene[k * nbnds + v]) +
                                              I * broading_Ha));
                                        complex_t gcc1 =
                                            gkkp_miq[((jl * nk + kpq) * nbnds +
                                                      nv + cp) *
                                                         nbnds +
                                                     nv + c];
                                        m1_tmp1_ea += gcc1 * G1;

                                        complex_t dS =
                                            conj(
                                                elec_dip[((pol2 * nk + k) * nc +
                                                          cp) *
                                                             nv +
                                                         v]) /
                                            (ome_light_Ha -
                                             (Qp_ene[k * nbnds + nv + cp] -
                                              Qp_ene[k * nbnds + v]) +
                                             I * broading_Ha);
                                        complex_t gcc2 = conj(
                                            gkkp_miq[((il * nk + kpq) * nbnds +
                                                      nv + cp) *
                                                         nbnds +
                                                     nv + c]);
                                        m1_tmp2_ea += gcc2 * dS;
                                    }
                                    sum_ea += m1_tmp2_ea * m1_tmp1_ea *
                                              D_kqc_kv_minus_il;

                                    // M2
                                    complex_t m2_tmp1_ea = 0, m2_tmp2_ea = 0;
                                    for (int_type vp = 0; vp < nv; vp++)
                                    {
                                        complex_t G1 =
                                            ram_fac_ea *
                                            elec_dip[((pol1 * nk + k) * nc +
                                                      c) *
                                                         nv +
                                                     vp] /
                                            (ph_sum_ea +
                                             (ome_light_Ha -
                                              (Qp_ene[k * nbnds + nv + c] -
                                               Qp_ene[k * nbnds + vp]) +
                                              I * broading_Ha));
                                        complex_t gvv1 =
                                            gkkp_miq[((jl * nk + k) * nbnds +
                                                      v) *
                                                         nbnds +
                                                     vp];
                                        m2_tmp1_ea += G1 * gvv1;

                                        complex_t dS =
                                            conj(
                                                elec_dip[((pol2 * nk + k) * nc +
                                                          c) *
                                                             nv +
                                                         vp]) /
                                            (ome_light_Ha -
                                             (Qp_ene[k * nbnds + nv + c] -
                                              Qp_ene[k * nbnds + vp]) +
                                             I * broading_Ha);
                                        complex_t gvv2 = conj(
                                            gkkp_miq[((il * nk + k) * nbnds +
                                                      v) *
                                                         nbnds +
                                                     vp]);
                                        m2_tmp2_ea += dS * gvv2;
                                    }
                                    sum_ea += m2_tmp2_ea * m2_tmp1_ea *
                                              D_kc_kmqv_minus_il;

                                    // M3
                                    complex_t m3_tmp2_ea = 0;
                                    for (int_type cp = 0; cp < nc; cp++)
                                    {
                                        complex_t m3_tmp1_ea_cp = 0;
                                        for (int_type vp = 0; vp < nv; vp++)
                                        {
                                            complex_t G1 =
                                                ram_fac_ea *
                                                elec_dip[((pol1 * nk + kpq) *
                                                              nc +
                                                          cp) *
                                                             nv +
                                                         vp] /
                                                (ph_sum_ea +
                                                 (ome_light_Ha -
                                                  (Qp_ene[kpq * nbnds + nv +
                                                          cp] -
                                                   Qp_ene[kpq * nbnds + vp]) +
                                                  I * broading_Ha));
                                            complex_t gvv1 = gkkp_miq
                                                [((jl * nk + kpq) * nbnds + v) *
                                                     nbnds +
                                                 vp];
                                            m3_tmp1_ea_cp += G1 * gvv1;
                                        }
                                        complex_t D_kqc_kv_minus_il_cp =
                                            1.0 /
                                            (ome_light_Ha -
                                             (Qp_ene[kpq * nbnds + nv + cp] -
                                              Qp_ene[k * nbnds + v]) +
                                             I * broading_Ha -
                                             ph_freq[minus_iq_idx * nmode +
                                                     il]);
                                        m3_tmp1_ea_cp *= D_kqc_kv_minus_il_cp;
                                        complex_t gcc = conj(
                                            gkkp_miq[((il * nk + kpq) * nbnds +
                                                      nv + c) *
                                                         nbnds +
                                                     nv + cp]);
                                        m3_tmp2_ea += gcc * m3_tmp1_ea_cp;
                                    }
                                    sum_ea -= dipS_res * m3_tmp2_ea;

                                    // M4
                                    complex_t m4_tmp2_ea = 0;
                                    for (int_type vp = 0; vp < nv; vp++)
                                    {
                                        complex_t m4_tmp1_ea_vp = 0;
                                        for (int_type cp = 0; cp < nc; cp++)
                                        {
                                            complex_t G1 =
                                                ram_fac_ea *
                                                elec_dip[((pol1 * nk + kmq) *
                                                              nc +
                                                          cp) *
                                                             nv +
                                                         vp] /
                                                (ph_sum_ea +
                                                 (ome_light_Ha -
                                                  (Qp_ene[kmq * nbnds + nv +
                                                          cp] -
                                                   Qp_ene[kmq * nbnds + vp]) +
                                                  I * broading_Ha));
                                            complex_t gcc1 =
                                                gkkp_miq[((jl * nk + k) *
                                                              nbnds +
                                                          nv + cp) *
                                                             nbnds +
                                                         nv + c];
                                            m4_tmp1_ea_vp += gcc1 * G1;
                                        }
                                        complex_t D_kc_kmqv_minus_il_vp =
                                            1.0 /
                                            (ome_light_Ha -
                                             (Qp_ene[k * nbnds + nv + c] -
                                              Qp_ene[kmq * nbnds + vp]) +
                                             I * broading_Ha -
                                             ph_freq[minus_iq_idx * nmode +
                                                     il]);
                                        m4_tmp1_ea_vp *= D_kc_kmqv_minus_il_vp;
                                        complex_t gvv = conj(
                                            gkkp_miq[((il * nk + k) * nbnds +
                                                      vp) *
                                                         nbnds +
                                                     v]);
                                        m4_tmp2_ea += m4_tmp1_ea_vp * gvv;
                                    }
                                    sum_ea -= dipS_res * m4_tmp2_ea;
                                }
                            }
                        }

                        double norm_factor =
                            1.0 / nk / sqrt(CellVol) / sqrt(nq);

                        int_type out_idx_aa =
                            ((((0 * nq + iq) * nmode + il) * nmode + jl) *
                                 npol +
                             pol2) *
                                npol +
                            pol1;
                        tmp_ten[out_idx_aa] = sum_aa * norm_factor;

                        int_type out_idx_ee =
                            ((((1 * nq + iq) * nmode + il) * nmode + jl) *
                                 npol +
                             pol2) *
                                npol +
                            pol1;
                        tmp_ten[out_idx_ee] = sum_ee * norm_factor;

                        int_type out_idx_ae =
                            ((((2 * nq + iq) * nmode + il) * nmode + jl) *
                                 npol +
                             pol2) *
                                npol +
                            pol1;
                        tmp_ten[out_idx_ae] = sum_ae * norm_factor;

                        int_type out_idx_ea =
                            ((((3 * nq + iq) * nmode + il) * nmode + jl) *
                                 npol +
                             pol2) *
                                npol +
                            pol1;
                        tmp_ten[out_idx_ea] = sum_ea * norm_factor;
                    }
                }
            }
        }
        free(idx_kplusq);
        free(idx_kminusq);
    }
}
