##  Has all the io related functions
import numpy as np
from netCDF4 import Dataset
from kpts import *
from exph_precision import *
import torch as pytorch


def get_ncvar_quick(file, var, cmplx=True):
    ### generic netcdf read
    aaaaaa = Dataset(file, 'r')[var][...].data
    if cmplx:
        return aaaaaa[..., 0] + 1.0 * 1j * aaaaaa[..., 1]
    else:
        return aaaaaa


def read_bse_eig(bse_folder='SAVE', iq=1, neig=1):
    ## read eigen_vectors and eigen_values
    ## Output: (neig, k,c,v)
    diago_file = Dataset(bse_folder.strip() + '/ndb.BS_diago_Q%d' % (iq), 'r')
    bs_bands = diago_file['Bands'][...].data
    eig_vals = diago_file['BS_Energies'][:neig, 0].data.astype(numpy_float)
    bs_table = np.rint(diago_file['BS_TABLE'][...].data.T).astype(
        int)  # (k,v,c)
    eig_wfcs = diago_file['BS_EIGENSTATES'][:neig, :, :].data
    eig_wfcs = eig_wfcs[..., 0] + 1j * eig_wfcs[..., 1]
    nk = np.unique(bs_table[:, 0]).shape[0]
    nv = np.unique(bs_table[:, 1]).shape[0]
    nc = np.unique(bs_table[:, 2]).shape[0]
    bs_table[:, 0] = bs_table[:, 0] - 1
    bs_table[:, 1] = bs_table[:, 1] - min(bs_bands)
    bs_table[:, 2] = bs_table[:, 2] - min(bs_bands) - nv
    eig_wfcs_returned = np.zeros(eig_wfcs.shape,
                                 dtype=numpy_Cmplx)  #(neig,nk,nc,nv)
    sort_idx = bs_table[:, 0] * nc * nv + bs_table[:, 2] * nv + bs_table[:, 1]
    eig_wfcs_returned[:, sort_idx] = eig_wfcs[...]
    diago_file.close()
    return bs_bands, eig_vals, eig_wfcs_returned.reshape(neig, nk, nc, nv)


def get_SAVE_Data(save_folder='SAVE'):
    ### read basic lattice data need for us
    ns_db1 = Dataset(save_folder.strip() + '/ns.db1', 'r')
    symm_mats = ns_db1['SYMMETRY'][...].data.transpose(0, 2,
                                                       1).astype(numpy_float)
    nk_ibz = ns_db1['K-POINTS'][...].data.T.astype(numpy_float)
    dims = ns_db1['DIMENSIONS'][...].data
    lat_vecs = ns_db1['LATTICE_VECTORS'][...].data.astype(numpy_float)
    ns_db1.close()
    return lat_vecs, nk_ibz.shape[0], symm_mats, int(np.rint(
        dims[9]))  ## dim[9] is time_rev is on/off


def get_dipoles(bands_range, nval_bnds, dip_file='ndb.dipoles', var='DIP_iR'):
    min_bnd = min(bands_range)
    max_bnd = max(bands_range)
    nbnds = max_bnd - min_bnd + 1
    ndb_dip = Dataset(dip_file, 'r')
    pars = ndb_dip['PARS'][...].data
    nvmin = int(pars[0])
    ncmmax = int(pars[1])
    nvmax = int(pars[2])
    ncmin = int(pars[3])
    dip_bands = [nvmin, ncmmax]
    assert (min_bnd >= nvmin)
    assert (max_bnd <= ncmmax)
    if nvmax == ncmmax and ncmin == nvmin:
        start_bnd_idx = min_bnd - nvmin
        end_bnd = start_bnd_idx + nbnds
        val_bnd_idx = start_bnd_idx + nval_bnds
        dipoles = ndb_dip[var][0, :, start_bnd_idx:val_bnd_idx,
                               val_bnd_idx:end_bnd, ...].data  # kvc
    else:
        assert (min_bnd <= nvmax)
        assert (max_bnd >= ncmin)
        v_start_bnd = min_bnd - nvmin
        c_end_bnd = max_bnd - ncmin + 1
        dipoles = ndb_dip[var][0, :, v_start_bnd:, :c_end_bnd, ...].data  # kvc
    ## we need photon absoprtion but yambo computes for emission, so conjugate
    dipoles = dipoles[..., 0] - 1j * dipoles[..., 1]
    ndb_dip.close()
    if var=='DIP_iR': dipoles = -1j*dipoles. ## we want r dipoles not ir 
    return dipoles.transpose(0, 2, 1, 3).astype(numpy_Cmplx)  # (k,c,v,pol)


def get_ph_data(bands_range, elph_file, dmat_file='ndb.Dmats'):
    ## first get convention
    conv = elph_file['convention'][...].data
    convlist = [iconv.decode('utf-8') for iconv in conv]
    conv = ''
    for iconv in convlist:
        conv = conv + iconv
    stard_conv = False
    if conv.strip() == 'standard':
        print("Convention used in Letzelphc : k -> k+q (standard)")
        stard_conv = True
    else:
        print("Convention used in Letzelphc : k-q -> k (yambo)")
    #
    ph_freq = np.abs(elph_file['FREQ'][...].data).astype(numpy_float)
    kpts = elph_file['kpoints'][...].data.astype(numpy_float)
    kmap = elph_file['kmap'][...].data
    qpts = elph_file['qpoints'][...].data.astype(numpy_float)
    qmap = elph_file['qmap'][...].data
    elph_cal_bands = elph_file['bands'][...].data
    ph_sym = elph_file['symmetry_matrices'][...].data.astype(numpy_float)
    time_rev = elph_file['time_reversal_phonon'][...].data
    min_bnd = min(bands_range)
    max_bnd = max(bands_range)
    nbnds = max_bnd - min_bnd + 1
    assert (min_bnd >= min(elph_cal_bands))
    assert (max_bnd <= max(elph_cal_bands))
    start_bnd_idx = min_bnd - min(elph_cal_bands)
    end_bnd = start_bnd_idx + nbnds
    #print(start_bnd_idx, end_bnd)
    Dmat_data = Dataset(dmat_file, 'r')
    Dmats = Dmat_data['Dmats'][:, :, 0, start_bnd_idx:end_bnd,
                               start_bnd_idx:end_bnd, :].data
    Dmats = Dmats[...,
                  0] + 1j * Dmats[..., 1]  # # nsym_ph, nkpts, Rk_band, k_band,
    Dmat_data.close()
    assert (nbnds == Dmats.shape[-1])  ## sanity check
    return ph_sym, time_rev, kpts, kmap, qpts, qmap, ph_freq * 0.5, stard_conv, [
        start_bnd_idx, end_bnd
    ], Dmats.astype(numpy_Cmplx)


def get_elph_data_iq(iq, elph_bands_range, stard_conv, ph_freq, ktree, kpts,
                     qpts, elph_file):
    start_bnd_idx = elph_bands_range[0]
    end_bnd = elph_bands_range[1]
    #print(start_bnd_idx, end_bnd)
    eph_mat = elph_file['elph_mat'][iq, :, :, 0, start_bnd_idx:end_bnd,
                                    start_bnd_idx:end_bnd, :].data
    eph_mat = eph_mat[..., 0] + eph_mat[
        ..., 1] * 1j  #nk, nmodes, initial_band, final_band_PH_abs 0,1,2,3
    eph_mat = eph_mat.transpose(1, 0, 3,
                                2)  #nmodes,nk, final_band_PH_abs, initial_band
    ## normalize with 1/sqrt(2*E_ph)
    sqrt_EPh = 1.0 / np.sqrt(2 * ph_freq)
    sqrt_EPh[np.isnan(sqrt_EPh)] = 0
    sqrt_EPh = (0.5)**1.5 * sqrt_EPh  ## 0.5 for Ry to Ha of ex-ph
    eph_mat = np.einsum('v...,v->v...', eph_mat, sqrt_EPh)
    ## if convtion is not standard, we need to rearrange:
    if not stard_conv:
        idx_q = find_kindx(qpts[iq][None, :] + kpts, ktree)
        for imode in range(eph_mat.shape[0]):
            elph_tmp_iq = pytorch.from_numpy(eph_mat[imode])[idx_q, ...].clone()
            eph_mat[imode][...] = elph_tmp_iq.numpy()
    return eph_mat.astype(numpy_Cmplx)
