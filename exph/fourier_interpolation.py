## Fourier interpolation of from coarse to fine (for ex Ex-ph matrix elements)
### Given a exciton-phonon on (k1,k2,k3) grid for
### (q1, q2, q3) phonon grid where q1/q2/q3 divides
#### k1/k2/k3. then this function interpolates
#### to (k1,k2,k3) phonon grid !

import numpy as np
import torch as pt
from kpts import get_kgrid, build_ktree, find_kpt, generate_kgrid
from exph_precision import pytor_Cmplx


def fourier_interpolate(kpts, qpts_co, data_co):
    ## return fourier interpolation for qpoints same as kpts
    ## if data is exph m.e they must in cartisian basis
    ## kpoints and qpts in crystal coordinates
    ## input exph shape (qpts_co, kpts, ....)
    ## output shape = (kpts, kpts,....)

    ## find the k grid
    nkpts = len(kpts)
    nqpts_co = len(qpts_co)
    #
    # Get the k and q grid dimensions
    kgrid = get_kgrid(kpts)
    qgrid_co = get_kgrid(qpts_co)
    assert np.abs(kgrid % qgrid_co).max() < 1e-7
    #
    # return the same if grids are same
    # if np.allclose(kgrid, qpts_co):
    #     return data_co
    #
    kgrid_fft = generate_kgrid(kgrid)
    qgrid_fft_co = generate_kgrid(qgrid_co)
    #
    ktree = build_ktree(kpts)
    qtree_co = build_ktree(qpts_co)
    #
    kfft_idx = find_kpt(ktree, kgrid_fft)
    qfft_idx_co = find_kpt(qtree_co, qgrid_fft_co)
    ##
    # Get inverse of kfft_idx
    kfft_idx_inv = pt.zeros(len(kfft_idx), dtype=int)
    kfft_idx_inv[kfft_idx] = pt.arange(len(kfft_idx), dtype=int)
    ##

    fft_shape_in = tuple(qgrid_co) + tuple(kgrid) + data_co.shape[2:]
    fft_shape_out = tuple(kgrid) + tuple(kgrid) + data_co.shape[2:]
    ##
    fft_tmp = pt.from_numpy(data_co)[qfft_idx_co,
                                     ...][:, kfft_idx,
                                          ...].reshape(fft_shape_in)
    #
    # Perform fft
    fft_in = pt.fft.fftn(fft_tmp, dim=(0, 1, 2, 3, 4, 5), norm="forward")
    ## Zero Pad
    fft_out = pt.zeros(fft_shape_out, dtype=pytor_Cmplx)
    #
    fft_idx1 = fft_map_idxs(qgrid_co[0], kgrid[0])
    fft_idx2 = fft_map_idxs(qgrid_co[1], kgrid[1])
    fft_idx3 = fft_map_idxs(qgrid_co[2], kgrid[2])
    #
    fft_idx1, fft_idx2, fft_idx3 = pt.meshgrid(fft_idx1,
                                               fft_idx2,
                                               fft_idx3,
                                               indexing='ij')
    #
    fft_idx = fft_idx1 * kgrid[1] * kgrid[2] + fft_idx2 * kgrid[2] + fft_idx3
    fft_idx = fft_idx.reshape(-1)
    #
    tmp_shape = (-1,) + fft_out.shape[3:]
    fft_out.reshape(tmp_shape)[fft_idx, ...] = fft_in.reshape(tmp_shape)
    #
    ## Perform inverse fft
    fft_out_tmp = pt.fft.ifftn(fft_out, dim=(0, 1, 2, 3, 4, 5),
                               norm="forward").reshape((nkpts, nkpts) +
                                                       data_co.shape[2:])
    fft_out = fft_out.reshape(fft_out_tmp.shape)
    fft_out[kfft_idx, :, ...] = fft_out_tmp[:, kfft_idx_inv, ...]
    return fft_out.cpu().detach().numpy()


def fft_map_idxs(g_cor, g_fine=0):
    assert g_cor > 0
    if g_cor > g_fine:
        g_fine = g_cor
    fft_idx1 = np.fft.fftfreq(g_cor, d=1.0 / g_cor)
    fft_idx1[fft_idx1 < 0] += g_fine
    return pt.from_numpy(np.rint(fft_idx1).astype(int))


## test
if __name__ == '__main__':
    from netCDF4 import Dataset

    elph_db = Dataset(
        '/Users/murali/phd/one_phonon_raman' +
        '/si/elph_test/interpolation/elph/ndb.elph', 'r')

    kpts = elph_db['kpoints'][...].data
    qpts = elph_db['qpoints'][...].data
    elph = elph_db['elph_mat'][...].data
    elph = elph[..., 0] + 1j * elph[..., 1]
    pol_v = elph_db['POLARIZATION_VECTORS'][...].data
    pol_v = pol_v[..., 0] + 1j * pol_v[..., 1]
    nq, nmodes = pol_v.shape[:2]
    pol_v = np.linalg.inv(pol_v.reshape(-1, nmodes, nmodes))
    elph = np.einsum('qkv...,qxv->qkx...', elph, pol_v, optimize=True)

    qgrid_co = np.array([4, 4, 4])
    #qgrid_co = np.array([8, 8, 8])
    qpts_co = generate_kgrid(qgrid_co)
    qpts_co = np.random.permutation(qpts_co - np.rint(qpts_co))
    qtree = build_ktree(qpts)
    idx_q = find_kpt(qtree, qpts_co)
    kidxs = find_kpt(qtree, kpts)
    elph_test_in = elph[idx_q, ...]

    #print(kpts.shape)
    ## close files
    elph_db.close()
    elph_test_out = np.zeros(elph.shape, dtype=np.complex64)
    elph_test_out[kidxs, ...] = fourier_interpolate(kpts, qpts_co, elph_test_in)

    #print(np.abs(elph_test_out-elph).max())
    print(elph_test_out[1, 145, 4, 0, 2:4, 6:7])
    print(elph[1, 145, 4, 0, 2:4, 6:7])
    # print(np.sum(elph_test_out))
    # print(np.sum(elph))
    # print(elph_test_out.shape)
    #print(exp1.shape)
