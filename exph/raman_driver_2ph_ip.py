import numpy as np
from netCDF4 import Dataset
from io_exph import *
from excitons import *
from tqdm import tqdm
from raman import *
from time import time
from exe_dips import exe_dipoles, dipole_expand
from exph_precision import *

# # ### Basic input
# SAVE_dir = '../bse/SAVE'
# BSE_dir  = '../bse/bse2Ry'
# elph_file = '../elph/ndb.elph'
# Dmat_file = '../elph/ndb.Dmats'
# nstates = 12000
# Raman = True ## compute luminescence if set to true
# Exph = True
# ome_range = [1.5,2.2,10000] ## (min max, numpoints)
# broading = 0.01 # in eV
# npol = 3
# modes = [121,122,123,124] # in empty all modes are computed, indexing start with 1

folder = '/Users/murali/phd/one_phonon_raman/si/2ph/ww'
SAVE_dir = folder + '/silicon.save/SAVE'
elph_file = folder + '/elph/ndb.elph'
Dmat_file = folder + '/elph/ndb.Dmats'
dipole_file = folder + '/silicon.save/dipoles/ndb.dipoles'
omega = [1.0]  ## (min max, numpoints)
broading = 0.01  # in eV
npol = 3
bands = [1, 8]  # fortran indexing

## read the lattice data
print('*' * 30, ' Program started ', '*' * 30)
print('Reading Lattice data')
lat_vecs, nibz, symm_mats, ele_time_rev, val_bnd_idx = get_SAVE_Data(
    save_folder=SAVE_dir)
blat_vecs = np.linalg.inv(lat_vecs.T)

nvalance_bnds = val_bnd_idx - min(bands) + 1
assert (nvalance_bnds > 0), "No valance bnds included"
#
print('Reading Phonon Data')
elph_file = Dataset(elph_file, 'r')
ph_sym, ph_time_rev, kpts, kmap, qpts, qmap, ph_freq, stard_conv, \
    elph_bnds_range, Dmats = get_ph_data(bands, elph_file, Dmat_file)

### Read dipoles
ele_dips = get_dipoles(bands, nvalance_bnds, dip_file=dipole_file, var='DIP_v')

nq, nmodes = ph_freq.shape
#nmodes,nk, final_band_PH_abs, initial_band
nbnds = max(bands) - min(bands) + 1
elph_mat = np.zeros((nq, nmodes, len(kpts), nbnds, nbnds), dtype=ele_dips.dtype)
kpt_tree = build_ktree(kpts)
for iq in range(len(qpts)):
    elph_mat[iq] = get_elph_data_iq(iq, elph_bnds_range, stard_conv, \
                              ph_freq[iq], kpt_tree, kpts, qpts, elph_file)

CellVol = np.linalg.det(lat_vecs)
## close el-ph file
elph_file.close()
## compute Raman

## read qp energies
qp_db = Dataset(SAVE_dir + '/ns.db1', 'r')
Qp_ene = qp_db['EIGENVALUES'][0, :, min(bands) - 1:max(bands)].data
Qp_ene = Qp_ene[kmap[:, 0], :].copy()
qp_db.close()

ele_dips = dipole_expand(ele_dips, kmap, symm_mats,
                         ele_time_rev).transpose(3, 0, 1, 2).conj()

ram_ten = []
omega_ran = np.linspace(1, 4, num=1000)
for iw in omega:
    tmp_ten = compute_Raman_twoph_iq(iw,
                                     ph_freq,
                                     Qp_ene,
                                     ele_dips,
                                     elph_mat,
                                     kpts,
                                     qpts,
                                     CellVol,
                                     broading,
                                     npol=npol,
                                     ktree=kpt_tree)

    # tmp_ten = compute_Raman_oneph_ip(iw,
    #                        ph_freq[0],
    #                        Qp_ene,
    #                        ele_dips,
    #                        elph_mat[0],
    #                        CellVol,
    #                        broading,
    #                        npol,
    #                        ph_fre_th=5)
    ram_ten.append(tmp_ten)
