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
omega_one_ph_range = [0.001,4,10000]  ## (min max, numpoints) in eV
omega_two_ph_freq = [1.0] ## list of frequecies in eV (not range liste omega_one_ph_range) 
broading = 0.01  # in eV
npol = 3
bands = [1, 8]  # fortran indexing
one_ph = True
two_ph = True


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
nq_read = nq
if not two_ph and one_ph: nq_read = 1
#
elph_mat = np.zeros((nq_read, nmodes, len(kpts), nbnds, nbnds), dtype=ele_dips.dtype)
kpt_tree = build_ktree(kpts)
for iq in range(nq_read):
    elph_mat[iq] = get_elph_data_iq(iq, elph_bnds_range, stard_conv, \
                              ph_freq[iq], kpt_tree, kpts, qpts, elph_file)

CellVol = np.fabs(np.linalg.det(lat_vecs))
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


Raman_ten_data_dict = {}

if one_ph:
    print("Computing one-phonon stokes Raman")
    omega_one_ph_freq = np.linspace(omega_one_ph_range[0],omega_one_ph_range[1],num=omega_one_ph_range[2])
    ram_ten_one_ph = []
    for iw in tqdm(range(len(omega_one_ph_freq)), desc="1ph IP Raman tensor"):
        tmp_ten = compute_Raman_oneph_ip(omega_one_ph_freq[iw],
                               ph_freq[0],
                               Qp_ene,
                               ele_dips,
                               elph_mat[0],
                               CellVol,
                               broading,
                               npol,
                               ph_fre_th=5)
        ram_ten_one_ph.append(tmp_ten)
    #
    ram_ten_one_ph = np.array(ram_ten_one_ph)
    Raman_ten_data_dict["1ph_light_omega_eV"] = omega_one_ph_freq
    Raman_ten_data_dict["1ph_ph_freq_cm-1"] = ph_freq[0]*219474.63
    Raman_ten_data_dict["1ph_Raman_tensor"] = ram_ten_one_ph


if two_ph:
    ram_ten_two_ph = []
    print("Computing two-phonon Raman")
    freq_out = True
    out_2ph_freq = []
    for iw in tqdm(range(len(omega_two_ph_freq)), desc="2ph IP Raman tensor"):
        tmp_ten = compute_Raman_twoph_iq(omega_two_ph_freq[iw],
                                     ph_freq,
                                     Qp_ene,
                                     ele_dips,
                                     elph_mat,
                                     kpts,
                                     qpts,
                                     CellVol,
                                     broading,
                                     npol=npol,
                                     ktree=kpt_tree,
                                     out_freq=freq_out)
        if freq_out:
            ram_ten_two_ph.append(tmp_ten[1])
            out_2ph_freq = tmp_ten[0]
            freq_out = False
        else:
            ram_ten_two_ph.append(tmp_ten)

    #
    ram_ten_two_ph = np.array(ram_ten_two_ph)
    Raman_ten_data_dict["2ph_light_omega_eV"] = np.array(omega_two_ph_freq) 
    Raman_ten_data_dict["2ph_ph_freq_cm-1"] = out_2ph_freq*219474.63
    Raman_ten_data_dict["2ph_Raman_tensor"] = ram_ten_two_ph

np.savez("Raman_tensors.npz", **Raman_ten_data_dict)
