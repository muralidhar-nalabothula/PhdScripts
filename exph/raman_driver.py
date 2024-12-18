import numpy as np
from netCDF4 import Dataset
from io_exph import *
from excitons import *
from tqdm import tqdm
from raman import *
from time import time
from exe_dips import exe_dipoles
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

SAVE_dir = '../bse/SAVE'
BSE_dir = '../bse/bse2Ry'
elph_file = '../elph/ndb.elph'
Dmat_file = '../elph/ndb.Dmats'
nstates = 12000
Raman = True  ## compute luminescence if set to true
Exph = True
ome_range = [1.2, 2.2, 10000]  ## (min max, numpoints)
broading = 0.01  # in eV
npol = 2
modes = [
    58, 71, 72
]  #[121,122,123,124] # in empty all modes are computed, indexing start with 1

# SAVE_dir = '../gw_bse/SAVE'
# BSE_dir  = '../gw_bse/GW_BSE'
# elph_file = '../ndb.elph'
# Dmat_file = '../ndb.Dmats'
# nstates = 1000
# Raman = True ## compute luminescence if set to true
# Exph = True
# ome_range = [1,5,1000] ## (min max, numpoints)
# broading = 0.02 # in eV
# npol = 3
# modes = [] # in empty all modes are computed, indexing start with 1

# ## test case
# calc_folder = '../nscf'
# SAVE_dir = calc_folder + '/si.save/SAVE'
# BSE_dir  = calc_folder + '/si.save/bse'
# elph_file =calc_folder + '/ndb.elph'
# Dmat_file =calc_folder + '/ndb.Dmats'
# nstates = 100
# Raman = True ## compute luminescence if set to true
# Exph = True  ## compute excion phonon at Gamman
# ome_range = [1,5,1000] ## (min max, numpoints)
# broading = 0.02 # in eV
# npol = 3

## read the lattice data
print('*' * 30, ' Program started ', '*' * 30)
print('Reading Lattice data')
lat_vecs, nibz, symm_mats, ele_time_rev = get_SAVE_Data(save_folder=SAVE_dir)
blat_vecs = np.linalg.inv(lat_vecs.T)

## read exciton eigen vectors
bs_bands, BS_eigs, BS_wfcs = read_bse_eig(BSE_dir, neig=nstates)

print('Reading Phonon Data')
elph_file = Dataset(elph_file, 'r')
ph_sym, ph_time_rev, kpts, kmap, qpts, qmap, ph_freq, stard_conv, \
    elph_bnds_range, Dmats = get_ph_data(bs_bands, elph_file, Dmat_file)

print('Total phonon modes:', ph_freq.shape[1])
if (len(modes) != 0):
    print("Requested phonon modes :", modes)
    modes = np.array(modes) - 1
else:
    print("All modes requested requested")
    modes = np.arange(ph_freq.shape[1])

### Read dipoles
nvalance_bnds = BS_wfcs.shape[-1]
dipole_file = BSE_dir.strip() + '/ndb.dipoles'
ele_dips = get_dipoles(bs_bands,
                       nvalance_bnds,
                       dip_file=dipole_file,
                       var='DIP_v')

### build a kdtree for kpoints
print('Building kD-tree for kpoints')
kpt_tree = build_ktree(kpts)

### compute ex-ph matrix elements:
if Exph:
    print('Computing Exciton-phonon matrix elements for phonon absorption ...')
    ## read elph_matrix elements
    eph_mat_iq = get_elph_data_iq(0, elph_bnds_range, stard_conv, \
            ph_freq[0], kpt_tree, kpts, qpts, elph_file)
    ## compute ex-ph matrix
    ex_ph = ex_ph_mat(BS_wfcs, BS_wfcs, eph_mat_iq[modes], np.array([0,0,0]), \
                      np.array([0,0,0]), kpts, kpt_tree)
    ## SAving exciton phonon matrix elements
    print('Saving exciton phonon matrix elements')
    ex_ph = np.array(ex_ph).astype(numpy_Cmplx)
    # (modes, initial state (i), final-states (f)) i.e <f|dv_Q|i> for phonon absoption
    np.save('Ex-ph_gamma', ex_ph)
else:
    print('Loading exciton phonon matrix elements')
    ex_ph = np.load('Ex-ph_gamma.npy')

print('Computing Exciton-photon matrix elements')
ex_dip = exe_dipoles(ele_dips, BS_wfcs, kmap, symm_mats, ele_time_rev)
np.save('Ex-dipole', ex_dip)

# ## Test  <0|S'><S'|S>* <0|S>^*
# print (np.sum(np.abs(np.einsum('ia,vab,jb->vij',ex_dip,ex_ph.conj(),ex_dip.conj()))**2,axis=(1,2)))

CellVol = np.linalg.det(lat_vecs)
## close el-ph file
elph_file.close()
## compute Raman
if Raman:
    ome_range = np.linspace(ome_range[0], ome_range[1], num=ome_range[2])
    BS_eigs
    Raman_inten = []
    print('Computing Raman tensor ...')
    raman_ten = []
    for i in tqdm(range(ome_range.shape[0]), desc="Raman tensor"):
        raman_ten_tmp = compute_Raman(ome_range[i], ph_freq[0][modes], BS_eigs, \
            ex_dip, ex_ph, len(kpts), CellVol, broading=broading, npol=npol, ph_fre_th=8)
        raman_ten.append(raman_ten_tmp)
    ## save intensties

    raman_ten = np.array(raman_ten)[:, :, :npol, :npol]

    raman_inte = np.sum(np.abs(raman_ten)**2, axis=(2, 3))
    np.savetxt('Raman_intensities.dat',
               np.concatenate((ome_range[:, None], raman_inte), axis=1))
    np.save('Raman_tensor', np.array(Raman_inten))
