import numpy as np
from netCDF4 import Dataset
from io_exph import *
from excitons import *
from tqdm import tqdm
from luminescence import *
from point_group_ops import get_pg_info, decompose_rep2irrep
import re

np.set_printoptions(suppress=True)

### Basic input
SAVE_dir = '../gw_bse/SAVE'
BSE_dir = '../gw_bse/GW_BSE'
elph_file = '../../gamma_ph_full/ndb.elph'
Dmat_file = '../../gamma_ph_full/ndb.Dmats'
iQ = 181  ## exciton number
nstates = 2
read_symm_from_ns_db_file = False
## if true, will read symmetry matrices from ns.db1 file else from ndb.elph (if present)
degen_thres = 0.001  # eV ## degenercy threshould
### end of input

## convert degen_thres to decimal digits
## read the lattice data
print('Reading lattice data')
lat_vecs, nibz, symm_mats, time_rev, _ = get_SAVE_Data(save_folder=SAVE_dir)
frac_trans = np.zeros(
    (symm_mats.shape[0], 3))  ## yambo does not support frac trans
blat_vecs = np.linalg.inv(lat_vecs.T)
bs_bands = []  ## bands that are involved in BSE

# print(symm_mats.reshape(symm_mats.shape[0],-1))
## read exciton eigen vectors
print('Reading BSE eigen vectors')
bands_range, BS_eigs, BS_wfcs = read_bse_eig(BSE_dir, iQ, nstates)

BS_eigs = BS_eigs * 27.2114079527
### get unique values upto threshould
uni_eigs, degen_eigs = np.unique((BS_eigs / degen_thres).astype(int),
                                 return_counts=True)
uni_eigs = uni_eigs * degen_thres
### get elph_data
print('Reading electron-phonon matrix elements')
elph_file = Dataset(elph_file, 'r')
kpts = elph_file['kpoints'][...].data
kmap = elph_file['kmap'][...].data
elph_cal_bands = elph_file['bands'][...].data
if not read_symm_from_ns_db_file:
    symm_mats = elph_file['symmetry_matrices'][...].data
    time_rev = elph_file['time_reversal_phonon'][...].data
    frac_trans = elph_file['fractional_translation'][
        ...].data  ## this is in cart units
    ## convert to crystal coordinates
    frac_trans = np.einsum('ij,nj->ni', blat_vecs.T, frac_trans)

min_bnd = min(bands_range)
max_bnd = max(bands_range)
nbnds = max_bnd - min_bnd + 1
assert (min_bnd >= min(elph_cal_bands))
assert (max_bnd <= max(elph_cal_bands))
start_bnd_idx = min_bnd - min(elph_cal_bands)
end_bnd = start_bnd_idx + nbnds
elph_file.close()
Dmat_data = Dataset(Dmat_file, 'r')
Dmats = Dmat_data['Dmats'][:, :, 0, start_bnd_idx:end_bnd,
                           start_bnd_idx:end_bnd, :].data
Dmats = Dmats[..., 0] + 1j * Dmats[..., 1]  # # nsym_ph, nkpts, Rk_band, k_band,
Dmat_data.close()

### build a kdtree for kpoints
print('Building kD-tree for kpoints')
kpt_tree = build_ktree(kpts)
sym_red = np.einsum('ij,njk,kl->nil',
                    lat_vecs.T,
                    symm_mats,
                    blat_vecs,
                    optimize=True)

kpts_ibz = np.zeros((len(np.unique(kmap[:, 0])), 3))
for i in range(kmap.shape[0]):
    ik_ibz, isym = kmap[i]
    if isym == 0:
        kpts_ibz[ik_ibz, :] = kpts[i]

## print some data about the degeneracies
print('=' * 40)
print('Group theory analysis for Q point : ', kpts_ibz[iQ - 1])
print('*' * 40)
# print('Unique eigen values : ', uni_eigs)
# print('# of degeneracies   : ', degen_eigs)

trace_all_real = []
trace_all_imag = []
little_group = []
for isym in range(int(sym_red.shape[0] / (time_rev + 1))):
    #isym = 2
    Sq_minus_q = np.einsum('ij,j->i', sym_red[isym],
                           kpts_ibz[iQ - 1]) - kpts_ibz[iQ - 1]
    #print(Sq_minus_q)
    #diff = Sq_minus_q.copy()
    Sq_minus_q = Sq_minus_q - np.rint(Sq_minus_q)
    ## check if Sq = q
    if np.linalg.norm(Sq_minus_q) > 10**-5:
        continue
    little_group.append(isym + 1)
    tau_dot_k = np.exp(1j * 2 * np.pi *
                       np.dot(kpts_ibz[iQ - 1], frac_trans[isym]))
    #assert(np.linalg.norm(Sq_minus_q)<10**-5)
    wfc_tmp = rotate_exc_wfc(BS_wfcs, sym_red[isym], kpts, \
    kpt_tree, kpts_ibz[iQ-1], Dmats[isym], False)
    #print(np.linalg.norm(wfc_tmp.reshape(wfc_tmp.shape[0],-1),axis=-1))
    rep = np.einsum('n...,m...->nm', wfc_tmp, BS_wfcs.conj(),
                    optimize=True) * tau_dot_k
    #print('Symmetry number : ',isym + 1)
    ## print characters
    irrep_sum = 0
    real_trace = []
    imag_trace = []
    for iirepp in range(len(uni_eigs)):
        idegen = degen_eigs[iirepp]
        idegen2 = irrep_sum + idegen
        trace_tmp = np.trace(rep[irrep_sum:idegen2, irrep_sum:idegen2])
        real_trace.append(trace_tmp.real.round(4))
        imag_trace.append(trace_tmp.imag.round(4))
        irrep_sum = idegen2
    # print('Real : ',real_trace)
    # print('Imag : ',imag_trace)
    trace_all_real.append(real_trace)
    trace_all_imag.append(imag_trace)

little_group = np.array(little_group, dtype=int)

pg_label, classes, class_dict, char_tab, irreps = get_pg_info(
    symm_mats[little_group - 1])

print('Little group : ', pg_label)
print('Little group symmetries : ', little_group)

# print class info
print('Classes (symmetry indices in each class): ')
req_sym_characters = np.zeros(len(classes), dtype=int)
class_orders = np.zeros(len(classes), dtype=int)
for ilab, iclass in class_dict.items():
    print("%16s    : " % (classes[ilab]), little_group[np.array(iclass)])
    req_sym_characters[ilab] = min(iclass)
    class_orders[ilab] = len(iclass)
print()
trace_all_real = np.array(trace_all_real)
trace_all_imag = np.array(trace_all_imag)
trace = trace_all_real + 1j * trace_all_imag
trace_req = trace[req_sym_characters, :].T
print("====== Exciton representations ======")
print("Energy (eV),  degenercy  : representation")
print('-' * 40)
for i in range(len(trace_req)):
    rep_str_tmp = decompose_rep2irrep(trace_req[i], char_tab, len(little_group),
                                      class_orders, irreps)
    print('%.4f        %9d  : ' % (uni_eigs[i], degen_eigs[i]), rep_str_tmp)

# print("Characters for excitonic states: ")
# print(trace_req.round(2))
print('*' * 40)
#print(sym_red[little_group-1])
