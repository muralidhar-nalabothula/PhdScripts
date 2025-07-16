import numpy as np
from netCDF4 import Dataset
from io_exph import *
from excitons import *
from tqdm import tqdm
from luminescence import *
from time import time
from exe_dips import exe_dipoles
from exph_precision import *
from kpts import find_kpatch

# # ### Basic input
# SAVE_dir = '../gw_bse/SAVE'
# BSE_dir  = '../gw_bse/GW_BSE'
# elph_file = '../ndb.elph'
# Dmat_file = '../ndb.Dmats'
# nstates = 50
# lumin = True ## compute luminescence if set to true
# Exph = True
# Temp = 20 ## temperature used in luminescence (in Kelvin)
# ome_range = [5.1,5.8,10000] ## (min max, numpoints)
# broading = 0.01 # in eV
# npol = 3

# ## test case
calc_folder = '../../'
SAVE_dir = calc_folder + 'gw_bse/SAVE'
BSE_dir = calc_folder + 'gw_bse/GW_BSE'
elph_file = calc_folder + 'elph/ndb.elph'
Dmat_file = calc_folder + 'elph/ndb.Dmats'
nstates = 10
lumin = True  #False  ## compute luminescence if set to true
Exph = False  #True
Temp = 20  ## temperature used in luminescence (in Kelvin)
ome_range = [1.35, 1.5, 40000]  ## (min max, numpoints)
broading = 0.004  # in eV
kcentre = [[1.0 / 6, 1.0 / 6, 0], 2 * np.pi / 7 * 0.1
          ]  ## only take kpoints around this point. [[kx,ky,kz],k_dist]
#             [kx,ky,kz] kpoint in crystal coorinates and k_dist in cart units (1/bohr)
#             if empty, consider all points.
npol = 3
# ### end of input

## read the lattice data
print('*' * 30, ' Program started ', '*' * 30)
print('Reading Lattice data')
lat_vecs, nibz, symm_mats, ele_time_rev = get_SAVE_Data(save_folder=SAVE_dir)
blat_vecs = np.linalg.inv(lat_vecs.T)
bs_bands = []  ## bands that are involved in BSE
## read all bse_eigs
BS_eigs = []
BS_wfcs = []

## read exciton eigen vectors
for iq in tqdm(range(nibz), desc="Loading Ex-wfcs "):
    bs_bands, tmp_eigs, tmp_wfcs = read_bse_eig(BSE_dir, iq + 1, nstates)
    BS_eigs.append(tmp_eigs)
    BS_wfcs.append(tmp_wfcs)

bs_bands = np.array(bs_bands)
BS_eigs = np.array(BS_eigs)
BS_wfcs = np.array(BS_wfcs)

np.save('BS_energies', BS_eigs)

### get elph_data
print('Reading Phonon Data')
elph_file = Dataset(elph_file, 'r')
ph_sym, ph_time_rev, kpts, kmap, qpts, qmap, ph_freq, stard_conv, \
    elph_bnds_range, Dmats = get_ph_data(bs_bands, elph_file, Dmat_file)

### Read dipoles
nvalance_bnds = BS_wfcs.shape[-1]
dipole_file = BSE_dir.strip() + '/ndb.dipoles'
ele_dips = get_dipoles(bs_bands,
                       nvalance_bnds,
                       dip_file=dipole_file,
                       var='DIP_iR')

### build a kdtree for kpoints
print('Building kD-tree for kpoints')
kpt_tree = build_ktree(kpts)

## check if phonon and electron symmetries are SAme.abs
## this is to make sure that we can make use of Dmats
assert (ele_time_rev == ph_time_rev)
assert (np.abs(symm_mats - ph_sym).max() < 10**-5)

### find the indices of qpoints in kpts
qidx_in_kpts = find_kindx(qpts, kpt_tree)

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

assert (kpts_ibz.shape[0] == nibz)

time_exph = 0
time_elph_io = 0
time_ex_rot = 0
time_exph_io = 0
### compute ex-ph matrix elements:
ex_ph = []
if Exph:
    print('Computing Exciton-phonon matrix elements for phonon absorption ...')
    for i in tqdm(range(qidx_in_kpts.shape[0]), desc="Exciton-phonon "):
        # < S|dv|0>
        ## first do a basic check
        kq_diff = qpts[i] - kpts[qidx_in_kpts[i]]
        kq_diff = kq_diff - np.rint(kq_diff)
        assert (np.linalg.norm(kq_diff) < 10**-5)
        tik = time()
        ## read elph_matrix elements
        eph_mat_iq = get_elph_data_iq(i, elph_bnds_range, stard_conv, \
            ph_freq[i], kpt_tree, kpts, qpts, elph_file)
        time_elph_io = time_elph_io + time() - tik
        ## get rotated ex-wfc
        tik = time()
        ik_ibz, isym = kmap[qidx_in_kpts[
            i]]  ## get the ibZ kpt and symmetry matrix for this q/k point
        is_sym_time_rev = False
        if (isym >= ph_sym.shape[0] / (int(ele_time_rev) + 1)):
            is_sym_time_rev = True
        wfc_tmp = rotate_exc_wfc(BS_wfcs[ik_ibz], sym_red[isym], kpts, \
            kpt_tree, kpts_ibz[ik_ibz], Dmats[isym], is_sym_time_rev)
        ## compute ex-ph matrix
        time_ex_rot = time_ex_rot + time() - tik
        tik = time()
        ex_ph_tmp = ex_ph_mat(wfc_tmp, BS_wfcs[0], eph_mat_iq, kpts[0],
                              kpts[qidx_in_kpts[i]], kpts, kpt_tree)
        ex_ph.append(ex_ph_tmp)
        time_exph = time_exph + time() - tik
    ## SAving exciton phonon matrix elements
    print('Saving exciton phonon matrix elements')
    tik_exph = time()
    ex_ph = np.array(ex_ph).astype(numpy_Cmplx)
    # (iq, modes, initial state (i), final-states (f)) i.e <f|dv_Q|i> for phonon absoption
    np.save('Ex-ph', ex_ph)
    time_exph_io = time_exph_io + time() - tik_exph
else:
    print('Loading exciton phonon matrix elements')
    tik_exph = time()
    ex_ph = np.load('Ex-ph.npy')
    time_exph_io = time_exph_io + time() - tik_exph

print('Computing Exciton-photon matrix elements')
ex_dip = exe_dipoles(ele_dips, BS_wfcs[0], kmap, symm_mats, ele_time_rev)
np.save('Ex-dipole', ex_dip)

## close el-ph file
elph_file.close()
## compute Lumenscence
if lumin:
    if len(kcentre) != 0:
        kcen = np.array(kcentre[0])
        kdist = kcentre[1]
        rot_kcen = np.einsum('nij,j->ni', sym_red, kcen)
        uniqe_kpts_centre = [kcen]
        for iikk in rot_kcen:
            for jjkk in uniqe_kpts_centre:
                ijkk = iikk - jjkk
                ijkk = np.linalg.norm(ijkk - np.rint(ijkk))
                if ijkk < 1e-5:
                    uniqe_kpts_centre.append(iikk)
                    break
        qinclude_idx = []
        for iqkk in uniqe_kpts_centre:
            qinclude_idx = qinclude_idx + list(
                find_kpatch(qpts, iqkk, kdist, lat_vecs))
        qinclude_idx = np.unique(qinclude_idx)
        print('Number of qpoints considered: ', len(qinclude_idx))
    else:
        qinclude_idx = np.arange(len(qpts), dtype=int)
    #
    ome_range = np.linspace(ome_range[0], ome_range[1], num=ome_range[2])
    exe_ene = BS_eigs[kmap[qidx_in_kpts[qinclude_idx], 0], :]
    lumen_inten = []

    Exe_min = np.min(exe_ene)
    print(f'Minimum energy of the exciton is        : {Exe_min*27.2114:.4f} eV')
    print(
        f'Minimum energy of the Direct exciton is : {min(BS_eigs[0])*27.2114:.4f} eV'
    )

    print('Computing luminescence intensities ...')
    for i in tqdm(range(ome_range.shape[0]), desc="Luminescence "):
        inte_tmp = compute_luminescence(ome_range[i], ph_freq[qinclude_idx], exe_ene, \
            Exe_min, ex_dip, ex_ph[qinclude_idx], temp=Temp, broading=broading,npol=npol)
        lumen_inten.append(inte_tmp)
    ## save intensties
    np.savetxt('luminescence_intensities.dat', np.c_[ome_range,
                                                     np.array(lumen_inten)])
    np.save('Intensties_lumen', np.array(lumen_inten))

print('** Timings **')
print(f'ELPH IO   : {time_elph_io:.4f} s')
print(f'Ex-ph IO  : {time_exph_io:.4f} s')
print(f'Ex rot    : {time_ex_rot:.4f} s')
print(f'Ex-ph     : {time_exph:.4f} s')
print('*' * 30, ' Program ended ', '*' * 30)
