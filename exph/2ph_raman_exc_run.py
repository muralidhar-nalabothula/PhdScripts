import numpy as np
import matplotlib.pyplot as plt
from yambopy.exciton_phonon.excph_luminescence import exc_ph_luminescence
from yambopy.exciton_phonon.excph_input_data import exc_ph_get_inputs
from raman import compute_two_ph_raman_exc
from netCDF4 import Dataset
from yambopy.kpoints import build_ktree, find_kpt
import os
from yambopy.tools.funcs import bose

path = '.'
# Path to BSE calculation (Lout--> response is Lfull)
bsepath = f'{path}/GW_BSE'  # ndb.BS_diago_Q* databases are needed
# Path to BSE calculation for optically active exciton (Lin --> response is Lbar)
elphpath = path  # ndb.elph is needed
# Path to unprojected dipoles matrix elements (optional)
dipolespath = bsepath  # ndb.dipoles is needed (optional)
# Path to lattice and k-space info
savepath = f'{path}/SAVE'  # ns.db1 database is needed

nexc = 12  # 12 excitonic states at Q=0 (Lin)
T_ph = 10  # Lattice temperature

emin = 6  # Energy range and plot details (in eV)
emax = 7
nsteps = 1000
broad = 0.005  # Broadening parameter for peak width (in eV)

# We calculate and load all the inputs:
# * Exciton-phonon matrix elements
# * Excitonic dipole matrix elements
# * Exciton energies
# * Phonon energies
# * We specify bse_path2=bseBARpath meaning we use Lbar calculation for Q=0 excitons
input_data = exc_ph_get_inputs(savepath,elphpath,bsepath,\
                               bse_path2=bsepath,dipoles_path=dipolespath,\
                               nexc_in=nexc,nexc_out=nexc)

ph_energies, exc_energies, exc_energies_in, exph_mat, exc_dipoles = input_data

ph_energies /= 27.21111
exc_energies /= 27.21111
exc_energies_in /= 27.21111
broad /= 27.21111
ome_light = np.linspace(emin, emax, nsteps) / 27.21111

elph_db = Dataset(os.path.join(elphpath, 'ndb.elph'), 'r')
qpoints = elph_db['qpoints'][...].data
elph_db.close()

qtree = build_ktree(qpoints)
idx_mq = find_kpt(qtree, -qpoints)

# we are assuming time reversal
# < 0 | dv-q| q> = (<q| dv^dagger_-q| 0>)^*
g_q_mq = exph_mat.transpose(0, 1, 3, 2).conj()
# Note here we are messing up with phonon phase, but when computing Raman intensities by \sum_modes|R|^2
exc_dipoles = exc_dipoles.conj()  # we need for photon absorption
freq_ram, two_ph_raman = compute_two_ph_raman_exc(ome_light,
                                                  qpoints,
                                                  ph_energies,
                                                  exc_energies_in,
                                                  exc_energies,
                                                  exc_dipoles,
                                                  exph_mat,
                                                  g_q_mq,
                                                  gamma=broad,
                                                  precision='s')

#
#Intensity_tensor = np.abs(M_tensor)**2

n_q = np.abs(bose(ph_energies, T_ph))
n_mq = np.abs(bose(ph_energies, T_ph))
#
bose_0 = np.sqrt(n_mq[:, :, None] * n_q[:, None, :])
two_ph_raman[:, :, 0, ...] *= bose_0[None, :, :, :, None, None]
#
bose_1 = np.sqrt((n_q[:, :, None] + 1.0) * n_q[:, None, :])
two_ph_raman[:, :, 1, ...] *= bose_1[None, :, :, :, None, None]
#
bose_2 = np.sqrt((n_q[:, :, None] + 1.0) * (n_mq[:, None, :] + 1.0))
two_ph_raman[:, :, 2, ...] *= bose_2[None, :, :, :, None, None]
#
# print(tensor)
# np.savetxt('data.txt',np.c_[ome_light*27.21111,tensor])
# (2, 18, 3, 12, 12, 3, 3)
#print(np.abs(two_ph_raman).max())
np.savez_compressed('raman_results.npz',
                    two_ph_raman=two_ph_raman,
                    freq_ram=freq_ram)

# loaded_data = np.load('raman_results.npz')
# two_ph_raman = loaded_data['two_ph_raman']
# freq_ram = loaded_data['freq_ram']
