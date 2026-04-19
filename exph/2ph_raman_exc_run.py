import numpy as np
import matplotlib.pyplot as plt
from yambopy.exciton_phonon.excph_luminescence import exc_ph_luminescence
from yambopy.exciton_phonon.excph_input_data import exc_ph_get_inputs
from .raman import compute_two_ph_raman_exc
from netCDF4 import Dataset
from yambopy.kpoints import build_ktree, find_kpt
import os

path = '3D_hBN'
# Path to BSE calculation (Lout--> response is Lfull)
bsepath =  f'{path}/bse_Lfull' # ndb.BS_diago_Q* databases are needed
# Path to BSE calculation for optically active exciton (Lin --> response is Lbar)
elphpath = path # ndb.elph is needed
# Path to unprojected dipoles matrix elements (optional)
dipolespath = bsepath # ndb.dipoles is needed (optional)
# Path to lattice and k-space info
savepath = f'{path}/SAVE' # ns.db1 database is needed

bands_range=[6,10] # 2 valence, 2 conduction bands
phonons_range=[0,12] # All phonons
nexc_out = 12 # 12 excitonic states at each momentum (Lout)
nexc_in  = 12 # 12 excitonic states at Q=0 (Lin)
T_ph  = 10 # Lattice temperature
T_exc = 10 # Effective excitonic temperature

emin=4.4      # Energy range and plot details (in eV)
emax=4.7
estep=0.0002
broad = 0.005 # Broadening parameter for peak width (in eV)

# We calculate and load all the inputs:
# * Exciton-phonon matrix elements
# * Excitonic dipole matrix elements
# * Exciton energies
# * Phonon energies
# * We specify bse_path2=bseBARpath meaning we use Lbar calculation for Q=0 excitons
input_data = exc_ph_get_inputs(savepath,elphpath,bsepath,\
                               bse_path2=bsepath,dipoles_path=dipolespath,\
                               nexc_in=nexc_in,nexc_out=nexc_out,\
                               bands_range=bands_range,phonons_range=phonons_range)

ph_energies, exc_energies, exc_energies_in, exph_mat, exc_dipoles = input_data

elph_db = Dataset(os.path.join(elphpath, 'ndb.elph'), 'r')
qpoints = elph_db['qpoints'][...].data
elph_db.close()

qtree = build_ktree(kpoints)
idx_mq = find_kpt(qtree,-qpoints)

# we are assuming time reversal
# < 0 | dv-q| q> = (<q| dv^dagger_-q| 0>)^*
g_q_mq = exph_mat.transpose(0,1,3,2).conj()
# Note here we are messing up with phonon phase, but when computing Raman intensities by \sum_modes|R|^2
exc_dipoles = exc_dipoles.conj() # we need for photon absorption
two_ph_raman = compute_two_ph_raman_exc(ome_light,
                             ph_energies,
                             ph_energies[idx_mq],
                             exc_energies_in,
                             exc_energies,
                             exc_energies[idx_mq],
                             ex_dip,
                             g_0_q,
                             g_0_q[idx_mq],
                             g_q_mq,
                             g_q_mq[idx_mq],
                             gamma=broad,
                             precision='s')


