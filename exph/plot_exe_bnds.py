import numpy as np
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from exph_precision import *

### Input
bse_folder = '/Users/murali/phd/one_phonon_raman/BN/data/hbn/GW_BSE'
## Job name of bse calculation
iQs = [1, 53, 97, 137, 169, 197, 217, 233, 241]
### list of iQ indices
iQ_name = ['$\\Gamma$', '', '', '', '$\\Omega$', '', '', '', 'K']
### name of points, no name set to  empty string ''
nbnds = 6
### number of bands to plot
shift = 0.4
## in eV. Constant shift applied to all bands
bnd_color = 'black'
### colour used when ploting bands
Title = 'hBN'
### Title of the plot
### end of input

## matplotlib parameters
txt_font_size = 12
plt.style.use(['science', 'nature'])
plt.rcParams['font.size'] = txt_font_size
plt.rcParams['axes.linewidth'] = 1
fig = plt.figure()
fig.set_size_inches(3, 4)
plt.tick_params(axis='both', which='major', labelsize=txt_font_size)


def read_bse_eig(bse_folder='SAVE', iq=1, neig=1):
    ## read eigen_vectors and eigen_values
    ## Output: (neig, k,c,v)
    diago_file = Dataset(bse_folder.strip() + '/ndb.BS_diago_Q%d' % (iq), 'r')
    bs_bands = diago_file['Bands'][...].data
    eig_vals = diago_file['BS_Energies'][:neig, 0].data
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


bse_eigs = []
bse_wfcs = []

if (len(iQ_name) == 0):
    iQ_name = [''] * len(iQs)

for iq in iQs:
    _, eigs_tmp, wfcs_tmp = read_bse_eig(bse_folder, iq, nbnds)
    bse_eigs.append(eigs_tmp)
    bse_wfcs.append(wfcs_tmp)

bse_eigs = np.array(bse_eigs) * 27.2114079527 + shift
bse_wfcs = np.array(bse_wfcs)

ixq_range = np.arange(bse_eigs.shape[0])
for i in range(bse_eigs.shape[1]):
    plt.plot(ixq_range, bse_eigs[:, i], color=bnd_color, marker='o')

plt.minorticks_off()
plt.ylabel('Energy (eV)', fontsize=txt_font_size)
plt.xticks(ixq_range, iQ_name)
plt.xlim(min(ixq_range), max(ixq_range))
Title = Title.strip()
#plt.title('Exciton band structure of '+ Title)
plt.savefig("tmp/exe_bandstruct_%s.pdf" % (Title),
            bbox_inches='tight',
            pad_inches=0.03)
plt.show()
