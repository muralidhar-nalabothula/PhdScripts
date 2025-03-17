## COmpute the exciton spin projection
import numpy as np
from yambopy.dbs import wfdb
from yambopy.dbs.latticedb import YamboLatticeDB
from netCDF4 import Dataset

wf_db = wfdb.YamboWFDB(bands_range=[5, 10])
# Expand wfcs in full BZ
wf_db.expand_fullBZ()

## Compute expectation values of Sz operator
assert wf_db.nspin == 1 and wf_db.nspinor == 2, \
    "Exciton spin is meaning to compute only for nspinor =2 case."

# wfc_bz = wf_db.wf_bz[:,0,...] # 'kbsg
# nk, nb, nspinor, ng = wfc_bz.shape
# Sz_elec =  np.zeros(wfc_bz.shape,dtype=wfc_bz.dtype)
# Sz_elec[:,:,0,:] =  wfc_bz[:,:,0,:]
# Sz_elec[:,:,1,:] = -wfc_bz[:,:,1,:]
# Sz_elec = wfc_bz.reshape(nk,nb,-1).conj() @ Sz_elec.reshape(nk,nb,-1).transpose(0,2,1)
# Sz_elec ( nk, nb, nb)

#
Sz_elec = wf_db.get_spin_m_e_BZ()
#
exe_wf = 1  #(S, k, c, v)
## 1) electron spin
exe_espin = exe_wf.transpose(0, 1, 3, 2).conj() @ Sz_elec[None, :, nval:, nval:]
exe_espin = exe_espin.transpose(0, 1, 3, 2)
exe_espin = np.sum(exe_espin * exe_wf, axis=(1, 2, 3))
## 2) hole spin
exe_hspin = exe_wf @ Sz_elec[None, :, :nval, :nval]
exe_hspin = np.sum(exe_wf.conj() * exe_hspin, axis=(1, 2, 3))
### Total spin
exe_spin = 0.5 * (exe_espin - exe_hspin)
