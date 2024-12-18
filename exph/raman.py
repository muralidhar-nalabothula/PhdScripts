## Compute resonant Raman intensities
## Sven Reichardt, etal Sci. Adv.6,eabb5915(2020)
import numpy as np
import torch as pytorch
from exph_precision import *
import math

def compute_Raman(ome_light, ph_freq, ex_ene, ex_dip, ex_ph, nkpts, CellVol, broading=0.1, npol=3, ph_fre_th=5):
    ## We need exciton dipoles for light emission (<0|r|S>)
    ## and exciton phonon matrix elements for phonon absorption <S',0|dV|S,0>
    ## except ph_fre_th, ome_light and broading, all are in Hartree
    ## ph_fre_th is phonon freqency threshould. Raman tensor for phonons with freq below ph_fre_th 
    ## will be set to 0. The unit of ph_fre_th is cm-1
    ## 
    nmode,nbnd_i,nbnd_f = ex_ph.shape
    ex_dip_absorp =  pytorch.from_numpy(ex_dip).conj() ## exciton dipoles for light absoption
    # 
    broading_Ha = numpy_float(broading/27.211/2)
    ome_light_Ha = numpy_float(ome_light/27.211)
    #
    BS_energies = pytorch.from_numpy(ex_ene) - 1j*broading_Ha
    ph_fre_th = ph_fre_th*0.12398/1000/27.211

    dipS_res  = ex_dip_absorp       /((ome_light_Ha - BS_energies)[None,:])
    dipS_ares = ex_dip_absorp.conj()/((ome_light_Ha + BS_energies)[None,:])

    ex_ph_pytorch = pytorch.from_numpy(ex_ph)
    ph_freq_pytorch = pytorch.from_numpy(ph_freq)

    Ram_ten = np.zeros((nmode,3,3),dtype=numpy_Cmplx)

    ram_fac = 1.0/nkpts/math.sqrt(CellVol)
    # Now compute raman tensor for each mode
    for i in range(nmode):
        if (abs(ph_freq_pytorch[i]) <= ph_fre_th):
            Ram_ten[i] = 0
        else :
            dipSp_res  = ex_dip_absorp.conj()/((ome_light_Ha - BS_energies - ph_freq_pytorch[i])[None,:])
            dipSp_ares = ex_dip_absorp       /((ome_light_Ha + BS_energies - ph_freq_pytorch[i])[None,:])
            # 1) Compute the resonant term
            Ram_ten[i] =  pytorch.einsum('jp,ps,is->ij',dipSp_res, ex_ph_pytorch[i].conj(), dipS_res).numpy()
            # 2) anti resonant
            Ram_ten[i] += pytorch.einsum('jp,ps,is->ij',dipSp_ares, ex_ph_pytorch[i], dipS_ares).numpy()
            ## scale to get raman tensor
            Ram_ten[i] *= (math.sqrt(math.fabs(ome_light_Ha-ph_freq_pytorch[i])/ome_light_Ha)*ram_fac)
    return Ram_ten
