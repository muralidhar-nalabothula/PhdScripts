import numpy as np
import numba as nb
import torch as pytorch

single_prec = True

if single_prec:
    numba_Cmplx = nb.complex64
    numpy_Cmplx = np.complex64
    pytor_Cmplx = pytorch.complex64
    numba_float = nb.float32
    numpy_float = np.float32
    pytor_float = pytorch.float32

else:
    numba_Cmplx = nb.complex128
    numpy_Cmplx = np.complex128
    pytor_Cmplx = pytorch.complex128
    numba_float = nb.float64
    numpy_float = np.float64
    pytor_float = pytorch.float64
