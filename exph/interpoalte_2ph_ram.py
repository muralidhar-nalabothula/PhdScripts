import numpy as np
from scipy.interpolate import RegularGridInterpolator
import sys
from netCDF4 import Dataset
from yambopy.kpoints import build_ktree, find_kpt

def main():
    input_file = sys.argv[1]
    qpoints_file = sys.argv[2]
    output_file = sys.argv[3]
    refinementx = int(sys.argv[4])
    refinementy = int(sys.argv[5])
    refinementz = int(sys.argv[6])

    data = np.load(input_file)
    two_ph_raman = data['two_ph_raman']
    freq_ram = data['freq_ram']
    omega_light = data['omega_light']
    elph = Dataset(qpoints_file,'r')
    qpoints = elph['qpoints'][...].data
    elph.close()

    tol = 1e-5
    qpoints = qpoints - np.floor(qpoints)
    qpoints = (qpoints + tol) % 1 - tol

    max_q = np.max(qpoints, axis=0)
    grid_dims = np.ones(3, dtype=int)
    
    for i in range(3):
        if max_q[i] > tol:
            grid_dims[i] = int(np.rint(1.0 / (1.0 - max_q[i])))

    nx, ny, nz = grid_dims

    tmp1 = np.arange(nx) / nx
    tmp2 = np.arange(ny) / ny
    tmp3 = np.arange(nz) / nz
    tmp_qpt = np.zeros((nx,ny,nz,3),dtype=float)
    tmp_qpt[...,0], tmp_qpt[...,2], tmp_qpt[...,2] = np.meshgrid(tmp1, tmp2, tmp3, indexing='ij')
    qpoints_tree = build_ktree(qpoints)
    indices = find_kpt(qpoints_tree,tmp_qpt) 

    qpoints = qpoints[indices].copy()

    tpr_q_first = np.moveaxis(two_ph_raman, 1, 0)

    shape_tpr = (nx, ny, nz) + tpr_q_first.shape[1:]
    shape_freq = (nx, ny, nz) + freq_ram.shape[1:]

    grid_tpr_real = two_ph_raman[:,indices,...].real.reshape(shape_tpr)
    grid_tpr_imag = two_ph_raman[:,indices,...].imag.reshape(shape_tpr)
    grid_freq = freq_ram[indices,...].copy()
    #
    interp_tpr_real = RegularGridInterpolator((tmp1, tmp2, tmp3), grid_tpr_real, bounds_error=False, fill_value=None)
    interp_tpr_imag = RegularGridInterpolator((tmp1, tmp2, tmp3), grid_tpr_imag, bounds_error=False, fill_value=None)
    interp_freq = RegularGridInterpolator((tmp1, tmp2, tmp3), grid_freq, bounds_error=False, fill_value=None)

    nx_fine = nx * refinementx if nx > 1 else 1
    ny_fine = ny * refinementy if ny > 1 else 1
    nz_fine = nz * refinementz if nz > 1 else 1
    print(refinementx,refinementy,refinementz)

    qx_fine = np.linspace(0, (nx - 1) / nx, nx_fine)
    qy_fine = np.linspace(0, (ny - 1) / ny, ny_fine)
    qz_fine = np.linspace(0, (nz - 1) / nz, nz_fine)

    QX, QY, QZ = np.meshgrid(qx_fine, qy_fine, qz_fine, indexing='ij')
    qpoints_fine = np.vstack([QX.ravel(), QY.ravel(), QZ.ravel()]).T

    tpr_fine_real = interp_tpr_real(qpoints_fine)
    tpr_fine_imag = interp_tpr_imag(qpoints_fine)
    tpr_fine = tpr_fine_real + 1j * tpr_fine_imag

    freq_fine = interp_freq(qpoints_fine)

    two_ph_raman_fine = np.moveaxis(tpr_fine, 0, 1)

    np.savez_compressed(
        output_file,
        two_ph_raman=two_ph_raman_fine,
        freq_ram=freq_fine,
        omega_light=omega_light,
        qpoints=qpoints_fine
    )

if __name__ == "__main__":
    main()

# python3 interpolate.py raman_results.npz ndb.elph raman_results_fine.npz 3 4 5

