import numpy as np
from scipy.interpolate import RegularGridInterpolator
import sys
from netCDF4 import Dataset
from yambopy.kpoints import build_ktree, find_kpt
from numba import njit, prange

@njit(parallel=True)
def get_broadened_spectrum(x_axis, peak_centers, peak_intensities, gamma=1.0):
    spectrum = np.zeros_like(x_axis)
    n_x = len(x_axis)
    n_peaks = len(peak_centers)
    
    for j in prange(n_x):
        x = x_axis[j]
        s = 0.0
        for i in range(n_peaks):
            w = peak_centers[i]
            I = peak_intensities[i]
            s += I * (gamma / np.pi) / ((x - w)**2 + gamma**2)
        spectrum[j] = s
        
    return spectrum

def main():
    if len(sys.argv) < 7:
        print("Usage: python script.py raman_results.npz ndb.elph output_prefix ref_x ref_y ref_z")
        sys.exit(1)

    input_file = sys.argv[1]
    qpoints_file = sys.argv[2]
    output_prefix = sys.argv[3]
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

    xpol = np.array([1, 1j, 0])
    I_co = np.abs(np.einsum('...ij,i,j->...', two_ph_raman, xpol, np.conj(xpol)))**2
    I_cross = np.abs(np.einsum('...ij,i,j->...', two_ph_raman, xpol, xpol))**2

    nq_orig = len(qpoints)
    q_axis_I = I_co.shape.index(nq_orig)
    q_axis_freq = freq_ram.shape.index(nq_orig)
    
    # The branch axis is the dimension of size 3 in freq_ram
    # In freq_ram, shape is e.g. (nq, 3, nmode, nmode)
    branch_axis_freq = [i for i, dim in enumerate(freq_ram.shape) if dim == 3 and i != q_axis_freq][0]
    
    # In I_co, shape is e.g. (N_ome, nq, 3, nmode, nmode)
    branch_axis_I = [i for i, dim in enumerate(I_co.shape) if dim == 3 and i != q_axis_I and i != 0][0]

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
    tmp_qpt[...,0], tmp_qpt[...,1], tmp_qpt[...,2] = np.meshgrid(tmp1, tmp2, tmp3, indexing='ij')
    qpoints_tree = build_ktree(qpoints)
    indices = find_kpt(qpoints_tree, tmp_qpt) 

    I_co_q_first = np.moveaxis(I_co, q_axis_I, 0)
    I_cross_q_first = np.moveaxis(I_cross, q_axis_I, 0)
    freq_q_first = np.moveaxis(freq_ram, q_axis_freq, 0)

    I_co_grid = I_co_q_first[indices, ...].reshape((nx, ny, nz) + I_co_q_first.shape[1:])
    I_cross_grid = I_cross_q_first[indices, ...].reshape((nx, ny, nz) + I_cross_q_first.shape[1:])
    freq_grid = freq_q_first[indices, ...].reshape((nx, ny, nz) + freq_q_first.shape[1:])

    interp_I_co = RegularGridInterpolator((tmp1, tmp2, tmp3), I_co_grid, bounds_error=False, fill_value=None)
    interp_I_cross = RegularGridInterpolator((tmp1, tmp2, tmp3), I_cross_grid, bounds_error=False, fill_value=None)
    interp_freq = RegularGridInterpolator((tmp1, tmp2, tmp3), freq_grid, bounds_error=False, fill_value=None)

    nx_fine = nx * refinementx if nx > 1 else 1
    ny_fine = ny * refinementy if ny > 1 else 1
    nz_fine = nz * refinementz if nz > 1 else 1

    qx_fine = np.linspace(0, (nx - 1) / nx, nx_fine)
    qy_fine = np.linspace(0, (ny - 1) / ny, ny_fine)
    qz_fine = np.linspace(0, (nz - 1) / nz, nz_fine)
    QX, QY, QZ = np.meshgrid(qx_fine, qy_fine, qz_fine, indexing='ij')
    qpoints_fine = np.vstack([QX.ravel(), QY.ravel(), QZ.ravel()]).T

    I_co_fine = interp_I_co(qpoints_fine)
    I_cross_fine = interp_I_cross(qpoints_fine)
    freq_fine = interp_freq(qpoints_fine)

    expansion_factor = (refinementx * refinementy * refinementz)
    I_co_fine /= expansion_factor
    I_cross_fine /= expansion_factor

    I_co_fine_restored = np.moveaxis(I_co_fine, 0, q_axis_I)
    I_cross_fine_restored = np.moveaxis(I_cross_fine, 0, q_axis_I)
    freq_fine_restored = np.moveaxis(freq_fine, 0, q_axis_freq)
        
    freq_fine_cm = freq_fine_restored * 27.21111 * 8065.5

    # Determine plotting range across ALL shifts
    all_shifts = freq_fine_cm.flatten()
    # We span from min shift (Stokes) to max shift (Anti-Stokes)
    x_min = np.min(all_shifts) - 0.05
    x_max = np.max(all_shifts) + 0.05
    x_axis = np.linspace(x_min, x_max, 4000)

    for i_ome in range(len(omega_light)):
        print(f"Processing omega_light = {omega_light[i_ome]}")
        
        # We slice along the branch axis: 0=AA, 1=EA, 2=EE
        spectra_co = {'AA': None, 'EA': None, 'EE': None}
        spectra_cross = {'AA': None, 'EA': None, 'EE': None}
        
        for branch_idx, branch_name in enumerate(['AA', 'EA', 'EE']):
            I_co_branch = np.take(I_co_fine_restored[i_ome], branch_idx, axis=branch_axis_I - 1) # -1 because we sliced omega
            I_cross_branch = np.take(I_cross_fine_restored[i_ome], branch_idx, axis=branch_axis_I - 1)
            freq_branch = np.take(freq_fine_cm, branch_idx, axis=branch_axis_freq)
            
            w = freq_branch.flatten()
            i_c = I_co_branch.flatten()
            i_x = I_cross_branch.flatten()
            
            spectra_co[branch_name] = get_broadened_spectrum(x_axis, w, i_c, gamma=2.0)
            spectra_cross[branch_name] = get_broadened_spectrum(x_axis, w, i_x, gamma=2.0)

        # Full is the sum
        full_co = spectra_co['AA'] + spectra_co['EA'] + spectra_co['EE']
        full_cross = spectra_cross['AA'] + spectra_cross['EA'] + spectra_cross['EE']

        header = "Shift(cm-1) Full EE(Stokes) AA(Anti-Stokes) EA(Mixed)"
        out_co = np.c_[-x_axis, full_co, spectra_co['EE'], spectra_co['AA'], spectra_co['EA']]
        out_cross = np.c_[-x_axis, full_cross, spectra_cross['EE'], spectra_cross['AA'], spectra_cross['EA']]
        
        np.savetxt(f"{output_prefix}_ome_{i_ome}_co.txt", out_co, header=header)
        np.savetxt(f"{output_prefix}_ome_{i_ome}_cross.txt", out_cross, header=header)

if __name__ == "__main__":
    main()
