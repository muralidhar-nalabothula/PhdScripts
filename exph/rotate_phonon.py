import numpy as np
from kpts import make_kpositive, build_ktree, find_kpt
from io_exph import get_SAVE_Data
import scipy.linalg as la

np.set_printoptions(suppress=True)


def rotate_eig_vecs(sym_Rmat, sym_tau, sym_time_rev, alat, atomic_pos, qcart, eig_q):
    # ai = alat[:,i]
    # q_cart must contain 2*pi
    blat = np.linalg.inv(alat.T)
    #
    atom_pos_crys = atomic_pos @ blat
    tree = build_ktree(atom_pos_crys)
    #
    rot_atom = atomic_pos @ sym_Rmat.T
    rot_atom += sym_tau[None,:]
    #
    if sym_time_rev:
        rot_atom = -rot_atom
    rot_atom_crys = rot_atom @ blat
    #
    rot_map = find_kpt(tree, rot_atom_crys)
    Sq_cart = sym_Rmat @ qcart
    qphase = np.exp(-1j * (sym_tau @ Sq_cart))
    exp_Sqr = np.exp(1j * (atomic_pos @ Sq_cart))
    exp_qr = np.exp(-1j * (atomic_pos @ qcart))
    rot_eig = eig_q @ sym_Rmat.T
    phase_factor = exp_qr * qphase
    rot_eig *= phase_factor[np.newaxis, :, np.newaxis]
    if sym_time_rev:
        rot_eig = -np.conj(rot_eig)
    eig_Sq = np.zeros_like(eig_q, dtype=np.complex128)
    eig_Sq[:, rot_map, :] = rot_eig * exp_Sqr[rot_map][np.newaxis, :, np.newaxis]
    return Sq_cart, eig_Sq

def read_dyn_qe_old(filename):
    with open(filename, 'r') as fp:
    # Open the specified dynamical matrix file for reading
        lines = fp.readlines()
        # Read all lines into memory

    line_idx = 0
    # Initialize a counter to track our position in the file lines

    line_idx += 2
    # Skip the first two comment lines

    read_buf = lines[line_idx]
    # Read the third line which contains basic system parameters
    line_idx += 1
    # Increment the line counter

    read_fbuf = [float(x) for x in read_buf.split()]
    # Parse the line into a list of floats

    if len(read_fbuf) != 9:
    # Check if exactly 9 floats were found as expected by the C code
        raise ValueError("Error reading line 3 in dyn file")
        # Throw an error if the format is unexpected

    ntype = int(round(read_fbuf[0]))
    # Extract the number of atom types
    natom = int(round(read_fbuf[1]))
    # Extract the total number of atoms
    ibrav = int(round(read_fbuf[2]))
    # Extract the Bravais lattice index
    alat = read_fbuf[3]
    # Extract the lattice parameter (alat)

    nmodes = natom * 3
    # Calculate the total number of vibrational modes

    if ibrav == 0:
    # If ibrav is 0, lattice vectors are explicitly defined next
        line_idx += 4
        # Skip the scratch line and the 3 lattice vector lines

    atm_mass_type = np.zeros(ntype)
    # Initialize an array to store the mass for each atom type

    for i in range(ntype):
    # Loop over the number of atom types
        read_buf = lines[line_idx]
        # Read the line containing the mass info
        line_idx += 1
        # Increment the line counter

        parts = read_buf.split()
        # Split the string by whitespace

        if len(parts) < 3:
        # Check if the line contains index, symbol, and mass
            raise ValueError("Failed to read atomic masses from dyn file")
            # Throw an error if parsing fails

        atm_mass_type[i] = float(parts[-1])
        # Extract the mass (the 3rd token) and store it

        if abs(atm_mass_type[i]) < 1e-12:
        # Check for zero mass (ELPH_EPS equivalent)
            raise ValueError("Zero masses in dynamical file")
            # Throw an error if mass is zero

    atm_mass = np.zeros(natom)
    # Initialize an array to store the mass of each individual atom

    atomic_positions = []
    # Initialize a list to store atomic coordinates

    for i in range(natom):
    # Loop over the total number of atoms
        read_buf = lines[line_idx]
        # Read the line containing atom type and coordinates
        line_idx += 1
        # Increment the line counter

        read_fbuf = [float(x) for x in read_buf.split()]
        # Parse the string into floats

        if len(read_fbuf) != 5:
        # Ensure we have atom index, type index, and 3 coordinates
            raise ValueError("Failed to read atomic masses from dyn file")
            # Throw an error if parsing fails

        itype = int(round(read_fbuf[1]))
        # Extract the atom type index

        atm_mass[i] = atm_mass_type[itype - 1]
        # Assign the correct mass to this specific atom based on its type

        atomic_positions.append([read_fbuf[2], read_fbuf[3], read_fbuf[4]])
        # Extract the x, y, z coordinates and append to the list

    nq_found = 0
    # Initialize counter for the number of q-points found
    qpts = []
    # List to store q-points
    omegas = []
    # List to store frequencies
    pol_vecs = []
    # List to store polarization vectors (eigenvectors)

    while line_idx < len(lines) and nq_found < 1:
    # Loop through the lines, strictly exiting after the first dynamical matrix is processed
        read_buf = lines[line_idx]
        # Read the current line
        line_idx += 1
        # Increment the line counter

        if not read_buf.strip():
        # Skip empty lines
            continue
            # Move to next iteration

        if "Dynamical  Matrix" in read_buf:
        # Identify the start of a dynamical matrix block
            line_idx += 1
            # Skip the empty line following the header

            read_buf = lines[line_idx]
            # Read the line defining the q-point
            line_idx += 1
            # Increment the line counter

            q_str = read_buf.split('(')[1].split(')')[0]
            # Extract the string between parentheses

            qpt = [float(val) for val in q_str.split()]
            # Parse the q-point coordinates

            if len(qpt) != 3:
            # Validate q-point length
                raise ValueError("error reading qpoint from dynamat files")
                # Throw error if parsing fails

            qpts.append(qpt)
            # Store the extracted q-point

            line_idx += 1
            # Skip the empty line following the q-point

            dyn_mat_tmp = np.zeros((nmodes, nmodes), dtype=complex)
            # Initialize an empty complex matrix for the dynamical matrix

            for ia in range(natom):
            # Loop over the first atom index
                for ib in range(natom):
                # Loop over the second atom index
                    read_buf = lines[line_idx]
                    # Read the block header (atom indices)
                    line_idx += 1
                    # Increment the line counter

                    indices = [int(x) for x in read_buf.split()]
                    # Parse the indices

                    itmp = indices[0] - 1
                    # Adjust to 0-based indexing
                    jtmp = indices[1] - 1
                    # Adjust to 0-based indexing

                    if itmp != ia or jtmp != ib:
                    # Validate block order matches expected loop order
                        raise ValueError("error reading dynamical matrix from dynamat files")
                        # Throw error if mismatch

                    inv_mass_sqtr = 1.0 / np.sqrt(atm_mass[ia] * atm_mass[ib])
                    # Calculate mass scaling factor 1/sqrt(Ma*Mb)

                    for ix in range(3):
                    # Loop over cartesian directions for atom ia
                        read_buf = lines[line_idx]
                        # Read the matrix row
                        line_idx += 1
                        # Increment the line counter

                        read_fbuf = [float(x) for x in read_buf.split()]
                        # Parse the floats

                        if len(read_fbuf) != 6:
                        # Validate exactly 6 elements (3 complex numbers)
                            raise ValueError("error reading dynamical matrix from dynamat files")
                            # Throw error if invalid

                        for i in range(6):
                        # Loop through elements to apply mass scaling
                            read_fbuf[i] *= inv_mass_sqtr
                            # Multiply element by the mass scaling factor

                        for iy in range(3):
                        # Loop over cartesian directions for atom ib
                            col_idx = ix + ia * 3
                            # Determine column index (equivalent to C code row/col logic)
                            row_idx = iy + ib * 3
                            # Determine row index

                            dyn_mat_tmp[row_idx, col_idx] = read_fbuf[2 * iy] + 1j * read_fbuf[2 * iy + 1]
                            # Assign complex value. Note: Python matrix indexing is row-major,
                            # C code used column-major array flattening, so indices are mapped accordingly to match zheev input.

            # Symmetrize the matrix
            for idim1 in range(nmodes):
            # Loop over rows
                for jdim1 in range(idim1 + 1):
                # Loop over columns up to the diagonal
                    dyn_mat_tmp[idim1, jdim1] = 0.5 * (dyn_mat_tmp[idim1, jdim1] + np.conj(dyn_mat_tmp[jdim1, idim1]))
                    # Average the element with its conjugate transpose
                    dyn_mat_tmp[jdim1, idim1] = np.conj(dyn_mat_tmp[idim1, jdim1])
                    # Assign the conjugate to the upper triangle to ensure exact Hermitian property

            # Diagonalize the dynamical matrix
            omega2, eig_vecs = la.eigh(dyn_mat_tmp)
            # Use scipy's eigh for Hermitian matrices (equivalent to LAPACK zheev)

            omega_q = np.zeros(nmodes)
            # Initialize array for frequencies

            for imode in range(nmodes):
            # Loop through computed eigenvalues
                omega_q[imode] = np.sqrt(abs(omega2[imode]))
                # Calculate frequency taking square root of absolute magnitude

                if omega2[imode] < 0:
                # If eigenvalue was negative
                    omega_q[imode] = -omega_q[imode]
                    # Make frequency negative to indicate imaginary frequency

            omegas.append(omega_q)
            # Store frequencies for this q-point

            # Divide eigenvectors with sqrt of masses
            eig_q = np.zeros((nmodes, nmodes), dtype=complex)
            # Initialize array for scaled eigenvectors

            for imode in range(nmodes):
            # Loop over modes
                for jmode in range(nmodes):
                # Loop over components
                    ia = jmode // 3
                    # Determine atom index for this component

                    eig_q[jmode, imode] = eig_vecs[jmode, imode] / np.sqrt(atm_mass[ia])
                    # Scale component by 1/sqrt(Mass) and store

            pol_vecs.append(eig_q)
            # Store scaled eigenvectors for this q-point

            nq_found += 1
            # Increment q-point counter

    if nq_found == 0:
    # Check if any matrices were actually read
        raise ValueError("No dynamical matrices found in the dyn file")
        # Throw error if file was empty or improperly formatted

    return alat, np.array(atomic_positions)*alat, 2*np.pi*np.array(qpts)[0]/alat,\
                np.array(omegas)[0]*109737.31,\
                np.array(pol_vecs)[0].T.reshape(-1,len(atomic_positions),3)
    #
    # Return all the extracted and computed data

def write_dyn_format(output_file, q_point, freqs, modes):
    with open(output_file, 'w') as f:
    # Open the target file for writing

        f.write("     diagonalizing the dynamical matrix ...\n\n")
        # Write the hardcoded header

        f.write(f" q = {q_point[0]:13.8f} {q_point[1]:11.8f} {q_point[2]:11.8f}\n")
        # Write the formatted q-point matching the spacing logic from your prompt

        f.write(" **************************************************************************\n")
        # Write the top border of the block

        for m in range(len(freqs)):
        # Loop through the total number of modes using the length of the frequencies array

            cm_val = freqs[m]
            # Retrieve the cm-1 frequency for the current mode

            thz_val = cm_val / 33.35641
            # Calculate the THz frequency using the standard conversion factor

            f.write(f"     freq ({m+1:5d}) = {thz_val:14.6f} [THz] = {cm_val:14.6f} [cm-1]\n")
            # Write the frequency line with specific format spacing

            for a in range(len(modes[m])):
            # Loop through the atoms for this specific mode (the second dimension of your array)

                v = modes[m][a]
                # Retrieve the 3-component complex vector (x, y, z) for this atom

                f.write(f" ( {v[0].real:9.6f} {v[0].imag:9.6f} {v[1].real:9.6f} {v[1].imag:9.6f} {v[2].real:9.6f} {v[2].imag:9.6f} )\n")
                # Write the 6 real and imaginary components adhering strictly to your layout

        f.write(" **************************************************************************\n")
        # Write the closing border to finish the file

# def rotate_dynmats():
#     lat_vecs, nk_ibz, symm_mats, trev, nval = \
#         get_SAVE_Data(save_folder='/Users/murali/Downloads/hbn_cp')
#     sym_Rmat = symm_mats[2]
#     lat_param, atomic_positions, qpts, omegas, pol_vecs =   \
#         read_dyn_qe_old('/Users/murali/Downloads/hbn_cp/bn.dyn1')
#     # print(atomic_positions.shape, qpts.shape, omegas, pol_vecs.shape)
#     # print(pol_vecs[1]/np.linalg.norm(pol_vecs[1]))
#     qpt_array = np.zeros((3,3))
#     pol_array = np.zeros((3,) + pol_vecs.shape,dtype=complex)

#     q_rot1, pol_rot_1 = rotate_eig_vecs(sym_Rmat, np.zeros(3), False, lat_vecs, atomic_positions, qpts, pol_vecs)
#     q_rot2, pol_rot_2 = rotate_eig_vecs(sym_Rmat@sym_Rmat, np.zeros(3), False, lat_vecs, atomic_positions, qpts, pol_vecs)

#     qpt_array[0,:] = qpts[...]
#     pol_array[0,...] = pol_vecs[...]

#     qpt_array[1,:] = q_rot1[...]
#     pol_array[1,...] = pol_rot_1[...]

#     qpt_array[2,:] = q_rot2[...]
#     pol_array[2,...] = pol_rot_2[...]

#     pol_array_original = pol_array.copy()

#     x = np.exp(-1j * 2 * np.pi / 3)
#     coeff_array = np.array([
#         [1, x, x**2],
#         [x, x**2, 1],
#         [x**2, 1, x]
#     ])
#     pol_array = np.einsum('ij,j...->i...',coeff_array,pol_array_original)

#     pol_norms = np.linalg.norm(pol_array,axis=(-1,-2))
#     pol_array  = pol_array/pol_norms[:,:,None,None]
#     print(qpt_array/2/np.pi@lat_vecs)

#     for i in range(3):
#         write_dyn_format('/Users/murali/Downloads/hbn_cp/bn.modes%d'%(i+1), 0.0*qpt_array[i]/2/np.pi*lat_param, omegas, pol_array[i])


def rotate_basis(save_folder, dyn_file, isym, out_file):
    #
    lat_vecs, nk_ibz, symm_mats, trev, nval = \
        get_SAVE_Data(save_folder=save_folder)
    #
    sym_Rmat = symm_mats[2]
    #
    lat_param, atomic_positions, qpts, omegas, pol_vecs =   \
        read_dyn_qe_old(dyn_file)
    #
    sym_Rmat = symm_mats[isym]
    #
    _, pol_rot = rotate_eig_vecs(sym_Rmat, np.zeros(3), False, lat_vecs, atomic_positions, qpts, pol_vecs)
    nmodes = pol_vecs.shape[0]
    pol_vecs = pol_vecs.reshape(nmodes,nmodes)
    pol_rot = pol_rot.reshape(nmodes,nmodes)
    rep =  np.linalg.inv(pol_vecs.T)@pol_rot.T
    tol = 1e-2
    new_pol = np.zeros_like(pol_vecs, dtype=complex)
    w = np.zeros(nmodes, dtype=complex)
    i = 0
    while i < nmodes:
        j = i + 1
        while j < nmodes and abs(omegas[j] - omegas[i]) < tol:
            j += 1
        rep_block = rep[i:j, i:j]
        det_mag = np.abs(np.linalg.det(rep_block))
        # Calculate the magnitude of the determinant of the submatrix
        if not np.isclose(det_mag, 1.0, atol=1e-2):
             print(f"Warning: Block {i}:{j} determinant magnitude is {det_mag:.4f}. Frequencies are improperly grouped!")
        # A valid symmetry block must be unitary (determinant magnitude of 1.0); if not, the tolerance sliced it in half
        w_block, v_block = np.linalg.eig(rep_block)
        w[i:j] = w_block
        new_pol[i:j, :] = np.einsum('ji,jx->ix', v_block, pol_vecs[i:j, :], optimize=True)
        i = j
    print(-3 * np.angle(w) / 2 / np.pi)
    omegas +=  np.linspace(0,1,len(omegas))
    print(omegas)
    norm = np.linalg.norm(new_pol,axis=(-1))
    write_dyn_format(out_file,qpts/2/np.pi*lat_param, omegas, new_pol.reshape(nmodes,-1,3)/norm[:,None,None])


if __name__ == "__main__":
    rotate_basis('/Users/murali/Downloads/scratch/mote2_cp', '/Users/murali/Downloads/scratch/mote2_cp/mote2.dyn1', 2, '/Users/murali/Downloads/scratch/mote2_cp/mote2_22.modes')
