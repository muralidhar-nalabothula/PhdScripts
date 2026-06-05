import numpy as np
from netCDF4 import Dataset
from yambopy.bse.exciton_matrix_elements import exciton_X_matelem
from yambopy import YamboLatticeDB, YamboWFDB, LetzElphElectronPhononDB, YamboElectronsDB, YamboExcitonDB


def compute_second_derivative_gemm(h_nl: np.ndarray,
                                   h_lm: np.ndarray,
                                   E_n: np.ndarray,
                                   E_m: np.ndarray,
                                   E_l: np.ndarray,
                                   tol: float = 1e-8) -> np.ndarray:
    """
    Computes the second derivative matrix element using optimized GEMM operations.
    #
    Expected shapes:
    h_nl : (natom, 3, nk, N_l, N_n)
    h_lm : (natom, 3, nk, N_m, N_l)
    E_n  : (nk, N_n)
    E_m  : (nk, N_m)
    E_l  : (nk, N_l)
    #
    # Everything in Ry units
    Returns:
    D_matrix : (natom, 3, 3, nk, N_n, N_m)  < nk | ddV} mk>
    $$\langle n\mathbf{k} | \partial^2_{\kappa\alpha, \kappa\beta} V | m\mathbf{k} \rangle =
    - \sum_{l \neq n,m}^{N_{bands}} h_{nl}^{\kappa\alpha}(\mathbf{k}) h_{lm}^{\kappa\beta}(\mathbf{k})
    \left( \frac{1}{\varepsilon_{n\mathbf{k}} - \varepsilon_{l\mathbf{k}}} +
    \frac{1}{\varepsilon_{m\mathbf{k}} - \varepsilon_{l\mathbf{k}}} \right)$$
    """
    diff_nl = E_n[:, :, None] - E_l[:, None, :]
    inv_diff_nl = np.zeros_like(diff_nl)
    mask_nl = np.abs(diff_nl) > tol
    inv_diff_nl[mask_nl] = 1.0 / diff_nl[mask_nl]
    diff_ml = E_m[:, :, None] - E_l[:, None, :]
    inv_diff_ml = np.zeros_like(diff_ml)
    mask_ml = np.abs(diff_ml) > tol
    inv_diff_ml[mask_ml] = 1.0 / diff_ml[mask_ml]
    inv_diff_nl_transposed = inv_diff_nl.transpose(0, 2, 1)
    h_nl_W1 = h_nl * inv_diff_nl_transposed[None, None, :, :, :]
    h_lm_W2 = h_lm * inv_diff_ml[None, None, :, :, :]
    A1_mat = h_nl_W1.transpose(0, 1, 2, 4, 3)
    B1_mat = h_lm.transpose(0, 1, 2, 4, 3)
    A1_batch = A1_mat[:, :, None, :, :, :]
    B1_batch = B1_mat[:, None, :, :, :, :]
    C1 = A1_batch @ B1_batch
    A2_mat = h_nl.transpose(0, 1, 2, 4, 3)
    B2_mat = h_lm_W2.transpose(0, 1, 2, 4, 3)
    A2_batch = A2_mat[:, :, None, :, :, :]
    B2_batch = B2_mat[:, None, :, :, :, :]
    C2 = A2_batch @ B2_batch
    return -(C1 + C2)


def compute_debye_exph(SAVE_folder, BSE_dir, elph_file, nexc):
    lattice = YamboLatticeDB.from_db_file(filename=f'{SAVE_folder}/ns.db1')
    yel = YamboElectronsDB.from_db_file(folder=SAVE_folder)
    excdb = YamboExcitonDB.from_db_file(lattice,
                                        filename='ndb.BS_diago_Q1',
                                        folder=BSE_dir,
                                        Load_WF=True,
                                        neigs=nexc)

    bands_range = [np.min(excdb.unique_vbands), np.max(excdb.unique_cbands) + 1]
    #print(bands_range)
    elph = Dataset(elph_file, 'r')
    kpoints = elph['kpoints'][...].data
    elph_ml = elph['elph_mat'][0, ..., :, bands_range[0]:bands_range[1], :].data
    elph_ml = elph_ml[..., 0] + 1j * elph_ml[..., 1]
    elph_lm = elph['elph_mat'][0, ..., bands_range[0]:bands_range[1], :, :].data
    elph_lm = elph_lm[..., 0] + 1j * elph_lm[..., 1]
    pol_vec = elph['POLARIZATION_VECTORS'][0, ...].data
    pol_vec = pol_vec[..., 0] + 1j * pol_vec[..., 1]
    nmodes, natom, _ = pol_vec.shape
    inv_pol_vec = np.linalg.inv(pol_vec.reshape(nmodes, -1))
    nk, _, ns, Nl, Nbands = elph_ml.shape
    #print(nk, ns, Nl, Nbands)
    elph_ml = np.einsum('kvsmn,xv->xksmn', elph_ml, inv_pol_vec,
                        optimize=True)[:, :, 0].reshape(natom, 3, nk, Nl,
                                                        Nbands)
    elph_lm = np.einsum('kvsmn,xv->xksmn', elph_lm, inv_pol_vec,
                        optimize=True)[:, :, 0].reshape(natom, 3, nk, Nbands,
                                                        Nl)
    elph.close()
    ene = yel.eigenvalues_ibz[0][lattice.kmap[:, 0], ...] / 13.60569807  # in Ry
    E_l = ene.copy()
    E_n = ene[:, bands_range[0]:bands_range[1]]
    Dwaller = compute_second_derivative_gemm(elph_ml,
                                             elph_lm,
                                             E_n,
                                             E_n,
                                             E_l,
                                             tol=1e-4)
    # (nlambda, nk, nspin, m_bnd, n_bnd)
    Dwaller = Dwaller.reshape(-1, nk, 1, Nbands, Nbands)
    kvec = np.zeros(3)
    Ak = excdb.get_Akcv()
    exph_debye = exciton_X_matelem(kvec, kvec, Ak, Ak, Dwaller, kpoints)
    return exph_debye.transpose(0,2,1)


SAVE_folder = 'SAVE'
BSE_dir = 'GW_BSE'
elph_file = 'ndb.elph'
debye_exph = compute_debye_exph(SAVE_folder, BSE_dir, elph_file, nexc=10)
np.save('debye_exph',debye_exph)
print(np.sum(np.abs(debye_exph)**2))
