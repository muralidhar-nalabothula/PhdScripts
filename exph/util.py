import numpy as np


def bose(Eb, Bose_Temp, thr=1e-10):
    """
    Bose-Einstein occupation function.

    Parameters
    ----------
    Eb : float or ndarray
        Boson energies in eV (must be >= 0 for phonons).
    Bose_Temp : float
        Temperature in Kelvin.
    thr : float, optional
        Small threshold used to treat T ~ 0 or E ~ 0.

    Returns
    -------
    ndarray
        Bose-Einstein occupation numbers:
            n_B = 1 / (exp(E / k_B T) - 1)

    Notes
    -----
    - For T <= thr, returns zero occupation.
    - For E <= thr (e.g. acoustic phonon mode at q=0), returns zero
      to avoid divergence in numerical loops.
    - Negative energies are unphysical for phonons and raise an error.
    """
    Eb = np.asarray(Eb, dtype=float)

    kb = 8.6173e-5
    # Zero temperature
    if Bose_Temp <= thr:
        return np.zeros_like(Eb)

    # Reject negative phonon energies
    if np.any(Eb < -thr):
        raise ValueError(
            "Negative boson energies found. Unphysical for phonons.")

    # Clamp tiny negatives from numerical noise
    Eb = np.maximum(Eb, 0.0)

    x = Eb / (kb * Bose_Temp)

    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        n_be = np.where(Eb <= thr, 0.0, 1.0 / np.expm1(x))

    return n_be
