"""
Bispectrum Time Series estimation
Mirrors btsestimate.m functionality
"""
import numpy as np
from scipy.signal import find_peaks
from bispectrum import bispectrum

def btsestimate(Sig, Phi, Rho, nLag, Fc, nFFT):
    """
    Extract bispectrum time series along parametric curve
    
    Parameters:
    -----------
    Sig : ndarray
        Input signals (can be multi-channel)
    Phi : float
        Slope parameter for curve extraction
    Rho : float
        Offset parameter for curve extraction
    nLag : int
        Maximum lag
    Fc : float
        Sampling frequency
    nFFT : int
        FFT size
    
    Returns:
    --------
    BTS : ndarray
        Bispectrum time series values
    Waxis : ndarray
        Frequency axis
    """
    if Sig.ndim == 1:
        Sig = Sig.reshape(1, -1)
    
    L = Sig.shape[0]
    Np = 2 * nLag + 1
    BTS = []
    
    for i in range(L):
        x = Sig[i, :]
        Bispec, Waxis = bispectrum(x, nLag, Fc, nFFT)
        
        W1 = Waxis.copy()
        W2 = Phi * Waxis + Rho
        
        # Filter out frequencies outside valid range
        valid_mask = np.abs(W2) <= np.max(np.abs(Waxis))
        W1 = W1[valid_mask]
        W2 = W2[valid_mask]
        
        # Convert to indices
        W1_ind = np.round(W1 * Np / Fc + nLag).astype(int)
        W2_ind = np.round(W2 * Np / Fc + nLag).astype(int)
        
        # Extract values from bispectrum
        bts_values = Bispec[W1_ind, W2_ind]
        BTS.append(bts_values)
    
    BTS = np.array(BTS)
    if BTS.shape[0] == 1:
        BTS = BTS.flatten()
    
    return BTS, Waxis
