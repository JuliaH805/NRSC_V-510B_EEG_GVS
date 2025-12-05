"""
Bispectrum estimation
Mirrors bispectrum.m functionality
"""
import numpy as np
from scipy.signal import windows
from cumulant import cumulant

def bispectrum(x, nLag, fc, nFFT):
    """
    Calculate bispectrum using indirect method
    
    Parameters:
    -----------
    x : array_like
        Input signal
    nLag : int
        Maximum lag value
    fc : float
        Sampling frequency
    nFFT : int
        FFT size
    
    Returns:
    --------
    Bspec : ndarray
        Bispectrum matrix (nFFT x nFFT)
    waxis : ndarray
        Frequency axis
    """
    # Initialize
    Np = 2 * nLag + 1
    x = np.array(x).flatten().astype(float)
    
    # Calculate cumulants
    Cum = cumulant(x, [nLag, nLag])
    
    # Lag Window - Parzen window
    Window = windows.parzen(Np)
    
    # Create 2D window
    BWind = np.zeros((Np, Np))
    
    for i in range(nLag, -1, -1):
        ind = np.arange(i, Np)
        indices = ind * Np + ind - i
        BWind.flat[indices] = Window[nLag - i]
    
    BWind = BWind + np.triu(BWind, 1).T
    Window_2d = np.outer(Window, Window)
    BWind = BWind * Window_2d
    
    # Apply window to cumulant
    WCum = Cum * BWind
    
    # Calculate bispectrum via 2D FFT
    Bspec = np.fft.fft2(WCum, s=(nFFT, nFFT))
    Bspec = np.fft.fftshift(Bspec)
    
    # Create frequency axis
    if nFFT % 2 == 0:
        waxis = np.arange(-nFFT/2, nFFT/2) * fc / nFFT
    else:
        waxis = np.arange(-(nFFT-1)/2, (nFFT-1)/2 + 1) * fc / nFFT
    
    return Bspec, waxis
