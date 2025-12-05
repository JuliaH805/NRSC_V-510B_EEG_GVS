"""
Feature Extraction Module
Extracts 69 features from EEG signal
Mirrors FeatEx.m functionality
"""
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
import pywt
from btsestimate import btsestimate

def extract_features(Sig, Fs, Phi=1, Rho=0, nLag=150, nFFT=512):
    """
    Extract 69 features from a single EEG epoch
    
    Parameters:
    -----------
    Sig : array_like
        EEG signal (single channel, single epoch)
    Fs : float
        Sampling frequency (Hz)
    Phi : float
        Bispectrum parameter (default=1)
    Rho : float
        Bispectrum parameter (default=0)
    nLag : int
        Lag for cumulant calculation (default=150)
    nFFT : int
        FFT size (default=512)
    
    Returns:
    --------
    Feats : ndarray
        Array of 69 features
    """
    Fnum = 69
    Feats = np.zeros(Fnum)
    n = len(Sig)
    X = np.array(Sig).flatten()
    
    # ========== A. Relative Spectral Power (RSP) - Features 1-11 ==========
    Ptot = bandpower(X, Fs, [0, 45])
    
    Feats[0] = bandpower(X, Fs, [0.5, 2]) / Ptot      # RSP_Delta1
    Feats[1] = bandpower(X, Fs, [2, 4]) / Ptot        # RSP_Delta2
    Feats[2] = bandpower(X, Fs, [4, 6]) / Ptot        # RSP_Theta1
    Feats[3] = bandpower(X, Fs, [6, 8]) / Ptot        # RSP_Theta2
    Feats[4] = bandpower(X, Fs, [8, 10]) / Ptot       # RSP_Alpha1
    Feats[5] = bandpower(X, Fs, [10, 12]) / Ptot      # RSP_Alpha2
    Feats[6] = bandpower(X, Fs, [12, 14]) / Ptot      # RSP_Sigma1
    Feats[7] = bandpower(X, Fs, [14, 16]) / Ptot      # RSP_Sigma2
    Feats[8] = bandpower(X, Fs, [16, 24]) / Ptot      # RSP_Beta1
    Feats[9] = bandpower(X, Fs, [24, 32]) / Ptot      # RSP_Beta2
    Feats[10] = bandpower(X, Fs, [32, 45]) / Ptot     # RSP_Gamma
    
    # ========== B. Harmonic Parameters (HP) - Features 12-29 ==========
    # Use full signal length for best frequency resolution
    nperseg = len(X)
    noverlap = nperseg // 2
    f, P = signal.welch(X, Fs, nperseg=nperseg, noverlap=noverlap, 
                        scaling='density')
    
    # Limit frequency range
    freq_mask = f <= Fs/2
    f = f[freq_mask]
    P = P[freq_mask]
    
    # --- Fc (Central Frequency) computation - Features 12-17
    bands = [(0.5, 4), (4, 8), (8, 12), (12, 16), (16, 32), (32, 45)]
    for i, (fl, fh) in enumerate(bands):
        Inds = (f >= fl) & (f < fh)
        if np.sum(P[Inds]) > 0:
            Feats[11 + i] = np.sum(P[Inds] * f[Inds]) / np.sum(P[Inds])
        else:
            Feats[11 + i] = (fl + fh) / 2  # Default to band center
    
    # --- Fsigma (Frequency Std) computation - Features 18-23
    for i, (fl, fh) in enumerate(bands):
        Inds = (f >= fl) & (f < fh)
        if np.sum(P[Inds]) > 0:
            fc = Feats[11 + i]
            Feats[17 + i] = np.sqrt(np.sum(P[Inds] * (f[Inds] - fc)**2) / np.sum(P[Inds]))
        else:
            Feats[17 + i] = 0
    
    # --- S(fc) computation - Features 24-29
    for i in range(6):
        fc = Feats[11 + i]
        # Find closest frequency index
        fc_idx = np.argmin(np.abs(f - fc))
        Feats[23 + i] = P[fc_idx]
    
    # ========== C. Slow-Wave Indices (SWI) - Features 30-32 ==========
    bspD = bandpower(X, Fs, [0.6, 4])      # Delta
    bspD2 = bandpower(X, Fs, [2, 4])       # Delta2
    bspT = bandpower(X, Fs, [4, 8])        # Theta
    bspA = bandpower(X, Fs, [8, 11.5])     # Alpha
    
    # DSI
    Feats[29] = bspD / (bspT + bspA + 1e-10)
    # TSI
    Feats[30] = bspT / (bspD + bspA + 1e-10)
    # ASI
    Feats[31] = bspA / (bspD2 + bspT + 1e-10)
    
    # ========== D. Hjorth Parameters - Features 33-35 ==========
    Xp = np.diff(X) * Fs         # 1st derivative
    Xpp = np.diff(X, n=2) * Fs**2  # 2nd derivative
    
    Feats[32] = np.var(X)                                    # Activity
    Feats[33] = np.sqrt(np.var(Xp) / (np.var(X) + 1e-10))  # Mobility
    Feats[34] = np.sqrt(np.var(Xpp) * np.var(X) / (np.var(Xp)**2 + 1e-10))  # Complexity
    
    # ========== E. Statistical Moments - Features 36-37 ==========
    M2 = 0
    M3 = 0
    M4 = 0
    m = np.mean(X)
    
    for i in range(n):
        M2 += (X[i] - m)**2
        M3 += (X[i] - m)**3
        M4 += (X[i] - m)**4
    
    M2 /= n
    M3 /= n
    M4 /= n
    
    Feats[35] = M3 / (np.sqrt(M2**3) + 1e-10)  # Skewness
    Feats[36] = M4 / (M2**2 + 1e-10)           # Kurtosis
    
    # ========== F. Bispectrum Time Series (BTS) - Features 38-61 ==========
    try:
        BTS, FFF = btsestimate(X.reshape(1, -1), Phi, Rho, nLag, Fs, nFFT)
        
        desiredFreqs = np.array([1, 3, 5, 7, 9, 11, 13, 15, 20, 28, 36, 40])
        
        # Find closest frequency indices
        IndsF = []
        for freq in desiredFreqs:
            idx = np.argmin(np.abs(FFF - freq))
            IndsF.append(idx)
        IndsF = np.array(IndsF)
        
        # Amplitude (magnitude)
        Feats[37:49] = np.abs(BTS[IndsF])
        # Phase (angle)
        Feats[49:61] = np.angle(BTS[IndsF])
    except Exception as e:
        print(f"Warning: BTS calculation failed: {e}")
        Feats[37:61] = 0
    
    # ========== G. Wavelet Coefficients - Features 62-69 ==========
    WaveletLvl = 8
    Wavename = 'sym6'
    
    coeffs = pywt.wavedec(X, Wavename, level=WaveletLvl)
    
    for i in range(WaveletLvl):
        # Reconstruct detail coefficients (skip approximation coefficients at index 0)
        coeffs_recon = [np.zeros_like(c) if j != (i + 1) else c 
                       for j, c in enumerate(coeffs)]
        reconstructed = pywt.waverec(coeffs_recon, Wavename)
        
        # Ensure same length as original signal
        if len(reconstructed) > len(X):
            reconstructed = reconstructed[:len(X)]
        elif len(reconstructed) < len(X):
            reconstructed = np.pad(reconstructed, (0, len(X) - len(reconstructed)))
        
        Feats[61 + i] = bandpower_array(reconstructed, Fs) / (Ptot + 1e-10)
    
    return Feats


def bandpower(data, fs, band):
    """
    Calculate bandpower using Welch's method
    
    Parameters:
    -----------
    data : array_like
        Input signal
    fs : float
        Sampling frequency
    band : list or tuple
        Frequency band [low, high] in Hz
    
    Returns:
    --------
    bp : float
        Band power
    """
    low, high = band
    
    # Use full signal length for best frequency resolution at low frequencies
    # For 2000 samples at 1000 Hz, this gives 0.5 Hz resolution
    nperseg = len(data)
    noverlap = nperseg // 2
    
    f, Pxx = signal.welch(data, fs, nperseg=nperseg, noverlap=noverlap,
                          scaling='density')
    
    # Find intersecting frequencies
    idx_band = np.logical_and(f >= low, f <= high)
    
    # Integrate power spectral density
    if np.sum(idx_band) > 0:
        bp = np.trapz(Pxx[idx_band], f[idx_band])
    else:
        bp = 0.0
    
    return bp


def bandpower_array(data, fs):
    """
    Calculate total power of a signal array
    
    Parameters:
    -----------
    data : array_like
        Input signal
    fs : float
        Sampling frequency
    
    Returns:
    --------
    power : float
        Total signal power
    """
    return np.sum(data**2) / len(data)
