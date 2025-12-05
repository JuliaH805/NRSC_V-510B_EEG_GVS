"""
Third-order cumulant calculation
Mirrors cumulant.m functionality
"""
import numpy as np

def cumulant(X, T):
    """
    Calculate 3rd order cumulant
    C(t1,t2) = E[X(k)*X(k+t1)*X(k+t2)]
    
    Parameters:
    -----------
    X : array_like
        Input signal vector
    T : list or tuple
        Lag values [t1, t2]
    
    Returns:
    --------
    C_X : ndarray
        Cumulant matrix of size (2*t1+1) x (2*t1+1)
    """
    N = len(X)
    X = X - np.mean(X)  # Center the signal
    
    t1 = T[0]
    t2 = T[1]
    L = 2 * t1 + 1
    
    # Initialize matrices
    Xr = np.tile(X.reshape(-1, 1), (1, t1 + 1))
    Xdf = np.zeros_like(Xr)
    C_X = np.zeros((L, L))
    
    # Compute cumulant
    for i in range(t1, -1, -1):
        if i > 0:
            Xdf[:N-i, i] = X[i:N]
        else:
            Xdf[:, i] = X
        
        ind1 = 0
        ind2 = i
        for j in range(t2 + i, L):
            if ind2 < t1 + 1:
                C_X[t1 + i, j] = np.dot(Xr[:, ind1] * Xdf[:, i], Xdf[:, ind2])
            ind1 += 1
            ind2 += 1
        
        # Use symmetry
        m = np.arange(i, t2 + 1)
        n = np.ones(t2 - i + 1, dtype=int) * i
        
        indices1 = (t2 - m) + (t1 + n - m) * L
        indices2 = (t2 - n) + (t1 + m - n) * L
        
        C_X.flat[indices1] = C_X[t1 + i, t2 + i:L]
        C_X.flat[indices2] = C_X[t1 + i, t2 + i:L]
    
    C_X = C_X / N
    
    # Add upper triangle to make symmetric
    C_X = C_X + np.triu(C_X, 1).T
    
    return C_X
