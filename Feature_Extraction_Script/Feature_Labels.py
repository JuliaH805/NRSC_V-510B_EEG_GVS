"""
Feature Labels
Generates human-readable labels for all 69 features
"""

def get_feature_labels():
    """
    Generate labels for all 69 features
    
    Returns:
    --------
    labels : list
        List of 69 feature label strings
    """
    labels = []
    
    # RSP features (1-11)
    labels.append("Feature_1: RSP_Delta1")
    labels.append("Feature_2: RSP_Delta2")
    labels.append("Feature_3: RSP_Theta1")
    labels.append("Feature_4: RSP_Theta2")
    labels.append("Feature_5: RSP_Alpha1")
    labels.append("Feature_6: RSP_Alpha2")
    labels.append("Feature_7: RSP_Sigma1")
    labels.append("Feature_8: RSP_Sigma2")
    labels.append("Feature_9: RSP_Beta1")
    labels.append("Feature_10: RSP_Beta2")
    labels.append("Feature_11: RSP_Gamma")
    
    # HP: Fc features (12-17)
    labels.append("Feature_12: HP_FC_Delta")
    labels.append("Feature_13: HP_FC_Theta")
    labels.append("Feature_14: HP_FC_Alpha")
    labels.append("Feature_15: HP_FC_Sigma")
    labels.append("Feature_16: HP_FC_Beta")
    labels.append("Feature_17: HP_FC_Gamma")
    
    # HP: Fsigma features (18-23)
    labels.append("Feature_18: HP_Fsigma_Delta")
    labels.append("Feature_19: HP_Fsigma_Theta")
    labels.append("Feature_20: HP_Fsigma_Alpha")
    labels.append("Feature_21: HP_Fsigma_Sigma")
    labels.append("Feature_22: HP_Fsigma_Beta")
    labels.append("Feature_23: HP_Fsigma_Gamma")
    
    # HP: S(fc) features (24-29)
    labels.append("Feature_24: HP_S(fc)_Delta")
    labels.append("Feature_25: HP_S(fc)_Theta")
    labels.append("Feature_26: HP_S(fc)_Alpha")
    labels.append("Feature_27: HP_S(fc)_Sigma")
    labels.append("Feature_28: HP_S(fc)_Beta")
    labels.append("Feature_29: HP_S(fc)_Gamma")
    
    # SWI features (30-32)
    labels.append("Feature_30: SWI_DSI")
    labels.append("Feature_31: SWI_TSI")
    labels.append("Feature_32: SWI_ASI")
    
    # Hjorth features (33-35)
    labels.append("Feature_33: Hjorth_Activity")
    labels.append("Feature_34: Hjorth_Mobility")
    labels.append("Feature_35: Hjorth_Complexity")
    
    # Statistical moments (36-37)
    labels.append("Feature_36: Skewness")
    labels.append("Feature_37: Kurtosis")
    
    # BTS Amplitude features (38-49)
    desiredFreqs = ["Delta", "Delta", "Theta", "Theta", "Alpha", "Alpha", 
                    "Sigma", "Sigma", "Beta", "Beta", "Gamma", "Gamma"]
    for i, band in enumerate(desiredFreqs):
        labels.append(f"Feature_{38+i}: BTS_Amplitude.{band}")
    
    # BTS Phase features (50-61)
    for i, band in enumerate(desiredFreqs):
        labels.append(f"Feature_{50+i}: BTS_Angle.{band}")
    
    # Wavelet features (62-69)
    for i in range(1, 9):
        labels.append(f"Feature_{61+i}: Wavelet_EstPower.{i}")
    
    return labels
