import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy.signal import welch, hilbert
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

data_path = Path(r"D:\UBC\510\neurodata_dataset")
output_path = Path(r"D:\UBC\510\features")
output_path.mkdir(parents=True, exist_ok=True)

mat_file = data_path / "stim7data_tlgo.mat"

sfreq = 1000  # Hz

ch_names = [f'Ch{i+1}' for i in range(27)]

freq_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha_low': (8, 12),
    'alpha_high': (12, 16),
    'beta': (16, 32),
    'gamma': (32, 45)}


mat_data = sio.loadmat(mat_file)
stim7_hc = mat_data['stim7hceeg'][0]
stim7_pd1 = mat_data['stim7pd1eeg'][0]
stim7_pd2 = mat_data['stim7pd2eeg'][0]

print(f"{mat_file.name}"loaded)
print(f"HC: {len(stim7_hc)} subjects")
print(f"PD Off Med: {len(stim7_pd1)} subjects")
print(f"PD On Med: {len(stim7_pd2)} subjects")


def extract_band_power(signal_data, sfreq, freq_bands):
    nperseg = min(int(2 * sfreq), len(signal_data))  
    noverlap = nperseg // 2 
    
    freqs, psd = welch(signal_data, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
    
    idx_total = np.logical_and(freqs >= 0.5, freqs <= 45)
    total_power = np.trapezoid(psd[idx_total], freqs[idx_total])
    
    band_powers = {}
    for band_name, (low_freq, high_freq) in freq_bands.items():
        
        idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        
        if np.sum(idx_band) == 0:
            # No frequencies in this band (shouldn't happen with correct settings)
            band_powers[f'rel_power_{band_name}'] = 0
            continue
        
        band_power = np.trapezoid(psd[idx_band], freqs[idx_band])
        
        relative_power = band_power / total_power if total_power > 0 else 0
        band_powers[f'rel_power_{band_name}'] = relative_power
    
    return band_powers

def extract_harmonic_parameters(signal_data, sfreq):
   
    nperseg = min(int(2 * sfreq), len(signal_data))
    noverlap = nperseg // 2
    freqs, psd = welch(signal_data, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
    
    idx_relevant = np.logical_and(freqs >= 0.5, freqs <= 45)
    freqs = freqs[idx_relevant]
    psd = psd[idx_relevant]
    
    psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
    
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    
    cumsum_psd = np.cumsum(psd)
    idx_95 = np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0]
    spectral_edge_freq = freqs[idx_95[0]] if len(idx_95) > 0 else freqs[-1]
    
    dominant_freq = freqs[np.argmax(psd)]
    
    spectral_centroid = np.sum(freqs * psd_norm)
    
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd_norm))
    
    harmonic_params = {
        'spectral_entropy': spectral_entropy,
        'spectral_edge_freq': spectral_edge_freq,
        'dominant_freq': dominant_freq,
        'spectral_centroid': spectral_centroid,
        'spectral_spread': spectral_spread
    }
    
    return harmonic_params

def compute_bispectrum(signal_data, sfreq, nfft=256):
    signal_data = signal_data - np.mean(signal_data)
    
    n = len(signal_data)
    fft_signal = np.fft.fft(signal_data, n=nfft)
    
    freq_resolution = sfreq / nfft
    max_freq_idx = int(50 / freq_resolution)
    
    bispectrum = np.zeros((max_freq_idx, max_freq_idx), dtype=complex)
    
    for i in range(max_freq_idx):
        for j in range(i, max_freq_idx):
            if i + j < nfft:
                bispectrum[i, j] = fft_signal[i] * fft_signal[j] * np.conj(fft_signal[i + j])
    
    return bispectrum, freq_resolution

def extract_bispectrum_features(signal_data, sfreq, freq_bands):
    
    bispectrum, freq_resolution = compute_bispectrum(signal_data, sfreq)
    
    bispec_features = {}
    
    for band_name, (low_freq, high_freq) in freq_bands.items():
        low_idx = int(low_freq / freq_resolution)
        high_idx = int(high_freq / freq_resolution)
        
        if high_idx < bispectrum.shape[0]:
            band_bispec = bispectrum[low_idx:high_idx, low_idx:high_idx]
            
            mean_amplitude = np.mean(np.abs(band_bispec))
            mean_phase = np.angle(np.mean(band_bispec))
            
            bispec_features[f'bispec_amp_{band_name}'] = mean_amplitude
            bispec_features[f'bispec_phase_{band_name}'] = mean_phase
        
        else:
            bispec_features[f'bispec_amp_{band_name}'] = 0
            bispec_features[f'bispec_phase_{band_name}'] = 0
    
    return bispec_features

def extract_all_features(signal_data, sfreq, freq_bands):
    features = {}
    
    band_power_features = extract_band_power(signal_data, sfreq, freq_bands)
    features.update(band_power_features)
    
    harmonic_features = extract_harmonic_parameters(signal_data, sfreq)
    features.update(harmonic_features)
    
    bispectrum_features = extract_bispectrum_features(signal_data, sfreq, freq_bands)
    features.update(bispectrum_features)
    
    features['theta_alpha_ratio'] = (features['rel_power_theta'] / 
                                     (features['rel_power_alpha_low'] + features['rel_power_alpha_high'] + 1e-10))
    features['theta_beta_ratio'] = features['rel_power_theta'] / (features['rel_power_beta'] + 1e-10)
    features['alpha_beta_ratio'] = ((features['rel_power_alpha_low'] + features['rel_power_alpha_high']) / 
                                    (features['rel_power_beta'] + 1e-10))
    
    nperseg = min(int(2 * sfreq), len(signal_data))
    noverlap = nperseg // 2
    freqs, psd = welch(signal_data, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
    
    for band_name, (low_freq, high_freq) in freq_bands.items():
        idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        if np.sum(idx_band) > 0:
            abs_power = np.trapz(psd[idx_band], freqs[idx_band])
            features[f'abs_power_{band_name}'] = abs_power
        else:
            features[f'abs_power_{band_name}'] = 0
    
    features['peak_to_peak_amplitude'] = np.ptp(signal_data)
    
    features['line_length'] = np.sum(np.abs(np.diff(signal_data)))
    
    zero_crossings = np.where(np.diff(np.sign(signal_data)))[0]
    features['zero_crossing_rate'] = len(zero_crossings) / len(signal_data)
    
    features['hjorth_activity'] = np.var(signal_data)
    
    diff1 = np.diff(signal_data)
    features['hjorth_mobility'] = np.sqrt(np.var(diff1) / (np.var(signal_data) + 1e-10))
    
    diff2 = np.diff(diff1)
    mobility2 = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10))
    features['hjorth_complexity'] = mobility2 / (features['hjorth_mobility'] + 1e-10)
    
    nperseg = min(int(2 * sfreq), len(signal_data))
    noverlap = nperseg // 2
    freqs, psd = welch(signal_data, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
    
    idx_relevant = np.logical_and(freqs >= 0.5, freqs <= 45)
    psd_relevant = psd[idx_relevant]
    freqs_relevant = freqs[idx_relevant]
    
    geometric_mean = np.exp(np.mean(np.log(psd_relevant + 1e-10)))
    arithmetic_mean = np.mean(psd_relevant)
    features['spectral_flatness'] = geometric_mean / (arithmetic_mean + 1e-10)
    
    cumsum_psd = np.cumsum(psd_relevant)
    rolloff_idx = np.where(cumsum_psd >= 0.85 * cumsum_psd[-1])[0]
    features['spectral_rolloff'] = freqs_relevant[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs_relevant[-1]
    
    features['rms_amplitude'] = np.sqrt(np.mean(signal_data ** 2))
    
    features['kurtosis'] = np.mean((signal_data - np.mean(signal_data)) ** 4) / (np.std(signal_data) ** 4 + 1e-10)
    
    features['skewness'] = np.mean((signal_data - np.mean(signal_data)) ** 3) / (np.std(signal_data) ** 3 + 1e-10)
    
    return features



def process_subject(subject_data, subject_id, group_name):
    n_channels, n_timepoints, n_trials = subject_data.shape
    
    all_features = []
    
    for ch_idx in range(n_channels):
        for trial_idx in range(n_trials):
            signal_data = subject_data[ch_idx, :, trial_idx]
            
            features = extract_all_features(signal_data, sfreq, freq_bands)
            
            features['subject_id'] = subject_id
            features['group'] = group_name
            features['channel'] = ch_names[ch_idx]
            features['channel_idx'] = ch_idx + 1
            features['trial'] = trial_idx + 1
            
            all_features.append(features)
    
    return pd.DataFrame(all_features)


all_dataframes = []

print("\nProcessing Healthy Controls...")
for i, subject_data in enumerate(tqdm(stim7_hc, desc="HC")):
    subject_id = i + 1
    df = process_subject(subject_data, subject_id, 'HC')
    all_dataframes.append(df)

print("\nProcessing PD Off Medication...")
for i, subject_data in enumerate(tqdm(stim7_pd1, desc="PD1")):
    subject_id = i + 1
    df = process_subject(subject_data, subject_id, 'PD1')
    all_dataframes.append(df)

print("\nProcessing PD On Medication...")
for i, subject_data in enumerate(tqdm(stim7_pd2, desc="PD2")):
    subject_id = i + 1
    df = process_subject(subject_data, subject_id, 'PD2')
    all_dataframes.append(df)

print("\nCombining all features...")
full_feature_df = pd.concat(all_dataframes, ignore_index=True)

metadata_cols = ['subject_id', 'group', 'channel', 'channel_idx', 'trial']
feature_cols = [col for col in full_feature_df.columns if col not in metadata_cols]
full_feature_df = full_feature_df[metadata_cols + feature_cols]



full_csv_path = output_path / "eeg_features_all_subjects.csv"
full_feature_df.to_csv(full_csv_path, index=False)
print(f"✓ Saved complete feature set to: {full_csv_path}")

for group_name in ['HC', 'PD1', 'PD2']:
    group_df = full_feature_df[full_feature_df['group'] == group_name]
    group_csv_path = output_path / f"eeg_features_{group_name}.csv"
    group_df.to_csv(group_csv_path, index=False)
    print(f"✓ Saved {group_name} features to: {group_csv_path}")



print(f"\nDataset Shape: {full_feature_df.shape}")
print(f"  - Total rows: {len(full_feature_df):,}")
print(f"  - Total features: {len(feature_cols)}")

print(f"\nBreakdown by group:")
for group_name in ['HC', 'PD1', 'PD2']:
    group_df = full_feature_df[full_feature_df['group'] == group_name]
    n_subjects = group_df['subject_id'].nunique()
    n_rows = len(group_df)
    print(f"  {group_name}: {n_subjects} subjects × 27 channels × 10 trials = {n_rows} rows")

print(f"\nExtracted Features ({len(feature_cols)} total):")
print("\nSpectral Power Features (12):")
spectral_power_features = [f for f in feature_cols if 'power' in f]
for i, feat in enumerate(spectral_power_features, 1):
    print(f"  {i}. {feat}")

print("\nHarmonic Parameters (5):")
harmonic_features = ['spectral_entropy', 'spectral_edge_freq', 'dominant_freq', 
                     'spectral_centroid', 'spectral_spread']
for i, feat in enumerate(harmonic_features, 1):
    print(f"  {i}. {feat}")

print("\nBispectrum Features (12):")
bispectrum_features = [f for f in feature_cols if 'bispec' in f]
for i, feat in enumerate(bispectrum_features, 1):
    print(f"  {i}. {feat}")

print("\nBand Ratios (3):")
ratio_features = [f for f in feature_cols if 'ratio' in f]
for i, feat in enumerate(ratio_features, 1):
    print(f"  {i}. {feat}")

print("\nTemporal Features (6):")
temporal_features = ['peak_to_peak_amplitude', 'line_length', 'zero_crossing_rate',
                     'rms_amplitude', 'kurtosis', 'skewness']
for i, feat in enumerate(temporal_features, 1):
    print(f"  {i}. {feat}")

print("\nHjorth Parameters (3):")
hjorth_features = ['hjorth_activity', 'hjorth_mobility', 'hjorth_complexity']
for i, feat in enumerate(hjorth_features, 1):
    print(f"  {i}. {feat}")

print("\nAdditional Spectral Features (3):")
additional_features = ['spectral_flatness', 'spectral_rolloff']
for i, feat in enumerate(additional_features, 1):
    print(f"  {i}. {feat}")

print("feature extration completed")



key_features = ['rel_power_alpha_low', 'rel_power_beta', 'spectral_entropy', 
                'dominant_freq', 'hjorth_mobility']

print("\nSummary statistics for selected features:")
summary_stats = full_feature_df.groupby('group')[key_features].agg(['mean', 'std'])
print(summary_stats.round(4))

summary_path = output_path / "feature_summary_statistics.csv"
summary_stats.to_csv(summary_path)
print(f"\n statistics summary saved to: {summary_path}")
