import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats
from scipy.signal import welch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")


data_path = Path(r"D:\UBC\510\neurodata_dataset")
output_path = Path(r"D:\UBC\510\features")
figure_path = Path(r"D:\UBC\510\Figures\prestim_poststim_analysis")
figure_path.mkdir(parents=True, exist_ok=True)

mat_file = data_path / "stim7data_tlgo.mat"
sfreq = 1000  
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

print(f"{mat_file.name} loaded")
print(f"HC: {len(stim7_hc)} subjects")
print(f"PD OFF: {len(stim7_pd1)} subjects")
print(f"PD ON: {len(stim7_pd2)} subjects")

def extract_band_power(signal_data, sfreq, freq_bands):
    nperseg = min(int(2 * sfreq), len(signal_data))
    noverlap = nperseg // 2
    freqs, psd = welch(signal_data, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
    
    idx_total = np.logical_and(freqs >= 0.5, freqs <= 45)
    total_power = np.trapz(psd[idx_total], freqs[idx_total])
    
    band_powers = {}
    for band_name, (low_freq, high_freq) in freq_bands.items():
        idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        if np.sum(idx_band) == 0:
            band_powers[f'rel_power_{band_name}'] = 0
            continue
        band_power = np.trapz(psd[idx_band], freqs[idx_band])
        relative_power = band_power / total_power if total_power > 0 else 0
        band_powers[f'rel_power_{band_name}'] = relative_power
    
    return band_powers

def extract_simple_features(signal_data, sfreq, freq_bands):
    features = {}
    
    band_power_features = extract_band_power(signal_data, sfreq, freq_bands)
    features.update(band_power_features)
    
    features['mean_amplitude'] = np.mean(signal_data)
    features['std_amplitude'] = np.std(signal_data)
    features['peak_to_peak'] = np.ptp(signal_data)
    features['rms'] = np.sqrt(np.mean(signal_data ** 2))
    
    zero_crossings = np.where(np.diff(np.sign(signal_data)))[0]
    features['zero_crossing_rate'] = len(zero_crossings) / len(signal_data)
    
    features['hjorth_activity'] = np.var(signal_data)
    diff1 = np.diff(signal_data)
    features['hjorth_mobility'] = np.sqrt(np.var(diff1) / (np.var(signal_data) + 1e-10))
    
    return features


def process_subject_prestim_poststim(subject_data, subject_id, group_name):
    n_channels, n_timepoints, n_trials = subject_data.shape
    
    all_features = []
    
    pre_stim_idx = slice(0, 1000)  
    post_stim_idx = slice(1000, 2000)  
    
    for ch_idx in range(n_channels):
        for trial_idx in range(n_trials):
            full_signal = subject_data[ch_idx, :, trial_idx]
            
            pre_signal = full_signal[pre_stim_idx]
            post_signal = full_signal[post_stim_idx]
            
            pre_features = extract_simple_features(pre_signal, sfreq, freq_bands)
            pre_features = {f'{k}_pre': v for k, v in pre_features.items()}
            
            post_features = extract_simple_features(post_signal, sfreq, freq_bands)
            post_features = {f'{k}_post': v for k, v in post_features.items()}
            
            features = {**pre_features, **post_features}
            
            features['subject_id'] = subject_id
            features['group'] = group_name
            features['channel'] = ch_names[ch_idx]
            features['channel_idx'] = ch_idx + 1
            features['trial'] = trial_idx + 1
            
            all_features.append(features)
    
    return pd.DataFrame(all_features)


all_dataframes = []
datasets = [
    (stim7_hc, 'HC'),
    (stim7_pd1, 'PD1'),
    (stim7_pd2, 'PD2')]

for data, group_name in datasets:
    print(f"\nProcessing {group_name}...")
    for i, subject_data in enumerate(tqdm(data, desc=group_name)):
        subject_id = i + 1
        df = process_subject_prestim_poststim(subject_data, subject_id, group_name)
        all_dataframes.append(df)

full_df = pd.concat(all_dataframes, ignore_index=True)

prestim_poststim_csv = output_path / "eeg_features_prestim_poststim.csv"
full_df.to_csv(prestim_poststim_csv, index=False)
print(f"Saved features to: {prestim_poststim_csv}")
print(f"\nDataset shape: {full_df.shape}")

base_features = ['rel_power_delta', 'rel_power_theta', 'rel_power_alpha_low',
                 'rel_power_alpha_high', 'rel_power_beta', 'rel_power_gamma',
                 'mean_amplitude', 'std_amplitude', 'peak_to_peak', 'rms',
                 'zero_crossing_rate', 'hjorth_activity', 'hjorth_mobility']

for feat in base_features:
    full_df[f'{feat}_delta'] = full_df[f'{feat}_post'] - full_df[f'{feat}_pre']
    full_df[f'{feat}_percent_change'] = ((full_df[f'{feat}_post'] - full_df[f'{feat}_pre']) / 
                                          (full_df[f'{feat}_pre'].abs() + 1e-10) * 100)




power_features = ['rel_power_delta', 'rel_power_theta', 'rel_power_alpha_low',
                  'rel_power_alpha_high', 'rel_power_beta', 'rel_power_gamma']

subject_avg = full_df.groupby(['group', 'subject_id']).agg({
    **{f'{feat}_pre': 'mean' for feat in power_features},
    **{f'{feat}_post': 'mean' for feat in power_features}
}).reset_index()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, power_feat in enumerate(power_features):
    ax = axes[i]
    
    plot_data = []
    for group in ['HC', 'PD1', 'PD2']:
        group_data = subject_avg[subject_avg['group'] == group]
        
        for _, row in group_data.iterrows():
            plot_data.append({
                'Group': group,
                'Period': 'Pre-stimulus',
                'Power': row[f'{power_feat}_pre']})
            plot_data.append({
                'Group': group,
                'Period': 'Post-stimulus',
                'Power': row[f'{power_feat}_post']})
    
    plot_df = pd.DataFrame(plot_data)
    
    sns.boxplot(data=plot_df, x='Group', y='Power', hue='Period', ax=ax,
                palette=['lightblue', 'lightcoral'])
    
    band_name = power_feat.replace('rel_power_', '').replace('_', '-').title()
    p_values = []
    
    for group in ['HC', 'PD1', 'PD2']:
        group_data = subject_avg[subject_avg['group'] == group]
        pre = group_data[f'{power_feat}_pre']
        post = group_data[f'{power_feat}_post']
        _, p = stats.ttest_rel(pre, post)
        p_values.append(p)
    
    ax.set_title(f'{band_name} Band\nPaired t-test p: HC={p_values[0]:.3f}, PD1={p_values[1]:.3f}, PD2={p_values[2]:.3f}',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Group', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Power', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(title='Period', fontsize=9)

plt.tight_layout()
plt.savefig(figure_path / 'pre_vs_post_spectral_power.png', dpi=300, bbox_inches='tight')
print(f"Saved: pre_vs_post_spectral_power.png")
plt.show()



subject_delta = full_df.groupby(['group', 'subject_id']).agg({
    f'{feat}_delta': 'mean' for feat in power_features
}).reset_index()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, power_feat in enumerate(power_features):
    ax = axes[i]
    
    delta_feat = f'{power_feat}_delta'
    
    sns.violinplot(data=subject_delta, x='group', y=delta_feat, ax=ax,
                   order=['HC', 'PD1', 'PD2'], palette=['skyblue', 'salmon', 'lightgreen'])
    
    sns.stripplot(data=subject_delta, x='group', y=delta_feat, ax=ax,
                  order=['HC', 'PD1', 'PD2'], color='black', alpha=0.4, size=4)
    
    ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    hc_delta = subject_delta[subject_delta['group'] == 'HC'][delta_feat]
    pd1_delta = subject_delta[subject_delta['group'] == 'PD1'][delta_feat]
    pd2_delta = subject_delta[subject_delta['group'] == 'PD2'][delta_feat]
    
    f_stat, p_val = stats.f_oneway(hc_delta, pd1_delta, pd2_delta)
    
    _, p_hc = stats.ttest_1samp(hc_delta, 0)
    _, p_pd1 = stats.ttest_1samp(pd1_delta, 0)
    _, p_pd2 = stats.ttest_1samp(pd2_delta, 0)
    
    band_name = power_feat.replace('rel_power_', '').replace('_', '-').title()
    ax.set_title(f'{band_name} Band - Stimulus Effect\nANOVA p={p_val:.3f} | vs Zero: HC={p_hc:.3f}, PD1={p_pd1:.3f}, PD2={p_pd2:.3f}',
                 fontsize=9, fontweight='bold')
    ax.set_xlabel('Group', fontsize=11, fontweight='bold')
    ax.set_ylabel('Change (Post - Pre)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(figure_path / 'stimulus_induced_changes_delta.png', dpi=300, bbox_inches='tight')
print(f"Saved: stimulus_induced_changes_delta.png")
plt.show()



subject_pct = full_df.groupby(['group', 'subject_id']).agg({
    f'{feat}_percent_change': 'mean' for feat in power_features
}).reset_index()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, power_feat in enumerate(power_features):
    ax = axes[i]
    
    pct_feat = f'{power_feat}_percent_change'
    
    sns.boxplot(data=subject_pct, x='group', y=pct_feat, ax=ax,
                order=['HC', 'PD1', 'PD2'], palette=['skyblue', 'salmon', 'lightgreen'])
    sns.swarmplot(data=subject_pct, x='group', y=pct_feat, ax=ax,
                  order=['HC', 'PD1', 'PD2'], color='black', alpha=0.4, size=4)
    
    ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    hc_pct = subject_pct[subject_pct['group'] == 'HC'][pct_feat]
    pd1_pct = subject_pct[subject_pct['group'] == 'PD1'][pct_feat]
    pd2_pct = subject_pct[subject_pct['group'] == 'PD2'][pct_feat]
    
    f_stat, p_val = stats.f_oneway(hc_pct, pd1_pct, pd2_pct)
    
    band_name = power_feat.replace('rel_power_', '').replace('_', '-').title()
    ax.set_title(f'{band_name} Band\n% Change (ANOVA p={p_val:.3f})',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Group', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percent Change (%)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(figure_path / 'stimulus_percent_change.png', dpi=300, bbox_inches='tight')
print(f"Saved: stimulus_percent_change.png")
plt.show()



channel_delta = full_df.groupby(['group', 'channel_idx']).agg({
    f'{feat}_delta': 'mean' for feat in power_features
}).reset_index()

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

for idx, group in enumerate(['HC', 'PD1', 'PD2']):
    group_data = channel_delta[channel_delta['group'] == group]
    
    matrix = np.zeros((27, len(power_features)))
    for i, ch_idx in enumerate(range(1, 28)):
        ch_data = group_data[group_data['channel_idx'] == ch_idx]
        if len(ch_data) > 0:
            for j, feat in enumerate(power_features):
                matrix[i, j] = ch_data[f'{feat}_delta'].values[0]
    
    ax = axes[idx]
    band_labels = [f.replace('rel_power_', '').replace('_', '-').upper() for f in power_features]
    
    im = sns.heatmap(matrix, ax=ax, cmap='RdBu_r', center=0, 
                     xticklabels=band_labels, yticklabels=range(1, 28),
                     cbar_kws={'label': 'Change (Post - Pre)'}, vmin=-0.05, vmax=0.05)
    
    ax.set_title(f'{group} - Stimulus-Induced Changes by Channel', fontsize=14, fontweight='bold')
    ax.set_xlabel('Frequency Band', fontsize=12, fontweight='bold')
    ax.set_ylabel('Channel', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(figure_path / 'channel_wise_change_heatmap.png', dpi=300, bbox_inches='tight')
print(f"Saved: channel_wise_change_heatmap.png")
plt.show()


def cohens_d_paired(pre, post):
    diff = post - pre
    return np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)

effect_sizes = []

for group in ['HC', 'PD1', 'PD2']:
    group_data = subject_avg[subject_avg['group'] == group]
    
    for feat in power_features:
        pre = group_data[f'{feat}_pre'].values
        post = group_data[f'{feat}_post'].values
        
        d = cohens_d_paired(pre, post)
        
        effect_sizes.append({
            'Group': group,
            'Feature': feat.replace('rel_power_', '').replace('_', '-').title(),
            'Effect Size': d})

effect_df = pd.DataFrame(effect_sizes)

fig, ax = plt.subplots(figsize=(14, 8))

pivot_df = effect_df.pivot(index='Feature', columns='Group', values='Effect Size')

pivot_df.plot(kind='bar', ax=ax, color=['skyblue', 'salmon', 'lightgreen'],
              edgecolor='black', alpha=0.8, width=0.8)

ax.axhline(0.2, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax.axhline(-0.2, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax.axhline(0.5, color='orange', linestyle=':', alpha=0.5, linewidth=1)
ax.axhline(-0.5, color='orange', linestyle=':', alpha=0.5, linewidth=1)
ax.axhline(0, color='black', linestyle='-', alpha=0.7, linewidth=2)

ax.set_xlabel('Frequency Band', fontsize=12, fontweight='bold')
ax.set_ylabel("Cohen's d (Effect Size)", fontsize=12, fontweight='bold')
ax.set_title("Stimulus Response Effect Size by Group\n(Pre vs Post Stimulus)\nSmall: ±0.2, Medium: ±0.5",
             fontsize=14, fontweight='bold')
ax.legend(title='Group', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(figure_path / 'stimulus_response_effect_size.png', dpi=300, bbox_inches='tight')
print(f"Saved: stimulus_response_effect_size.png")
plt.show()



subject_magnitude = subject_delta.copy()
for feat in power_features:
    subject_magnitude[f'{feat}_magnitude'] = np.abs(subject_magnitude[f'{feat}_delta'])

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, power_feat in enumerate(power_features):
    ax = axes[i]
    mag_feat = f'{power_feat}_magnitude'
    
    sns.boxplot(data=subject_magnitude, x='group', y=mag_feat, ax=ax,
                order=['HC', 'PD1', 'PD2'], palette=['skyblue', 'salmon', 'lightgreen'])
    sns.swarmplot(data=subject_magnitude, x='group', y=mag_feat, ax=ax,
                  order=['HC', 'PD1', 'PD2'], color='black', alpha=0.4, size=4)
    
    hc_mag = subject_magnitude[subject_magnitude['group'] == 'HC'][mag_feat]
    pd1_mag = subject_magnitude[subject_magnitude['group'] == 'PD1'][mag_feat]
    pd2_mag = subject_magnitude[subject_magnitude['group'] == 'PD2'][mag_feat]
    
    f_stat, p_val = stats.f_oneway(hc_mag, pd1_mag, pd2_mag)
    
    band_name = power_feat.replace('rel_power_', '').replace('_', '-').title()
    ax.set_title(f'{band_name} - Response Magnitude\n(ANOVA p={p_val:.3f})',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Group', fontsize=11, fontweight='bold')
    ax.set_ylabel('Absolute Change Magnitude', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(figure_path / 'response_magnitude_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: response_magnitude_comparison.png")
plt.show()



pd_only = subject_delta[subject_delta['group'].isin(['PD1', 'PD2'])]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, power_feat in enumerate(power_features):
    ax = axes[i]
    delta_feat = f'{power_feat}_delta'
    
    sns.violinplot(data=pd_only, x='group', y=delta_feat, ax=ax,
                   order=['PD1', 'PD2'], palette=['salmon', 'lightgreen'])
    sns.stripplot(data=pd_only, x='group', y=delta_feat, ax=ax,
                  order=['PD1', 'PD2'], color='black', alpha=0.4, size=5)
    
    ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    pd1_delta = pd_only[pd_only['group'] == 'PD1'][delta_feat]
    pd2_delta = pd_only[pd_only['group'] == 'PD2'][delta_feat]
    
    t_stat, p_val = stats.ttest_ind(pd1_delta, pd2_delta)
    
    pd1_mean = pd1_delta.mean()
    pd2_mean = pd2_delta.mean()
    
    band_name = power_feat.replace('rel_power_', '').replace('_', '-').title()
    ax.set_title(f'{band_name} - Medication Effect\nt-test p={p_val:.3f} | PD1: {pd1_mean:.4f}, PD2: {pd2_mean:.4f}',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Group', fontsize=11, fontweight='bold')
    ax.set_ylabel('Stimulus Response (Post - Pre)', fontsize=11, fontweight='bold')
    ax.set_xticklabels(['PD Off Med', 'PD On Med'])
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(figure_path / 'medication_effect_pd1_vs_pd2.png', dpi=300, bbox_inches='tight')
print(f"Saved: medication_effect_pd1_vs_pd2.png")
plt.show()


response_direction = []

for group in ['HC', 'PD1', 'PD2']:
    group_data = subject_delta[subject_delta['group'] == group]
    
    for feat in power_features:
        delta_feat = f'{feat}_delta'
        
        n_increase = np.sum(group_data[delta_feat] > 0)
        n_decrease = np.sum(group_data[delta_feat] < 0)
        n_no_change = np.sum(group_data[delta_feat] == 0)
        total = len(group_data)
        
        response_direction.append({
            'Group': group,
            'Band': feat.replace('rel_power_', '').replace('_', '-').title(),
            'Increase': n_increase / total * 100,
            'Decrease': n_decrease / total * 100,
            'No Change': n_no_change / total * 100})

direction_df = pd.DataFrame(response_direction)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, group in enumerate(['HC', 'PD1', 'PD2']):
    ax = axes[idx]
    group_data = direction_df[direction_df['Group'] == group]
    
    bands = group_data['Band'].values
    increase = group_data['Increase'].values
    decrease = group_data['Decrease'].values
    
    x = np.arange(len(bands))
    width = 0.6
    
    ax.bar(x, increase, width, label='Increase', color='green', alpha=0.7, edgecolor='black')
    ax.bar(x, -decrease, width, label='Decrease', color='red', alpha=0.7, edgecolor='black')
    
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(bands, rotation=45, ha='right')
    ax.set_ylabel('Percentage of Subjects (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'{group} - Response Direction', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-100, 100)

plt.tight_layout()
plt.savefig(figure_path / 'response_direction_analysis.png', dpi=300, bbox_inches='tight')
print(f"Saved: response_direction_analysis.png")
plt.show()
