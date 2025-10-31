import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import sklearn
print(sklearn.__version__)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
sns.set_palette("husl")

figure_path = Path(r"D:\UBC\510\Figures\feature_analysis")
figure_path.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(feature_path / "eeg_features_all_subjects.csv")
print(f"features: {df.shape} loaded")
print(f"Subjects: {df['subject_id'].nunique()}")
print(f"Groups: {df['group'].unique()}")
print(f"Channels: {df['channel'].nunique()}")
print(f"Trials: {df['trial'].nunique()}")

metadata_cols = ['subject_id', 'group', 'channel', 'channel_idx', 'trial']
feature_cols = [col for col in df.columns if col not in metadata_cols]
print(f"  - Features: {len(feature_cols)}")



power_features = ['rel_power_delta', 'rel_power_theta', 'rel_power_alpha_low', 
                  'rel_power_alpha_high', 'rel_power_beta', 'rel_power_gamma']

subject_power = df.groupby(['group', 'subject_id'])[power_features].mean().reset_index()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, power_feat in enumerate(power_features):
    ax = axes[i]
    
    sns.violinplot(data=subject_power, x='group', y=power_feat, ax=ax, 
                   order=['HC', 'PD1', 'PD2'], palette=['skyblue', 'salmon', 'lightgreen'])
    
    sns.stripplot(data=subject_power, x='group', y=power_feat, ax=ax,
                  order=['HC', 'PD1', 'PD2'], color='black', alpha=0.3, size=4)
    
    hc_vals = subject_power[subject_power['group'] == 'HC'][power_feat]
    pd1_vals = subject_power[subject_power['group'] == 'PD1'][power_feat]
    pd2_vals = subject_power[subject_power['group'] == 'PD2'][power_feat]
    
    f_stat, p_val = stats.f_oneway(hc_vals, pd1_vals, pd2_vals)
    
    band_name = power_feat.replace('rel_power_', '').replace('_', '-').title()
    ax.set_title(f'{band_name} Band\nANOVA p={p_val:.4f}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Group', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Power', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(figure_path / 'spectral_power_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: spectral_power_comparison.png")
plt.show()


channel_power = df.groupby(['group', 'channel_idx'])[power_features].mean().reset_index()

fig, axes = plt.subplots(3, 6, figsize=(20, 10))

for row, group in enumerate(['HC', 'PD1', 'PD2']):
    group_data = channel_power[channel_power['group'] == group]
    
    for col, power_feat in enumerate(power_features):
        ax = axes[row, col]
        
        channels = group_data['channel_idx'].values
        powers = group_data[power_feat].values
        
        bars = ax.bar(channels, powers, color=f'C{col}', alpha=0.7, edgecolor='black')
        
        max_idx = np.argmax(powers)
        min_idx = np.argmin(powers)
        bars[max_idx].set_color('red')
        bars[max_idx].set_alpha(1.0)
        bars[min_idx].set_color('blue')
        bars[min_idx].set_alpha(1.0)
        
        band_name = power_feat.replace('rel_power_', '').replace('_', '-').upper()
        if row == 0:
            ax.set_title(f'{band_name}', fontsize=11, fontweight='bold')
        
        if col == 0:
            ax.set_ylabel(f'{group}\nPower', fontsize=10, fontweight='bold')
        
        if row == 2:
            ax.set_xlabel('Channel', fontsize=9)
        
        ax.set_xticks([1, 7, 14, 21, 27])
        ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(figure_path / 'topographic_power_maps.png', dpi=300, bbox_inches='tight')
print(f"Saved: topographic_power_maps.png")
plt.show()


key_features = power_features + ['spectral_entropy', 'dominant_freq', 'spectral_centroid',
                                  'hjorth_mobility', 'hjorth_complexity', 'theta_alpha_ratio',
                                  'alpha_beta_ratio', 'peak_to_peak_amplitude', 'rms_amplitude']

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for i, group in enumerate(['HC', 'PD1', 'PD2']):
    group_data = df[df['group'] == group][key_features]
    
    corr_matrix = group_data.corr()
    
    sns.heatmap(corr_matrix, ax=axes[i], cmap='RdBu_r', center=0, 
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={'label': 'Correlation'})
    
    axes[i].set_title(f'{group} - Feature Correlations', fontsize=14, fontweight='bold')
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right', fontsize=8)
    axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0, fontsize=8)

plt.tight_layout()
plt.savefig(figure_path / 'feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print(f"Saved: feature_correlation_heatmap.png")
plt.show()


subject_features = df.groupby(['group', 'subject_id'])[feature_cols].mean().reset_index()

X = subject_features[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'PC3': X_pca[:, 2],
    'group': subject_features['group'].values,
    'subject_id': subject_features['subject_id'].values})

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax1 = axes[0]
for group, color in [('HC', 'blue'), ('PD1', 'red'), ('PD2', 'green')]:
    group_data = pca_df[pca_df['group'] == group]
    ax1.scatter(group_data['PC1'], group_data['PC2'], 
                c=color, label=group, s=100, alpha=0.6, edgecolors='black')

ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12, fontweight='bold')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12, fontweight='bold')
ax1.set_title('PCA: Subject Clustering (PC1 vs PC2)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
for group, color in [('HC', 'blue'), ('PD1', 'red'), ('PD2', 'green')]:
    group_data = pca_df[pca_df['group'] == group]
    ax2.scatter(group_data['PC1'], group_data['PC3'], 
                c=color, label=group, s=100, alpha=0.6, edgecolors='black')

ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12, fontweight='bold')
ax2.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)', fontsize=12, fontweight='bold')
ax2.set_title('PCA: Subject Clustering (PC1 vs PC3)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figure_path / 'pca_subject_clustering.png', dpi=300, bbox_inches='tight')
print(f"Saved: pca_subject_clustering.png")
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
n_components = min(10, len(pca.explained_variance_ratio_))
pca_full = PCA(n_components=n_components)
pca_full.fit(X_scaled)

ax.bar(range(1, n_components+1), pca_full.explained_variance_ratio_ * 100, 
       alpha=0.7, color='steelblue', edgecolor='black')
ax.plot(range(1, n_components+1), np.cumsum(pca_full.explained_variance_ratio_ * 100),
        'ro-', linewidth=2, markersize=8, label='Cumulative')

ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
ax.set_ylabel('Explained Variance (%)', fontsize=12, fontweight='bold')
ax.set_title('PCA Explained Variance', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(figure_path / 'pca_explained_variance.png', dpi=300, bbox_inches='tight')
print(f"Saved: pca_explained_variance.png")
plt.show()


tsne = TSNE(n_components=2, random_state=42, perplexity=15)
X_tsne = tsne.fit_transform(X_scaled)

tsne_df = pd.DataFrame({
    'tsne1': X_tsne[:, 0],
    'tsne2': X_tsne[:, 1],
    'group': subject_features['group'].values,
    'subject_id': subject_features['subject_id'].values})

fig, ax = plt.subplots(figsize=(10, 8))

for group, color in [('HC', 'blue'), ('PD1', 'red'), ('PD2', 'green')]:
    group_data = tsne_df[tsne_df['group'] == group]
    ax.scatter(group_data['tsne1'], group_data['tsne2'], 
               c=color, label=group, s=150, alpha=0.6, edgecolors='black', linewidth=1.5)
    
    for _, row in group_data.iterrows():
        ax.annotate(f"{int(row['subject_id'])}", 
                   (row['tsne1'], row['tsne2']),
                   fontsize=8, ha='center', va='center')

ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
ax.set_title('t-SNE: Subject Clustering (Non-linear)', fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figure_path / 'tsne_subject_clustering.png', dpi=300, bbox_inches='tight')
print(f"Saved: tsne_subject_clustering.png")
plt.show()



fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, group in enumerate(['HC', 'PD1', 'PD2']):
    group_subjects = subject_features[subject_features['group'] == group]
    
    cv_values = []
    feature_names_short = []
    
    for feat in power_features + ['spectral_entropy', 'dominant_freq', 'hjorth_mobility']:
        mean_val = group_subjects[feat].mean()
        std_val = group_subjects[feat].std()
        cv = (std_val / mean_val * 100) if mean_val != 0 else 0
        cv_values.append(cv)
        feature_names_short.append(feat.replace('rel_power_', '').replace('_', '-'))
    
    bars = axes[i].barh(feature_names_short, cv_values, color=f'C{i}', alpha=0.7, edgecolor='black')
    axes[i].set_xlabel('Coefficient of Variation (%)', fontsize=11, fontweight='bold')
    axes[i].set_title(f'{group}\nInter-Subject Variability', fontsize=12, fontweight='bold')
    axes[i].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(figure_path / 'inter_subject_variability.png', dpi=300, bbox_inches='tight')
print(f"Saved: inter_subject_variability.png")
plt.show()


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std != 0 else 0

analysis_features = power_features + ['spectral_entropy', 'dominant_freq', 'spectral_centroid',
                                       'hjorth_mobility', 'theta_alpha_ratio', 'alpha_beta_ratio']

effect_sizes = []

for feat in analysis_features:
    hc_vals = subject_features[subject_features['group'] == 'HC'][feat].values
    pd1_vals = subject_features[subject_features['group'] == 'PD1'][feat].values
    pd2_vals = subject_features[subject_features['group'] == 'PD2'][feat].values
    
    d_hc_pd1 = cohens_d(hc_vals, pd1_vals)
    d_hc_pd2 = cohens_d(hc_vals, pd2_vals)
    
    effect_sizes.append({
        'feature': feat.replace('rel_power_', '').replace('_', ' ').title(),
        'HC vs PD1': d_hc_pd1,
        'HC vs PD2': d_hc_pd2})

effect_df = pd.DataFrame(effect_sizes)

fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(effect_df))
width = 0.35

bars1 = ax.barh(x - width/2, effect_df['HC vs PD1'], width, label='HC vs PD Off Med', 
                color='salmon', alpha=0.8, edgecolor='black')
bars2 = ax.barh(x + width/2, effect_df['HC vs PD2'], width, label='HC vs PD On Med', 
                color='lightgreen', alpha=0.8, edgecolor='black')

ax.axvline(0.2, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(-0.2, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(-0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(0.8, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(-0.8, color='red', linestyle='--', alpha=0.5, linewidth=1)

ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12, fontweight='bold')
ax.set_title("Feature Effect Sizes Between Groups\n(Small: 0.2, Medium: 0.5, Large: 0.8)", 
             fontsize=14, fontweight='bold')
ax.set_yticks(x)
ax.set_yticklabels(effect_df['feature'])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(figure_path / 'feature_effect_sizes.png', dpi=300, bbox_inches='tight')
print(f"Saved: feature_effect_sizes.png")
plt.show()


fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

key_features_plot = ['spectral_entropy', 'dominant_freq', 'spectral_centroid',
                     'hjorth_mobility', 'hjorth_complexity', 'theta_alpha_ratio',
                     'alpha_beta_ratio', 'peak_to_peak_amplitude', 'rms_amplitude']

for i, feat in enumerate(key_features_plot):
    ax = axes[i]
    
    sns.boxplot(data=subject_features, x='group', y=feat, ax=ax,
                order=['HC', 'PD1', 'PD2'], palette=['skyblue', 'salmon', 'lightgreen'])
    sns.swarmplot(data=subject_features, x='group', y=feat, ax=ax,
                  order=['HC', 'PD1', 'PD2'], color='black', alpha=0.5, size=3)
    
    hc_vals = subject_features[subject_features['group'] == 'HC'][feat]
    pd1_vals = subject_features[subject_features['group'] == 'PD1'][feat]
    pd2_vals = subject_features[subject_features['group'] == 'PD2'][feat]
    
    f_stat, p_val = stats.f_oneway(hc_vals, pd1_vals, pd2_vals)
    
    feat_name = feat.replace('_', ' ').title()
    ax.set_title(f'{feat_name}\np={p_val:.4f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Group', fontsize=10, fontweight='bold')
    ax.set_ylabel(feat_name, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(figure_path / 'detailed_feature_distributions.png', dpi=300, bbox_inches='tight')
print(f"Saved: detailed_feature_distributions.png")
plt.show()



from sklearn.preprocessing import MinMaxScaler

radar_features = power_features
scaler = MinMaxScaler()

radar_data = {}
for group in ['HC', 'PD1', 'PD2']:
    group_means = subject_features[subject_features['group'] == group][radar_features].mean()
    radar_data[group] = group_means.values

all_values = np.array([radar_data['HC'], radar_data['PD1'], radar_data['PD2']])
all_values_norm = scaler.fit_transform(all_values.T).T

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))  

for i, (group, color) in enumerate([('HC', 'blue'), ('PD1', 'red'), ('PD2', 'green')]):
    values = np.concatenate((all_values_norm[i], [all_values_norm[i][0]]))
    ax.plot(angles, values, 'o-', linewidth=2, label=group, color=color)
    ax.fill(angles, values, alpha=0.15, color=color)

ax.set_xticks(angles[:-1])
labels = [f.replace('rel_power_', '').replace('_', '-').upper() for f in radar_features]
ax.set_xticklabels(labels, size=11)
ax.set_ylim(0, 1)
ax.set_title('Spectral Power Profile Comparison\n(Normalized)', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.grid(True)

plt.tight_layout()
plt.savefig(figure_path / 'radar_chart_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: radar_chart_comparison.png")
plt.show()
