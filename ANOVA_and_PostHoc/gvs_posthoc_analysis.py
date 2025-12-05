import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy import stats
import os
import pingouin as pg

# Create output directory
output_dir = r'D:\UBC\510\Figures\PostHoc_Analysis'
os.makedirs(output_dir, exist_ok=True)

# Load data files
sham = loadmat('task_gvssham.mat')
gvs7 = loadmat('task_gvsstim7.mat')
gvs8 = loadmat('task_gvsstim8.mat')

# Define metric indices
metrics = {
    'Grip Strength': 0,
    'Strength Velocity': 4,
    'Movement Time': 6,
    'Squeeze Time': 7,
    'Peak Time': 11,
    'Reaction Time': 12
}

# Define subjects to exclude
exclude_hc = [7, 8, 9, 13]
exclude_pd = [2, 6, 7, 11]

# Define groups and their data
groups = {
    'HC': {
        'sham': sham['hcoffmed'][0],
        'low_gvs': gvs7['hcoffmed'][0],
        'high_gvs': gvs8['hcoffmed_gvs8'][0],
        'n_subjects': 22,
        'exclude': exclude_hc
    },
    'PD_ON': {
        'sham': sham['pdonmed'][0],
        'low_gvs': gvs7['pdonmed'][0],
        'high_gvs': gvs8['pdonmed_gvs8'][0],
        'n_subjects': 20,
        'exclude': exclude_pd
    },
    'PD_OFF': {
        'sham': sham['pdoffmed'][0],
        'low_gvs': gvs7['pdoffmed'][0],
        'high_gvs': gvs8['pdoffmed_gvs8'][0],
        'n_subjects': 20,
        'exclude': exclude_pd
    }
}

# Verify data loaded correctly
print("=" * 80)
print("DATA VERIFICATION")
print("=" * 80)
print("\nVariable names in files:")
print(f"  task_gvssham.mat: {[k for k in sham.keys() if not k.startswith('__')]}")
print(f"  task_gvsstim7.mat: {[k for k in gvs7.keys() if not k.startswith('__')]}")
print(f"  task_gvsstim8.mat: {[k for k in gvs8.keys() if not k.startswith('__')]}")

print("\nGroup data structure check:")
for group_name, group_data in groups.items():
    print(f"\n  {group_name}:")
    print(f"    sham shape: {len(group_data['sham'])}")
    print(f"    low_gvs shape: {len(group_data['low_gvs'])}")
    print(f"    high_gvs shape: {len(group_data['high_gvs'])}")
    print(f"    n_subjects: {group_data['n_subjects']}")
    print(f"    exclude: {group_data['exclude']}")

print("\n" + "=" * 80)

def create_long_format_data(groups, metric_idx):
    """Create long-format dataframe for analysis"""
    all_data = []
    subject_counter = 0
    
    for group_name in ['HC', 'PD_ON', 'PD_OFF']:
        group_data = groups[group_name]
        
        for subj_idx in range(group_data['n_subjects']):
            if subj_idx in group_data['exclude']:
                continue
            
            sham_mean = np.nanmean(group_data['sham'][subj_idx][:, metric_idx])
            low_mean = np.nanmean(group_data['low_gvs'][subj_idx][:, metric_idx])
            high_mean = np.nanmean(group_data['high_gvs'][subj_idx][:, metric_idx])
            
            if np.isnan(sham_mean) or np.isnan(low_mean) or np.isnan(high_mean):
                continue
            
            all_data.append({
                'Subject': subject_counter,
                'Group': group_name,
                'Stimulation': 'Sham',
                'Value': sham_mean
            })
            all_data.append({
                'Subject': subject_counter,
                'Group': group_name,
                'Stimulation': 'Low',
                'Value': low_mean
            })
            all_data.append({
                'Subject': subject_counter,
                'Group': group_name,
                'Stimulation': 'High',
                'Value': high_mean
            })
            
            subject_counter += 1
    
    return pd.DataFrame(all_data)

def perform_pairwise_tests_parametric(df, group_name=None):
    """Perform pairwise t-tests with Bonferroni correction"""
    if group_name:
        df = df[df['Group'] == group_name].copy()
    
    # Use pingouin for pairwise t-tests
    posthoc = pg.pairwise_tests(data=df, dv='Value', within='Stimulation', 
                                 subject='Subject', parametric=True, 
                                 padjust='bonf', effsize='cohen')
    
    return posthoc

def perform_pairwise_tests_nonparametric(df, group_name=None):
    """Perform pairwise Wilcoxon tests with Bonferroni correction"""
    if group_name:
        df = df[df['Group'] == group_name].copy()
    
    # Use pingouin for pairwise Wilcoxon tests
    posthoc = pg.pairwise_tests(data=df, dv='Value', within='Stimulation',
                                subject='Subject', parametric=False,
                                padjust='bonf', effsize='CLES')
    
    return posthoc

def calculate_percent_change(df, group_name=None):
    """Calculate percent change from Sham baseline"""
    if group_name:
        df = df[df['Group'] == group_name].copy()
    
    changes = []
    
    for subj in df['Subject'].unique():
        subj_data = df[df['Subject'] == subj]
        sham_val = subj_data[subj_data['Stimulation'] == 'Sham']['Value'].values[0]
        low_val = subj_data[subj_data['Stimulation'] == 'Low']['Value'].values[0]
        high_val = subj_data[subj_data['Stimulation'] == 'High']['Value'].values[0]
        
        # Calculate percent change from sham
        low_change = ((low_val - sham_val) / sham_val) * 100 if sham_val != 0 else 0
        high_change = ((high_val - sham_val) / sham_val) * 100 if sham_val != 0 else 0
        
        changes.append({
            'Subject': subj,
            'Low_vs_Sham_%': low_change,
            'High_vs_Sham_%': high_change
        })
    
    return pd.DataFrame(changes)

def create_posthoc_barplot(df, metric_name, output_dir):
    """Create bar plot with individual points for post-hoc comparisons"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Post-Hoc Analysis: {metric_name}', fontsize=16, fontweight='bold')
    
    group_names = ['HC', 'PD_ON', 'PD_OFF']
    colors = {'Sham': '#E8E8E8', 'Low': '#A8D5E2', 'High': '#548CA8'}
    
    for idx, group_name in enumerate(group_names):
        ax = axes[idx]
        group_data = df[df['Group'] == group_name]
        
        # Calculate means and SEMs
        summary = group_data.groupby('Stimulation')['Value'].agg(['mean', 'sem']).reset_index()
        summary = summary.set_index('Stimulation').reindex(['Sham', 'Low', 'High']).reset_index()
        
        # Bar plot
        x_pos = [0, 1, 2]
        bars = ax.bar(x_pos, summary['mean'], yerr=summary['sem'],
                     color=[colors['Sham'], colors['Low'], colors['High']],
                     capsize=5, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Overlay individual points
        for stim_idx, stim in enumerate(['Sham', 'Low', 'High']):
            stim_data = group_data[group_data['Stimulation'] == stim]
            y_values = stim_data['Value'].values
            x_values = np.random.normal(stim_idx, 0.04, size=len(y_values))
            ax.scatter(x_values, y_values, color='black', alpha=0.6, s=30, zorder=3)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Sham', 'Low GVS', 'High GVS'])
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{group_name} (n={len(group_data["Subject"].unique())})', 
                    fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    filename = f'PostHoc_Bars_{metric_name.replace(" ", "_")}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def create_percent_change_plot(changes_dict, metric_name, output_dir):
    """Create plot showing percent changes from baseline"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Percent Change from Sham Baseline: {metric_name}', 
                fontsize=16, fontweight='bold')
    
    group_names = ['HC', 'PD_ON', 'PD_OFF']
    
    for idx, group_name in enumerate(group_names):
        ax = axes[idx]
        changes = changes_dict[group_name]
        
        # Calculate means and SEMs
        low_mean = changes['Low_vs_Sham_%'].mean()
        low_sem = changes['Low_vs_Sham_%'].sem()
        high_mean = changes['High_vs_Sham_%'].mean()
        high_sem = changes['High_vs_Sham_%'].sem()
        
        # Bar plot
        x_pos = [0, 1]
        means = [low_mean, high_mean]
        sems = [low_sem, high_sem]
        colors_list = ['#A8D5E2', '#548CA8']
        
        bars = ax.bar(x_pos, means, yerr=sems, capsize=5,
                     color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Overlay individual points
        for i, col in enumerate(['Low_vs_Sham_%', 'High_vs_Sham_%']):
            y_values = changes[col].values
            x_values = np.random.normal(i, 0.04, size=len(y_values))
            ax.scatter(x_values, y_values, color='black', alpha=0.6, s=30, zorder=3)
        
        # Zero line
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Low vs Sham', 'High vs Sham'])
        ax.set_ylabel('% Change from Sham', fontsize=11, fontweight='bold')
        ax.set_title(f'{group_name} (n={len(changes)})', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    filename = f'PercentChange_{metric_name.replace(" ", "_")}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

# Main analysis
print("=" * 80)
print("POST-HOC PAIRWISE COMPARISONS")
print("=" * 80)
print("\nAnalyzing pairwise differences between stimulation conditions")
print("Correction: Bonferroni (α = 0.05/3 = 0.017 per comparison)")
print("=" * 80)

all_results = []

for metric_name, metric_idx in metrics.items():
    print(f"\n{'=' * 60}")
    print(f"{metric_name}")
    print(f"{'=' * 60}")
    
    # Create long-format data
    df = create_long_format_data(groups, metric_idx)
    
    if len(df) == 0:
        print("  WARNING: No valid data. Skipping.")
        continue
    
    # Overall analysis (across all groups)
    print(f"\n{'─' * 60}")
    print("OVERALL (All Groups Combined)")
    print(f"{'─' * 60}")
    
    print("\nParametric (Paired t-tests, Bonferroni corrected):")
    try:
        posthoc_param = perform_pairwise_tests_parametric(df)
        for _, row in posthoc_param.iterrows():
            comparison = f"{row['A']} vs {row['B']}"
            sig = "***" if row['p-corr'] < 0.001 else "**" if row['p-corr'] < 0.01 else "*" if row['p-corr'] < 0.05 else "ns"
            print(f"  {comparison}: t = {row['T']:.3f}, p = {row['p-corr']:.4f}, d = {row['cohen']:.3f} {sig}")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print("\nNon-parametric (Wilcoxon, Bonferroni corrected):")
    try:
        posthoc_nonparam = perform_pairwise_tests_nonparametric(df)
        for _, row in posthoc_nonparam.iterrows():
            comparison = f"{row['A']} vs {row['B']}"
            sig = "***" if row['p-corr'] < 0.001 else "**" if row['p-corr'] < 0.01 else "*" if row['p-corr'] < 0.05 else "ns"
            print(f"  {comparison}: W = {row['W-val']:.1f}, p = {row['p-corr']:.4f}, CLES = {row['CLES']:.3f} {sig}")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Group-specific analyses
    changes_dict = {}
    
    for group_name in ['HC', 'PD_ON', 'PD_OFF']:
        print(f"\n{'─' * 60}")
        print(f"{group_name} Group")
        print(f"{'─' * 60}")
        
        group_df = df[df['Group'] == group_name]
        n = len(group_df['Subject'].unique())
        
        print(f"Sample size: n={n}")
        
        # Descriptive statistics
        desc = group_df.groupby('Stimulation')['Value'].agg(['mean', 'std']).round(3)
        print(f"\nDescriptive statistics:")
        print(desc)
        
        # Parametric tests
        print(f"\nParametric (Paired t-tests, Bonferroni corrected):")
        try:
            posthoc_param = perform_pairwise_tests_parametric(df, group_name)
            for _, row in posthoc_param.iterrows():
                comparison = f"{row['A']} vs {row['B']}"
                sig = "***" if row['p-corr'] < 0.001 else "**" if row['p-corr'] < 0.01 else "*" if row['p-corr'] < 0.05 else "ns"
                direction = "↑" if row['T'] > 0 else "↓"
                print(f"  {comparison}: t = {row['T']:.3f}, p = {row['p-corr']:.4f}, d = {row['cohen']:.3f} {sig} {direction}")
                
                # Store results
                all_results.append({
                    'Metric': metric_name,
                    'Group': group_name,
                    'Comparison': comparison,
                    'Test': 'Paired t-test',
                    'Statistic': row['T'],
                    'p_corrected': row['p-corr'],
                    'Effect_size': row['cohen'],
                    'Significant': row['p-corr'] < 0.05
                })
        except Exception as e:
            print(f"  ERROR: {e}")
        
        # Non-parametric tests
        print(f"\nNon-parametric (Wilcoxon, Bonferroni corrected):")
        try:
            posthoc_nonparam = perform_pairwise_tests_nonparametric(df, group_name)
            for _, row in posthoc_nonparam.iterrows():
                comparison = f"{row['A']} vs {row['B']}"
                sig = "***" if row['p-corr'] < 0.001 else "**" if row['p-corr'] < 0.01 else "*" if row['p-corr'] < 0.05 else "ns"
                print(f"  {comparison}: W = {row['W-val']:.1f}, p = {row['p-corr']:.4f}, CLES = {row['CLES']:.3f} {sig}")
        except Exception as e:
            print(f"  ERROR: {e}")
        
        # Percent change analysis
        changes = calculate_percent_change(df, group_name)
        changes_dict[group_name] = changes
        
        print(f"\nPercent change from Sham (mean ± SEM):")
        print(f"  Low GVS:  {changes['Low_vs_Sham_%'].mean():+.2f}% ± {changes['Low_vs_Sham_%'].sem():.2f}%")
        print(f"  High GVS: {changes['High_vs_Sham_%'].mean():+.2f}% ± {changes['High_vs_Sham_%'].sem():.2f}%")
    
    # Create visualizations
    print(f"\n{'─' * 60}")
    print("Creating visualizations...")
    print(f"{'─' * 60}")
    
    bar_path = create_posthoc_barplot(df, metric_name, output_dir)
    print(f"  Bar plot: {os.path.basename(bar_path)}")
    
    pct_path = create_percent_change_plot(changes_dict, metric_name, output_dir)
    print(f"  Percent change plot: {os.path.basename(pct_path)}")

# Save results
print(f"\n{'=' * 80}")
print("SAVING RESULTS")
print(f"{'=' * 80}\n")

if all_results:
    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, 'PostHoc_Pairwise_Results.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}\n")
    
    # Summary of significant effects
    sig_results = df_results[df_results['Significant'] == True]
    print(f"Significant comparisons (p < 0.05): {len(sig_results)} out of {len(all_results)}")
    
    if len(sig_results) > 0:
        print("\nSignificant effects by group:")
        for group in ['HC', 'PD_ON', 'PD_OFF']:
            group_sig = sig_results[sig_results['Group'] == group]
            print(f"  {group}: {len(group_sig)} significant comparisons")
else:
    print("No results to save.")

print(f"\n{'=' * 80}")
print("Post-hoc analysis complete!")
print(f"Figures saved to: {output_dir}")
print(f"{'=' * 80}")