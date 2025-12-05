import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy import stats
import os
from statsmodels.stats.anova import AnovaRM
import pingouin as pg

# Note: You'll need to install pingouin if you haven't already
# pip install pingouin

# Create output directory
output_dir = r'D:\UBC\510\Figures\Mixed_Design_Analysis'
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

def create_long_format_data(metric_idx):
    """Create long-format dataframe for mixed ANOVA"""
    all_data = []
    subject_counter = 0
    
    for group_name in ['HC', 'PD_ON', 'PD_OFF']:
        group_data = groups[group_name]
        
        for subj_idx in range(group_data['n_subjects']):
            if subj_idx in group_data['exclude']:
                continue
            
            # Get means for each condition
            sham_mean = np.nanmean(group_data['sham'][subj_idx][:, metric_idx])
            low_mean = np.nanmean(group_data['low_gvs'][subj_idx][:, metric_idx])
            high_mean = np.nanmean(group_data['high_gvs'][subj_idx][:, metric_idx])
            
            # Skip if any condition is completely missing
            if np.isnan(sham_mean) or np.isnan(low_mean) or np.isnan(high_mean):
                continue
            
            # Add three rows (one per condition) for this subject
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

def perform_mixed_anova(df, metric_name):
    """Perform mixed-design ANOVA using pingouin"""
    # Mixed ANOVA: Between-subjects factor (Group), Within-subjects factor (Stimulation)
    aov = pg.mixed_anova(data=df, dv='Value', within='Stimulation', 
                         between='Group', subject='Subject')
    
    return aov

def perform_aligned_rank_transform(df):
    """
    Aligned Rank Transform (ART) - non-parametric alternative to mixed ANOVA
    This is a simplified version. For proper ART, consider using the ARTool R package.
    """
    # For each effect, align the data then rank
    results = {}
    
    # Main effect of Group (between-subjects)
    # Average across stimulation conditions
    group_means = df.groupby(['Subject', 'Group'])['Value'].mean().reset_index()
    # Rank the values
    group_means['Rank'] = group_means['Value'].rank()
    # Run one-way ANOVA on ranks
    groups_data = [group_means[group_means['Group'] == g]['Rank'].values 
                   for g in ['HC', 'PD_ON', 'PD_OFF']]
    f_stat, p_val = stats.f_oneway(*groups_data)
    results['Group'] = {'F': f_stat, 'p': p_val}
    
    # Main effect of Stimulation (within-subjects)
    # For each subject, rank their three stimulation values
    stim_data = []
    for subj in df['Subject'].unique():
        subj_data = df[df['Subject'] == subj].sort_values('Stimulation')
        sham_val = subj_data[subj_data['Stimulation'] == 'Sham']['Value'].values[0]
        low_val = subj_data[subj_data['Stimulation'] == 'Low']['Value'].values[0]
        high_val = subj_data[subj_data['Stimulation'] == 'High']['Value'].values[0]
        stim_data.append([sham_val, low_val, high_val])
    
    stim_data = np.array(stim_data)
    stat, p_val = stats.friedmanchisquare(stim_data[:, 0], stim_data[:, 1], stim_data[:, 2])
    results['Stimulation'] = {'chi2': stat, 'p': p_val}
    
    return results

def create_interaction_plot(df, metric_name, output_dir):
    """Create interaction plot for Group x Stimulation"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate means and SEMs
    summary = df.groupby(['Group', 'Stimulation'])['Value'].agg(['mean', 'sem']).reset_index()
    
    colors = {'HC': '#E74C3C', 'PD_ON': '#3498DB', 'PD_OFF': '#2ECC71'}
    markers = {'HC': 'o', 'PD_ON': 's', 'PD_OFF': '^'}
    
    stim_order = ['Sham', 'Low', 'High']
    x_pos = {'Sham': 0, 'Low': 1, 'High': 2}
    
    for group in ['HC', 'PD_ON', 'PD_OFF']:
        group_data = summary[summary['Group'] == group]
        x = [x_pos[s] for s in group_data['Stimulation']]
        y = group_data['mean'].values
        err = group_data['sem'].values
        
        ax.errorbar(x, y, yerr=err, marker=markers[group], markersize=10,
                   linewidth=2, capsize=5, capthick=2, 
                   label=group, color=colors[group])
    
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(stim_order)
    ax.set_xlabel('Stimulation Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric_name}', fontsize=12, fontweight='bold')
    ax.set_title(f'Group × Stimulation Interaction: {metric_name}', 
                fontsize=14, fontweight='bold')
    ax.legend(title='Group', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    filename = f'Interaction_Plot_{metric_name.replace(" ", "_")}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def perform_simple_effects(df):
    """Test simple effects - effect of Stimulation within each Group"""
    simple_effects = []
    
    for group in ['HC', 'PD_ON', 'PD_OFF']:
        group_data = df[df['Group'] == group]
        
        # Reshape for Friedman test
        stim_data = []
        for subj in group_data['Subject'].unique():
            subj_data = group_data[group_data['Subject'] == subj].sort_values('Stimulation')
            sham_val = subj_data[subj_data['Stimulation'] == 'Sham']['Value'].values[0]
            low_val = subj_data[subj_data['Stimulation'] == 'Low']['Value'].values[0]
            high_val = subj_data[subj_data['Stimulation'] == 'High']['Value'].values[0]
            stim_data.append([sham_val, low_val, high_val])
        
        stim_data = np.array(stim_data)
        n = len(stim_data)
        
        if n >= 3:
            # Friedman test
            stat, p_val = stats.friedmanchisquare(stim_data[:, 0], stim_data[:, 1], stim_data[:, 2])
            
            simple_effects.append({
                'Group': group,
                'n': n,
                'chi2': stat,
                'p': p_val,
                'significant': p_val < 0.05
            })
    
    return simple_effects

# Main analysis
print("=" * 80)
print("MIXED-DESIGN ANALYSIS: GROUP × STIMULATION")
print("=" * 80)
print("\nDesign:")
print("  Between-subjects factor: Group (HC, PD_ON, PD_OFF)")
print("  Within-subjects factor: Stimulation (Sham, Low GVS, High GVS)")
print("=" * 80)

all_results = []

for metric_name, metric_idx in metrics.items():
    print(f"\n{'=' * 60}")
    print(f"{metric_name}")
    print(f"{'=' * 60}")
    
    # Create long-format data
    df = create_long_format_data(metric_idx)
    
    if len(df) == 0:
        print("  WARNING: No valid data. Skipping.")
        continue
    
    n_subjects = len(df['Subject'].unique())
    n_per_group = df.groupby('Group')['Subject'].nunique()
    
    print(f"\nSample sizes:")
    print(f"  Total: N={n_subjects}")
    for group in ['HC', 'PD_ON', 'PD_OFF']:
        print(f"  {group}: n={n_per_group.get(group, 0)}")
    
    # Descriptive statistics
    print(f"\nDescriptive Statistics:")
    desc_stats = df.groupby(['Group', 'Stimulation'])['Value'].agg(['mean', 'std', 'count'])
    print(desc_stats)
    
    # Mixed ANOVA (parametric)
    print(f"\n{'─' * 60}")
    print("PARAMETRIC: Mixed-Design ANOVA")
    print(f"{'─' * 60}")
    
    try:
        aov = perform_mixed_anova(df, metric_name)
        print("\nFull ANOVA table:")
        print(aov)
        
        # Extract key results
        group_effect = aov[aov['Source'] == 'Group'].iloc[0] if 'Group' in aov['Source'].values else None
        stim_effect = aov[aov['Source'] == 'Stimulation'].iloc[0] if 'Stimulation' in aov['Source'].values else None
        interaction = aov[aov['Source'] == 'Interaction'].iloc[0] if 'Interaction' in aov['Source'].values else None
        
        print(f"\nKey Effects:")
        if group_effect is not None:
            # For within-subjects effects, use GG-corrected p if sphericity violated
            # For between-subjects, use p-unc
            group_p = group_effect['p-unc']
            stim_p = stim_effect['p-GG-corr'] if pd.notna(stim_effect['p-GG-corr']) and stim_effect['sphericity'] == False else stim_effect['p-unc']
            inter_p = interaction['p-GG-corr'] if pd.notna(interaction['p-GG-corr']) and interaction['sphericity'] == False else interaction['p-unc']
            
            stim_note = " (GG-corrected)" if pd.notna(stim_effect['p-GG-corr']) and stim_effect['sphericity'] == False else ""
            inter_note = " (GG-corrected)" if pd.notna(interaction['p-GG-corr']) and interaction['sphericity'] == False else ""
            
            print(f"  Group:        F({group_effect['DF1']:.0f},{group_effect['DF2']:.0f}) = {group_effect['F']:.4f}, p = {group_p:.4f}, η²p = {group_effect['np2']:.4f}")
            print(f"  Stimulation:  F({stim_effect['DF1']:.0f},{stim_effect['DF2']:.0f}) = {stim_effect['F']:.4f}, p = {stim_p:.4f}, η²p = {stim_effect['np2']:.4f}{stim_note}")
            print(f"  Interaction:  F({interaction['DF1']:.0f},{interaction['DF2']:.0f}) = {interaction['F']:.4f}, p = {inter_p:.4f}, η²p = {interaction['np2']:.4f}{inter_note}")
            
            # Interpretation
            print(f"\n  Interpretation:")
            print(f"    Group effect: {'SIGNIFICANT' if group_p < 0.05 else 'Not significant'}")
            print(f"    Stimulation effect: {'SIGNIFICANT' if stim_p < 0.05 else 'Not significant'}")
            print(f"    Interaction: {'SIGNIFICANT' if inter_p < 0.05 else 'Not significant'}")
            
            if inter_p < 0.05:
                print(f"    → Groups respond differently to stimulation conditions")
            elif stim_p < 0.05:
                print(f"    → Stimulation has an effect, similar across all groups")
            
        else:
            print("  Could not extract effects - check ANOVA table above")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        group_effect = stim_effect = interaction = None
    
    # Non-parametric alternative
    print(f"\n{'─' * 60}")
    print("NON-PARAMETRIC: Aligned Rank Transform (simplified)")
    print(f"{'─' * 60}")
    
    try:
        art_results = perform_aligned_rank_transform(df)
        print(f"  Group effect:       F = {art_results['Group']['F']:.4f}, p = {art_results['Group']['p']:.4f}")
        print(f"  Stimulation effect: χ² = {art_results['Stimulation']['chi2']:.4f}, p = {art_results['Stimulation']['p']:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Simple effects analysis if interaction is significant
    if interaction is not None:
        try:
            p_col = 'p-unc' if 'p-unc' in aov.columns else 'p-value'
            if interaction[p_col] < 0.05:
                print(f"\n{'─' * 60}")
                print("SIMPLE EFFECTS: Effect of Stimulation within each Group")
                print(f"{'─' * 60}")
                
                simple_effects = perform_simple_effects(df)
                for se in simple_effects:
                    sig_marker = '***' if se['significant'] else 'ns'
                    print(f"  {se['Group']} (n={se['n']}): χ²(2) = {se['chi2']:.4f}, p = {se['p']:.4f} {sig_marker}")
        except Exception as e:
            print(f"\n  Could not perform simple effects: {e}")
    
    # Create interaction plot
    plot_path = create_interaction_plot(df, metric_name, output_dir)
    print(f"\n  Interaction plot saved: {os.path.basename(plot_path)}")
    
    # Store results
    if group_effect is not None:
        try:
            # Choose appropriate p-values
            group_p = group_effect['p-unc']
            stim_p = stim_effect['p-GG-corr'] if pd.notna(stim_effect['p-GG-corr']) and stim_effect['sphericity'] == False else stim_effect['p-unc']
            inter_p = interaction['p-GG-corr'] if pd.notna(interaction['p-GG-corr']) and interaction['sphericity'] == False else interaction['p-unc']
            
            all_results.append({
                'Metric': metric_name,
                'N': n_subjects,
                'HC_n': n_per_group.get('HC', 0),
                'PDON_n': n_per_group.get('PD_ON', 0),
                'PDOFF_n': n_per_group.get('PD_OFF', 0),
                'Group_F': group_effect['F'],
                'Group_p': group_p,
                'Group_np2': group_effect['np2'],
                'Stim_F': stim_effect['F'],
                'Stim_p': stim_p,
                'Stim_np2': stim_effect['np2'],
                'Interaction_F': interaction['F'],
                'Interaction_p': inter_p,
                'Interaction_np2': interaction['np2']
            })
        except Exception as e:
            print(f"\n  WARNING: Could not store results - {e}")

# Save results
print(f"\n{'=' * 80}")
print("SAVING RESULTS")
print(f"{'=' * 80}\n")

if all_results:
    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, 'Mixed_ANOVA_Results.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}\n")
    
    print("Summary of Results:")
    print(df_results.to_string(index=False))
else:
    print("No results to save.")

print(f"\n{'=' * 80}")
print("Analysis complete!")
print(f"Figures saved to: {output_dir}")
print(f"{'=' * 80}")