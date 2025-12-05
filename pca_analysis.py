"""
PCA Analysis on EEG Features
Averages trials per subject, performs PCA, and visualizes results
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

# Configuration
DATA_FOLDER = r"D:\UBC\510\neurodata_dataset"
INPUT_FILE = "AllFeatures_Extracted.csv"
OUTPUT_FOLDER = "PCA_Results"

def load_and_prepare_data(filepath):
    """Load extracted features and prepare for PCA"""
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    print(f"Original data shape: {df.shape}")
    print(f"Groups: {df['Health'].unique()}")
    print(f"Stimulations: {df['Stim'].unique()}")
    print(f"Channels: {df['Channel'].nunique()}")
    print(f"Trials: {df['Trial'].nunique()}")
    
    return df


def average_trials(df):
    """
    Average features across 10 trials for each subject/channel/condition
    
    Returns:
    --------
    df_avg : DataFrame
        Averaged data with one row per subject/channel/stim/health
    """
    print("\nAveraging across trials...")
    
    # Get feature columns (all columns starting with "Feature_")
    feature_cols = [col for col in df.columns if col.startswith("Feature_")]
    
    print(f"Number of features: {len(feature_cols)}")
    
    # Group by subject, health, stim, and channel, then average across trials
    groupby_cols = ['SID', 'Health', 'Stim', 'Channel']
    
    # Average the features
    df_avg = df.groupby(groupby_cols)[feature_cols].mean().reset_index()
    
    print(f"After averaging trials: {df_avg.shape}")
    print(f"  Original rows: {len(df)}")
    print(f"  Averaged rows: {len(df_avg)}")
    print(f"  Reduction: {len(df) / len(df_avg):.1f}x")
    
    return df_avg, feature_cols


def perform_pca(df_avg, feature_cols, n_components=None):
    """
    Perform PCA on averaged features
    
    Parameters:
    -----------
    df_avg : DataFrame
        Averaged data
    feature_cols : list
        List of feature column names
    n_components : int or None
        Number of PCA components (None = all)
    
    Returns:
    --------
    pca : PCA object
    X_pca : ndarray
        Transformed data
    scaler : StandardScaler
        Fitted scaler
    """
    print("\n" + "="*80)
    print("PERFORMING PCA")
    print("="*80)
    
    # Extract feature matrix
    X = df_avg[feature_cols].values
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    
    # Check for NaN or Inf
    n_nan = np.sum(np.isnan(X))
    n_inf = np.sum(np.isinf(X))
    
    if n_nan > 0 or n_inf > 0:
        print(f"\nWARNING: Found {n_nan} NaN and {n_inf} Inf values")
        print("Replacing NaN/Inf with 0...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Standardize features (zero mean, unit variance)
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    if n_components is None:
        n_components = min(X.shape[0], X.shape[1])
    
    print(f"\nFitting PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"PCA output shape: {X_pca.shape}")
    
    return pca, X_pca, scaler, X_scaled


def plot_scree_elbow(pca, output_folder):
    """Create scree plot (elbow plot) showing explained variance"""
    
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scree plot
    ax1 = axes[0]
    n_components = len(explained_var)
    ax1.plot(range(1, n_components + 1), explained_var * 100, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance (%)', fontsize=12)
    ax1.set_title('Scree Plot (Elbow Plot)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, min(30, n_components + 1))
    
    # Mark common thresholds
    for i in [5, 10, 15, 20]:
        if i < n_components:
            ax1.axvline(x=i, color='red', linestyle='--', alpha=0.3)
            ax1.text(i, ax1.get_ylim()[1] * 0.95, f'PC{i}', ha='center', fontsize=9)
    
    # Cumulative variance
    ax2 = axes[1]
    ax2.plot(range(1, n_components + 1), cumulative_var * 100, 'ro-', linewidth=2, markersize=8)
    ax2.axhline(y=80, color='g', linestyle='--', linewidth=2, label='80% threshold')
    ax2.axhline(y=90, color='b', linestyle='--', linewidth=2, label='90% threshold')
    ax2.axhline(y=95, color='m', linestyle='--', linewidth=2, label='95% threshold')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=12)
    ax2.set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, min(30, n_components + 1))
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    output_path = output_folder / "scree_plot_elbow.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Scree plot saved to: {output_path}")
    plt.close()
    
    # Find number of components for thresholds
    n_80 = np.argmax(cumulative_var >= 0.80) + 1
    n_90 = np.argmax(cumulative_var >= 0.90) + 1
    n_95 = np.argmax(cumulative_var >= 0.95) + 1
    
    print(f"\nComponents needed:")
    print(f"  80% variance: {n_80} components")
    print(f"  90% variance: {n_90} components")
    print(f"  95% variance: {n_95} components")
    
    return n_80, n_90, n_95


def plot_loadings(pca, feature_cols, output_folder, n_top_pcs=5):
    """
    Plot feature loadings for top principal components
    
    Parameters:
    -----------
    pca : PCA object
    feature_cols : list
        Feature column names
    output_folder : Path
        Output directory
    n_top_pcs : int
        Number of top PCs to analyze
    """
    
    loadings = pca.components_  # Shape: (n_components, n_features)
    
    # Create loadings heatmap for top PCs
    fig, ax = plt.subplots(figsize=(20, 8))
    
    n_pcs_to_plot = min(n_top_pcs, loadings.shape[0])
    
    # Get simplified feature names (remove "Feature_X: " prefix)
    simple_names = [col.split(': ')[1] if ': ' in col else col for col in feature_cols]
    
    sns.heatmap(loadings[:n_pcs_to_plot, :], 
                xticklabels=simple_names,
                yticklabels=[f'PC{i+1}' for i in range(n_pcs_to_plot)],
                cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Loading'},
                ax=ax)
    
    ax.set_title(f'PCA Loadings Heatmap (Top {n_pcs_to_plot} Components)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Principal Components', fontsize=12)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    output_path = output_folder / "loadings_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Loadings heatmap saved to: {output_path}")
    plt.close()
    
    # Create bar plots for individual PCs
    fig, axes = plt.subplots(n_pcs_to_plot, 1, figsize=(15, 4 * n_pcs_to_plot))
    
    if n_pcs_to_plot == 1:
        axes = [axes]
    
    for i in range(n_pcs_to_plot):
        ax = axes[i]
        
        # Get loadings for this PC
        pc_loadings = loadings[i, :]
        
        # Sort by absolute value
        sorted_indices = np.argsort(np.abs(pc_loadings))[::-1]
        top_n = 20  # Show top 20 features
        top_indices = sorted_indices[:top_n]
        
        # Plot
        colors = ['red' if x < 0 else 'blue' for x in pc_loadings[top_indices]]
        ax.barh(range(top_n), pc_loadings[top_indices], color=colors, alpha=0.7)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([simple_names[idx] for idx in top_indices], fontsize=9)
        ax.set_xlabel('Loading', fontsize=11)
        ax.set_title(f'PC{i+1} - Top 20 Feature Loadings (Explains {pca.explained_variance_ratio_[i]*100:.2f}% variance)', 
                     fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
    
    plt.tight_layout()
    output_path = output_folder / "loadings_top_features.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Top features plot saved to: {output_path}")
    plt.close()


def save_loadings_csv(pca, feature_cols, output_folder, n_components=20):
    """Save loadings to CSV for detailed analysis"""
    
    n_comps = min(n_components, pca.components_.shape[0])
    
    # Create DataFrame
    loadings_df = pd.DataFrame(
        pca.components_[:n_comps, :].T,
        columns=[f'PC{i+1}' for i in range(n_comps)],
        index=feature_cols
    )
    
    # Add variance explained
    var_explained = pd.Series(
        pca.explained_variance_ratio_[:n_comps],
        index=[f'PC{i+1}' for i in range(n_comps)],
        name='Variance_Explained'
    )
    
    # Save
    output_path = output_folder / "pca_loadings.csv"
    loadings_df.to_csv(output_path)
    print(f"Loadings CSV saved to: {output_path}")
    
    # Save variance explained
    var_path = output_folder / "variance_explained.csv"
    var_explained.to_csv(var_path, header=True)
    print(f"Variance explained saved to: {var_path}")
    
    # Print top contributors for each PC
    print("\n" + "="*80)
    print("TOP 10 FEATURES FOR EACH PC")
    print("="*80)
    
    for i in range(min(5, n_comps)):
        print(f"\n--- PC{i+1} (Explains {pca.explained_variance_ratio_[i]*100:.2f}% variance) ---")
        pc_loadings = loadings_df[f'PC{i+1}'].abs().sort_values(ascending=False)
        print(pc_loadings.head(10))


def plot_pc_scores(X_pca, df_avg, output_folder):
    """Plot scatter of PC scores colored by groups"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # PC1 vs PC2 by Health
    ax = axes[0, 0]
    for health in df_avg['Health'].unique():
        mask = df_avg['Health'] == health
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=health, alpha=0.6, s=30)
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('PC1 vs PC2 (by Health Group)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # PC1 vs PC2 by Stim
    ax = axes[0, 1]
    for stim in df_avg['Stim'].unique():
        mask = df_avg['Stim'] == stim
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=stim, alpha=0.6, s=30)
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('PC1 vs PC2 (by Stimulation)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # PC2 vs PC3 by Health
    ax = axes[1, 0]
    for health in df_avg['Health'].unique():
        mask = df_avg['Health'] == health
        ax.scatter(X_pca[mask, 1], X_pca[mask, 2], label=health, alpha=0.6, s=30)
    ax.set_xlabel('PC2', fontsize=12)
    ax.set_ylabel('PC3', fontsize=12)
    ax.set_title('PC2 vs PC3 (by Health Group)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # PC2 vs PC3 by Stim
    ax = axes[1, 1]
    for stim in df_avg['Stim'].unique():
        mask = df_avg['Stim'] == stim
        ax.scatter(X_pca[mask, 1], X_pca[mask, 2], label=stim, alpha=0.6, s=30)
    ax.set_xlabel('PC2', fontsize=12)
    ax.set_ylabel('PC3', fontsize=12)
    ax.set_title('PC2 vs PC3 (by Stimulation)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_folder / "pc_scores_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"PC scores scatter plot saved to: {output_path}")
    plt.close()


def main():
    """Main PCA analysis pipeline"""
    
    print("="*80)
    print("PCA ANALYSIS ON EEG FEATURES")
    print("="*80)
    
    data_path = Path(DATA_FOLDER)
    input_filepath = data_path / INPUT_FILE
    
    # Create output folder
    output_folder = data_path / OUTPUT_FOLDER
    output_folder.mkdir(exist_ok=True)
    print(f"\nOutput folder: {output_folder}")
    
    # Load data
    df = load_and_prepare_data(input_filepath)
    
    # Average across trials
    df_avg, feature_cols = average_trials(df)
    
    # Save averaged data
    avg_output = output_folder / "features_averaged_by_trial.csv"
    df_avg.to_csv(avg_output, index=False)
    print(f"\nAveraged data saved to: {avg_output}")
    
    # Perform PCA
    pca, X_pca, scaler, X_scaled = perform_pca(df_avg, feature_cols)
    
    # Plot scree/elbow plot
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    n_80, n_90, n_95 = plot_scree_elbow(pca, output_folder)
    
    # Plot loadings
    plot_loadings(pca, feature_cols, output_folder, n_top_pcs=10)
    
    # Save loadings to CSV
    save_loadings_csv(pca, feature_cols, output_folder, n_components=20)
    
    # Plot PC scores
    plot_pc_scores(X_pca, df_avg, output_folder)
    
    # Save PCA-transformed data
    pc_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    df_pca = pd.concat([
        df_avg[['SID', 'Health', 'Stim', 'Channel']].reset_index(drop=True),
        pd.DataFrame(X_pca, columns=pc_cols)
    ], axis=1)
    
    pca_output = output_folder / "pca_transformed_data.csv"
    df_pca.to_csv(pca_output, index=False)
    print(f"\nPCA-transformed data saved to: {pca_output}")
    
    # Final summary
    print("\n" + "="*80)
    print("PCA ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nRecommendations for ML models:")
    print(f"  - Use {n_80} components for 80% variance")
    print(f"  - Use {n_90} components for 90% variance")
    print(f"  - Use {n_95} components for 95% variance")
    print(f"\nCheck the following files:")
    print(f"  1. scree_plot_elbow.png - Decide number of components")
    print(f"  2. loadings_heatmap.png - See feature importance")
    print(f"  3. loadings_top_features.png - Top features per PC")
    print(f"  4. pca_loadings.csv - Detailed loadings for analysis")
    print(f"  5. pca_transformed_data.csv - Use this for ML models")
    print("="*80)


if __name__ == "__main__":
    main()
