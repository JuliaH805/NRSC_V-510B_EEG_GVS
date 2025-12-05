"""
Main script to run feature extraction on GVS EEG data
Processes all subjects and saves results to CSV
"""
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from FeatEx import extract_features
from Feature_Labels import get_feature_labels
from tqdm import tqdm

# Configuration
DATA_FOLDER = r"D:\UBC\510\neurodata_dataset"
Fs = 1000  # Sampling frequency (Hz)
ChanNum = 27
Phi = 1
Rho = 0
nLag = 150
nFFT = 512

# Subjects to exclude
HC_EXCLUDE = [8, 9, 10, 14]  # 1-indexed
PD_EXCLUDE = [3, 7, 8, 12]   # 1-indexed

# Initialize data containers
all_data = []

def load_mat_file(filepath, var_name):
    """Load variable from .mat file"""
    try:
        mat_data = sio.loadmat(filepath)
        return mat_data[var_name]
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def process_group(data_cells, health_str, stim_str, exclude_subjects):
    """
    Process a group of subjects (HC or PD) for one stimulation condition
    
    Parameters:
    -----------
    data_cells : ndarray
        Cell array from MATLAB (1 x N_subjects)
    health_str : str
        Health status ('HC', 'PD1', 'PD2')
    stim_str : str
        Stimulation type ('Sham', 'GVS7', 'GVS8')
    exclude_subjects : list
        List of subject indices to exclude (1-indexed)
    
    Returns:
    --------
    results : list
        List of dictionaries containing features and metadata
    """
    results = []
    n_subjects = data_cells.shape[1]
    
    print(f"\nProcessing {health_str} - {stim_str}: {n_subjects} subjects")
    
    for subIdx in tqdm(range(n_subjects), desc=f"{health_str}-{stim_str}"):
        # Check if subject should be excluded (convert to 1-indexed)
        if (subIdx + 1) in exclude_subjects:
            print(f"  Skipping subject {subIdx + 1} (excluded)")
            continue
        
        # Extract data for this subject
        # data_cells is (1, n_subjects), each cell contains (27, 2000, 10)
        signal = data_cells[0, subIdx]
        
        if signal is None or signal.size == 0:
            print(f"  Warning: No data for subject {subIdx + 1}")
            continue
        
        # Process each channel and trial
        n_channels, n_timepoints, n_trials = signal.shape
        
        for chanIdx in range(n_channels):
            for trialIdx in range(n_trials):
                # Extract single epoch: (2000,)
                Sig = signal[chanIdx, :, trialIdx]
                
                # Extract features
                try:
                    Feats = extract_features(Sig, Fs, Phi, Rho, nLag, nFFT)
                    
                    # Store result
                    result = {
                        'SID': subIdx + 1,  # 1-indexed subject ID
                        'Health': health_str,
                        'Stim': stim_str,
                        'Channel': chanIdx + 1,  # 1-indexed channel
                        'Trial': trialIdx + 1,   # 1-indexed trial
                        'Features': Feats
                    }
                    results.append(result)
                    
                except Exception as e:
                    print(f"  Error processing Subject {subIdx+1}, Channel {chanIdx+1}, Trial {trialIdx+1}: {e}")
                    continue
    
    return results

def main():
    """Main processing pipeline"""
    print("=" * 80)
    print("EEG Feature Extraction Pipeline")
    print("=" * 80)
    
    data_path = Path(DATA_FOLDER)
    
    # File names and their variables
    files_config = [
        ('shamdata_tlgo.mat', 'Sham', [
            ('shamhceeg', 'HC', HC_EXCLUDE),
            ('shampd1eeg', 'PD1', PD_EXCLUDE),
            ('shampd2eeg', 'PD2', PD_EXCLUDE)
        ]),
        ('stim7data_tlgo.mat', 'GVS7', [
            ('stim7hceeg', 'HC', HC_EXCLUDE),
            ('stim7pd1eeg', 'PD1', PD_EXCLUDE),
            ('stim7pd2eeg', 'PD2', PD_EXCLUDE)
        ]),
        ('stim8data_tlgo.mat', 'GVS8', [
            ('stim8hceeg', 'HC', HC_EXCLUDE),
            ('stim8pd1eeg', 'PD1', PD_EXCLUDE),
            ('stim8pd2eeg', 'PD2', PD_EXCLUDE)
        ])
    ]
    
    # Process all files
    for filename, stim_type, variables in files_config:
        filepath = data_path / filename
        
        if not filepath.exists():
            print(f"\nWarning: File not found - {filepath}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Loading {filename}")
        print(f"{'='*80}")
        
        # Load the .mat file
        mat_data = sio.loadmat(filepath)
        
        # Process each variable (HC, PD1, PD2)
        for var_name, health_str, exclude_list in variables:
            if var_name not in mat_data:
                print(f"Warning: Variable {var_name} not found in {filename}")
                continue
            
            data_cells = mat_data[var_name]
            
            # Process this group
            results = process_group(data_cells, health_str, stim_type, exclude_list)
            all_data.extend(results)
    
    # Create DataFrame
    print("\n" + "="*80)
    print("Creating output DataFrame...")
    print("="*80)
    
    if len(all_data) == 0:
        print("ERROR: No data was processed!")
        return
    
    # Get feature labels
    feature_labels = get_feature_labels()
    
    # Create rows for DataFrame
    rows = []
    for entry in all_data:
        row = {
            'SID': entry['SID'],
            'Health': entry['Health'],
            'Stim': entry['Stim'],
            'Channel': entry['Channel'],
            'Trial': entry['Trial']
        }
        
        # Add features
        for i, feat_val in enumerate(entry['Features']):
            row[feature_labels[i]] = feat_val
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    output_file = data_path / "AllFeatures_Extracted.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"Feature extraction complete!")
    print(f"Total samples processed: {len(df)}")
    print(f"Output saved to: {output_file}")
    print(f"{'='*80}")
    
    # Print summary statistics
    print("\nSummary by group:")
    print(df.groupby(['Health', 'Stim']).size())
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nDataFrame shape:", df.shape)
    print(f"Columns: {df.columns.tolist()[:10]}... (showing first 10)")

if __name__ == "__main__":
    main()
