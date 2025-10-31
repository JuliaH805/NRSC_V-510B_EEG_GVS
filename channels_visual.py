import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from pathlib import Path

sns.set_style("whitegrid")
mne.set_log_level('WARNING')


data_path = Path(r"D:\UBC\510\neurodata_dataset")
figure_path = Path(r"D:\UBC\510\Figures\channel_visualization_3")
figure_path.mkdir(parents=True, exist_ok=True)

mat_file = data_path / "stim7data_tlgo.mat"

sfreq = 1000

ch_names = [f'Ch{i+1}' for i in range(27)]

mat_data = sio.loadmat(mat_file)
stim7_hc = mat_data['stim7hceeg'][0] #explain code note: change to a more generic name
stim7_pd1 = mat_data['stim7pd1eeg'][0]
stim7_pd2 = mat_data['stim7pd2eeg'][0]

print(f"{mat_file.name}loaded") 
print(f"HC: {len(stim7_hc)} subjects")
print(f"PD Off: {len(stim7_pd1)} subjects")
print(f"PD On: {len(stim7_pd2)} subjects")


def visualize_all_channels(data, subject_id, group_name, save_path):
    
    data_avg = np.mean(data, axis=2)
    
    fig, ax = plt.subplots(figsize=(20, 12))
    
    time_vec = np.arange(data_avg.shape[1]) / sfreq * 1000
    
    # Plot all channels with offset
    offset = 0
    offsets = []
    colors = plt.cm.tab20(np.linspace(0, 1, 27))  
    
    for i in range(27):
        signal = data_avg[i, :]
        scale = np.std(signal)
        
        ax.plot(time_vec, signal + offset, 
                color=colors[i], alpha=0.8, linewidth=1.2, 
                label=ch_names[i])
        
        offsets.append(offset)
        offset += 5 * scale
    
 
    ax.axvline(1000, color='red', linestyle='--', linewidth=2.5, 
               alpha=0.8, label='Go Signal (1000ms)')
    

    ax.set_xlabel('Time (ms)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Channels (offset for visibility)', fontsize=14, fontweight='bold')
    ax.set_title(f'{group_name} - Subject {subject_id}\nAll 27 Channels (Trial Average)', 
                 fontsize=16, fontweight='bold')
    
    ax.set_yticks(offsets)
    ax.set_yticklabels(ch_names, fontsize=10)
    
    ax.plot([], [], color='red', linestyle='--', linewidth=2.5, label='Go Signal (1000ms)')
    ax.legend('')
    
    ax.grid(True, alpha=0.3, axis='x')
    
    ax.axvspan(0, 1000, alpha=0.1, color='blue', label='Pre-stimulus')
    ax.axvspan(1000, 2000, alpha=0.1, color='orange', label='Post-stimulus')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


for i, subject_data in enumerate(stim7_hc):
    subject_id = i + 1
    save_path = figure_path / f"HC_Subject_{subject_id:02d}.png"
    visualize_all_channels(subject_data, subject_id, "Healthy Control", save_path)
    print(f"Subject {subject_id:02d} saved to {save_path.name}")

for i, subject_data in enumerate(stim7_pd1):
    subject_id = i + 1
    save_path = figure_path / f"PD1_Subject_{subject_id:02d}.png"
    visualize_all_channels(subject_data, subject_id, "PD Off Medication", save_path)
    print(f"Subject {subject_id:02d} saved to {save_path.name}")

for i, subject_data in enumerate(stim7_pd2):
    subject_id = i + 1
    save_path = figure_path / f"PD2_Subject_{subject_id:02d}.png"
    visualize_all_channels(subject_data, subject_id, "PD On Medication", save_path)
    print(f"Subject {subject_id:02d} saved to {save_path.name}")
