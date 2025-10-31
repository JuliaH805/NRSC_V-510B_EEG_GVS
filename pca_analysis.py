import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def analyze_pca_components(data, labels, pc_to_analyze=1):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    pca = PCA()
    scores = pca.fit_transform(data_scaled)
    loadings = pca.components_.T  # Transpose to have neurons as rows

    pc_idx = pc_to_analyze - 1
    
    num_neurons = data.shape[1]
    loadings_df = pd.DataFrame({
        'neuron_index': np.arange(num_neurons),
        'loading': loadings[:, pc_idx]})
    
    sorted_loadings = loadings_df.sort_values(by='loading', ascending=False)
    
    top_pos_neurons = sorted_loadings.head(5)['neuron_index'].values
    top_neg_neurons = sorted_loadings.tail(5)['neuron_index'].values
    
    mean_pos_activity = data[:, top_pos_neurons].mean(axis=1)
    mean_neg_activity = data[:, top_neg_neurons].mean(axis=1)
    
    fig = go.Figure()

    pc_scores_scaled = (scores[:, pc_idx] - scores[:, pc_idx].min()) / (scores[:, pc_idx].max() - scores[:, pc_idx].min())
    mean_pos_scaled = (mean_pos_activity - mean_pos_activity.min()) / (mean_pos_activity.max() - mean_pos_activity.min())
    mean_neg_scaled = (mean_neg_activity - mean_neg_activity.min()) / (mean_neg_activity.max() - mean_neg_activity.min())
    
    time_points = np.arange(data.shape[0])

    for label_val in np.unique(labels):
        mask = labels == label_val
        fig.add_trace(go.Scatter(
            x=time_points[mask],
            y=pc_scores_scaled[mask],
            mode='markers',
            name=f'PC {pc_to_analyze} (Lever Press {int(label_val)})',
            marker=dict(size=8)))

    fig.add_trace(go.Scatter(
        x=time_points,
        y=mean_pos_scaled,
        mode='lines',
        name='Mean of Top 5 Positive Neurons',
        line=dict(color='firebrick', width=2)))
    fig.add_trace(go.Scatter(
        x=time_points,
        y=mean_neg_scaled,
        mode='lines',
        name='Mean of Top 5 Negative Neurons',
        line=dict(color='royalblue', width=2)))

    fig.update_layout(
        title_text=f'Analysis of Principal Component {pc_to_analyze}',
        xaxis_title='Time Bins (Samples)',
        yaxis_title='Normalized Activity / Score',
        legend_title='Trace',
        template='plotly_white')
    
    fig.show()

def reconstruct_and_plot_pc1(data):
    data_mean = data.mean(axis=0)
    data_centered = data - data_mean
    
    pca = PCA(n_components=1)
    original_pc1_scores = pca.fit_transform(data)
    pc1_loadings = pca.components_[0, :]

    reconstructed_pc1 = np.dot(data_centered, pc1_loadings)
    
    fig1 = make_subplots(rows=1, cols=2, subplot_titles=('Original PC1 from PCA', 'Reconstructed PC1'))
    
    fig1.add_trace(go.Scatter(y=original_pc1_scores.flatten(), mode='lines', name='Original PC1', line=dict(color='black')), row=1, col=1)
    fig1.add_trace(go.Scatter(y=reconstructed_pc1, mode='lines', name='Reconstructed PC1', line=dict(color='rgba(50,50,50,0.8)')), row=1, col=2)
    
    fig1.update_layout(
        title_text='Comparison of Original and Reconstructed PC1',
        template='plotly_white')
    fig1.show()
    
    loadings_df = pd.DataFrame({'loading': pc1_loadings, 'neuron_index': np.arange(data.shape[1])})
    top_5_neurons = loadings_df.sort_values('loading', ascending=False).head(5)['neuron_index'].values

    scaled_pc = (original_pc1_scores - original_pc1_scores.min()) / (original_pc1_scores.max() - original_pc1_scores.min())
    
    weighted_centered_data = data_centered * pc1_loadings
    mean_top5_activity = weighted_centered_data[:, top_5_neurons].mean(axis=1)
    
    scaled_nrn_activity = (mean_top5_activity - mean_top5_activity.min()) / (mean_top5_activity.max() - mean_top5_activity.min())
    
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=('Scaled PC1', 'Scaled Avg. Response of Top 5 Neurons'))
    
    fig2.add_trace(go.Scatter(y=scaled_pc.flatten(), mode='lines', name='Scaled PC1', line=dict(color='black')), row=1, col=1)
    fig2.add_trace(go.Scatter(y=scaled_nrn_activity, mode='lines', name='Scaled Neuron Avg.', line=dict(color='darkcyan')), row=1, col=2)
    
    fig2.update_layout(
        title_text='Scaled Comparison of PC1 and Top Neuron Activity',
        template='plotly_white',
        yaxis=dict(range=[-0.1, 1.1]),
        yaxis2=dict(range=[-0.1, 1.1]))
    fig2.show()


if __name__ == '__main__':
    # Load the processed data
    try:
        with np.load('session2_sampledata.npz') as npz_file:
            data = npz_file['data']
            labels = npz_file['labels']
        
        print("--- Running PCA Component Analysis ---")
        analyze_pca_components(data, labels)
        
        print("\n--- Running PC1 Reconstruction Analysis ---")
        reconstruct_and_plot_pc1(data)
        
        print("\nPC Analysis completed")

    except FileNotFoundError:
        print("Error: 'session2_sampledata.npz' not found.")
        print("Please run 'data_converter.py' first to generate the required data file.")

