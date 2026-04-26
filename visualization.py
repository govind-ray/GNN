"""
Visualization utilities for GTN-EEG model
Includes graph visualization, attention weights, and feature maps
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.patches import FancyBboxPatch


class ModelVisualizer:
    """
    Visualization tools for GTN-EEG model interpretation
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        
    def visualize_channel_graph(self, eeg_data, save_path=None):
        """
        Visualize the channel correlation graph
        
        Args:
            eeg_data: [1, num_channels, seq_length]
            save_path: Path to save the visualization
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get correlation graph
            eeg_tensor = torch.FloatTensor(eeg_data).unsqueeze(0).to(self.device)
            adj_matrix = self.model.graph_constructor(eeg_tensor).cpu().numpy()
        
        # Create network graph
        G = nx.from_numpy_array(adj_matrix)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Network graph
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                             node_color='lightblue',
                             node_size=500,
                             alpha=0.9,
                             ax=ax1)
        
        # Draw edges with weights
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(G, pos,
                             width=[w * 3 for w in weights],
                             alpha=0.5,
                             edge_color=weights,
                             edge_cmap=plt.cm.Blues,
                             ax=ax1)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, 
                              font_size=8,
                              ax=ax1)
        
        ax1.set_title('EEG Channel Correlation Graph', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Plot 2: Adjacency matrix heatmap
        sns.heatmap(adj_matrix, 
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   cbar_kws={'label': 'Correlation'},
                   ax=ax2)
        ax2.set_title('Channel Correlation Matrix', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Channel')
        ax2.set_ylabel('Channel')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
        
        return adj_matrix
    
    def visualize_gtn_attention(self, eeg_data, layer_idx=0, save_path=None):
        """
        Visualize GTN layer attention weights
        """
        self.model.eval()
        
        # Hook to capture intermediate outputs
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output
            return hook
        
        # Register hooks
        hooks = []
        for i, layer in enumerate(self.model.gtn_layers):
            if i == layer_idx:
                hook = layer.register_forward_hook(hook_fn(f'gtn_{i}'))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            eeg_tensor = torch.FloatTensor(eeg_data).unsqueeze(0).to(self.device)
            _ = self.model(eeg_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Get attention weights if available
        if f'gtn_{layer_idx}' in activations:
            attention = activations[f'gtn_{layer_idx}']
            
            if isinstance(attention, torch.Tensor):
                attention = attention.cpu().numpy()
                
                plt.figure(figsize=(10, 8))
                
                if len(attention.shape) == 3:  # [batch, nodes, features]
                    # Visualize as heatmap
                    sns.heatmap(attention[0], 
                              cmap='viridis',
                              cbar_kws={'label': 'Activation'},
                              xticklabels=False)
                    plt.title(f'GTN Layer {layer_idx} Node Representations', 
                            fontsize=14, fontweight='bold')
                    plt.xlabel('Feature Dimension')
                    plt.ylabel('Channel/Node')
                
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                else:
                    plt.show()
                
                plt.close()
    
    def plot_eeg_signals(self, eeg_data, channels_to_plot=None, save_path=None):
        """
        Plot raw EEG signals
        
        Args:
            eeg_data: [num_channels, seq_length]
            channels_to_plot: List of channel indices to plot (default: first 8)
        """
        if channels_to_plot is None:
            channels_to_plot = list(range(min(8, eeg_data.shape[0])))
        
        num_channels = len(channels_to_plot)
        
        fig, axes = plt.subplots(num_channels, 1, figsize=(14, 2 * num_channels))
        
        if num_channels == 1:
            axes = [axes]
        
        time = np.arange(eeg_data.shape[1]) / 500  # Assuming 500 Hz sampling rate
        
        for idx, (ax, ch) in enumerate(zip(axes, channels_to_plot)):
            signal = eeg_data[ch]
            ax.plot(time, signal, linewidth=0.5, color='darkblue')
            ax.set_ylabel(f'Ch {ch}\n(μV)', rotation=0, ha='right', va='center')
            ax.set_xlim(time[0], time[-1])
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                ax.set_title('EEG Signals', fontsize=14, fontweight='bold')
            if idx == num_channels - 1:
                ax.set_xlabel('Time (s)')
            else:
                ax.set_xticklabels([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_maps(self, eeg_data, save_path=None):
        """
        Visualize extracted features from CNN
        """
        self.model.eval()
        
        with torch.no_grad():
            eeg_tensor = torch.FloatTensor(eeg_data).unsqueeze(0).to(self.device)
            
            # Extract features for each channel
            channel_features = []
            for i in range(eeg_data.shape[0]):
                channel_signal = eeg_tensor[:, i:i+1, :]
                feat = self.model.feature_extractor(channel_signal)
                channel_features.append(feat.cpu().numpy())
            
            features = np.array(channel_features).squeeze()  # [num_channels, feature_dim]
        
        # Visualize
        plt.figure(figsize=(12, 8))
        
        sns.heatmap(features,
                   cmap='coolwarm',
                   center=0,
                   cbar_kws={'label': 'Feature Value'})
        
        plt.title('Extracted CNN Features per Channel', fontsize=14, fontweight='bold')
        plt.xlabel('Feature Dimension')
        plt.ylabel('EEG Channel')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
        
        return features
    
    def visualize_prediction(self, eeg_data, true_label=None, save_path=None):
        """
        Visualize model prediction with confidence scores
        """
        self.model.eval()
        
        with torch.no_grad():
            eeg_tensor = torch.FloatTensor(eeg_data).unsqueeze(0).to(self.device)
            outputs = self.model(eeg_tensor)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(probs)
        
        class_names = ['Normal', 'MCI', 'AD']
        colors = ['green', 'orange', 'red']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Prediction probabilities
        bars = ax1.bar(class_names, probs, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Probability', fontsize=12)
        ax1.set_title('Class Prediction Probabilities', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Highlight predicted class
        bars[pred_class].set_alpha(1.0)
        bars[pred_class].set_linewidth(3)
        
        # Add probability values on bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prob:.2%}',
                   ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Prediction summary
        ax2.axis('off')
        
        # Create prediction box
        pred_text = f"Predicted: {class_names[pred_class]}\nConfidence: {probs[pred_class]:.2%}"
        if true_label is not None:
            true_text = f"\nTrue Label: {class_names[true_label]}"
            correct = "✓ CORRECT" if pred_class == true_label else "✗ INCORRECT"
            pred_text += true_text + f"\n\n{correct}"
        
        bbox = FancyBboxPatch((0.1, 0.3), 0.8, 0.4,
                             boxstyle="round,pad=0.05",
                             edgecolor=colors[pred_class],
                             facecolor=colors[pred_class],
                             alpha=0.2,
                             linewidth=3)
        ax2.add_patch(bbox)
        
        ax2.text(0.5, 0.5, pred_text,
                ha='center', va='center',
                fontsize=14, fontweight='bold',
                transform=ax2.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
        
        return pred_class, probs


def create_model_architecture_diagram(save_path='model_architecture.png'):
    """
    Create a diagram showing the model architecture
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define components
    components = [
        {'name': 'EEG Input\n[channels × time]', 'pos': (5, 11), 'color': 'lightblue'},
        {'name': 'CNN Feature\nExtractor', 'pos': (5, 9.5), 'color': 'lightgreen'},
        {'name': 'Channel Features\n[channels × features]', 'pos': (5, 8), 'color': 'lightblue'},
        {'name': 'Graph Constructor', 'pos': (2, 6.5), 'color': 'lightyellow'},
        {'name': 'Adjacency Matrix', 'pos': (2, 5), 'color': 'lightcoral'},
        {'name': 'GTN Layer 1', 'pos': (5, 6.5), 'color': 'lightgreen'},
        {'name': 'GTN Layer 2', 'pos': (5, 5), 'color': 'lightgreen'},
        {'name': 'Graph Embeddings', 'pos': (5, 3.5), 'color': 'lightblue'},
        {'name': 'Global Pooling', 'pos': (5, 2), 'color': 'lightyellow'},
        {'name': 'Classification Head', 'pos': (5, 0.5), 'color': 'lightgreen'},
    ]
    
    # Draw components
    for comp in components:
        bbox = FancyBboxPatch(
            (comp['pos'][0] - 0.8, comp['pos'][1] - 0.3),
            1.6, 0.6,
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor=comp['color'],
            linewidth=2
        )
        ax.add_patch(bbox)
        ax.text(comp['pos'][0], comp['pos'][1], comp['name'],
               ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrows
    arrows = [
        (5, 10.7, 5, 9.8),
        (5, 9.2, 5, 8.3),
        (5, 7.7, 4, 6.8),
        (5, 7.7, 5, 6.8),
        (2.8, 6.5, 4.2, 6.5),
        (2, 5.3, 4.5, 6.2),
        (5, 6.2, 5, 5.3),
        (5, 4.7, 5, 3.8),
        (5, 3.2, 5, 2.3),
        (5, 1.7, 5, 0.8),
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    # Add title
    ax.text(5, 11.8, 'Hybrid GTN-EEG Model Architecture',
           ha='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Architecture diagram saved to {save_path}")


if __name__ == "__main__":
    print("Model Visualization Utilities")
    print("=" * 50)
    
    # Create architecture diagram
    create_model_architecture_diagram('model_architecture.png')
    
    print("\nArchitecture diagram created!")
