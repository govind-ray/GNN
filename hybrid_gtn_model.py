"""
Hybrid GTN Model for EEG-based Alzheimer's Detection
Combines Graph Transformer Networks with EEG signal processing

Based on:
- Graph Transformer Networks (Yun et al., NeurIPS 2019)
- Custom CNN-based Feature Extraction for EEG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Linear
from typing import List, Tuple, Optional, Union

# Set device agnostic code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GTLayer(nn.Module):
    """
    Graph Transformer Layer
    Learns to transform meta-path graphs
    """
    def __init__(self, in_channels: int, out_channels: int, num_edges: int, num_nodes: int):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edges = num_edges
        self.num_nodes = num_nodes
        
        # Learnable weight matrices for each channel
        self.weight = nn.Parameter(torch.Tensor(num_edges, in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(num_edges, 1, out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, A: List[torch.Tensor], X: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            A: List of adjacency matrices [num_edges, num_nodes, num_nodes]
            X: Node features [num_nodes, in_channels] (optional)
        Returns:
            Ws: Stacked transformed features [num_edges, num_nodes, out_channels]
        """
        # If no node features provided, use identity
        if X is None:
            X = torch.eye(self.num_nodes).to(A[0].device)
        
        # Transform each edge type
        Ws = []
        for i in range(self.num_edges):
            # Compute weighted adjacency: A[i] * weight[i]
            # Matrix multiplication: (N, N) x (N, C_in) -> (N, C_in)
            H = torch.mm(A[i], X)
            # Linear transform: (N, C_in) x (C_in, C_out) -> (N, C_out)
            H = torch.mm(H, self.weight[i]) + self.bias[i]
            Ws.append(H)
        
        # Stack and process
        Ws = torch.stack(Ws, dim=0)  # [num_edges, num_nodes, out_channels]
        
        return Ws


class FastGTNLayer(nn.Module):
    """
    Fast GTN Layer with channel aggregation.
    Efficiently computes interactions without explicitly constructing all meta-paths.
    """
    def __init__(self, in_channels: int, out_channels: int, num_edges: int, num_nodes: int, num_channels: int):
        super(FastGTNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edges = num_edges
        self.num_nodes = num_nodes
        self.num_channels = num_channels
        
        # Soft selection parameters for meta-paths
        self.weight = nn.Parameter(torch.Tensor(num_edges, num_channels))
        
        # Transformation for aggregated graph
        self.linear = Linear(in_channels, out_channels)
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, A: List[torch.Tensor], X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A: List of adjacency matrices
            X: Node features [num_channels, feature_dim]
        Returns:
            H: Updated node features [num_nodes, out_channels]
        """
        # Soft selection of meta-paths
        filter_weights = F.softmax(self.weight, dim=0)
        
        # Aggregate adjacency matrices
        A_agg = torch.zeros_like(A[0])
        for i in range(self.num_edges):
            for j in range(self.num_channels):
                A_agg += filter_weights[i, j] * A[i]
        
        # Apply graph convolution
        H = torch.mm(A_agg, X)
        H = self.linear(H) + self.bias
        
        return H


class EEGFeatureExtractor(nn.Module):
    """
    Extract features from raw EEG signals
    Uses 1D CNN for temporal feature extraction from time-series data.
    """
    def __init__(self, num_channels: int = 64, seq_length: int = 1000, feature_dim: int = 128):
        super(EEGFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, feature_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(feature_dim)
        self.pool3 = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_channels, seq_length]
        Returns:
            features: [batch, feature_dim]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        return x.squeeze(-1)  # [batch, feature_dim]


class ChannelCorrelationGraph(nn.Module):
    """
    Constructs graph based on EEG channel correlations dynamically.
    """
    def __init__(self, num_channels: int = 64, threshold: float = 0.3):
        super(ChannelCorrelationGraph, self).__init__()
        self.num_channels = num_channels
        self.threshold = threshold
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_channels, seq_length]
        Returns: 
            adjacency matrix [num_channels, num_channels]
        """
        batch_size = x.size(0)
        
        # Compute correlation matrix across batch
        correlations = []
        for i in range(batch_size):
            corr = torch.corrcoef(x[i])
            correlations.append(corr)
        
        avg_corr = torch.stack(correlations).mean(dim=0)
        
        # Threshold to create adjacency matrix
        adj = (torch.abs(avg_corr) > self.threshold).float()
        
        # Add self-loops
        adj = adj + torch.eye(self.num_channels).to(x.device)
        
        # Normalize
        deg = adj.sum(dim=1, keepdim=True)
        adj = adj / (deg + 1e-6)
        
        return adj


class HybridGTN_EEG(nn.Module):
    """
    Hybrid Model: GTN + EEG Feature Extraction
    
    Architecture:
    1. EEG Feature Extractor (CNN) applied per channel
    2. Channel Correlation Graph Construction
    3. GTN Layers for learning channel relationships
    4. Classification head
    """
    def __init__(self, 
                 num_eeg_channels: int = 64,
                 seq_length: int = 1000,
                 num_classes: int = 3,  # Normal, MCI, AD
                 feature_dim: int = 128,
                 gtn_hidden_dim: int = 64,
                 num_gtn_layers: int = 2,
                 num_graph_channels: int = 2,
                 dropout: float = 0.5):
        super(HybridGTN_EEG, self).__init__()
        
        self.num_eeg_channels = num_eeg_channels
        self.num_classes = num_classes
        
        # EEG Feature Extraction
        # Note: We extract features for each channel individually
        self.feature_extractor = EEGFeatureExtractor(
            num_channels=1,
            seq_length=seq_length,
            feature_dim=feature_dim
        )
        
        # Graph Construction
        self.graph_constructor = ChannelCorrelationGraph(
            num_channels=num_eeg_channels
        )
        
        # GTN Layers - FIXED: Complete initialization
        self.gtn_layers = nn.ModuleList()
        in_dim = feature_dim
        for i in range(num_gtn_layers):
            self.gtn_layers.append(
                FastGTNLayer(
                    in_channels=in_dim,
                    out_channels=gtn_hidden_dim,
                    num_edges=3,  # Different types of connections
                    num_nodes=num_eeg_channels,
                    num_channels=num_graph_channels
                )
            )
            in_dim = gtn_hidden_dim
        
        # Classification Head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = Linear(gtn_hidden_dim * num_eeg_channels, 256)
        self.fc2 = Linear(256, 128)
        self.fc3 = Linear(128, num_classes)
        
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
        
    def create_multi_edge_graphs(self, base_adj: torch.Tensor) -> List[torch.Tensor]:
        """
        Create multiple types of adjacency matrices
        1. Original correlation-based
        2. Distance-based (spatial proximity - currently simplified)
        3. Frequency-based (global connectivity)
        """
        num_nodes = base_adj.size(0)
        device = base_adj.device
        
        # Edge type 1: Correlation-based (already computed)
        A1 = base_adj
        
        # Edge type 2: Spatial proximity (simplified - can be improved with actual electrode positions)
        A2 = torch.eye(num_nodes).to(device)
        # Add connections to neighboring channels
        for i in range(num_nodes - 1):
            A2[i, i + 1] = 0.5
            A2[i + 1, i] = 0.5
        
        # Edge type 3: Global connectivity
        A3 = torch.ones(num_nodes, num_nodes).to(device) / num_nodes
        
        return [A1, A2, A3]
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, num_channels, seq_length]
        Returns:
            Tuple: (logits, gtn_embeddings)
        """
        batch_size = x.size(0)
        
        # Step 1: Extract temporal features for each channel
        # Reshape to process each channel's signal
        # We need to apply feature extraction per channel, treating each channel as an independent sample first
        # But wait, feature_extractor expects (Batch, Channels, Time).
        # We want to use it for EACH channel of the ORIGINAL batch.
        
        # Reshape input: (B, C, T) -> (B*C, 1, T)
        x_reshaped = x.view(-1, 1, x.size(-1))
        
        # Extract features: (B*C, 1, T) -> (B*C, feature_dim)
        features_flat = self.feature_extractor(x_reshaped)
        
        # Reshape back: (B, C, feature_dim)
        X = features_flat.view(batch_size, self.num_eeg_channels, -1)
        
        # Step 2: Construct channel correlation graph
        base_adj = self.graph_constructor(x)  # [num_channels, num_channels]
        
        # Create multiple edge types
        A_list = self.create_multi_edge_graphs(base_adj)
        
        # Step 3: Apply GTN layers
        # Process each sample in batch
        batch_outputs = []
        for b in range(batch_size):
            node_features = X[b]  # [num_channels, feature_dim]
            
            for gtn_layer in self.gtn_layers:
                node_features = gtn_layer(A_list, node_features)
                node_features = F.relu(node_features)
            
            batch_outputs.append(node_features)
        
        # Stack batch outputs: [batch, num_channels, gtn_hidden_dim]
        gtn_output = torch.stack(batch_outputs, dim=0)
        
        # Step 4: Global pooling and classification
        # Flatten: [batch, num_channels * gtn_hidden_dim]
        out = gtn_output.reshape(batch_size, -1)
        
        out = self.dropout(out)
        out = F.relu(self.batch_norm1(self.fc1(out)))
        
        out = self.dropout(out)
        out = F.relu(self.batch_norm2(self.fc2(out)))
        
        out = self.fc3(out)
        
        return out, gtn_output  # Return both predictions and graph embeddings


class SimpleGTN_EEG(nn.Module):
    """
    Simplified version using original GTN architecture
    """
    def __init__(self, 
                 num_eeg_channels: int = 64,
                 seq_length: int = 1000,
                 num_classes: int = 3,
                 feature_dim: int = 128,
                 num_gtn_layers: int = 2,
                 dropout: float = 0.5):
        super(SimpleGTN_EEG, self).__init__()
        
        self.num_eeg_channels = num_eeg_channels
        
        # Feature extraction
        self.feature_extractor = EEGFeatureExtractor(
            num_channels=1,
            seq_length=seq_length,
            feature_dim=feature_dim
        )
        
        # Graph construction
        self.graph_constructor = ChannelCorrelationGraph(
            num_channels=num_eeg_channels
        )
        
        # GTN layers
        self.gtn_layers = nn.ModuleList()
        for i in range(num_gtn_layers):
            self.gtn_layers.append(
                GTLayer(
                    in_channels=feature_dim if i == 0 else 64,
                    out_channels=64,
                    num_edges=3,
                    num_nodes=num_eeg_channels
                )
            )
        
        # Pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Extract features per channel (optimized)
        x_reshaped = x.view(-1, 1, x.size(-1))
        features_flat = self.feature_extractor(x_reshaped)
        X = features_flat.view(batch_size, self.num_eeg_channels, -1)
        
        # Construct graph
        base_adj = self.graph_constructor(x)
        A1 = base_adj
        A2 = torch.eye(self.num_eeg_channels).to(x.device)
        A3 = torch.ones(self.num_eeg_channels, self.num_eeg_channels).to(x.device) / self.num_eeg_channels
        A_list = [A1, A2, A3]
        
        # Apply GTN
        batch_outputs = []
        for b in range(batch_size):
            node_features = X[b]
            
            for gtn_layer in self.gtn_layers:
                Ws = gtn_layer(A_list, node_features)
                # Aggregate across edge types
                node_features = Ws.mean(dim=0)
                node_features = F.relu(node_features)
            
            # Global pooling across channels
            # node_features: [num_channels, feature_dim]
            pooled = node_features.mean(dim=0)  # [feature_dim]
            batch_outputs.append(pooled)
        
        out = torch.stack(batch_outputs, dim=0)  # [batch, feature_dim]
        out = self.classifier(out)
        
        return out


if __name__ == "__main__":
    # Test the models
    print("Testing Hybrid GTN-EEG Models")
    print("=" * 50)
    
    # Create dummy data
    batch_size = 4
    num_channels = 32  # Reduced for testing
    seq_length = 1000
    
    x = torch.randn(batch_size, num_channels, seq_length)
    
    # Test HybridGTN_EEG
    print("\n1. Testing HybridGTN_EEG:")
    model1 = HybridGTN_EEG(
        num_eeg_channels=num_channels,
        seq_length=seq_length,
        num_classes=3,
        feature_dim=128,
        gtn_hidden_dim=64,
        num_gtn_layers=2
    )
    
    output1, embeddings1 = model1(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output1.shape}")
    print(f"   Embeddings shape: {embeddings1.shape}")
    print(f"   Total parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    # Test SimpleGTN_EEG
    print("\n2. Testing SimpleGTN_EEG:")
    model2 = SimpleGTN_EEG(
        num_eeg_channels=num_channels,
        seq_length=seq_length,
        num_classes=3,
        feature_dim=128,
        num_gtn_layers=2
    )
    
    output2 = model2(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output2.shape}")
    print(f"   Total parameters: {sum(p.numel() for p in model2.parameters()):,}")
    
    print("\n" + "=" * 50)
    print("Models tested successfully!")
