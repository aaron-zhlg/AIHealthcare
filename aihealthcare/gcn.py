"""Simple graph convolution network for FC-based classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """Symmetric normalization with self-loops. Input shape: (B, N, N)."""
    adj = adjacency.clone()
    batch_size, num_nodes, _ = adj.shape
    eye = torch.eye(num_nodes, device=adj.device).unsqueeze(0).expand(batch_size, -1, -1)
    adj = adj + eye
    degree = adj.sum(dim=-1).clamp(min=1e-6)
    inv_sqrt = degree.pow(-0.5)
    adj = inv_sqrt.unsqueeze(-1) * adj * inv_sqrt.unsqueeze(-2)
    return adj


class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # node_features: (B, N, F), adjacency: (B, N, N)
        support = self.linear(node_features)
        return torch.bmm(adjacency, support)


class SimpleGCN(nn.Module):
    """Two-layer GCN with global mean pooling for graph classification."""

    def __init__(
        self,
        in_features: int = 111,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv1 = GraphConvLayer(in_features, hidden_dim)
        self.conv2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        adj_norm = normalize_adjacency(adjacency)

        x = self.conv1(node_features, adj_norm)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, adj_norm)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        graph_embedding = x.mean(dim=1)
        return self.classifier(graph_embedding)
