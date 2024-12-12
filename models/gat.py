import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax, scatter
# from torch_scatter import scatter
from typing import Optional, Tuple
from config import config


class GAT(MessagePassing):
    """
    A single Graph Attention Network (GAT) layer using multi-head attention.
    """

    def __init__(self, **kwargs):
        super().__init__(node_dim=0, aggr='add', **kwargs)

        self.in_channels = config["in_channels"]
        self.out_channels = config["out_channels"]
        self.heads = config["heads"]
        self.negative_slope = config["negative_slope"]
        self.dropout = config["dropout"]

        # Linear transformations for node embeddings
        self.lin_l = nn.Linear(self.in_channels, self.heads * self.out_channels, bias=False)
        self.lin_r = nn.Linear(self.in_channels, self.heads * self.out_channels, bias=False)

        # Attention coefficients
        self.att_l = nn.Parameter(torch.Tensor(1, self.heads, self.out_channels))
        self.att_r = nn.Parameter(torch.Tensor(1, self.heads, self.out_channels))

        # Initialization of weights
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, size: Optional[Tuple[int]] = None):
        """
        Forward pass of GAT layer.

        Args:
            x: Node features of shape [num_nodes, in_channels].
            edge_index: Edge indices of shape [2, num_edges].
            size: Optional tuple containing the sizes of source and target nodes.

        Returns:
            Updated node embeddings of shape [num_nodes, heads * out_channels].
        """
        H, C = self.heads, self.out_channels

        # Transform node embeddings for left and right nodes
        x_l = self.lin_l(x).view(-1, H, C)  # [num_nodes, heads, out_channels]
        x_r = self.lin_r(x).view(-1, H, C)  # [num_nodes, heads, out_channels]

        # Compute initial attention scores
        alpha_l = (x_l * self.att_l).sum(dim=-1)  # [num_nodes, heads]
        alpha_r = (x_r * self.att_r).sum(dim=-1)  # [num_nodes, heads]

        # Propagate attention scores and node embeddings
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)

        # Flatten multi-head outputs
        out = out.view(-1, H * C)
        return out

    def message(self, x_j: torch.Tensor, alpha_j: torch.Tensor, alpha_i: torch.Tensor, index: torch.Tensor,
                ptr: Optional[torch.Tensor], size_i: Optional[int]):
        """
        Compute attention scores and apply them to node embeddings.
        """
        # Compute attention weights
        alpha = F.leaky_relu(alpha_i + alpha_j, self.negative_slope)
        # Normalize attention scores
        alpha = softmax(alpha, index=index, ptr=ptr, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout,
                          training=self.training)  # Apply dropout

        # Multiply attention weights with embeddings
        out = x_j * alpha.unsqueeze(-1)
        return out

    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor, dim_size: Optional[int] = None):
        """
        Aggregate messages from neighbors.
        """
        return scatter(inputs, index, dim=0, dim_size=dim_size, reduce='sum')


class GATModel(nn.Module):
    """
    A multi-layer Graph Attention Network (GAT) model.
    Outputs Q-values (`output_dim` == )
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=2, num_layers=2, dropout=0.1):
        super().__init__()

        assert num_layers >= 1, "GATModel requires at least one layer."

        self.layers = nn.ModuleList()
        self.layers.append(
            GAT(input_dim, hidden_dim, heads=num_heads, dropout=dropout))

        for _ in range(num_layers - 1):
            self.layers.append(
                GAT(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))

        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.dropout = dropout

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor):
        """
        Forward pass for the GAT model.

        Args:
            node_features: Node feature matrix [num_nodes, input_dim].
            edge_index: Edge index tensor [2, num_edges].

        Returns:
            Final output logits [num_nodes, output_dim].
        """
        x = node_features

        # Pass through GAT layers
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Post message-passing (e.g., classification or Q-value prediction)
        x = self.post_mp(x)
        return x


# Optional: Define a loss function if needed
def gat_loss(pred, target):
    return F.mse_loss(pred, target)
