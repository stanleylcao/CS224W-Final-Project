import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter
# from torch_scatter import scatter
from config import config


class GraphSAGE(MessagePassing):
    """
    GraphSAGE layer that computes embeddings for graph nodes using neighborhood aggregation.
    """

    def __init__(self, input_dim, output_dim, normalize=True, bias=True, **kwargs):
        # Aggregates messages using 'mean'
        super(GraphSAGE, self).__init__(aggr='mean', **kwargs)

        self.in_channels = input_dim
        self.out_channels = output_dim
        self.normalize = normalize

        # Linear transformations for the node's own features and aggregated neighborhood features
        self.lin_l = nn.Linear(self.in_channels, self.out_channels, bias=bias)
        self.lin_r = nn.Linear(self.in_channels, self.out_channels, bias=bias)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Resets parameters to their initial values."""
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size=None):
        """
        Forward pass for GraphSAGE.

        Args:
            x: Node features of shape [num_nodes, in_channels].
            edge_index: Edge index of the graph [2, num_edges].
            size: Size of source and target node sets (for bipartite graphs).

        Returns:
            Node embeddings of shape [num_nodes, out_channels].
        """
        # Perform message passing
        out = self.propagate(edge_index, x=(x, x), size=size)

        # Apply linear transformations and skip connections
        out = self.lin_l(x) + self.lin_r(out)

        # Normalize output if specified
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out

    def message(self, x_j):
        """
        Constructs messages for all edges. Here, simply passes neighbor features (x_j).
        """
        return x_j

    def aggregate(self, inputs, index, dim_size=None):
        """
        Aggregates messages from neighbors using 'mean'.
        """
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')


class GraphSAGEModel(nn.Module):
    """
    A multi-layer GraphSAGE model for node-level embeddings or graph-level tasks.
    """

    def __init__(self, num_layers=2, normalize=True, dropout=0.1):
        super(GraphSAGEModel, self).__init__()

        input_dim = config["input_dim"]
        hidden_dim = config["hidden_dim"]
        output_dim = config["output_dim"]

        assert num_layers >= 1, "GraphSAGEModel requires at least one layer."

        # Build layers
        self.layers = nn.ModuleList()
        self.layers.append(
            GraphSAGE(input_dim, hidden_dim, normalize=normalize))

        for _ in range(num_layers - 1):
            self.layers.append(
                GraphSAGE(hidden_dim, hidden_dim, normalize=normalize))

        # Post-message-passing module
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.dropout = dropout

    def forward(self, node_features, edge_index):
        """
        Forward pass for the GraphSAGE model.

        Args:
            node_features: Node feature matrix [num_nodes, input_dim].
            edge_index: Edge index tensor [2, num_edges].

        Returns:
            Final node embeddings or logits [num_nodes, output_dim].
        """
        x = node_features

        # Pass through GraphSAGE layers
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Post-message-passing transformation
        x = self.post_mp(x)
        return x


# Optional: Define a loss function for training
def graphsage_loss(pred, target):
    """
    Loss function for GraphSAGE model.
    """
    return F.mse_loss(pred, target)
