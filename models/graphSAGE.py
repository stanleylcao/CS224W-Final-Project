"""
graphSAGE.py
Defines the GraphSAGE and GraphSAGEModel classes. Based on code from Colab 3.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter
# from torch_scatter import scatter
from config import config


class GraphSAGE(MessagePassing):
    """
    A single GraphSAGE layer that computes embeddings for graph nodes by aggregating 
    features from their local neighborhood.

    Supports neighborhood aggregation using the "mean" operation and optional 
    feature normalization.
    """

    def __init__(self, input_dim, output_dim, normalize=True, bias=True, **kwargs):
        """
        Initializes a GraphSAGE layer.

        Args:
            input_dim (int): Number of input features for each node.
            output_dim (int): Number of output features for each node.
            normalize (bool, optional): Whether to normalize the output embeddings. Defaults to True.
            bias (bool, optional): Whether to include a bias term in the linear transformations. Defaults to True.
            **kwargs: Additional arguments passed to the MessagePassing superclass.
        """
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
        """
        Resets the parameters of the linear layers to their initial values.
        """
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size=None):
        """
        Forward pass for GraphSAGE.

        Args:
            x (torch.Tensor): Node features of shape [num_nodes, in_channels].
            edge_index (torch.Tensor): Edge indices of the graph [2, num_edges].
            size (tuple, optional): Size of source and target node sets (for bipartite graphs). Defaults to None.

        Returns:
            torch.Tensor: Node embeddings of shape [num_nodes, out_channels].
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
        Constructs messages to be passed along edges during message propagation.

        Args:
            x_j (torch.Tensor): Features of neighboring nodes [num_edges, in_channels].

        Returns:
            torch.Tensor: Messages to be aggregated.
        """
        return x_j

    def aggregate(self, inputs, index, dim_size=None):
        """
        Aggregates messages from neighbors using the "mean" operation.

        Args:
            inputs (torch.Tensor): Messages to aggregate [num_edges, in_channels].
            index (torch.Tensor): Indices of target nodes for each edge.
            dim_size (int, optional): Total number of nodes. Defaults to None.

        Returns:
            torch.Tensor: Aggregated messages [num_nodes, in_channels].
        """
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')


class GraphSAGEModel(nn.Module):
    """
    A multi-layer GraphSAGE model for node-level embeddings or graph-level tasks.

    This model stacks multiple GraphSAGE layers and includes a post-message-passing
    module for classification or other downstream tasks.
    """
    def __init__(self, num_layers=2, normalize=True, dropout=0.1):
        """
        Initializes the GraphSAGE model.

        Args:
            num_layers (int, optional): Number of GraphSAGE layers. Defaults to 2.
            normalize (bool, optional): Whether to normalize the output embeddings. Defaults to True.
            dropout (float, optional): Dropout probability for regularization. Defaults to 0.1.
        """
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
            node_features (torch.Tensor): Node feature matrix [num_nodes, input_dim].
            edge_index (torch.Tensor): Edge index tensor [2, num_edges].

        Returns:
            torch.Tensor: Final node embeddings or logits [num_nodes, output_dim].
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