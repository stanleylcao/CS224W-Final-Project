import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, return_embeds=False):
        super(GCN, self).__init__()
        
        # Initialize GCN layers and batch normalization
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # Add GCN layers
        for i in range(num_layers):
            self.convs.append(GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim))
            if i < num_layers - 1:
                self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Final classification layer (linear transformation)
        self.final_layer = GCNConv(hidden_dim, output_dim)
        
        # Dropout probability
        self.dropout = dropout
        
        # Flag to skip final classification and return embeddings
        self.return_embeds = return_embeds
    
    def reset_parameters(self):
        """
        Reset parameters of all layers.
        """
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.final_layer.reset_parameters()
    
    def forward(self, x, edge_index):
        """
        Forward pass through the GCN model.
        
        Args:
            x (Tensor): Node features of shape [num_nodes, input_dim].
            edge_index (Tensor): Edge indices of shape [2, num_edges].
        
        Returns:
            Tensor: Node-level embeddings or classification output.
        """
        # Apply GCN layers with ReLU, BatchNorm, and Dropout
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.bns):  # BatchNorm is applied to all but the last layer
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer (classification or output embeddings)
        out = self.final_layer(x, edge_index)
        if not self.return_embeds:
            out = F.log_softmax(out, dim=1)  # For classification
        return out
