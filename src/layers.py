"""
GIN, RGIN, attention
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl.nn.pytorch import RelGraphConv

class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction

        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction

        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN(nn.Module):
    """GIN model"""
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 learn_eps, neighbor_pooling_type):
        """model parameters setting

        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)

        """
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, g, h, *args):
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        # return score_over_layer
        return hidden_rep


class RGCN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, num_rels,
                 regularizer="basis", num_bases=30, self_loop=False):
        super(RGCN, self).__init__()
        self.num_layers = num_layers
        self.num_rels = num_rels

        if num_bases == -1:
            num_bases = num_rels

        self.rgcnlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.rgcnlayers.append(
                    RelGraphConv(input_dim, hidden_dim, num_rels, regularizer,
                                num_bases, self_loop=self_loop)
                )
            else:
                self.rgcnlayers.append(
                    RelGraphConv(hidden_dim, hidden_dim, num_rels, regularizer,
                                num_bases, self_loop=self_loop)
                )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, g, h, etypes, norm=None):
        hidden_rep = [h]

        for i in range(self.num_layers - 1):
            h = self.rgcnlayers[i](g, h, etypes, norm)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        
        return hidden_rep



class RGIN(nn.Module):
    """ The only difference of our implementation to RGCN is to use an additional MLP to 
    each layer.
    """
    def __init__(self, num_layers, input_dim, hidden_dim, num_rels,
                 regularizer="basis", num_bases=30, self_loop=False, num_mlp_layers=2):
        super(RGIN, self).__init__()
        self.num_layers = num_layers
        self.num_rels = num_rels

        if num_bases == -1:
            num_bases = num_rels

        self.rgcnlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
                self.rgcnlayers.append(
                    RelGraphConv(input_dim, hidden_dim, num_rels, regularizer,
                                num_bases, self_loop=self_loop, activation=mlp)
                )
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
                self.rgcnlayers.append(
                    RelGraphConv(hidden_dim, hidden_dim, num_rels, regularizer,
                                num_bases, self_loop=self_loop, activation=mlp)
                )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, g, h, etypes, norm=None):
        hidden_rep = [h]

        for i in range(self.num_layers - 1):
            h = self.rgcnlayers[i](g, h, etypes, norm)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        
        return hidden_rep


from dgl.nn.pytorch.conv.hgtconv import HGTConv
class HGT(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, num_heads=4, num_ntypes=1, num_etypes=16):
        super(HGT, self).__init__()
        assert hidden_dim % num_heads == 0
        self.num_layers = num_layers
        # output size = head size * number of heads
        head_size = hidden_dim // num_heads
        self.num_ntypes = num_ntypes
        self.num_etypes = num_etypes

        self.hgtlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.hgtlayers.append(
                    # RelGraphConv(input_dim, hidden_dim, num_rels, regularizer,
                    #             num_bases, self_loop=self_loop)
                    HGTConv(input_dim, head_size, num_heads, num_ntypes, num_etypes, use_norm=True),
                )
            else:
                self.hgtlayers.append(
                    # RelGraphConv(hidden_dim, hidden_dim, num_rels, regularizer,
                    #             num_bases, self_loop=self_loop)
                    HGTConv(hidden_dim, head_size, num_heads, num_ntypes, num_etypes, use_norm=True),
                )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, g, h, etypes):
        etypes = g.edata['etypes'].to(h.device)
        if self.num_ntypes == 1:
            ntypes = torch.zeros(g.ndata['ntypes'].size(), dtype=torch.long, device=h.device)
        else:
            ntypes = g.ndata['ntypes'].to(h.device)
        
        hidden_rep = [h]

        for i in range(self.num_layers - 1):
            h = self.hgtlayers[i](g, h, ntypes, etypes)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        
        return hidden_rep