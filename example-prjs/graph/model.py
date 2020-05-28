"""
This module implements the PyTorch modules that define the
message-passing graph neural networks for hit or segment classification.
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

cuda=True

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
    
    def set_mask(self, mask):
        self.mask = mask
        self.weight.data = self.weight.data*self.mask.data
        self.mask_flag = True
    
    def get_mask(self):
        print(self.mask_flag)
        return self.mask
    
    def forward(self, x):
        if self.mask_flag == True:
            weight = self.weight*self.mask
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)
        

class EdgeNetwork(nn.Module):
    """
    A module which computes weights for edges of the graph.
    For each edge, it selects the associated nodes' features
    and applies some fully-connected network layers with a final
    sigmoid activation.
    """
    def __init__(self, input_dim, hidden_dim=8, hidden_activation=nn.Tanh, mask=None):
        super(EdgeNetwork, self).__init__()
        self.network = nn.Sequential(
            MaskedLinear(input_dim*2, hidden_dim),
            hidden_activation(),
            MaskedLinear(hidden_dim, 1),
            nn.Sigmoid())
        self.mask = mask
        if self.mask is not None:
            self.network[0].set_mask(self.mask[0])
            self.network[2].set_mask(self.mask[1])
        #self.network[0].weight.register_hook(self.maskgrads0)
        #self.network[2].weight.register_hook(self.maskgrads1)
        
    def maskgrads0(self, grad):
        if self.mask is not None:
            print('grad', grad.size())
            print('self.mask[0]', self.mask[0].size())
            return grad * self.mask[0]
    
    def maskgrads1(self, grad):
        if self.mask is not None:
            print('grad', grad.size())
            print('self.mask[1]', self.mask[1].size())
            return grad * self.mask[1]
    
    def forward(self, X, Ri, Ro):
        # Select the features of the associated nodes
        bo = torch.bmm(Ro.transpose(1, 2), X)
        bi = torch.bmm(Ri.transpose(1, 2), X)
        B = torch.cat([bo, bi], dim=2)
        # Mask these weights
        #if self.mask is not None:
            #self.network[0].weight.data[self.mask[0]!=1] = 0
            #self.network[2].weight.data[self.mask[1]!=1] = 0
            #self.network[0].weight.register_hook(self.maskgrads0)
            #self.network[2].weight.register_hook(self.maskgrads1)
        # Apply the network to each edge
        return self.network(B).squeeze(-1)


class NodeNetwork(nn.Module):
    """
    A module which computes new node features on the graph.
    For each node, it aggregates the neighbor node features
    (separately on the input and output side), and combines
    them with the node's previous features in a fully-connected
    network to compute the new features.
    """
    def __init__(self, input_dim, output_dim, hidden_activation=nn.Tanh, mask=None):
        super(NodeNetwork, self).__init__()
        self.network = nn.Sequential(
            MaskedLinear(input_dim*3, output_dim),
            hidden_activation(),
            MaskedLinear(output_dim, output_dim),
            hidden_activation())
        self.mask = mask
        if mask is not None:
            self.network[0].set_mask(self.mask[0])
            self.network[2].set_mask(self.mask[1])
        #self.network[0].weight.register_hook(self.maskgrads0)
        #self.network[2].weight.register_hook(self.maskgrads1)
        
    def maskgrads0(self, grad):
        if self.mask is not None:
            return grad * self.mask[0]
    
    def maskgrads1(self, grad):
        if self.mask is not None:
            return grad * self.mask[1]
        
    def forward(self, X, e, Ri, Ro):
        bo = torch.bmm(Ro.transpose(1, 2), X)
        bi = torch.bmm(Ri.transpose(1, 2), X)
        Rwo = Ro * e[:,None]
        Rwi = Ri * e[:,None]
        mi = torch.bmm(Rwi, bo)
        mo = torch.bmm(Rwo, bi)
        M = torch.cat([mi, mo, X], dim=2)
        # Mask these weights
        #if self.mask is not None:
            #self.network[0].weight = torch.nn.Parameter(self.network[0].weight * self.mask[0])
            #self.network[2].weight = torch.nn.Parameter(self.network[2].weight * self.mask[1])
        return self.network(M)

class SegmentClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=8, n_iters=3, hidden_activation=nn.Tanh, masks_e=None, masks_n=None):
        super(SegmentClassifier, self).__init__()
        self.n_iters = n_iters
        # Setup the input network
        self.input_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            hidden_activation())
        # Setup the edge network
        self.edge_network = EdgeNetwork(input_dim+hidden_dim, hidden_dim, hidden_activation, masks_e)
        # Setup the node layers
        self.node_network = NodeNetwork(input_dim+hidden_dim, hidden_dim, hidden_activation, masks_n)

    def forward(self, inputs):
        """Apply forward pass of the model"""
        X, Ri, Ro = inputs
        # Apply input network to get hidden representation
        H = self.input_network(X)
        # Shortcut connect the inputs onto the hidden representation
        H = torch.cat([H, X], dim=-1)
        # Loop over iterations of edge and node networks
        for i in range(self.n_iters):
            # Apply edge network
            e = self.edge_network(H, Ri, Ro)
            # Apply node network
            H = self.node_network(H, e, Ri, Ro)
            # Shortcut connect the inputs onto the hidden representation
            H = torch.cat([H, X], dim=-1)
        # Apply final edge network
        return self.edge_network(H, Ri, Ro)

