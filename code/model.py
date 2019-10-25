import torch
import torch.nn as nn
import torch.nn.functional as F
from rgcn_layer import RGCN_Layer


class Model(torch.nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.node_embedding_g2 = nn.Parameter(data.x)
        self.rgcn_layer_1 = RGCN_Layer(args.node_dim, 0, data.num_nodes, args.hidden_dim, data.num_relations, args.num_bases)
        self.rgcn_layer_2 = RGCN_Layer(args.node_dim, 1, args.hidden_dim, args.num_classes, data.num_relations, args.num_bases)

    def forward(self, edge_index, edge_type):
        x = self.rgcn_layer_1(self.node_embedding_g2, edge_index, edge_type)
        x = F.relu(x)

        x = self.rgcn_layer_2(x, edge_index, edge_type)
        return F.softmax(x, dim=1)