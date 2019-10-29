import torch
import torch.nn as nn
import torch.nn.functional as F
from rgcn_layer import RGCN_Layer
from concept_layer import Concept_Layer


class Model(torch.nn.Module):
    def __init__(self, args, data, all_node_embedding):
        super(Model, self).__init__()
        # self.node_embedding_g2 = nn.Parameter(data.x)
        self.all_node_embedding = nn.Parameter(all_node_embedding)
        self.rgcn_layer_1 = RGCN_Layer(args.node_dim, 0, data.num_nodes, args.hidden_dim, data.num_relations, args.num_bases)
        self.rgcn_layer_2 = RGCN_Layer(args.node_dim, 1, args.hidden_dim, args.num_classes, data.num_relations, args.num_bases)
        self.relu_use = args.relu_use
        self.concept_layer = Concept_Layer()
        self.num_nodes_g2 = data.num_nodes

    def forward(self, edge_index_g2, edge_type_g2, edge_index_g1):
        node_embedding_g1 = self.all_node_embedding
        x_g1 = self.concept_layer(node_embedding_g1, edge_index_g1)
        x_g1 = F.relu(x_g1)

        node_embedding_g2 = x_g1[0:self.num_nodes_g2,:]
        x_g2 = self.rgcn_layer_1(node_embedding_g2, edge_index_g2, edge_type_g2)
        if self.relu_use == 1:
            x_g2 = F.relu(x_g2)
        x_g2 = self.rgcn_layer_2(x_g2, edge_index_g2, edge_type_g2)

        return F.softmax(x_g2, dim=1)