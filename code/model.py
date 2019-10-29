
from concept_layer import Concept_Layer
from rgcn_layer import RGCN_Layer
import torch.nn.functional as F
import torch.nn as nn
from inits import *

class Model(torch.nn.Module):
    def __init__(self, args, data, all_node_embedding, left_common):
        super(Model, self).__init__()
        self.all_node_embedding = nn.Parameter(all_node_embedding)
        self.rgcn_layer_1 = RGCN_Layer(args.node_dim, 0, data.num_nodes, args.hidden_dim, data.num_relations, args.num_bases)
        self.rgcn_layer_2 = RGCN_Layer(args.node_dim, 1, args.hidden_dim, args.final_dim, data.num_relations, args.num_bases)
        self.relu_use = args.relu_use
        self.concept_layer = Concept_Layer()
        self.num_nodes_g2 = data.num_nodes
        self.weights = nn.Parameter(torch.rand((args.final_dim, 1)))  # 200*1
        self.final_dim = args.final_dim
        self.left_common = left_common


    def forward(self, edge_index_g2, edge_type_g2, edge_index_g1):
        node_embedding_g1 = self.all_node_embedding
        x_g1 = self.concept_layer(node_embedding_g1, edge_index_g1)
        x_g1 = F.relu(x_g1)  # 26989*200

        node_embedding_g2 = x_g1[0:self.num_nodes_g2,:]
        x_g2 = self.rgcn_layer_1(node_embedding_g2, edge_index_g2, edge_type_g2)
        if self.relu_use == 1:
            x_g2 = F.relu(x_g2)
        x_g2 = self.rgcn_layer_2(x_g2, edge_index_g2, edge_type_g2) # 26078*200
        ty_embedding_matrix = x_g1[self.left_common, :]  # 106*200

        tmp = mul_def(x_g2, ty_embedding_matrix, self.final_dim)

        x_final = torch.matmul(tmp, self.weights.to(torch.float64)).squeeze(-1) # 26078*106*200 200*1 ----> 26078*106

        return F.softmax(x_final, dim=1)