
from concept_layer import Concept_Layer
from rgcn_layer import RGCN_Layer
import torch.nn.functional as F
import torch.nn as nn
import datetime
import torch

class Model(torch.nn.Module):
    def __init__(self, args, data, all_node_embedding, left_common,target_train):
        super(Model, self).__init__()
        self.all_node_embedding = nn.Parameter(all_node_embedding)
        self.rgcn_layer_1 = RGCN_Layer(args.node_dim, 0, data.num_nodes, args.hidden_dim, data.num_relations, args.num_bases)
        self.rgcn_layer_2 = RGCN_Layer(args.node_dim, 1, args.hidden_dim, args.final_dim, data.num_relations, args.num_bases)
        self.relu_use_rgcn_layer1 = args.relu_use_rgcn_layer1
        self.relu_use_concept_layer = args.relu_use_concept_layer
        self.concept_layer = Concept_Layer()
        self.num_nodes_g2 = data.num_nodes
        self.weights = nn.Parameter(torch.rand((200, 1)))  #
        self.final_dim = args.final_dim
        self.left_common = left_common
        self.target_train = target_train  # 6178*911
        self.sample_num = args.sample_num
        self.concept_clamp = args.concept_clamp
        self.weights_clamp = args.weights_clamp
        self.sigmoid = args.sigmoid


    def forward(self, edge_index_g2, edge_type_g2, edge_index_g1, index_list, sample_index):


        # concept layer
        start = datetime.datetime.now()
        x_g1 = self.concept_layer(self.all_node_embedding, edge_index_g1)  # 26989*20
        if self.relu_use_concept_layer == 1:
            x_g1 = F.relu(x_g1)
        end = datetime.datetime.now()
        print("running time in Concept layer:" + str((end - start).seconds) + " seconds")


        # RGCN-layer
        start = datetime.datetime.now()
        node_embedding_g2 = x_g1[0:self.num_nodes_g2,:]
        # node_embedding_g2 = self.all_node_embedding[0:self.num_nodes_g2, :]
        x_g2 = self.rgcn_layer_1(node_embedding_g2, edge_index_g2, edge_type_g2)
        if self.relu_use_rgcn_layer1 == 1:
            x_g2 = F.relu(x_g2)
        x_g2 = self.rgcn_layer_2(x_g2, edge_index_g2, edge_type_g2) # 26078*200
        en_embedding_matrix = x_g2[index_list, :]  # 6178*200
        end = datetime.datetime.now()
        print("running time in RGCN layer:" + str((end - start).seconds) + " seconds")

        # prediction layer
        #  sample_index: 6178*150
        start = datetime.datetime.now()
        tmp_ = torch.zeros((len(index_list),self.sample_num))
        for i in range(len(index_list)):
            if self.concept_clamp == 1:
                sample_em = torch.clamp(F.relu(x_g1[list(sample_index[i]),:]), 0, 1)  # 150*200
            else:
                sample_em = F.relu(x_g1[list(sample_index[i]), :])  # 150*200
            en_em = en_embedding_matrix[i,:]  # 1*200
            if self.weights_clamp == 1:
                weights = torch.clamp(self.weights, 0, 1)
                tmp_[i,:] = torch.matmul((en_em.mul(sample_em)).mul(en_em),weights).squeeze(1)  # 150*200 * 200*1--->150*1--->150
            else:
                tmp_[i,:] = torch.matmul((en_em.mul(sample_em)).mul(en_em), self.weights).squeeze(1)  # 150*200 * 200*1--->150*1--->150

        end = datetime.datetime.now()
        print("running time in prediction layer:"+str((end-start).seconds)+" seconds")
        if self.sigmoid == 0:
            return F.softmax(tmp_, dim=1)
        else:
            return F.sigmoid(tmp_)
