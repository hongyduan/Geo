
from concept_layer import Concept_Layer
from rgcn_layer import RGCN_Layer
import torch.nn.functional as F
import torch.nn as nn
import logging
import torch


class Model(torch.nn.Module):
    def __init__(self, args, data, all_node_embedding, left_common,target_train):
        super(Model, self).__init__()
        self.all_node_embedding = nn.Parameter(all_node_embedding)
        self.rgcn_layer_1 = RGCN_Layer(args.node_dim, 0, data.num_nodes, args.hidden_dim, data.num_relations, args.num_bases)
        self.rgcn_layer_2 = RGCN_Layer(args.node_dim, 1, args.hidden_dim, args.final_dim, data.num_relations, args.num_bases)
        self.concept_layer = Concept_Layer()
        self.num_nodes_g2 = data.num_nodes
        if args.prediction_layer == 1:
            self.weights = nn.Parameter(torch.rand((200, 1)))  #
        self.left_common = left_common
        self.target_train = target_train  # 6178*911


    def forward(self, args, edge_index_g2, edge_type_g2, edge_index_g1, index_list, sample_index, sample_index_min):

        # Concept layer
        if args.concept_layer == 1:
            # concept layer
            logging.info("begin concept layer... ...")
            x_g1 = self.concept_layer(self.all_node_embedding, edge_index_g1)  # 26989*20
            entity_x_g1 = x_g1[0:self.num_nodes_g2,:]
            concept_x_g1 = x_g1[self.num_nodes_g2:26989,:]
            concept_x_g1 = F.relu(concept_x_g1)
        if args.concept_layer == 0:
            entity_x_g1 = self.all_node_embedding[0:self.num_nodes_g2, :]

        # RGCN layer
        if args.RGCN_layer == 1:
            logging.info("begin RGCN layer... ...")
            logging.info("in RGCN_layer_1... ...")
            x_g2 = self.rgcn_layer_1(entity_x_g1, edge_index_g2, edge_type_g2)
            if args.relu_use_rgcn_layer1 == 1:
                x_g2 = F.relu(x_g2)
            logging.info("in RGCN_layer_2... ...")
            x_g2 = self.rgcn_layer_2(x_g2, edge_index_g2, edge_type_g2) # 26078*200
            x_g2_mini = x_g2[index_list, :]  # 6178*200

        # Prediction layer
        if args.prediction_layer == 1:
            logging.info("begin prediction layer... ...")
            # prediction layer
            #  sample_index: 6178*150
            tmp_ = torch.zeros((len(index_list),args.sample_num))
            for i in range(len(index_list)):
                if args.concept_clamp == 1:
                    sample_em = torch.clamp(concept_x_g1[list(sample_index_min[i]),:], 0, 1)  # 150*200
                else:
                    sample_em = concept_x_g1[list(sample_index_min[i]), :]  # 150*200
                en_em = x_g2_mini[i,:]  # 1*200
                if args.weights_clamp == 1:
                    tmp_[i,:] = torch.matmul((en_em.mul(sample_em)).mul(en_em),torch.clamp(self.weights, 0, 1)).squeeze(1)  # 150*200 * 200*1--->150*1--->150
                else:
                    tmp_[i,:] = torch.matmul((en_em.mul(sample_em)).mul(en_em), self.weights).squeeze(1)  # 150*200 * 200*1--->150*1--->150

        if args.sigmoid == 1:
            return torch.sigmoid(tmp_)
        else:
            return F.softmax(tmp_, dim=1)


        logging.info("out model.. ...")

