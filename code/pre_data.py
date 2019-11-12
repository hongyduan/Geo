from torch_geometric.data import Data
from collections import OrderedDict
from load_data import Load_Data
from load_data_new import Load_Data_NEW
import numpy as np
import torch
import os


class Pre_Data():
    def __init__(self, args):
        self.entity_path = args.entity_path
        self.type_path = args.type_path
        self.data_path = args.data_path
        self.data_path_bef = args.data_path_bef
        self.entity_relation_path = args.entity_relation_path
        self.dim = args.node_dim
        self.G2_val_file_name = args.G2_val_file_name
        self.G2_test_file_name = args.G2_test_file_name
        self.leaf_node_entity = args.leaf_node_entity
        self.num_classes = args.num_classes
        #  G1_graph_sub2_tmp_2 == G1_graph_sub2_final_is_a == G1_graph_sub2_original : G1_train, 7723entity, 8467edges

        node2id_G2, relation2id_G2, type_node2id_G1, G1_graph_sub3, G1_graph_sub2_tmp_2, G1_graph_sub2_final_is_a, G1_graph_sub2_original, G1_graph_sub1, G1_graph_sub1_is_a, triples_test_G1_tmp, triples_test_G1_is_a, triples_test_G1_oroginal, list_list = Load_Data_NEW(self.data_path, self.data_path_bef)
        self.train_edge_clear = list_list[0]
        self.train_edge = list_list[1]
        self.test_edge_clear = list_list[2]
        self.test_edge = list_list[3]
        # test, 3985 edges; 3774 entity;
        node_embedding_entity = np.load(self.entity_path)
        node_embedding_type = np.load(self.type_path)
        all_node_embedding = torch.from_numpy(np.vstack((node_embedding_entity, node_embedding_type)))
        self.all_node_embedding = all_node_embedding
        self.node2id_G2 = node2id_G2  # 26078
        self.relation2id_G2 = relation2id_G2  # 34
        self.type_node2id_G1 = type_node2id_G1  # 911
        self.G1_graph_sub3 = G1_graph_sub3  # 106  ty_en

        self.G1_graph_sub1 = G1_graph_sub1  # 894  ty_ty
        self.G1_graph_sub1_is_a = G1_graph_sub1_is_a  # 676  ty_ty

        # G1_graph_sub2_tmp_2 == G1_graph_sub2_final_is_a == G1_graph_sub2_original
        self.G1_graph_sub2_final_is_a = G1_graph_sub2_final_is_a
        self.G1_graph_sub2_original = G1_graph_sub2_original  # train, 5977 edges; 5549 entity;
        for key2, tmppop2 in G1_graph_sub2_tmp_2.items():
            tmp_list2 = []
            while len(tmppop2) > 0:
                value2 = tmppop2[0]
                if value2 in G1_graph_sub1_is_a.keys() and value2 not in tmp_list2:
                    tmppop2[0:0] = G1_graph_sub1_is_a[value2]
                    self.G1_graph_sub2_final_is_a[key2][0:0] = G1_graph_sub1_is_a[value2]
                    self.G1_graph_sub2_final_is_a[key2] = list(set(self.G1_graph_sub2_final_is_a[key2]))
                    tmppop2 = list(set(tmppop2))
                tmppop2.remove(value2)
                tmp_list2.append(value2)

        # triples_test_G1_tmp == triples_test_G1_is_a == triples_test_G1_oroginal
        self.triples_test_G1_is_a = triples_test_G1_is_a
        self.triples_test_G1_oroginal = triples_test_G1_oroginal  # train, 5977 edges; 5549 entity;
        for key2, tmppop2 in triples_test_G1_tmp.items():
            tmp_list2 = []
            while len(tmppop2) > 0:
                value2 = tmppop2[0]
                if value2 in G1_graph_sub1_is_a.keys() and value2 not in tmp_list2:
                    tmppop2[0:0] = G1_graph_sub1_is_a[value2]
                    self.triples_test_G1_is_a[key2][0:0] = G1_graph_sub1_is_a[value2]
                    self.triples_test_G1_is_a[key2] = list(set(self.triples_test_G1_is_a[key2]))
                    tmppop2 = list(set(tmppop2))
                tmppop2.remove(value2)
                tmp_list2.append(value2)



    def G1_node_embedding(self):
        # G1_node_embedding_type:  906 type embedding
        G1_node_embedding_type = torch.from_numpy(np.load(self.type_path))
        # G1_node_embedding_type_small:  106 type embedding
        # G1_ndoe_embedding_entity:  8948 entity embedding
        left_common = list()  # 106
        right_specific = list()  # 8948
        for left_comm, right_speci in self.G1_graph_sub3.items():
            if left_comm not in left_common:
                left_common.append(left_comm)
            for values in right_speci:
                values = int(values)
                if values not in right_specific:
                    right_specific.append(values)
        G1_node_embedding_type_small = self.all_node_embedding[left_common]
        G1_ndoe_embedding_entity = self.all_node_embedding[right_specific]
        return G1_node_embedding_type, G1_node_embedding_type_small, G1_ndoe_embedding_entity, left_common

    def G2_node_embedding(self):
        # 26078 entity embedding
        G2_node_embedding_entity = torch.from_numpy(np.load(self.entity_path))
        return G2_node_embedding_entity

    def edge_index_G1(self):
        # templist1(en is_instance_of ty)  G1_graph_sub1(ty1 is_a ty2)
        templist1 = list()
        for source, targets in self.G1_graph_sub2_original.items():
            for target in targets:
                templist1.append([int(source), int(target)]) # edges: 5977
        templist2 = list()
        for source, targets in self.G1_graph_sub1.items():
            for target in targets:
                templist2.append([int(source), int(target)])
                templist1.append([int(source), int(target)])
        edge_index_G1_sub1 = torch.tensor(templist2, dtype=torch.long)  # edges: 8962
        edge_index_G1 = torch.tensor(templist1, dtype=torch.long)  # edge_index_G1_sub1 + edge_index_G1_sub2
        return edge_index_G1, edge_index_G1_sub1

    def edge_index_attr_G2(self):
        # G2_graph(en1 relation en2)
        f = open(os.path.join(self.data_path, 'train_entity_Graph.txt'), "r")
        num_lines = len(f.readlines()) * 2
        i = 0
        templist_index = [None] * num_lines
        edge_attr_G2 = torch.empty(num_lines, self.dim)
        edge_type_G2 = torch.empty(num_lines, dtype=torch.long)
        num_relations = list()
        G2_relation_embedding = np.load(self.entity_relation_path) # 34*200
        t_i = G2_relation_embedding.shape[0]
        G2_relation_embedding_re = torch.Tensor(G2_relation_embedding.shape[0], G2_relation_embedding.shape[1])
        with open(os.path.join(self.data_path, 'train_entity_Graph.txt')) as fin:
            for line in fin:
                en1, r, en2 = line.strip().split('\t')
                en1id = self.node2id_G2[en1]
                rid = self.relation2id_G2[r]
                en2id = self.node2id_G2[en2]
                if int(rid) not in num_relations:
                    num_relations.append(int(rid))
                templist_index[i] = [int(en1id), int(en2id)]
                edge_attr_G2[i] = torch.from_numpy(G2_relation_embedding[int(rid)]).unsqueeze(0)
                edge_type_G2[i] = int(rid)
                i = i + 1
                if int(int(rid)+G2_relation_embedding.shape[0]) not in num_relations:
                    num_relations.append(int(int(rid)+G2_relation_embedding.shape[0]))
                templist_index[i] = [int(en2id), int(en1id)]
                # edge_attr_G2[i] = torch.rand(1, 200)
                edge_attr_G2[i] = G2_relation_embedding_re[int(rid),:]
                edge_type_G2[i] = int(int(rid)+G2_relation_embedding.shape[0])
                i = i + 1
        num_relations = len(num_relations)
        edge_index_G2 = torch.tensor(templist_index, dtype=torch.long)  # edges: 2*332127
        edge_attr_G2 = edge_attr_G2  # attr: (2*332127)*200

        return edge_index_G2, edge_attr_G2, edge_type_G2, num_relations

    def G3(self):
        ty_index_G3 = list(self.G1_graph_sub3.keys())
        ty_index_G3_dict = OrderedDict()
        ind=0
        for i in ty_index_G3:
            ty_index_G3_dict[i] = int(ind)
            ind = ind + 1
        G1_graph_sub2_is_a_new = OrderedDict()
        for key, values in self.G1_graph_sub2_final_is_a.items():
            if key not in G1_graph_sub2_is_a_new.keys():
                G1_graph_sub2_is_a_new[key] = list()
            for value in values:
                if int(value-len(self.node2id_G2)) not in G1_graph_sub2_is_a_new[key]:
                    G1_graph_sub2_is_a_new[key].append(int(value-len(self.node2id_G2)))

        G1_graph_sub2_original_new = OrderedDict()
        for key, values in self.G1_graph_sub2_original.items():
            if key not in G1_graph_sub2_original_new.keys():
                G1_graph_sub2_original_new[key] = list()
            for value in values:
                if int(value-len(self.node2id_G2)) not in G1_graph_sub2_original_new[key]:
                    G1_graph_sub2_original_new[key].append(int(value-len(self.node2id_G2)))

        triples_test_G1_is_a_new = OrderedDict()
        for key, values in self.triples_test_G1_is_a.items():
            if key not in triples_test_G1_is_a_new.keys():
                triples_test_G1_is_a_new[key] = list()
            for value in values:
                if int(value-len(self.node2id_G2)) not in triples_test_G1_is_a_new[key]:
                    triples_test_G1_is_a_new[key].append(int(value-len(self.node2id_G2)))

        triples_test_G1_new = OrderedDict()
        for key, values in self.triples_test_G1_oroginal.items():
            if key not in triples_test_G1_new.keys():
                triples_test_G1_new[key] = list()
            for value in values:
                if int(value-len(self.node2id_G2)) not in triples_test_G1_new[key]:
                    triples_test_G1_new[key].append(int(value-len(self.node2id_G2)))


        en_index_G3 = list(self.G1_graph_sub2_original.keys())   # train: 5549 entity, 5977 edges
        en_index_G3_list = list()
        for i in en_index_G3:
            en_index_G3_list.append(int(i))
        en_index_G3_list_train_bef = en_index_G3_list

        en_index_G3 = list(self.triples_test_G1_oroginal.keys())   # test: 3774 entity, 3985 edges
        en_index_G3_list = list()
        for i in en_index_G3:
            en_index_G3_list.append(int(i))
        en_index_G3_list_test_bef = en_index_G3_list

        # big
        target_train = torch.zeros(len(en_index_G3_list_train_bef), len(self.type_node2id_G1))
        i=0
        final_max_1_train = 0  # 113
        for inn in en_index_G3_list_train_bef:
            max_1_train = 0
            for value in G1_graph_sub2_is_a_new[str(inn)]:
                target_train[i,int(value)] = 1
                max_1_train = max_1_train + 1
            if max_1_train>final_max_1_train:
                final_max_1_train = max_1_train
            i = i + 1

        # small
        target_train_small = torch.zeros(len(en_index_G3_list_train_bef), len(self.type_node2id_G1))
        i=0
        final_max_1_train_s = 0  # 3
        for inn in en_index_G3_list_train_bef:
            max_1_train_s = 0
            for value in G1_graph_sub2_original_new[str(inn)]:
                target_train_small[i,int(value)] = 1
                max_1_train_s = max_1_train_s + 1
            if max_1_train_s>final_max_1_train_s:
                final_max_1_train_s = max_1_train_s
            i = i + 1

        # small
        final_max_1_test = 0  # 2
        target_test = torch.zeros(len(en_index_G3_list_test_bef), len(self.type_node2id_G1))
        i=0
        for inn in en_index_G3_list_test_bef:
            max_1_test = 0
            for value in triples_test_G1_new[str(inn)]:
                target_test[i, int(value)] = 1
                max_1_test = max_1_test +1
            if max_1_test>final_max_1_test:
                final_max_1_test = max_1_test
            i = i + 1

        # big
        final_max_1_test_b = 0  # 113
        target_test_big = torch.zeros(len(en_index_G3_list_test_bef), len(self.type_node2id_G1))
        i=0
        for inn in en_index_G3_list_test_bef:
            max_1_test_b = 0
            for value in triples_test_G1_is_a_new[str(inn)]:
                target_test_big[i, int(value)] = 1
                max_1_test_b = max_1_test_b +1
            if max_1_test_b>final_max_1_test_b:
                final_max_1_test_b = max_1_test_b
            i = i + 1

        return target_train, target_train_small, target_test, target_test_big, en_index_G3_list_train_bef, en_index_G3_list_test_bef

    def G1(self):
        G1_node_embedding_ = self.G1_node_embedding()
        G1_node_embedding_type_, G1_node_embedding_type_small_, G1_node_embedding_entity_, left_common = G1_node_embedding_
        edge_index_G1_, edge_index_G1_sub1_ = self.edge_index_G1()
        if self.leaf_node_entity == 1:
            G1_x = self.all_node_embedding
            G1_edge_index = edge_index_G1_
        else:
            G1_x = G1_node_embedding_type_
            G1_edge_index = edge_index_G1_sub1_
        data_G1 = Data(x = G1_x, edge_index = G1_edge_index.t().contiguous())

        return data_G1, left_common

    def G2(self):
        G2_node_embedding_ = self.G2_node_embedding()
        G2_x = G2_node_embedding_
        G2_edge_index, G2_edge_attr, G2_edge_type, G2_num_relations = self.edge_index_attr_G2()
        data_G2 = Data(x = G2_x, edge_index=G2_edge_index.t().contiguous(), edge_attr=G2_edge_attr, edge_type=G2_edge_type, num_relations=G2_num_relations)

        return data_G2


    def embedding(self):
        return self.all_node_embedding
