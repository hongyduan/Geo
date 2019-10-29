from torch_geometric.data import Data
from collections import OrderedDict
from load_data import Load_Data
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
        node2id_G2, relation2id_G2, type_node2id_G1, G1_graph_sub3, G1_graph_sub2, G1_graph_sub1, val_data, test_data = Load_Data(self.data_path, self.data_path_bef)
        node_embedding_entity = np.load(self.entity_path)
        node_embedding_type = np.load(self.type_path)
        all_node_embedding = torch.from_numpy(np.vstack((node_embedding_entity, node_embedding_type)))
        self.all_node_embedding = all_node_embedding
        self.node2id_G2 = node2id_G2  # 26078
        self.relation2id_G2 = relation2id_G2  # 34
        self.type_node2id_G1 = type_node2id_G1  # 911
        self.G1_graph_sub3 = G1_graph_sub3  # 106  ty_en
        self.G1_graph_sub2 = G1_graph_sub2  # 8948 en_ty
        self.G1_graph_sub1 = G1_graph_sub1  # 894  ty_ty
        self.val_data = val_data  # (triples_val_G1, triples_val_G2, G2_links_val)        | (triples_val_G1: en_ty/499;   triples_val_G2: en_en/19538;)
        self.test_data = test_data  # (triples_test_G1, triples_test_G2, G2_links_test)   | (triples_test_G1: en_ty/996;  triples_test_G2: en_en/39073;

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
        # G1_graph_sub2(en is_instance_of ty)  G1_graph_sub1(ty1 is_a ty2)
        templist1 = list()
        for source, targets in self.G1_graph_sub2.items():
            for target in targets:
                templist1.append([int(source), int(target)])
        edge_index_G1_sub2 = torch.tensor(templist1, dtype=torch.long)  # edges: 9962
        templist2 = list()
        for source, targets in self.G1_graph_sub1.items():
            for target in targets:
                templist2.append([int(source), int(target)])
                templist1.append([int(source), int(target)])
        edge_index_G1_sub1 = torch.tensor(templist2, dtype=torch.long)  # edges: 8962
        edge_index_G1 = torch.tensor(templist1, dtype=torch.long)
        return edge_index_G1, edge_index_G1_sub2, edge_index_G1_sub1

    def edge_index_G1_val_test(self):
        # G1_val: en_ty/499; G1_test: en_ty/996
        G1_val = self.val_data[0]
        G1_test = self.test_data[0]
        templist = list()
        for source, targets in G1_val.items():
            for target in targets:
                templist.append([int(source), int(target)])
        G1_edge_index_val = torch.tensor(templist, dtype=torch.long)
        templist = list()
        for source, targets in G1_test.items():
            for target in targets:
                templist.append([int(source), int(target)])
        G1_edge_index_test = torch.tensor(templist, dtype=torch.long)
        return G1_edge_index_val, G1_edge_index_test

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

    def edge_index_attr_G2_val_test(self,file_name):
        f = open(os.path.join(self.data_path, file_name), 'r')
        num_lines = len(f.readlines())
        i = 0
        templist_index = [None] * num_lines
        edge_attr_G2_val_test = torch.empty(num_lines, self.dim)
        G2_relation_embedding = np.load(self.entity_relation_path)
        with open(os.path.join(self.data_path, file_name)) as fin:
            for line in fin:
                en1, r, en2 = line.strip().split('\t')
                en1id = self.node2id_G2[en1]
                rid = self.relation2id_G2[r]
                en2id = self.node2id_G2[en2]

                templist_index[i] = [int(en1id), int(en2id)]
                edge_attr_G2_val_test[i] = torch.from_numpy(G2_relation_embedding[int(rid)]).unsqueeze(0)
                i = i + 1
        edge_index_G2_val_test = torch.tensor(templist_index, dtype=torch.long)  # edges: 19538
        edge_attr_G2_val_test = edge_attr_G2_val_test  # attr: 19538*200
        return edge_index_G2_val_test, edge_attr_G2_val_test

    def G3(self):
        ty_index_G3 = list(self.G1_graph_sub3.keys())
        ty_index_G3_dict = OrderedDict()
        ind=0
        for i in ty_index_G3:
            ty_index_G3_dict[i] = int(ind)
            ind = ind + 1
        G1_graph_sub2_new = OrderedDict()
        for key, values in self.G1_graph_sub2.items():
            if key not in G1_graph_sub2_new.keys():
                G1_graph_sub2_new[key] = list()
            for value in values:
                if ty_index_G3_dict[value] not in G1_graph_sub2_new[key]:
                    G1_graph_sub2_new[key].append(int(ty_index_G3_dict[value]))

        en_index_G3 = list(self.G1_graph_sub2.keys())
        en_index_G3_list = list()
        for i in en_index_G3:
            en_index_G3_list.append(int(i))
        en_index_G3_list_train_bef = en_index_G3_list[0:6178]  # 6178
        en_index_G3_list_test_bef = en_index_G3_list[6178:len(en_index_G3_list)]  # 1545

        en_index_G3_list = torch.tensor(en_index_G3_list, dtype=torch.long)  # 6178+1544 entity
        en_index_G3_list_train = torch.tensor(en_index_G3_list_train_bef, dtype=torch.long)  # 6178 entity for train
        en_index_G3_list_test = torch.tensor(en_index_G3_list_test_bef, dtype=torch.long)  # 1544 entity for test

        en_embedding_G3 = torch.index_select(  # 7723*200
            self.all_node_embedding,
            dim=0,
            index=en_index_G3_list
        )
        en_embedding_G3_train = torch.index_select(  # 6178*200
            self.all_node_embedding,
            dim=0,
            index=en_index_G3_list_train
        )
        en_embedding_G3_test = torch.index_select(  # 1545*200
            self.all_node_embedding,
            dim=0,
            index=en_index_G3_list_test
        )
        target_train = torch.zeros(len(en_index_G3_list_train_bef), self.num_classes)
        i=0
        for inn in en_index_G3_list_train_bef:
            for value in G1_graph_sub2_new[str(inn)]:
                target_train[i,int(value)] = 1
            i = i + 1
        target_test = torch.zeros(len(en_index_G3_list_test_bef), self.num_classes)
        i=0
        for inn in en_index_G3_list_test_bef:
            for value in G1_graph_sub2_new[str(inn)]:
                target_test[i, int(value)] = 1
            i = i + 1

        return target_train, target_test, G1_graph_sub2_new, en_index_G3_list_train_bef, en_index_G3_list_test_bef, en_index_G3_list_train, en_index_G3_list_test, en_embedding_G3, en_embedding_G3_train, en_embedding_G3_test

    def G1(self):
        G1_node_embedding_ = self.G1_node_embedding()
        G1_node_embedding_type_, G1_node_embedding_type_small_, G1_node_embedding_entity_, left_common = G1_node_embedding_
        edge_index_G1_, edge_index_G1_sub2_, edge_index_G1_sub1_ = self.edge_index_G1()
        if self.leaf_node_entity:
            G1_x = self.all_node_embedding
            G1_edge_index = edge_index_G1_
        else:
            G1_x = G1_node_embedding_type_
            G1_edge_index = edge_index_G1_sub1_
        data_G1 = Data(x = G1_x, edge_index = G1_edge_index.t().contiguous())
        G1_edge_index_val, G1_edge_index_test = self.edge_index_G1_val_test()
        data_G1_val = Data(edge_index = G1_edge_index_val.t().contiguous())
        data_G1_test = Data(edge_index = G1_edge_index_test.t().contiguous())

        return G1_node_embedding_, G1_node_embedding_type_, G1_node_embedding_type_small_, G1_node_embedding_entity_, data_G1, data_G1_val, data_G1_test, left_common



    def G2(self):
        G2_node_embedding_ = self.G2_node_embedding()
        G2_x = G2_node_embedding_
        G2_edge_index, G2_edge_attr, G2_edge_type, G2_num_relations = self.edge_index_attr_G2()
        data_G2 = Data(x = G2_x, edge_index=G2_edge_index.t().contiguous(), edge_attr=G2_edge_attr, edge_type=G2_edge_type, num_relations=G2_num_relations)
        G2_edge_index_val, G2_edge_attr_val = self.edge_index_attr_G2_val_test(self.G2_val_file_name)
        data_G2_val = Data(edge_index=G2_edge_index_val.t().contiguous(), edge_attr=G2_edge_attr_val)
        G2_edge_index_test, G2_edge_attr_test = self.edge_index_attr_G2_val_test(self.G2_test_file_name)
        data_G2_test = Data(edge_index=G2_edge_index_test.t().contiguous(), edge_attr=G2_edge_attr_test)

        return G2_edge_index, G2_node_embedding_, data_G2, data_G2_val, data_G2_test

    def embedding(self):
        return self.all_node_embedding
















