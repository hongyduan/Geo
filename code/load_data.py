from collections import OrderedDict
import os


def Load_Data(data_path, data_path_bef):

    # entity_G2   26078
    node2id_G2 = OrderedDict()
    node2id_G2_re = OrderedDict()
    with open(os.path.join(data_path, 'final_entity_order.txt')) as fin:  # 26078
        for line in fin:
            eid, entity = line.strip().split('\t')
            node2id_G2[eid] = entity
            node2id_G2_re[entity] = eid
    num_node_G2 = len(node2id_G2_re)  # num_node_G2: 26078

    # relation_G2  34
    relation2id_G2 = OrderedDict()
    relation2id_G2_re = OrderedDict()
    with open(os.path.join(data_path, 'ffinal_en_relation_order.txt')) as fin:  # 34
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id_G2[rid] = relation
            relation2id_G2_re[relation] = rid

    # type_G1   911
    type_node2id_G1 = OrderedDict()
    type_node2id_G1_re = OrderedDict()
    with open(os.path.join(data_path, 'final_type_order.txt')) as fin:  # 911
        for line in fin:
            tyid, type = line.strip().split('\t')
            type_node2id_G1[int(tyid) + num_node_G2] = type
            type_node2id_G1_re[type] = int(tyid) + num_node_G2





    # G1_graph_sub1    894type
    G1_graph_sub1 = OrderedDict()
    G1_graph_sub1_is_a = OrderedDict()
    with open(os.path.join(data_path_bef, 'yago_ontonet.txt')) as fin:  # 894type 8962edge
        for line in fin:
            # ty1 is_a ty2
            ty1, _, ty2 = line.strip().split('\t')
            ty1id = type_node2id_G1_re[ty1]
            ty2id = type_node2id_G1_re[ty2]
            if ty1id not in G1_graph_sub1.keys():
                G1_graph_sub1[ty1id] = list()
            G1_graph_sub1[ty1id].append(ty2id)
            if _ == "isa":
                if ty1id not in G1_graph_sub1_is_a.keys():
                    G1_graph_sub1_is_a[ty1id] = list()
                G1_graph_sub1_is_a[ty1id].append(ty2id)

    # G1_graph_sub2     8948entity 9962edges
    G1_graph_sub2 = OrderedDict()
    G1_graph_sub2_2 = OrderedDict()
    G1_graph_sub2_final = OrderedDict()
    G1_graph_sub2_final_is_a = OrderedDict()
    G1_graph_sub2_original = OrderedDict()
    with open(os.path.join(data_path, 'train_entity_typeG.txt')) as fin:   # train, 8467edges, 7723entity
        for line in fin:
            # en is_instance_of ty
            en, _, ty = line.strip().split('\t')
            enid = node2id_G2_re[en]
            tyid = type_node2id_G1_re[ty]
            if enid not in G1_graph_sub2.keys():
                G1_graph_sub2[enid] = list()
                G1_graph_sub2_2[enid] = list()
                G1_graph_sub2_final[enid] = list()
                G1_graph_sub2_final_is_a[enid] = list()
                G1_graph_sub2_original[enid] = list()
            G1_graph_sub2[enid].append(tyid)
            G1_graph_sub2_2[enid].append(tyid)
            G1_graph_sub2_final[enid].append(tyid)
            G1_graph_sub2_final_is_a[enid].append(tyid)
            G1_graph_sub2_original[enid].append(tyid)
    triples_val_G1 = OrderedDict()    # val, 484 entity; 499 edges;
    with open(os.path.join(data_path, 'val_entity_typeG.txt')) as fin:
        for line in fin:
            en, _, ty = line.strip().split('\t')
            enid = node2id_G2_re[en]
            tyid = type_node2id_G1_re[ty]
            if enid not in triples_val_G1.keys():
                triples_val_G1[enid] = list()
            triples_val_G1[enid].append(tyid)
    triples_test_G1 = OrderedDict()   # test, 968 entity; 996 edges;
    with open(os.path.join(data_path, 'test_entity_typeG.txt')) as fin:
        for line in fin:
            en, _, ty = line.strip().split('\t')
            enid = node2id_G2_re[en]
            tyid = type_node2id_G1_re[ty]
            if enid not in triples_test_G1.keys():
                triples_test_G1[enid] = list()
            triples_test_G1[enid].append(tyid)

    # type_G1_sub3      106
    G1_graph_sub3 = OrderedDict()
    with open(os.path.join(data_path_bef, 'yago_InsType_mini.txt')) as fin:   # 106
        for line in fin:
            en, _, ty = line.strip().split('\t')
            enid = node2id_G2_re[en]
            tyid = type_node2id_G1_re[ty]
            if tyid not in G1_graph_sub3.keys():
                G1_graph_sub3[tyid] = list()
            G1_graph_sub3[tyid].append(enid)




    # G2_graph:   21197entity, 390738edges
    G2_graph = OrderedDict()
    G2_links = OrderedDict()
    with open(os.path.join(data_path, 'train_entity_Graph.txt')) as fin:  # 21105entity, 332127edges
        for line in fin:
            en1, r, en2 = line.strip().split('\t')
            en1id = node2id_G2_re[en1]
            rid = relation2id_G2_re[r]
            en2id = node2id_G2_re[en2]
            if en1id not in G2_graph.keys():
                G2_graph[en1id] = list()
            G2_graph[en1id].append(en2id)

            if (en1id,en2id) not in G2_links:
                G2_links[(en1id,en2id)] = list()
            G2_links[(en1id,en2id)].append(rid)
    triples_val_G2 = OrderedDict()
    G2_links_val = OrderedDict()
    with open(os.path.join(data_path, 'val_entity_Graph.txt')) as fin:  # 11833entity, 19538edges
        for line in fin:
            en1, r, en2 = line.strip().split('\t')
            en1id = node2id_G2_re[en1]
            rid = relation2id_G2_re[r]
            en2id = node2id_G2_re[en2]
            if en1id not in triples_val_G2.keys():
                triples_val_G2[en1id] = list()
            triples_val_G2[en1id].append(en2id)
            if (en1id, en2id) not in G2_links_val:
                G2_links_val[(en1id,en2id)] = list()
            G2_links_val[(en1id,en2id)].append(rid)
    triples_test_G2 = OrderedDict()
    G2_links_test = OrderedDict()
    with open(os.path.join(data_path, 'test_entity_Graph.txt')) as fin:  # 16204entity, 39073edges
        for line in fin:
            en1, r, en2 = line.strip().split('\t')
            en1id = node2id_G2_re[en1]
            rid = relation2id_G2_re[r]
            en2id = node2id_G2_re[en2]
            if en1id not in triples_test_G2.keys():
                triples_test_G2[en1id] = list()
            triples_test_G2[en1id].append(en2id)
            if (en1id, en2id) not in G2_links_test:
                G2_links_test[(en1id,en2id)] = list()
            G2_links_test[(en1id,en2id)].append(rid)

    val_data = (triples_val_G1, triples_val_G2, G2_links_val)
    test_data = (triples_test_G1, triples_test_G2, G2_links_test)

    G1_graph_sub2_train = G1_graph_sub2
    # G1_graph_sub2_train == G1_graph_sub2_2 == G1_graph_sub2_final == G1_graph_sub2_final_is_a == G1_graph_sub2_original
    return node2id_G2_re, relation2id_G2_re, type_node2id_G1_re, G1_graph_sub3, G1_graph_sub2_train, G1_graph_sub2_2, G1_graph_sub2_final, G1_graph_sub2_final_is_a, G1_graph_sub2_original,G1_graph_sub1, G1_graph_sub1_is_a, val_data, test_data
