from collections import OrderedDict
import os


def Load_Data(data_path, data_path_bef):
    # node2id_G2_re   26078
    node2id_G2 = OrderedDict()
    node2id_G2_re = OrderedDict()
    with open(os.path.join(data_path, 'final_entity_order.txt')) as fin:  # 26078
        for line in fin:
            eid, entity = line.strip().split('\t')
            node2id_G2[eid] = entity
            node2id_G2_re[entity] = eid
    num_node_G2 = len(node2id_G2_re)  # num_node_G2: 26078


    # relation2id_G2_re  34
    relation2id_G2 = OrderedDict()
    relation2id_G2_re = OrderedDict()
    with open(os.path.join(data_path, 'ffinal_en_relation_order.txt')) as fin:  # 34
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id_G2[rid] = relation
            relation2id_G2_re[relation] = rid


    # type_node2id_G1_re   911
    type_node2id_G1 = OrderedDict()
    type_node2id_G1_re = OrderedDict()
    with open(os.path.join(data_path, 'final_type_order.txt')) as fin:  # 911
        for line in fin:
            tyid, type = line.strip().split('\t')
            type_node2id_G1[int(tyid) + num_node_G2] = type
            type_node2id_G1_re[type] = int(tyid) + num_node_G2


    # G1_graph_sub3      106
    G1_graph_sub3 = OrderedDict()
    with open(os.path.join(data_path_bef, 'yago_InsType_mini.txt')) as fin:   # 106
        for line in fin:
            en, _, ty = line.strip().split('\t')
            enid = node2id_G2_re[en]
            tyid = type_node2id_G1_re[ty]
            if tyid not in G1_graph_sub3.keys():
                G1_graph_sub3[tyid] = list()
            G1_graph_sub3[tyid].append(enid)


    # G1_graph_sub2     8948entity
    G1_graph_sub2 = OrderedDict()
    with open(os.path.join(data_path, 'train_entity_typeG.txt')) as fin:   # 8948entity 8467edge
        for line in fin:
            # en is_instance_of ty
            en, _, ty = line.strip().split('\t')
            enid = node2id_G2_re[en]
            tyid = type_node2id_G1_re[ty]
            if enid not in G1_graph_sub2.keys():
                G1_graph_sub2[enid] = list()
            G1_graph_sub2[enid].append(tyid)


    # G1_graph_sub1    894type
    G1_graph_sub1 = OrderedDict()
    with open(os.path.join(data_path_bef, 'yago_ontonet.txt')) as fin:  # 894type 8962edge
        for line in fin:
            # ty1 is_a ty2
            ty1, _, ty2 = line.strip().split('\t')
            ty1id = type_node2id_G1_re[ty1]
            ty2id = type_node2id_G1_re[ty2]
            if ty1id not in G1_graph_sub1.keys():
                G1_graph_sub1[ty1id] = list()
            G1_graph_sub1[ty1id].append(ty2id)


    # G2_graph:   21197;   G2_links:   228619
    G2_graph = OrderedDict()
    G2_links = OrderedDict()
    with open(os.path.join(data_path, 'train_entity_Graph.txt')) as fin:  # 332127
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

    # G1_val(en_ty_val/entity typing val): 499 triples;     G2_val(en_en_val/relation prediction val): 19538 triples
    # triples_val_G1: en_ty/499;   triples_val_G2: en_en/19538;    G2_links_val: relation
    triples_val_G1 = OrderedDict()
    with open(os.path.join(data_path, 'val_entity_typeG.txt')) as fin:
        for line in fin:
            en, _, ty = line.strip().split('\t')
            enid = node2id_G2_re[en]
            tyid = type_node2id_G1_re[ty]
            if enid not in triples_val_G1.keys():
                triples_val_G1[enid] = list()
            triples_val_G1[enid].append(tyid)

    triples_val_G2 = OrderedDict()
    G2_links_val = OrderedDict()
    with open(os.path.join(data_path, 'val_entity_Graph.txt')) as fin:
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

    # G1_test(en_ty_test/entity typing test): 996 triples;     G2_test(en_en_test/relation prediction test): 39073 triples
    # triples_test_G1: en_ty/996;  triples_test_G2: en_en/39073;
    triples_test_G1 = OrderedDict()
    with open(os.path.join(data_path, 'test_entity_typeG.txt')) as fin:
        for line in fin:
            en, _, ty = line.strip().split('\t')
            enid = node2id_G2_re[en]
            tyid = type_node2id_G1_re[ty]
            if enid not in triples_test_G1.keys():
                triples_test_G1[enid] = list()
            triples_test_G1[enid].append(tyid)
    triples_test_G2 = OrderedDict()
    G2_links_test = OrderedDict()
    with open(os.path.join(data_path, 'test_entity_Graph.txt')) as fin:
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

    return node2id_G2_re, relation2id_G2_re, type_node2id_G1_re, G1_graph_sub3, G1_graph_sub2, G1_graph_sub1, val_data, test_data


