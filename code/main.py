import torch.nn.functional as F
from model import Model
from pre_data import *
import argparse
import torch


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--type_path', type=str, default="/Users/bubuying/PycharmProjects/Geo/data/type_embedding/node_embedding.npy")
    parser.add_argument('--entity_path', type=str, default="/Users/bubuying/PycharmProjects/Geo/data/entity_embedding/node_embedding.npy")
    parser.add_argument('--entity_relation_path', type=str, default="/Users/bubuying/PycharmProjects/Geo/data/entity_embedding/node_re_embedding.npy")
    parser.add_argument('--data_path', type=str, default="/Users/bubuying/PycharmProjects/Geo/data/yago_result")
    parser.add_argument('--data_path_bef', type=str, default="/Users/bubuying/PycharmProjects/Geo/data/yago")
    parser.add_argument('--G2_val_file_name', type=str, default="val_entity_Graph.txt")
    parser.add_argument('--G2_test_file_name', type=str, default="test_entity_Graph.txt")
    parser.add_argument('--leaf_node_entity', action='store_true', default=True)

    parser.add_argument('--node_dim', type=int, default=500)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--num_bases', type=int, default=30)
    parser.add_argument('--num_classes', type=int, default=106)
    parser.add_argument('--epoch', type=int, default=500)
    return parser.parse_args(args)


def main(args):
    pre_data = Pre_Data(args)

    all_node_embedding = pre_data.embedding()  # 26989*500
    # G1_node_embedding = [G1_node_embedding_type, G1_node_embedding_type_small, G1_node_embedding_entity] # G1_node_embedding_type: 911*500;   G1_node_embedding_type_small:106*500;   G1_node_embedding_entity:8948*500;   data_G1:2*17429;   data_G1_val:2*499;   data_G1_test:2*996
    G1_node_embedding, G1_node_embedding_type, G1_node_embedding_type_small, G1_node_embedding_entity, data_G1, data_G1_val, data_G1_test = pre_data.G1()
    # G2_edge_index: 664254*2;   G2_node_embedding_: 26078*500;   data_G2: 2*664254;   data_G2_val: 2*499;  data_G2_test: 2*996
    G2_edge_index, G2_node_embedding_, data_G2, data_G2_val, data_G2_test = pre_data.G2()
    # G1_graph_sub2_new: 7723; en_index_G3_list_train_bef: 6178; en_index_G3_list_test_bef: 1545; en_embedding_G3: 7723*500;  en_embedding_G3_train: 6178*500; en_embedding_G3_test: 1545*500
    target_train, target_test, G1_graph_sub2_new, en_index_G3_list_train_bef, en_index_G3_list_test_bef, en_index_G3_list_train, en_index_G3_list_test, en_embedding_G3, en_embedding_G3_train, en_embedding_G3_test = pre_data.G3()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_G2 = data_G2.to(device)
    model = Model(args, data_G2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0.0005)


    for epoch in range(args.epoch):
        print('... ... ... epoch:{} ... ... ...'.format(epoch))
        # print("before optimizer, the node_embedding:{}".format(torch.mean(model.node_embedding_g2 ** 2)))
        model.train()
        optimizer.zero_grad()
        out = model(data_G2.edge_index, data_G2.edge_type)
        out_train = out[en_index_G3_list_train_bef,:]
        loss = F.binary_cross_entropy(out_train, target_train)
        print('train_loss:{}'.format(loss))
        loss.backward()
        optimizer.step()
        # print("after optimizer, the node_embedding:{}".format(torch.mean(model.node_embedding_g2 ** 2)))


        model.eval()
        acc = 0
        out = model(data_G2.edge_index, data_G2.edge_type)
        out_test = out[en_index_G3_list_test_bef,:]  # 1544*106
        for i in range(out_test.shape[0]):
            acc_temp = 0
            out_line = out_test[i,:] # 106
            target_line = target_test[i,:]
            aim_top_pos = torch.nonzero(target_line)
            aim_top_num = aim_top_pos.shape[0]
            # maxk = max((aim_top_num*10,))
            maxk = max((aim_top_num,))
            out_top, out_top_pos = out_line.topk(maxk, 0, True, True)
            for ii in out_top_pos:
                if ii in aim_top_pos:
                    acc_temp = acc_temp + 1
            acc = acc_temp/aim_top_num + acc


        final_acc = acc/len(en_index_G3_list_test_bef)
        print('test_score:{}'.format(final_acc))



if __name__ == '__main__':
    main(parse_args())