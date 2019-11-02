
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import Model
from pre_data import *
import argparse
import datetime
import random
import torch
import json


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
    parser.add_argument('--save_path_g2', type=str, default="/Users/bubuying/PycharmProjects/Geo/save")
    parser.add_argument('--G2_val_file_name', type=str, default="val_entity_Graph.txt")
    parser.add_argument('--G2_test_file_name', type=str, default="test_entity_Graph.txt")
    parser.add_argument('--leaf_node_entity', action='store_true', default=True)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--relu_use_rgcn_layer1', type=int, default=1)
    parser.add_argument('--relu_use_concept_layer', type=int, default=1)
    parser.add_argument('--concept_clamp', type=int, default=1)
    parser.add_argument('--weights_clamp', type=int, default=1)
    parser.add_argument('--sigmoid', type=int, default=0)  # 0: softmax; 1:sigmoid
    parser.add_argument('--which_dataset', type=int, default=0)

    parser.add_argument('--learn_rate', type=float, default=0.01)
    parser.add_argument('--node_dim', type=int, default=200)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--final_dim', type=int, default=200)
    parser.add_argument('--num_bases', type=int, default=30)
    parser.add_argument('--num_classes', type=int, default=106)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--sample_num', type=int, default=150)
    return parser.parse_args(args)


def save_model(model, optimizer, args, biggest_score):
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path_g2, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)
    torch.save({
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path_g2, 'checkpoint_en')
    )

    all_node_embedding = model.all_node_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path_g2, 'node_embedding'),
        all_node_embedding
    )
    with open(os.path.join(args.save_path_g2, 'final_test_score.json'), 'w') as scorejson:
        json.dump(biggest_score, scorejson)


def sample(args, target_train, g2_num_nodes):
    sample_index = torch.zeros((target_train.size(0), args.sample_num))  # 6178*150
    sample_label = torch.zeros((target_train.size(0), args.sample_num))  # 6178*150
    target_train_nonzero = target_train.nonzero()
    start = -1
    tmp_id = 0
    tmp_dict = OrderedDict()

    for item in target_train_nonzero:
        if item[0] == start:
            sample_index[item[0], tmp_id] = item[1] + g2_num_nodes
            sample_label[item[0], tmp_id] = 1
            tmp_dict[int(item[0])].append(int(item[1]))
            tmp_id = tmp_id + 1

        else:
            tmp_dict[int(item[0])] = list()
            start = item[0]
            tmp_id = 0
            sample_index[item[0], tmp_id] = item[1] + g2_num_nodes
            sample_label[item[0], tmp_id] = 1
            tmp_dict[int(item[0])].append(int(item[1]))
            tmp_id = tmp_id + 1

    for key, values in tmp_dict.items():
        a = args.sample_num - len(values)
        list1 = [x for x in range(target_train.shape[1])]
        tt = list(set(list1).difference(set(values)))

        slice = random.sample(tt, a)

        start_id = len(values)
        for ii in slice:
            sample_index[key, start_id] = ii + g2_num_nodes
            start_id = start_id + 1
    return sample_index, sample_label  # 6178*150



def main(args):
    if args.which_dataset == 0:
        train_loss_list = []
        test_score_list = []

        big_score = 0
        pre_data = Pre_Data(args)

        all_node_embedding = pre_data.embedding()  # 26989*200

        # G1_node_embedding = [G1_node_embedding_type, G1_node_embedding_type_small, G1_node_embedding_entity] # G1_node_embedding_type: 911*200;   G1_node_embedding_type_small:106*200;   G1_node_embedding_entity:8948*200;   data_G1:2*17429;   data_G1_val:2*499;   data_G1_test:2*996
        G1_node_embedding, G1_node_embedding_type, G1_node_embedding_type_small, G1_node_embedding_entity, data_G1, data_G1_val, data_G1_test, left_common, G1_graph_sub1 = pre_data.G1()
        # G2_edge_index: 664254*2;   G2_node_embedding_: 26078*200;   data_G2: 2*664254;   data_G2_val: 2*499;  data_G2_test: 2*996
        G2_edge_index, G2_node_embedding_, data_G2, data_G2_val, data_G2_test = pre_data.G2()
        # G1_graph_sub2_new: 7723; en_index_G3_list_train_bef: 6178; en_index_G3_list_test_bef: 1545; en_embedding_G3: 7723*200;  en_embedding_G3_train: 6178*200; en_embedding_G3_test: 1545*200
        # target_train, target_test, G1_graph_sub2_new, G1_graph_sub2_new_is_a, G1_graph_sub2_new_mini, G1_graph_sub2_new_mini_is_a, en_index_G3_list_train_bef, en_index_G3_list_test_bef, en_index_G3_list_train, en_index_G3_list_test, en_embedding_G3, en_embedding_G3_train, en_embedding_G3_test = pre_data.G3()
        target_train, target_test, G1_graph_sub2_new_is_a, G1_graph_sub2_new_mini, G1_graph_sub2_new_mini_is_a, en_index_G3_list_train_bef, en_index_G3_list_test_bef, en_index_G3_list_train, en_index_G3_list_test, en_embedding_G3, en_embedding_G3_train, en_embedding_G3_test = pre_data.G3()



        device = torch.device('cuda' if args.cuda==True else 'cpu')

        data_G2 = data_G2.to(device)
        target_train = target_train.to(device)

        model = Model(args, data_G2, all_node_embedding, left_common, target_train).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learn_rate, weight_decay = 0.0005)


        for epoch in range(args.epoch):
            print('_________________ epoch:{} _________________ '.format(epoch))

            sample_index, sample_label = sample(args, target_train, data_G2.num_nodes)  # 6178*150


            print("before optimizer, the node_embedding:{}".format(torch.mean(model.all_node_embedding ** 2)))
            model.train()
            optimizer.zero_grad()

            out_train = model(data_G2.edge_index, data_G2.edge_type, data_G1.edge_index, en_index_G3_list_train_bef, sample_index)  # 6178*106

            start = datetime.datetime.now()
            loss = F.binary_cross_entropy(out_train, sample_label)
            end = datetime.datetime.now()
            print("running time in calculate loss:" + str((end - start).seconds) + " seconds")

            train_loss_list.append(loss)
            print('train_loss:{}'.format(loss))
            start = datetime.datetime.now()
            loss.backward()
            end = datetime.datetime.now()
            print("running time in loss.backward:"+str((end-start).seconds)+" seconds")

            start = datetime.datetime.now()
            optimizer.step()
            end = datetime.datetime.now()
            print("running time in optimizer.step:" + str((end - start).seconds) + " seconds")
            print("after optimizer, the node_embedding:{}".format(torch.mean(model.all_node_embedding ** 2)))


            model.eval()
            acc = 0
            sample_index, sample_label = sample(args, target_test, data_G2.num_nodes)  # 1544*150
            start = datetime.datetime.now()
            out_test = model(data_G2.edge_index, data_G2.edge_type, data_G1.edge_index, en_index_G3_list_test_bef, sample_index)  # 1544*150
            end = datetime.datetime.now()
            print("running time in test.model:" + str((end - start).seconds) + " seconds")

            for i in range(out_test.shape[0]):  # 1544
                acc_temp = 0
                out_line = out_test[i,:] # 150
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
            test_score_list.append(final_acc)
            print('test_score:{}'.format(final_acc))
            if final_acc > big_score:
                print("current score is bigger, before:{}, current:{}, save model ... ".format(big_score, final_acc))
                big_score = final_acc
                save_model(model, optimizer, args, big_score)

            else:
                print("biggest acore:{} ... ".format(big_score))


        print("for debug")

        # plot
        x1 = range(0, args.epoch)
        x2 = range(0, args.epoch)
        y1 = test_score_list
        y2 = train_loss_list
        plt.subplot(2,1,1)
        plt.plot(x1, y1, 'b*')
        plt.title('Test score vs. epoches')
        plt.ylabel('Test score')
        plt.subplot(2,1,2)
        plt.plot(x2, y2, 'b*')
        plt.xlabel('Train loss vs. epoches')
        plt.ylabel('Train loss')
        plt.savefig(os.path.join(args.save_path_g2, 'Train_loss_and_Test_score_*.png'))
        # plt.show()

        plt.subplot(2,1,1)
        plt.plot(x1, y1, 'b-')
        plt.title('Test score vs. epoches')
        plt.ylabel('Test score')
        plt.subplot(2,1,2)
        plt.plot(x2, y2, 'b-')
        plt.xlabel('Train loss vs. epoches')
        plt.ylabel('Train loss')
        plt.savefig(os.path.join(args.save_path_g2, 'Train_loss_and_Test_score__.png'))
        # plt.show()


if __name__ == '__main__':
    main(parse_args())