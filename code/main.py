
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import Model
from pre_data import *
import numpy as np
import argparse
import datetime
import logging
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

    parser.add_argument('--leaf_node_entity', type=int, default=1)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--init_checkpoint', type=int, default=0)

    parser.add_argument('--init_embedding_path', type=str, default="/Users/bubuying/PycharmProjects/Geo/save/node_embedding.npy")
    parser.add_argument('--init_biggest_score', type=str, default="/Users/bubuying/PycharmProjects/Geo/save/final_test_score.json")

    parser.add_argument('--relu_use_rgcn_layer1', type=int, default=1)
    parser.add_argument('--relu_use_concept_layer', type=int, default=1)

    parser.add_argument('--concept_clamp', type=int, default=1)
    parser.add_argument('--weights_clamp', type=int, default=1)

    parser.add_argument('--sigmoid', type=int, default=1)  # 0: softmax; 1:sigmoid
    parser.add_argument('--which_dataset', type=int, default=0)

    parser.add_argument('--concept_layer', type=int, default=1)
    parser.add_argument('--RGCN_layer', type=int, default=1)
    parser.add_argument('--prediction_layer', type=int, default=1)

    parser.add_argument('--learn_rate', type=float, default=0.1)
    parser.add_argument('--node_dim', type=int, default=200)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--final_dim', type=int, default=200)
    parser.add_argument('--num_bases', type=int, default=30)
    parser.add_argument('--num_classes', type=int, default=106)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--sample_num', type=int, default=150)

    return parser.parse_args(args)


def set_logger(args):
    log_file = os.path.join(args.save_path_g2, 'train&test.log')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def save_model(model, optimizer, args, biggest_score):
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path_g2, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)
    with open(os.path.join(args.save_path_g2, 'final_test_score.json'), 'w') as scorejson:
        json.dump(biggest_score, scorejson)
    torch.save({
        # 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path_g2, 'checkpoint')
    )
    torch.save(model, os.path.join(args.save_path_g2,'model.pkl'))
    all_node_embedding = model.all_node_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path_g2, 'node_embedding'),
        all_node_embedding
    )


def sample(args, target_train, g2_num_nodes):
    sample_index = torch.zeros((target_train.size(0), args.sample_num))  # 6178*150
    sample_index_min = torch.zeros((target_train.size(0), args.sample_num))  # 6178*150
    sample_label = torch.zeros((target_train.size(0), args.sample_num))  # 6178*150
    target_train_nonzero = target_train.nonzero()
    start = -1
    tmp_id = 0
    tmp_dict = OrderedDict()

    for item in target_train_nonzero:
        if item[0] == start:
            sample_index[item[0], tmp_id] = item[1] + g2_num_nodes
            sample_index_min[item[0], tmp_id] = item[1]
            sample_label[item[0], tmp_id] = 1
            tmp_dict[int(item[0])].append(int(item[1]))
            tmp_id = tmp_id + 1

        else:
            tmp_dict[int(item[0])] = list()
            start = item[0]
            tmp_id = 0
            sample_index[item[0], tmp_id] = item[1] + g2_num_nodes
            sample_index_min[item[0], tmp_id] = item[1]
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
            sample_index_min[key, start_id] = ii
            start_id = start_id + 1
    return sample_index, sample_index_min, sample_label  # 6178*150


def load_data_and_pre_data(args):
    pre_data = Pre_Data(args)
    # G1_node_embedding = [G1_node_embedding_type, G1_node_embedding_type_small, G1_node_embedding_entity] # G1_node_embedding_type: 911*200;   G1_node_embedding_type_small:106*200;   G1_node_embedding_entity:8948*200;   data_G1:2*17429;   data_G1_val:2*499;   data_G1_test:2*996
    G1_node_embedding, G1_node_embedding_type, G1_node_embedding_type_small, G1_node_embedding_entity, data_G1, data_G1_val, data_G1_test, left_common, G1_graph_sub1 = pre_data.G1()
    # G2_edge_index: 664254*2;   G2_node_embedding_: 26078*200;   data_G2: 2*664254;   data_G2_val: 2*499;  data_G2_test: 2*996
    G2_edge_index, G2_node_embedding_, data_G2, data_G2_val, data_G2_test = pre_data.G2()
    # G1_graph_sub2_new: 7723; en_index_G3_list_train_bef: 6178; en_index_G3_list_test_bef: 1545; en_embedding_G3: 7723*200;  en_embedding_G3_train: 6178*200; en_embedding_G3_test: 1545*200
    # target_train, target_test, G1_graph_sub2_new, G1_graph_sub2_new_is_a, G1_graph_sub2_new_mini, G1_graph_sub2_new_mini_is_a, en_index_G3_list_train_bef, en_index_G3_list_test_bef, en_index_G3_list_train, en_index_G3_list_test, en_embedding_G3, en_embedding_G3_train, en_embedding_G3_test = pre_data.G3()
    target_train, target_test, G1_graph_sub2_new_is_a, G1_graph_sub2_new_mini, G1_graph_sub2_new_mini_is_a, en_index_G3_list_train_bef, en_index_G3_list_test_bef, en_index_G3_list_train, en_index_G3_list_test, en_embedding_G3, en_embedding_G3_train, en_embedding_G3_test = pre_data.G3()
    all_node_embedding = pre_data.embedding()  # 26989*200
    return target_train, target_test, all_node_embedding, data_G2, left_common, data_G1, en_index_G3_list_train_bef, en_index_G3_list_test_bef


def draw(args, test_score_list, train_loss_list):
    x1 = range(0, args.epoch)
    x2 = range(0, args.epoch)
    y1 = test_score_list
    y2 = train_loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'b*')
    plt.title('Test score vs. epochs')
    plt.ylabel('Test score')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, 'b*')
    plt.xlabel('Train loss vs. epochs')
    plt.ylabel('Train loss')
    plt.savefig(os.path.join(args.save_path_g2, 'Train_loss_and_Test_score_*.png'))


    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'b-')
    plt.title('Test score vs. epochs')
    plt.ylabel('Test score')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, 'b-')
    plt.xlabel('Train loss vs. epochs')
    plt.ylabel('Train loss')
    plt.savefig(os.path.join(args.save_path_g2, 'Train_loss_and_Test_score__.png'))




def main(args):
    set_logger(args)
    if args.which_dataset == 0:

        train_loss_list = []
        test_score_list = []

        logging.info("load data and pre-process data... ...")
        target_train, target_test, all_node_embedding, data_G2, left_common, data_G1, en_index_G3_list_train_bef, en_index_G3_list_test_bef = load_data_and_pre_data(args)
        device = torch.device('cuda' if args.cuda == 1 else 'cpu')
        data_G2 = data_G2.to(device)
        target_train = target_train.to(device)
        target_test = target_test.to(device)


        # load the saved model, if init_checkpoint is true. Otherwise initial model.
        if args.init_checkpoint == 1:
            logging.info("Loading model and optimizer... ...")
            with open(args.init_biggest_score) as fin:  # 26078
                for line in fin:
                    big_score = float(line)
            checkpoint = torch.load(os.path.join(args.save_path_g2, 'checkpoint'))
            model = torch.load(os.path.join(args.save_path_g2, 'model.pkl'))
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate, weight_decay=0.0005)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            logging.info("Initialing model and optimizer... ...")
            big_score = 0
            model = Model(args, data_G2, all_node_embedding, left_common, target_train).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate, weight_decay=0.0005)


        for epoch in range(args.epoch):
            start = datetime.datetime.now()
            logging.info("_________________ epoch:{} _________________ ".format(epoch))
            logging.info("Randomly sample {} types for each entity( {} train entities and {} test entities) ... ...".format(args.sample_num, target_train.shape[0], target_test.shape[0]))
            sample_index_train, sample_index_min_train, sample_label_train = sample(args, target_train, data_G2.num_nodes)  # 6178*150
            sample_index_test, sample_index_min_test, sample_label_test = sample(args, target_test, data_G2.num_nodes)  # 1544*150

            # print("before optimizer, the node_embedding:{}".format(torch.mean(model.all_node_embedding ** 2)))
            logging.info("training... ...")
            model.train()
            optimizer.zero_grad()
            out_train = model(args, data_G2.edge_index, data_G2.edge_type, data_G1.edge_index, en_index_G3_list_train_bef, sample_index_train, sample_index_min_train)  # 6178*106

            logging.info("calculate loss... ...")
            loss = F.binary_cross_entropy(out_train, sample_label_train)
            train_loss_list.append(loss)
            logging.info("train_loss:{}".format(loss))

            logging.info("loss backward... ...")
            loss.backward()
            logging.info("optimizer... ...")
            optimizer.step()
            # print("after optimizer, the node_embedding:{}".format(torch.mean(model.all_node_embedding ** 2)))


            logging.info("test... ...")
            model.eval()
            acc = 0
            out_test = model(args, data_G2.edge_index, data_G2.edge_type, data_G1.edge_index, en_index_G3_list_test_bef, sample_index_test, sample_index_min_test)  # 1544*150

            logging.info("calculate test score... ...")
            for i in range(out_test.shape[0]):  # 1544
                acc_temp = 0
                out_line = out_test[i,:] # 150
                target_line = sample_label_test[i,:]
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
            logging.info("test_score:{}".format(final_acc))
            if final_acc > big_score:
                logging.info("current score is bigger, before:{}, current:{}, save model... ...".format(big_score, final_acc))
                big_score = final_acc
                save_model(model, optimizer, args, big_score)
            else:
                logging.info("biggest score:{} ... ".format(big_score))
            end = datetime.datetime.now()
            logging.info("running time in epoch {} is {} seconds:".format(epoch, str((end - start).seconds)))

        logging.info("finished {} epochs... ...".format(args.epoch))
        logging.info("mean score:{} for {} epochs... ...".format(np.mean(test_score_list), args.epoch))
        logging.info("biggest score:{} ... for {} epochs... ...".format(big_score, args.epoch))
        logging.info("ploting... ...")
        draw(args, test_score_list, train_loss_list)
        logging.info("_________________ finished {} epochs _________________ ".format(args.epoch))


if __name__ == '__main__':
    main(parse_args())