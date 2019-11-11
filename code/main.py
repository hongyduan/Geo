
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
    parser.add_argument('--specific_test', type=int, default=0)
    parser.add_argument('--specific_train', type=int, default=0)

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
    parser.add_argument('--epoch', type=int, default=2)
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


def save_model(model, optimizer, args, biggest_score, biggest_mrr, biggest_top3):
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path_g2, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)
    with open(os.path.join(args.save_path_g2, 'final_test_score.json'), 'w') as scorejson:
        json.dump(biggest_score, scorejson)
    with open(os.path.join(args.save_path_g2, 'final_test_mrr.json'), 'w') as mrrjson:
        json.dump(biggest_mrr, mrrjson)
    with open(os.path.join(args.save_path_g2, 'final_test_top3.json'), 'w') as top3json:
        json.dump(biggest_top3, top3json)
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


def sample(args, target_small, target_big, g2_num_nodes, mode):
    if (mode == "train" and args.specific_train == 0) or (mode == "test" and args.specific_test == 0):
        sample_index = torch.zeros((target_big.size(0), args.sample_num))  # 6178*150
        sample_index_min = torch.zeros((target_big.size(0), args.sample_num))  # 6178*150
        sample_label = torch.zeros((target_big.size(0), args.sample_num))  # 6178*150
        target_train_nonzero = target_big.nonzero()
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
            list1 = [x for x in range(target_big.shape[1])]
            tt = list(set(list1).difference(set(values)))
            slice = random.sample(tt, a)
            start_id = len(values)
            for ii in slice:
                sample_index[key, start_id] = ii + g2_num_nodes
                sample_index_min[key, start_id] = ii
                start_id = start_id + 1

    elif (mode=="train" and args.specific_train == 1) or (mode == "test" and args.specific_test == 1):
        sample_index = torch.zeros((target_small.size(0), args.sample_num))  # 6178*150
        sample_index_min = torch.zeros((target_small.size(0), args.sample_num))  # 6178*150
        sample_label = torch.zeros((target_small.size(0), args.sample_num))  # 6178*150
        target_train_nonzero = target_small.nonzero()
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
        # big
        target_train_nonzero_tmp = target_big.nonzero()
        ttt = target_train_nonzero_tmp[0:10,:]
        start_tmp = -1
        tmp_id_tmp = 0
        tmp_dict_tmp = OrderedDict()
        for item in target_train_nonzero_tmp:
            if item[0] == start_tmp:
                tmp_dict_tmp[int(item[0])].append(int(item[1]))
                tmp_id_tmp = tmp_id_tmp + 1
            else:
                tmp_dict_tmp[int(item[0])] = list()
                start_tmp = item[0]
                tmp_id_tmp = 0
                tmp_dict_tmp[int(item[0])].append(int(item[1]))
                tmp_id_tmp = tmp_id_tmp + 1
        for key, values in tmp_dict_tmp.items():
            a = args.sample_num - len(tmp_dict[key])
            list1 = [x for x in range(target_big.shape[1])]
            tt = list(set(list1).difference(set(values)))
            slice = random.sample(tt, a)
            start_id = len(tmp_dict[key])
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
    target_train, target_train_small, target_test, target_test_big, G1_graph_sub2_new_is_a, G1_graph_sub2_new_mini, G1_graph_sub2_new_mini_is_a, en_index_G3_list_train_bef, en_index_G3_list_test_bef, en_index_G3_list_train, en_index_G3_list_test, en_embedding_G3, en_embedding_G3_train, en_embedding_G3_test = pre_data.G3()
    all_node_embedding = pre_data.embedding()  # 26989*200
    return target_train, target_train_small, target_test, target_test_big, all_node_embedding, data_G2, left_common, data_G1, en_index_G3_list_train_bef, en_index_G3_list_test_bef

def evalua(args, out_test, sample_label_test, en_index_G3_list_test_bef, test_score_list, big_score, model, optimizer, test_mrr_list, big_mrr, acc, top3, big_top3, test_top3_list):

    # MRR
    logging.info("calculate MRR... ...")
    MRR_final = 0
    for i in range(out_test.shape[0]):  # 1544
        target_line = sample_label_test[i, :]
        aim_top_pos = torch.nonzero(target_line)
        aim_top_num = aim_top_pos.shape[0]
        out_line = out_test[i, :]  # 150
        argsort = torch.argsort(out_line, dim=0, descending=True)
        print("for debug")
        MRR_tmp = 0
        for cur_tmp in aim_top_pos:
            aim_top_pos_copy = aim_top_pos.squeeze(1)
            t_list = aim_top_pos_copy.numpy().tolist()
            t_i = cur_tmp.item()
            t_list.remove(t_i)
            argsort_list = argsort.numpy().tolist()
            argsort_list_tmp = [i for i in argsort_list if i not in t_list]
            argsort_tensor_tmp = torch.Tensor(argsort_list_tmp)
            ranking_tmp = (argsort_tensor_tmp == cur_tmp).nonzero()
            assert ranking_tmp.size(0) == 1
            ranking_tmp = ranking_tmp.item() + 1
            mrr_tmp = 1.0 / ranking_tmp
            MRR_tmp = MRR_tmp + mrr_tmp
        MRR_final_tmp = MRR_tmp / aim_top_num
        MRR_final = MRR_final + MRR_final_tmp
    MRR = MRR_final / out_test.shape[0]
    test_mrr_list.append(MRR)
    logging.info("MRR:{}".format(MRR))
    if MRR > big_mrr:
        logging.info("current MRR is bigger, before:{}, current:{}... ...".format(big_mrr, MRR))
        big_mrr = MRR
    else:
        logging.info("biggest MRR:{} ... ".format(big_mrr))


    # top3
    logging.info("calculate Hit@3... ...")
    for i in range(out_test.shape[0]):  # 1544
        top3_temp = 0
        out_line = out_test[i, :]  # 150
        target_line = sample_label_test[i, :]
        aim_top_pos = torch.nonzero(target_line)
        # aim_top_num = aim_top_pos.shape[0]
        # maxk = max((aim_top_num*10,))
        # maxk = max((aim_top_num,))
        out_top, out_top_pos = out_line.topk(3, 0, True, True)
        for ii in out_top_pos:
            if ii in aim_top_pos:
                top3_temp = top3_temp + 1
        top3 = top3_temp / 3 + top3
    final_top3 = top3 / len(en_index_G3_list_test_bef)
    test_top3_list.append(final_top3)
    logging.info("top@3:{}".format(final_top3))
    if final_top3 > big_top3:
        logging.info(
            "current top@3 is bigger, before:{}, current:{}... ...".format(big_top3, final_top3))
        big_top3 = final_top3
    else:
        logging.info("biggest top@3:{} ... ".format(big_top3))

    # accuracy
    logging.info("calculate accuracy... ...")
    for i in range(out_test.shape[0]):  # 1544
        acc_temp = 0
        out_line = out_test[i, :]  # 150
        target_line = sample_label_test[i, :]
        aim_top_pos = torch.nonzero(target_line)
        aim_top_num = aim_top_pos.shape[0]
        # maxk = max((aim_top_num*10,))
        maxk = max((aim_top_num,))
        out_top, out_top_pos = out_line.topk(maxk, 0, True, True)
        for ii in out_top_pos:
            if ii in aim_top_pos:
                acc_temp = acc_temp + 1
        acc = acc_temp / aim_top_num + acc
    final_acc = acc / len(en_index_G3_list_test_bef)
    test_score_list.append(final_acc)
    logging.info("accuracy:{}".format(final_acc))
    if final_acc > big_score:
        logging.info(
            "current accuracy is bigger, before:{}, current:{}... ...".format(big_score, final_acc))
        big_score = final_acc
        save_model(model, optimizer, args, big_score, big_mrr, big_top3)
    else:
        logging.info("biggest accuracy:{} ... ".format(big_score))

    return test_score_list, big_score, test_mrr_list, big_mrr, test_top3_list, big_top3



def draw(args, test_score_list, train_loss_list, test_mrr_list, test_top3_list):
    x1 = range(0, args.epoch)
    x2 = range(0, args.epoch)
    x3 = range(0, args.epoch)
    x4 = range(0, args.epoch)
    y1 = test_score_list
    y2 = train_loss_list
    y3 = test_mrr_list
    y4 = test_top3_list

    plt.subplot(4, 1, 1)
    plt.plot(x1, y1, 'b*')
    plt.title('Accuracy vs. epochs')
    plt.ylabel('Accuracy')
    plt.subplot(4, 1, 2)
    plt.plot(x2, y2, 'b*')
    plt.xlabel('Train loss vs. epochs')
    plt.ylabel('Train loss')
    plt.subplot(4, 1, 3)
    plt.plot(x3, y3, 'b*')
    plt.xlabel('Test MRR vs. epochs')
    plt.ylabel('Test MRR')
    plt.subplot(4, 1, 4)
    plt.plot(x4, y4, 'b*')
    plt.xlabel('Test Top@3 vs. epochs')
    plt.ylabel('Test Top@3')
    plt.savefig(os.path.join(args.save_path_g2, 'Train_loss_MRR_Top@3_Accuracy_*.png'))


    plt.subplot(4, 1, 1)
    plt.plot(x1, y1, 'b-')
    plt.title('Accuracy vs. epochs')
    plt.ylabel('Accuracy')
    plt.subplot(4, 1, 2)
    plt.plot(x2, y2, 'b-')
    plt.xlabel('Train loss vs. epochs')
    plt.ylabel('Train loss')
    plt.subplot(4, 1, 3)
    plt.plot(x3, y3, 'b-')
    plt.xlabel('Test MRR vs. epochs')
    plt.ylabel('Test MRR')
    plt.subplot(4, 1, 4)
    plt.plot(x4, y4, 'b-')
    plt.xlabel('Test Top@3 vs. epochs')
    plt.ylabel('Test Top@3')
    plt.savefig(os.path.join(args.save_path_g2, 'Train_loss_MRR_Top@3_Accuracy__.png'))


def main(args):
    set_logger(args)
    if args.which_dataset == 0:

        train_loss_list = []
        test_score_list = []
        test_mrr_list = []
        test_top3_list = []

        logging.info("load data and pre-process data... ...")
        target_train, target_train_small, target_test, target_test_big, all_node_embedding, data_G2, left_common, data_G1, en_index_G3_list_train_bef, en_index_G3_list_test_bef = load_data_and_pre_data(args)
        device = torch.device('cuda' if args.cuda == 1 else 'cpu')
        data_G2 = data_G2.to(device)
        target_train = target_train.to(device)
        target_test = target_test.to(device)


        # load the saved model, if init_checkpoint is true. Otherwise initial model.
        if args.init_checkpoint == 1:
            logging.info("Loading model and optimizer... ...")
            with open(os.path.join(args.save_path_g2, 'final_test_score.json')) as fin:  # 26078
                for line in fin:
                    big_score = float(line)
            with open(os.path.join(args.save_path_g2, 'final_test_mrr.json')) as fin:  # 26078
                for line in fin:
                    big_mrr = float(line)
            with open(os.path.join(args.save_path_g2, 'final_test_top3.json')) as fin:  # 26078
                for line in fin:
                    big_top3 = float(line)
            checkpoint = torch.load(os.path.join(args.save_path_g2, 'checkpoint'))
            model = torch.load(os.path.join(args.save_path_g2, 'model.pkl'))
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate, weight_decay=0.0005)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            logging.info("Initialing model and optimizer... ...")
            big_score = 0
            big_mrr = 0
            big_top3 = 0
            model = Model(args, data_G2, all_node_embedding, left_common, target_train).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate, weight_decay=0.0005)


        for epoch in range(args.epoch):
            start = datetime.datetime.now()
            logging.info("_________________ epoch:{} _________________ ".format(epoch))
            logging.info("Randomly sample {} types for each entity( {} train entities and {} test entities) ... ...".format(args.sample_num, target_train.shape[0], target_test.shape[0]))
            sample_index_train, sample_index_min_train, sample_label_train = sample(args, target_train_small, target_train, data_G2.num_nodes, "train")  # 6178*150
            sample_index_test, sample_index_min_test, sample_label_test = sample(args, target_test, target_test_big, data_G2.num_nodes, "test")  # 1544*150

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
            top3 = 0
            out_test = model(args, data_G2.edge_index, data_G2.edge_type, data_G1.edge_index, en_index_G3_list_test_bef, sample_index_test, sample_index_min_test)  # 1544*150

            # evaluation
            test_score_list, big_score, test_mrr_list, big_mrr, test_top3_list, big_top3 = evalua(args, out_test, sample_label_test, en_index_G3_list_test_bef, test_score_list, big_score, model, optimizer, test_mrr_list, big_mrr, acc, top3, big_top3, test_top3_list)
            end = datetime.datetime.now()
            logging.info("running time in epoch {} is {} seconds:".format(epoch, str((end - start).seconds)))

        logging.info("finished {} epochs... ...".format(args.epoch))
        logging.info("mean accuracy:{} for {} epochs... ...".format(np.mean(test_score_list), args.epoch))
        logging.info("biggest accuracy:{} ... for {} epochs... ...".format(big_score, args.epoch))
        logging.info("mean MRR:{} for {} epochs... ...".format(np.mean(test_mrr_list), args.epoch))
        logging.info("biggest MRR:{} ... for {} epochs... ...".format(big_mrr, args.epoch))
        logging.info("ploting... ...")
        draw(args, test_score_list, train_loss_list, test_mrr_list, test_top3_list)
        logging.info("_________________ finished {} epochs _________________ ".format(args.epoch))


if __name__ == '__main__':
    main(parse_args())