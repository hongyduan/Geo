

from torch_geometric.datasets import Entities
import torch.nn.functional as F
from model import Model
import os.path as osp
import numpy as np
import argparse
import torch
import json
import os


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--save_path', type=str, default="/Users/bubuying/PycharmProjects/Geo/save")

    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--relu_use', type=int, default=0)
    parser.add_argument('--which_dataset', type=int, default=4)  # AIFB 1; MUTAG 2; BGS 3; AM 4;   # 95;80;87;89

    parser.add_argument('--learn_rate', type=float, default=0.01)
    parser.add_argument('--node_dim', type=int, default=500)
    parser.add_argument('--hidden_dim', type=int, default=10)  # AIFB 16;   MUTAG 16; BGS 16; AM 10;
    parser.add_argument('--num_bases', type=int, default=40)  # AIFB 0; MUTAG 30; BGS 40; AM 40;
    parser.add_argument('--num_classes', type=int, default=11)  # AIFB 4; MUTAG 2; BGS 2; AM 11;
    parser.add_argument('--epoch', type=int, default=50)
    return parser.parse_args(args)



def save_model(model, optimizer, args, biggest_score):
    if args.which_dataset == 1:
        save_path = os.path.join(args.save_path, 'AIFB')
    elif args.which_dataset == 2:
        save_path = os.path.join(args.save_path, 'MUTAG')
    elif args.which_dataset == 4:
        save_path = os.path.join(args.save_path, 'BGS')
    elif args.which_dataset == 3:
        save_path = os.path.join(args.save_path, 'AM')



    argparse_dict = vars(args)
    with open(os.path.join(save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)
    torch.save({
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(save_path, 'checkpoint_en')
    )
    # # for print
    # for var_name in optimizer.state_dict():
    #     print(var_name,'\t',optimizer.state_dict()[var_name])

    entity_embedding_g2 = model.node_embedding_g2.detach().cpu().numpy()
    np.save(
        os.path.join(save_path, 'node_embedding'),
        entity_embedding_g2
    )
    with open(os.path.join(save_path, 'final_test_score.json'), 'w') as scorejson:
        json.dump(biggest_score, scorejson)



def main(args):

    device = torch.device('cuda' if args.cuda == True else 'cpu')

    # AIFB
    if args.which_dataset == 1:
        name = 'AIFB'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Entities', name)
        dataset = Entities(path, name)
        data = dataset[0]

    # MUTAG
    elif args.which_dataset == 2:
        name = 'MUTAG'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..' ,'data' ,'Entities' ,name)
        dataset = Entities(path, name)
        data = dataset[0]

    # BGS
    elif args.which_dataset == 3:
        name = 'BGS'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Entities', name)
        dataset = Entities(path, name)
        data = dataset[0]

    # AM
    elif args.which_dataset == 4:
        name = 'AM'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Entities', name)
        dataset = Entities(path, name)
        data = dataset[0]



    node_embedding = torch.rand((data.num_nodes, args.node_dim))
    data.x = node_embedding
    data.num_relations = dataset.num_relations
    data = data.to(device)
    model = Model(args, data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate, weight_decay=0.0005)

    big_score = 0
    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        out = model(data.edge_index, data.edge_type)
        out_train = out[data.train_idx, :]
        loss = F.binary_cross_entropy(out_train, data.train_y)
        loss.backward()
        optimizer.step()

        # test
        model.eval()
        out = model(data.edge_index, data.edge_type)
        out_test = out[data.test_idx, :]
        pred = out_test.max(1)[1]
        acc = pred.eq(data.test_y).sum().item() / data.test_y.size(0)
        if acc > big_score:
            print('epoch:{}'.format(epoch))
            print("current score is bigger, before:{}, current:{}, save model ... ".format(big_score, acc))
            big_score = acc
            save_model(model, optimizer, args, big_score)
        else:
            print('epoch:{}'.format(epoch))
            print("biggest acore:{} ... ".format(big_score))




if __name__ == '__main__':
    main(parse_args())



















