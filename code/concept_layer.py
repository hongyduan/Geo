import torch
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from inits import uniform

class Concept_Layer(MessagePassing):
    # def __init__(self, in_chanels, out_chanels, aggr='add', **kwargs):
    def __init__(self, aggr='add', **kwargs):
        super(Concept_Layer, self).__init__(aggr=aggr, **kwargs)

        # self.in_chanels = in_chanels
        # self.out_chanels = out_chanels

        # self.weight = Parameter(torch.Tensor(in_chanels,out_chanels))

        # self.reset_parameters()

    # def reset_parameters(self):
    #     uniform(self.in_chanels, self.weight)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        # h = torch.matmul(x, self.weight)

        # return self.propagate(edge_index, size=size, x=x, h=h, edge_weight=edge_weight)
        return self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight)

    # def message(self, h_j, edge_weight):
    #     #     return h_j if edge_weight is None else edge_weight.view(-1,1) * h_j

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1,1) * x_j

    def update(self, aggr_out,x):
        return aggr_out + x

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_chanels, self.out_chanels)