from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter as Param
from inits import uniform
import torch

class RGCN_Layer(MessagePassing):
    def __init__(self, dim_node, flag_of_x, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True,**kwargs):
        super(RGCN_Layer, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.flag_of_x = flag_of_x

        self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = Param(torch.Tensor(num_relations, num_bases))

        if root_weight:
            if flag_of_x == 1:
                self.root = Param(torch.Tensor(in_channels, out_channels))
            if flag_of_x == 0:
                self.root = Param(torch.Tensor(dim_node, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type, edge_norm=edge_norm)

    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if self.flag_of_x == 0:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
            # print('out_0:{}'.format(torch.mean(out**2)))
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)
        return out if edge_norm is None else out * edge_norm.view(-1, 1)


    def update(self, aggr_out, x):
        if self.root is not None:
            out = aggr_out + torch.matmul(x, self.root)
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)