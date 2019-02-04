import collections

import torch
import torch.nn.functional as F
from torch import autograd
from torch import cuda as tcuda
from torch import nn


class RNTN(nn.Module):

    def __init__(self, word2index, hidden_size, output_size, use_gpu=False):
        super(RNTN, self).__init__()

        self.use_gpu = use_gpu

        self.word2index = word2index
        self.embed = nn.Embedding(len(word2index), hidden_size)
        self.V = nn.ParameterList(
            [nn.Parameter(torch.randn(hidden_size * 2, hidden_size * 2)) for _ in range(hidden_size)])  # Tensor
        self.W = nn.Parameter(torch.randn(hidden_size * 2, hidden_size))
        self.b = nn.Parameter(torch.randn(1, hidden_size))
        #         self.W_out = nn.Parameter(torch.randn(hidden_size,output_size))
        self.W_out = nn.Linear(hidden_size, output_size)

    def init_weight(self):
        nn.init.xavier_uniform(self.embed.state_dict()['weight'])
        nn.init.xavier_uniform(self.W_out.state_dict()['weight'])
        for param in self.V.parameters():
            nn.init.xavier_uniform(param)
        nn.init.xavier_uniform(self.W)
        self.b.data.fill_(0)

    def tree_propagation(self, node):

        LongTensor = tcuda.LongTensor if self.use_gpu else torch.LongTensor

        recursive_tensor = collections.OrderedDict()
        if node.isLeaf:
            tensor = autograd.Variable(LongTensor([self.word2index[node.word]])) if node.word in self.word2index.keys() \
                else autograd.Variable(LongTensor([self.word2index['<UNK>']]))
            current = self.embed(tensor)  # 1xD
        else:
            recursive_tensor.update(self.tree_propagation(node.left))
            recursive_tensor.update(self.tree_propagation(node.right))

            concated = torch.cat([recursive_tensor[node.left], recursive_tensor[node.right]], 1)  # 1x2D
            xVx = []
            for i, v in enumerate(self.V):
                #                 xVx.append(torch.matmul(v(concated),concated.transpose(0,1)))
                xVx.append(torch.matmul(torch.matmul(concated, v), concated.transpose(0, 1)))

            xVx = torch.cat(xVx, 1)  # 1xD
            #             Wx = self.W(concated)
            Wx = torch.matmul(concated, self.W)  # 1xD

            current = F.tanh(xVx + Wx + self.b)  # 1xD
        recursive_tensor[node] = current
        return recursive_tensor

    def forward(self, trees, root_only=False):

        propagated = []
        if not isinstance(trees, list):
            trees = [trees]

        for Tree in trees:
            recursive_tensor = self.tree_propagation(Tree.root)
            if root_only:
                recursive_tensor = recursive_tensor[Tree.root]
                propagated.append(recursive_tensor)
            else:
                recursive_tensor = [tensor for node, tensor in recursive_tensor.items()]
                propagated.extend(recursive_tensor)

        propagated = torch.cat(propagated)  # (num_of_node in batch, D)

        return F.log_softmax(self.W_out(propagated), 1)