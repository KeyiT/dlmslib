import collections
import itertools

import numpy as np
import torch
import torch.nn.functional as tfunc
from torch import autograd
from torch import cuda as tcuda
from torch import nn


class RNTN(nn.Module):

    def __init__(self, word2index, embed_matrix, output_size, trainable_embed=True, use_gpu=False):
        super(RNTN, self).__init__()

        self.word2index = word2index
        self.trainable_embed = trainable_embed
        self.use_gpu = use_gpu

        if isinstance(embed_matrix, np.ndarray):
            voc_size, hidden_size = embed_matrix.shape
            self.embed = nn.Embedding(voc_size, hidden_size)
            self.embed.load_state_dict({'weight': embed_matrix})
            self.embed.weight.requires_grad = trainable_embed
        elif isinstance(embed_matrix, (int, np.int8, np.int16, np.int32, np.int64, np.int128)):
            hidden_size = embed_matrix
            voc_size = len(word2index)
            self.embed = nn.Embedding(voc_size, hidden_size)
            self.embed.weight.requires_grad = trainable_embed
        else:
            raise ValueError("embed matrix must be either 2d numpy array or integer")

        self.V = nn.ParameterList(
            [nn.Parameter(torch.randn(hidden_size * 2, hidden_size * 2)) for _ in range(hidden_size)])  # Tensor
        self.W = nn.Parameter(torch.randn(hidden_size * 2, hidden_size))
        self.b = nn.Parameter(torch.randn(1, hidden_size))
        self.W_out = nn.Linear(hidden_size, output_size)

    def init_weight(self):
        if self.trainable_embed:
            nn.init.xavier_uniform(self.embed.state_dict()['weight'])

        nn.init.xavier_uniform(self.W_out.state_dict()['weight'])
        for param in self.V.parameters():
            nn.init.xavier_uniform(param)
        nn.init.xavier_uniform(self.W)
        self.b.data.fill_(0)

    def tree_propagation(self, node):

        LongTensor = tcuda.LongTensor if self.use_gpu else torch.LongTensor

        recursive_tensor = collections.OrderedDict()
        if node.is_leaf():
            tensor = autograd.Variable(LongTensor([self.word2index[node.text]]))
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

            current = torch.tanh(xVx + Wx + self.b)  # 1xD
        recursive_tensor[node] = current
        return recursive_tensor

    def forward(self, tree_roots, root_only=False):

        propagated = []
        if not isinstance(tree_roots, list):
            tree_roots = [tree_roots]

        for tree_root in tree_roots:
            recursive_tensor = self.tree_propagation(tree_root)
            if root_only:
                recursive_tensor = recursive_tensor[tree_root]
                propagated.append(recursive_tensor)
            else:
                recursive_tensor = [tensor for node, tensor in recursive_tensor.items()]
                propagated.extend(recursive_tensor)

        propagated = torch.cat(propagated)  # (num_of_node in batch, D)

        return tfunc.log_softmax(self.W_out(propagated), 1)


class ThinStackHybridLSTM(nn.Module):
    SHIFT_SYMBOL = 1
    REDUCE_SYMBOL = 2

    def __init__(self, embed_matrix, hidden_size, tracker_size, output_size, pad_token_index, alph_droput=0.5, trainable_embed=True,
                 use_gpu=False, train_phase=True):
        super(ThinStackHybridLSTM, self).__init__()

        self.trainable_embed = trainable_embed
        self.use_gpu = use_gpu
        self.train_phase = train_phase
        self.alph_dropout = alph_droput
        self.pad_index = pad_token_index

        if isinstance(embed_matrix, np.ndarray):
            voc_size, embed_size = embed_matrix.shape
            self.embed = nn.Embedding(voc_size, embed_size)
            self.embed.load_state_dict({'weight': embed_matrix})
            self.embed.weight.requires_grad = trainable_embed
        elif isinstance(embed_matrix, (int, np.int8, np.int16, np.int32, np.int64, np.int128)):
            embed_size = embed_matrix
            voc_size = hidden_size
            self.embed = nn.Embedding(voc_size, embed_size)
            self.embed.weight.requires_grad = trainable_embed
        else:
            raise ValueError("embed matrix must be either 2d numpy array or integer")

        self.W_in = nn.Linear(voc_size, hidden_size)
        self.reduce = Reduce(hidden_size, tracker_size)
        self.tracker = Tracker(hidden_size, tracker_size)
        self.W_out = nn.Linear(hidden_size, output_size)

    def prepare_data(self, trees, word2index, max_len):
        max_len_tran = 2 * max_len - 1

        words_batch, transitions_batch = list(), list()
        for tree in trees:
            words, transitions = self.__from_tree(tree, word2index, max_len_tran, self.pad_index, self.pad_index)
            words_batch.append(words)
            transitions_batch.append(transitions_batch)

        words_batch = autograd.Variable(torch.from_numpy(np.asarray(words_batch)))
        transitions_batch = autograd.Variable(torch.from_numpy(np.asarray(transitions_batch)))
        return words_batch, transitions_batch

    @staticmethod
    def __from_tree(tree, word2index, max_len_tran, pre_pad_index, post_pad_index):
        words = tree.get_leaf_texts()
        words = list(map(lambda word: word2index[word], words))
        transitions = tree.get_transitions(
            shift_symbol=ThinStackHybridLSTM.SHIFT_SYMBOL, reduce_symbol=ThinStackHybridLSTM.REDUCE_SYMBOL)

        num_words = len(words)
        num_transitions = len(transitions)

        if len(transitions) < max_len_tran:
            # pad transitions with shift
            num_pad_shifts = max_len_tran - num_transitions
            transitions = [ThinStackHybridLSTM.SHIFT_SYMBOL, ] * num_pad_shifts + transitions
            words = [pre_pad_index] * num_pad_shifts + \
                    words + \
                    [post_pad_index] * (max_len_tran - num_pad_shifts - num_words)

        elif len(transitions) > max_len_tran:
            num_shift_before_crop = num_words

            transitions = transitions[len(transitions) - max_len_tran:]
            trans = np.asarray(transitions)
            num_shift_after_crop = np.sum(trans[trans == ThinStackHybridLSTM.SHIFT_SYMBOL])

            words = words[num_shift_before_crop - num_shift_after_crop:]
            words = words + [post_pad_index, ] * (max_len_tran - len(words))

        # pre-pad every data with one empty tokens and shift
        transitions = [ThinStackHybridLSTM.SHIFT_SYMBOL,] + transitions
        words = [pre_pad_index,] + words

        return words, transitions


    def init_weight(self):
        if self.trainable_embed:
            nn.init.xavier_uniform(self.embed.state_dict()['weight'])

        nn.init.xavier_uniform(self.W_out.state_dict()['weight'])
        nn.init.xavier_uniform(self.W_in.state_dict()['weight'])

    def forward(self, token_index_sequences, transitions):

        buffers = self.embed(token_index_sequences)
        buffers = tfunc.alpha_dropout(tfunc.selu(self.W_in(buffers)), self.alph_dropout, training=self.train_phase)

        outputs0 = tfunc.log_softmax(self.W_out(buffers), 2).transpose(1, 0)

        buffers = [
            list(torch.split(b.squeeze(1), 1, 0))[::-1]
            for b in torch.split(buffers, 1, 0)
        ]

        transitions.transpose_(1, 0)

        # The input comes in as a single tensor of word embeddings;
        # I need it to be a list of stacks, one for each example in
        # the batch, that we can pop from independently. The words in
        # each example have already been reversed, so that they can
        # be read from left to right by popping from the end of each
        # list; they have also been prefixed with a null value.

        # shape = (max_len, batch, embed_dims)
        buffers = [list(map(lambda vec_: torch.cat([vec_, vec_], 0), buf)) for buf in buffers]

        pad_embed = buffers[0][0]
        stacks = [[pad_embed, pad_embed] for _ in buffers]

        self.tracker.reset_state()

        # TODO
        # shape = (max_len, batch)
        num_transitions = transitions.size(0)

        outputs1 = list()
        for i in range(num_transitions):
            trans = transitions[i]
            tracker_states = self.tracker(buffers, stacks)

            lefts, rights, trackings = [], [], []
            batch = zip(trans.data, buffers, stacks, tracker_states)

            for bi in range(len(batch)):
                transition, buf, stack, tracking = batch[bi]
                if transition == self.SHIFT_SYMBOL:  # shift
                    stack.append(buf.pop())
                elif transition == self.REDUCE_SYMBOL:  # reduce
                    rights.append(stack.pop())
                    lefts.append(stack.pop())
                    trackings.append(tracking)

                # make sure tree are good
                while len(stack) < 2:
                    stack.append(pad_embed)

            if rights:
                hc_list, hc_tensor = self.reduce(lefts, rights, trackings)
                reduced = iter(hc_list)
                for transition, stack in zip(trans.data, stacks):
                    if transition == 2:
                        stack.append(next(reduced))

                outputs1.append(tfunc.log_softmax(self.W_out(hc_tensor[0]), 1))

        # shape2 = (max_len, batch_size, output_dim)
        # shape1 = (max_len, [num_reduce], output_dim)
        return outputs0, outputs1


class Reduce(nn.Module):
    """TreeLSTM composition module for SPINN.
    The TreeLSTM has two or three inputs: the first two are the left and right
    children being composed; the third is the current state of the tracker
    LSTM if one is present in the SPINN model.
    Args:
        size: The size of the model state.
        tracker_size: The size of the tracker LSTM hidden state, or None if no
            tracker is present.
    """

    def __init__(self, size, tracker_size):
        super(Reduce, self).__init__()
        self.left = nn.Linear(size, 5 * size)
        self.right = nn.Linear(size, 5 * size, bias=False)
        self.track = nn.Linear(tracker_size, 5 * size, bias=False)

    def forward(self, left_in, right_in, tracking=None):
        """Perform batched TreeLSTM composition.
        This implements the REDUCE operation of a SPINN in parallel for a
        batch of nodes. The batch size is flexible; only provide this function
        the nodes that actually need to be REDUCEd.
        The TreeLSTM has two or three inputs: the first two are the left and
        right children being composed; the third is the current state of the
        tracker LSTM if one is present in the SPINN model. All are provided as
        iterables and batched internally into tensors.
        Additionally augments each new node with pointers to its children.
        Args:
            left_in: Iterable of ``B`` ~autograd.Variable objects containing
                ``c`` and ``h`` concatenated for the left child of each node
                in the batch.
            right_in: Iterable of ``B`` ~autograd.Variable objects containing
                ``c`` and ``h`` concatenated for the right child of each node
                in the batch.
            tracking: Iterable of ``B`` ~autograd.Variable objects containing
                ``c`` and ``h`` concatenated for the tracker LSTM state of
                each node in the batch, or None.
        Returns:
            out: Tuple of ``B`` ~autograd.Variable objects containing ``c`` and
                ``h`` concatenated for the LSTM state of each new node. These
                objects are also augmented with ``left`` and ``right``
                attributes.
        """
        left, right = _bundle(left_in), _bundle(right_in)
        tracking = _bundle(tracking)
        lstm_in = self.left(left[0])
        lstm_in += self.right(right[0])
        lstm_in += self.track(tracking[0])
        hcs = Reduce.tree_lstm(left[1], right[1], lstm_in)
        out = _unbundle(hcs)
        return out, hcs

    @classmethod
    def tree_lstm(cls, c1, c2, lstm_in):
        a, i, f1, f2, o = lstm_in.chunk(5, 1)
        c = a.tanh() * i.sigmoid() + f1.sigmoid() * c1 + f2.sigmoid() * c2
        h = o.sigmoid() * c.tanh()
        return h, c


class Tracker(nn.Module):

    def __init__(self, size, tracker_size):
        super(Tracker, self).__init__()
        self.rnn = nn.LSTMCell(3 * size, tracker_size)
        self.state_size = tracker_size

    def reset_state(self):
        self.state = None

    def forward(self, bufs, stacks):
        buf = _bundle([buf[-1] for buf in bufs])[0]
        stack1 = _bundle(stack[-1] for stack in stacks)[0]
        stack2 = _bundle(stack[-2] for stack in stacks)[0]
        x = torch.cat((buf, stack1, stack2), 1)
        if self.state is None:
            self.state = 2 * [autograd.Variable(
                x.data.new(x.size(0), self.state_size).zero_())]
        self.state = self.rnn(x, self.state)
        return _unbundle(self.state)


def _bundle(states):
    if states is None:
        return None
    states = tuple(states)
    if states[0] is None:
        return None

    # states is a list of B tensors of dimension (1, 2H)
    # this returns 2 tensors of dimension (B, H)
    return torch.cat(states, 0).chunk(2, 1)


def _unbundle(state):
    if state is None:
        return itertools.repeat(None)
    # state is a pair of tensors of dimension (B, H)
    # this returns a list of B tensors of dimension (1, 2H)
    return torch.split(torch.cat(state, 1), 1, 0)
