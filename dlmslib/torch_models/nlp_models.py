import collections
import itertools

import numpy as np
import sklearn.metrics as sm
import torch
import torch.nn.functional as tfunc
from torch import autograd
from torch import cuda as tcuda
from torch import nn
from torch import optim


class ThinStackHybridLSTM(nn.Module):
    SHIFT_SYMBOL = 1
    REDUCE_SYMBOL = 2

    def __init__(self, embed_matrix, hidden_size, tracker_size, output_size, pad_token_index, alph_droput=0.5,
                 trainable_embed=True,
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
            self.embed.weight = nn.Parameter(torch.from_numpy(embed_matrix).float())
            self.embed.weight.requires_grad = trainable_embed
        elif isinstance(embed_matrix, (int, np.int8, np.int16, np.int32, np.int64, np.int128)):
            embed_size = embed_matrix
            voc_size = hidden_size
            self.embed = nn.Embedding(voc_size, embed_size)
            self.embed.weight.requires_grad = trainable_embed
        else:
            raise ValueError("embed matrix must be either 2d numpy array or integer")

        self.W_in = nn.Linear(embed_size, hidden_size)
        self.reduce = Reduce(hidden_size, tracker_size)
        self.tracker = Tracker(hidden_size, tracker_size)
        self.W_out = nn.Linear(hidden_size, output_size)

    def forward(self, token_index_sequences, transitions):

        buffers = self.embed(token_index_sequences)
        buffers = tfunc.alpha_dropout(tfunc.selu(self.W_in(buffers)), self.alph_dropout, training=self.train_phase)

        outputs0 = tfunc.log_softmax(self.W_out(buffers), 2).transpose(1, 0)

        buffers = [
            list(torch.split(b.squeeze(0), 1, 0))[::-1]
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
        buffers = [list(map(lambda vec_: torch.cat([vec_, vec_], 1), buf)) for buf in buffers]

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
            batch = list(zip(trans.data, buffers, stacks, tracker_states))

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

    def predict_label_for_trees(self, trees, train_phase=False):
        if not hasattr(self, 'word2index_'):
            raise AttributeError('train the model from trees first!')

        max_len = 0
        for tree in trees:
            max_len = max(len(tree.get_leaf_texts()), max_len)

        data = ThinStackHybridLSTM.prepare_data(
            trees, self.word2index_, max_len=max_len, pre_pad_index=self.pad_index, post_pad_index=self.pad_index, for_train=False
        )

        self.train_phase = train_phase
        preds, nodes = self._predict_and_pack_tensor(
            data[0], data[1], data[2], data[3], for_train=False
        )

        preds = preds.max(1)[1].data.tolist()
        for i in range(len(preds)):
            nodes[i].label = preds[i]

    def train_model_from_trees(self, train_trees, word2index, max_len, validation_trees=None, epochs=30, batch_size=32):
        self.word2index_ = word2index

        train_data = ThinStackHybridLSTM.prepare_data(
            train_trees, word2index, max_len=max_len, pre_pad_index=self.pad_index, post_pad_index=self.pad_index
        )

        if validation_trees is not None:
            valid_data = ThinStackHybridLSTM.prepare_data(
                validation_trees, word2index, max_len=max_len, pre_pad_index=self.pad_index,
                post_pad_index=self.pad_index
            )
        else:
            valid_data = [None] * 4

        self.train_model(train_data[0], train_data[1], train_data[2], train_data[3],
                         epochs=epochs, batch_size=batch_size,
                         validation_tokens=valid_data[0], validation_transitions=valid_data[1],
                         validation_labels=valid_data[2], validation_token_labels=valid_data[3])

    def train_model(self, train_tokens, train_transitions, train_labels, train_token_labels,
                    epochs=100, batch_size=30,
                    validation_tokens=None, validation_transitions=None, validation_labels=None,
                    validation_token_labels=None):

        def get_batch(train_tokens_, train_transitions_, train_labels_, train_token_labels_, batch_size_=batch_size):
            indices = np.arange(0, train_tokens.shape[0], step=1, dtype=np.int32).tolist()
            np.random.shuffle(indices)

            train_tokens_ = train_tokens_[indices]
            train_transitions_ = train_transitions_[indices]
            train_labels_ = list(map(
                lambda i_: train_labels_[i_],
                indices
            ))
            train_token_labels_ = list(map(
                lambda i_: train_token_labels_[i_],
                indices
            ))

            idx = 0
            while idx < train_tokens_.shape[0]:
                end_idx = min(idx + batch_size_, train_tokens_.shape[0])
                batch_tokens_ = train_tokens_[idx: end_idx]
                batch_trans_ = train_transitions_[idx: end_idx]
                batch_labels_ = train_labels_[idx: end_idx]
                batch_token_labels_ = train_token_labels_[idx: end_idx]
                idx = end_idx
                yield batch_tokens_, batch_trans_, batch_labels_, batch_token_labels_

        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.RMSprop(self.parameters())
        self.train_phase = True
        for epoch in range(epochs):
            losses = []

            batch_index = 0
            for batch_tokens, batch_transitions, batch_labels, batch_token_labels in \
                    get_batch(train_tokens, train_transitions, train_labels, train_token_labels):
                preds, labels = self._predict_and_pack_tensor(
                    batch_tokens, batch_transitions, batch_labels, batch_token_labels
                )

                loss = loss_func(preds, labels)
                losses.append(loss.data.tolist())

                loss.backward()
                optimizer.step()

                if batch_index % 100 == 0:
                    preds, labels = preds.max(1)[1].data.tolist(), labels.data.tolist()
                    prec_score = sm.precision_score(labels, preds, average='weighted')
                    reca_score = sm.recall_score(labels, preds, average='weighted')

                    print('[%d/%d] mean_loss: %.4f; weighted_precision: %.4f; weighted_recall: %.4f' % (
                    epoch, epochs, np.mean(losses), prec_score, reca_score))
                    losses = []

                batch_index += 1

            if validation_labels is not None and \
                    validation_token_labels is not None and \
                    validation_tokens is not None and \
                    validation_transitions is not None:
                preds, labels = self._predict_and_pack_tensor(
                    validation_tokens, validation_transitions, validation_labels, validation_token_labels
                )

                preds, labels = preds.max(1)[1].data.tolist(), labels.data.tolist()
                print(sm.classification_report(labels, preds))

    def _predict_and_pack_tensor(self, batch_tokens, batch_transitions, batch_labels, batch_token_labels, for_train=True):

        batch_tokens = torch.from_numpy(batch_tokens)
        batch_transitions = torch.from_numpy(batch_transitions)

        LongTensor = tcuda.LongTensor if self.use_gpu else torch.LongTensor
        model = self.cuda() if self.use_gpu else self
        flatten = lambda l: [item for sublist in l for item in sublist]

        model.zero_grad()
        batch_token_pred, batch_pred = model(batch_tokens, batch_transitions)

        preds_list = list()
        label_list = list()
        # -------- add token prediction ---------
        # [batch_size, max_len, [1, vec_dim]]
        batch_token_pred_list = list(map(
            lambda token_pred: list(map(
                lambda vec: vec,
                torch.split(token_pred.squeeze(1), 1, 0)
            )),
            torch.split(batch_token_pred, 1, 1)
        ))
        # flatten
        # TODO: test
        print((len(batch_token_pred_list), len(batch_token_pred_list[0])))
        print((len(batch_token_labels), len(batch_token_labels[0])))

        batch_token_pred_list = flatten(batch_token_pred_list)
        batch_token_labels = flatten(batch_token_labels)

        # filter out padding leaf nodes
        for ti in range(len(batch_token_pred_list)):
            if batch_token_labels[ti] is not None:
                preds_list.append(batch_token_pred_list[ti])
                label_list.append(batch_token_labels[ti])

        # -------- add prediction ---------
        for li in range(len(batch_labels[0])):
            for bi in range(len(batch_labels)):
                if batch_labels[bi][li] is not None:
                    label_list.append(batch_labels[bi][li])

        # [max_len, [num_reduces], [1, vec_dim]]
        batch_pred = list(map(
            lambda preds: torch.split(preds, 1, 0),
            batch_pred
        ))
        batch_pred = flatten(batch_pred)
        preds_list.extend(batch_pred)


        # make them tensors
        preds = torch.cat(preds_list, 0)
        if for_train:
            labels = autograd.Variable(LongTensor(np.asarray(label_list, dtype=np.int32)))
        else:
            labels = label_list

        # [token_labels, non_leaf_labels]
        return preds, labels

    @classmethod
    def prepare_data(cls, trees, word2index, max_len, pre_pad_index, post_pad_index, for_train=True):
        max_len_tran = 2 * max_len - 1

        words_batch, transitions_batch = list(), list()
        non_leaf_labels_batch, leaf_labels_batch = list(), list()

        for tree in trees:
            words, transitions, non_leaf_labels, leaf_labels = cls.__from_tree(
                tree, word2index, max_len_tran, pre_pad=pre_pad_index, post_pad=post_pad_index, for_train=for_train)
            words_batch.append(words)
            transitions_batch.append(transitions)

            non_leaf_labels_batch.append(non_leaf_labels)
            leaf_labels_batch.append(leaf_labels)

        words_batch = np.array(words_batch)
        transitions_batch = np.array(transitions_batch)

        return words_batch, transitions_batch, non_leaf_labels_batch, leaf_labels_batch


    @staticmethod
    def __from_tree(tree, word2index, max_len_tran, pre_pad, post_pad, for_train=True):
        words = tree.get_leaf_texts()

        if word2index is not None:
            words = list(map(lambda word: word2index[word], words))

        transitions = tree.get_transitions(
            shift_symbol=ThinStackHybridLSTM.SHIFT_SYMBOL, reduce_symbol=ThinStackHybridLSTM.REDUCE_SYMBOL)

        if for_train:
            non_leaf_labels = tree.get_labels_in_transition_order()
            leaf_labels = tree.get_leaf_labels()
        else:
            non_leaf_labels = tree.get_nodes_in_transition_order()
            leaf_labels = tree.get_leaf_nodes()

        num_words = len(words)
        num_transitions = len(transitions)

        if len(transitions) <= max_len_tran:
            # pad transitions with shift
            num_pad_shifts = max_len_tran - num_transitions
            transitions = [ThinStackHybridLSTM.SHIFT_SYMBOL, ] * num_pad_shifts + transitions
            words = [pre_pad] * num_pad_shifts + \
                    words + \
                    [post_pad] * (max_len_tran - num_pad_shifts - num_words)

            # leaf_labels should has the same length as words
            # pad with None
            leaf_labels = [None] * num_pad_shifts + \
                          leaf_labels + \
                          [None] * (max_len_tran - num_pad_shifts - num_words)

            # non_leaf_labels must follow the same operations as transitions
            non_leaf_labels = [None, ] * num_pad_shifts + non_leaf_labels


        elif len(transitions) > max_len_tran:
            num_shift_before_crop = num_words

            transitions = transitions[len(transitions) - max_len_tran:]
            # non_leaf_labels must follow the same operations as transitions
            non_leaf_labels = non_leaf_labels[len(non_leaf_labels) - max_len_tran:]


            trans = np.asarray(transitions)
            num_shift_after_crop = np.sum(trans[trans == ThinStackHybridLSTM.SHIFT_SYMBOL])

            words = words[num_shift_before_crop - num_shift_after_crop:]
            words = words + [post_pad, ] * (max_len_tran - len(words))

            # leaf_labels should has the same length as words
            # pad with None
            leaf_labels = leaf_labels[num_shift_before_crop - num_shift_after_crop:]
            leaf_labels = leaf_labels + [None] * (max_len_tran - len(leaf_labels))

        # pre-pad every data with one empty tokens and shift
        transitions = [ThinStackHybridLSTM.SHIFT_SYMBOL, ] + transitions
        words = [pre_pad, ] + words

        leaf_labels = [None] + leaf_labels
        non_leaf_labels = [None] + non_leaf_labels

        return words, transitions, non_leaf_labels, leaf_labels

    def init_weight(self):
        if self.trainable_embed:
            nn.init.xavier_uniform(self.embed.state_dict()['weight'])

        nn.init.xavier_uniform(self.W_out.state_dict()['weight'])
        nn.init.xavier_uniform(self.W_in.state_dict()['weight'])


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
