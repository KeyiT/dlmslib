import os
import unittest

import numpy as np

import tests
from dlmslib.torch_models import nlp_models, trees as trees_module


class NLPModelTests(unittest.TestCase):

    def setUp(self):
        self.x_dims = 4
        self.time_steps = 10
        self.y_dims = 1

        self.test_ptb_file_path = os.path.join(tests.TEST_ROOT, "resources/torch_models/ptb_trees.txt")

    def test_thin_stack_hybrid_lstm(self):
        voca_dim = 10
        output_dim = 2
        w2v = np.ones(shape=(voca_dim, 10))

        model = nlp_models.ThinStackHybridLSTM(
            w2v, self.x_dims, self.x_dims, output_dim, 0, trainable_embed=False
        )

        self.assertIsNotNone(model)

    def test_prepare_data(self):
        trees = trees_module.read_parse_ptb_tree_bank_file(self.test_ptb_file_path)
        words, trans, non_leaf_labels, leaf_labels = nlp_models.ThinStackHybridLSTM.prepare_data(trees, None, 30, 'UNK',
                                                                                                 'UNK')
        self.assertIsNotNone(words)
        self.assertIsNotNone(trans)
        self.assertIsNotNone(non_leaf_labels)
        self.assertIsNotNone(leaf_labels)

    def test_train_model(self):

        train_trees = trees_module.read_parse_ptb_tree_bank_file(self.test_ptb_file_path)
        dev_trees = trees_module.read_parse_ptb_tree_bank_file(self.test_ptb_file_path)

        # build vocab
        w2v_fasttext = {
            'good': np.ones(shape=30),
            'bad': np.ones(shape=30)
        }
        UNKNOWN_TOKEN = '<UNK>'
        EMB_DIM = 30

        def map_unknown_token(tree, embeddings_index):
            if tree is None:
                return

            word = tree.text
            if word not in embeddings_index:
                tree.text = UNKNOWN_TOKEN

            map_unknown_token(tree.left, embeddings_index)
            map_unknown_token(tree.right, embeddings_index)

        for tree in train_trees:
            map_unknown_token(tree, w2v_fasttext)
        for tree in dev_trees:
            map_unknown_token(tree, w2v_fasttext)

        flatten = lambda l: [item for sublist in l for item in sublist]
        vocab = list(set(flatten([t.get_leaf_texts() for t in (train_trees + dev_trees)])))

        word2index = {'<UNK>': 0}
        wv = np.zeros(shape=(len(vocab), EMB_DIM))
        for vo in vocab:
            if word2index.get(vo) is None:
                word2index[vo] = len(word2index)

                wv[word2index[vo]] = w2v_fasttext[vo]

        # model training
        MAX_LEN = 0
        for tree in (train_trees + dev_trees):
            MAX_LEN = max(len(tree.get_leaf_texts()), MAX_LEN)

        hidden_size = 10
        tracker_size = 10
        output_size = 5
        pad_token_index = 0
        train_data = nlp_models.ThinStackHybridLSTM.prepare_data(train_trees, word2index, max_len=MAX_LEN,
                                                                 pre_pad_index=pad_token_index,
                                                                 post_pad_index=pad_token_index)
        dev_data = nlp_models.ThinStackHybridLSTM.prepare_data(dev_trees, word2index, max_len=MAX_LEN,
                                                               pre_pad_index=pad_token_index,
                                                               post_pad_index=pad_token_index)

        model = nlp_models.ThinStackHybridLSTM(wv, hidden_size, tracker_size, output_size, pad_token_index,
                                               trainable_embed=True)
        model.train_model(train_data[0], train_data[1], train_data[2], train_data[3],
                          epochs=1, batch_size=6,
                          validation_tokens=dev_data[0], validation_transitions=dev_data[1],
                          validation_labels=dev_data[2], validation_token_labels=dev_data[3])
