import unittest
import os

from dlmslib.torch_models import nlp_models, trees as trees_module
import numpy as np
import tests

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
        words, trans = nlp_models.ThinStackHybridLSTM.prepare_data(trees, None, 30, 'UNK', 'UNK')
        self.assertIsNotNone(words)
        self.assertIsNotNone(trans)
