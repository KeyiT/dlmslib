import unittest

from dlmslib.torch_models import nlp_models
import numpy as np

class NLPModelTests(unittest.TestCase):

    def setUp(self):
        self.x_dims = 4
        self.time_steps = 10
        self.y_dims = 1

    def test_thin_stack_hybrid_lstm(self):
        voca_dim = 10
        output_dim = 2
        w2v = np.ones(shape=(voca_dim, 10))

        model = nlp_models.ThinStackHybridLSTM(
            w2v, self.x_dims, self.x_dims, output_dim, 0, trainable_embed=False
        )

        self.assertIsNotNone(model)