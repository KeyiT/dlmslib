import unittest

from dlmslib.keras_models import nlp_models
import numpy as np

class NLPModelTests(unittest.TestCase):

    def setUp(self):
        self.x_dims = 4
        self.time_steps = 10
        self.y_dims = 1

    def test_build_cnn_model(self):
        voca_dim = 10
        time_steps = 10
        output_dim = 2
        mlp_dim = 10
        num_filters = 5
        filter_sizes = [2, 3, 5]
        item_embedding = np.ones(shape=(voca_dim, 10))
        mlp_depth = 2

        model = nlp_models.build_cnn_model(
            voca_dim, time_steps, output_dim, mlp_dim, num_filters, filter_sizes,
            item_embedding=item_embedding, mlp_depth=mlp_depth
        )

        self.assertIsNotNone(model)

    def test_build_birnn_attention_model(self):
        voca_dim = 10
        time_steps = 10
        output_dim = 2
        rnn_dim = 30
        mlp_dim = 30
        item_embedding = np.ones(shape=(voca_dim, 10))
        rnn_depth = 1
        mlp_depth = 2
        num_att_channel = 3
        gpu = False

        model = nlp_models.build_birnn_attention_model(
            voca_dim, time_steps, output_dim, rnn_dim, mlp_dim,
            item_embedding=item_embedding, rnn_depth=rnn_depth, mlp_depth=mlp_depth, num_att_channel = num_att_channel,
            gpu=gpu
        )

        self.assertIsNotNone(model)

    def test_build_birnn_cnn_attention_model(self):
        voca_dim = 10
        time_steps = 10
        output_dim = 2
        rnn_dim = 30
        mlp_dim = 30
        num_filters = 5
        filter_sizes = [2, 3, 5]
        item_embedding = np.ones(shape=(voca_dim, 10))
        rnn_depth = 1
        mlp_depth = 2
        gpu = False

        model = nlp_models.build_birnn_cnn_model(
            voca_dim, time_steps, output_dim, rnn_dim, mlp_dim, num_filters, filter_sizes,
            item_embedding=item_embedding, rnn_depth=rnn_depth, mlp_depth=mlp_depth, gpu=gpu
        )

        self.assertIsNotNone(model)

