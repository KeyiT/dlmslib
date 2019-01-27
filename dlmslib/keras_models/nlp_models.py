from keras import models
from keras import layers
from keras import backend

import numpy as np

from dlmslib import utils
from dlmslib.keras_models import layers as clayers


def predict(keras_model, x, learning_phase=0):

    if isinstance(keras_model.input, list):
        f = backend.function(
            keras_model.input + [backend.learning_phase()],
            [keras_model.output, ]
        )
        y = f(tuple(x) + (learning_phase,))[0]
    else:
        f = backend.function(
            [keras_model.input, backend.learning_phase()],
            [keras_model.output, ]
        )
        y = f((x, learning_phase))[0]
    return y
    

def build_birnn_attention_model(
        voca_dim, time_steps, output_dim, rnn_dim, mlp_dim, 
        item_embedding=None, rnn_depth=1, mlp_depth=1, num_att_channel=1,
        drop_out=0.5, rnn_drop_out=0., rnn_state_drop_out=0.,
        trainable_embedding=False, gpu=False):
    """
    Create A Bidirectional Attention Model.

    :param voca_dim: vocabulary dimension size.
    :param time_steps: the length of input
    :param output_dim: the output dimension size
    :param rnn_dim: rrn dimension size
    :param mlp_dim: the dimension size of fully connected layer
    :param item_embedding: integer, numpy 2D array, or None (default=None)
        If item_embedding is a integer, connect a randomly initialized embedding matrix to the input tensor.
        If item_embedding is a matrix, this matrix will be used as the embedding matrix.
        If item_embedding is None, then connect input tensor to RNN layer directly.
    :param rnn_depth: rnn depth
    :param mlp_depth: the depth of fully connected layers
    :param num_att_channel: the number of attention channels, this can be used to mimic multi-head attention mechanism
    :param drop_out: dropout rate of fully connected layers
    :param rnn_drop_out: dropout rate of rnn layers
    :param rnn_state_drop_out: dropout rate of rnn state tensor
    :param trainable_embedding: boolean
    :param gpu: boolean, default=False
        If True, CuDNNLSTM is used instead of LSTM for RNN layer.
    :return: keras model
    """
    
    if item_embedding is not None:
        inputs = models.Input(shape=(time_steps,), dtype='int32', name='input0')
        x = inputs

        # item embedding
        if isinstance(item_embedding, np.ndarray):
            assert voca_dim == item_embedding.shape[0]
            x = layers.Embedding(
                voca_dim, item_embedding.shape[1], input_length=time_steps,
                weights=[item_embedding, ], trainable=trainable_embedding,
                mask_zero=False, name='embedding_layer0'
            )(x)
        elif utils.is_integer(item_embedding):
            x = layers.Embedding(
                voca_dim, item_embedding, input_length=time_steps,
                trainable=trainable_embedding,
                mask_zero=False, name='embedding_layer0'
            )(x)
        else:
            raise ValueError("item_embedding must be either integer or numpy matrix")
    else:
        inputs = models.Input(shape=(time_steps, voca_dim), dtype='float32', name='input0')
        x = inputs

    if gpu:
        # rnn encoding
        for i in range(rnn_depth):
            x = layers.Bidirectional(
                layers.CuDNNLSTM(rnn_dim, return_sequences=True),
                name='bi_lstm_layer' + str(i))(x)
            x = layers.BatchNormalization(name='rnn_batch_norm_layer' + str(i))(x)
            x = layers.Dropout(name="rnn_dropout_layer" + str(i))(x)
    else:
        # rnn encoding
        for i in range(rnn_depth):
            x = layers.Bidirectional(
                layers.LSTM(rnn_dim, return_sequences=True, dropout=rnn_drop_out, recurrent_dropout=rnn_state_drop_out),
                name='bi_lstm_layer' + str(i))(x)
            x = layers.BatchNormalization(name='rnn_batch_norm_layer' + str(i))(x)

    # attention
    attention_heads = []
    x_per = layers.Permute((2, 1), name='permuted_attention_x')(x)
    for h in range(max(1, num_att_channel)):
        attention = clayers.AttentionWeight(name="attention_weights_layer" + str(h))(x)
        xx = layers.Dot(name='focus_head' + str(h) + '_layer0')([x_per, attention])
        attention_heads.append(xx)

    if num_att_channel > 1:
        x = layers.Concatenate(name='focus_layer0')(attention_heads)
    else:
        x = attention_heads[0]

    x = layers.BatchNormalization(name='focused_batch_norm_layer')(x)

    # MLP Layers
    for i in range(mlp_depth - 1):
        x = layers.Dense(mlp_dim, activation='selu', kernel_initializer='lecun_normal', name='selu_layer' + str(i))(x)
        x = layers.AlphaDropout(drop_out, name='alpha_layer' + str(i))(x)

    outputs = layers.Dense(output_dim, activation="softmax", name="softmax_layer0")(x)

    model = models.Model(inputs, outputs)
    return model


def build_cnn_model(
        voca_dim, time_steps, output_dim, mlp_dim, num_filters, filter_sizes,
        item_embedding=None, mlp_depth=1,
        drop_out=0.5, cnn_drop_out=0.5, pooling='max',
        trainable_embedding=False):
    """
    Create A CNN Model.

    :param voca_dim: vocabulary dimension size.
    :param time_steps: the length of input
    :param output_dim: the output dimension size
    :param num_filters: the number of filters
    :param filter_sizes: list of integers
        The kernel size.
    :param mlp_dim: the dimension size of fully connected layer
    :param item_embedding: integer, numpy 2D array, or None (default=None)
        If item_embedding is a integer, connect a randomly initialized embedding matrix to the input tensor.
        If item_embedding is a matrix, this matrix will be used as the embedding matrix.
        If item_embedding is None, then connect input tensor to RNN layer directly.
    :param mlp_depth: the depth of fully connected layers
    :param drop_out: dropout rate of fully connected layers
    :param cnn_drop_out: dropout rate of between cnn layer and fully connected layers
    :param pooling: str, either 'max' or 'average'
        Pooling method.
    :param trainable_embedding: boolean
    :return: keras model
    """

    if item_embedding is not None:
        inputs = models.Input(shape=(time_steps,), dtype='int32', name='input0')
        x = inputs

        # item embedding
        if isinstance(item_embedding, np.ndarray):
            assert voca_dim == item_embedding.shape[0]
            x = layers.Embedding(
                voca_dim, item_embedding.shape[1], input_length=time_steps,
                weights=[item_embedding, ], trainable=trainable_embedding,
                mask_zero=False, name='embedding_layer0'
            )(x)
        elif utils.is_integer(item_embedding):
            x = layers.Embedding(
                voca_dim, item_embedding, input_length=time_steps,
                trainable=trainable_embedding,
                mask_zero=False, name='embedding_layer0'
            )(x)
        else:
            raise ValueError("item_embedding must be either integer or numpy matrix")
    else:
        inputs = models.Input(shape=(time_steps, voca_dim), dtype='float32', name='input0')
        x = inputs

    pooled_outputs = []
    for i in range(len(filter_sizes)):
        conv = layers.Conv1D(num_filters, kernel_size=filter_sizes[i], padding='valid', activation='relu')(x)
        if pooling == 'max':
            conv = layers.MaxPooling1D(pool_size=time_steps - filter_sizes[i] + 1, strides=1, padding='valid')(conv)
        else:
            conv = layers.AveragePooling1D(pool_size=time_steps - filter_sizes[i] + 1, strides=1, padding='valid')(conv)
        pooled_outputs.append(conv)

    x = layers.Concatenate(name='concated_layer')(pooled_outputs)
    x = layers.Flatten()(x)
    x = layers.Dropout(cnn_drop_out, name='conv_dropout_layer')(x)
    x = layers.BatchNormalization(name="batch_norm_layer")(x)

    # MLP Layers
    for i in range(mlp_depth - 1):
        x = layers.Dense(mlp_dim, activation='selu', kernel_initializer='lecun_normal', name='selu_layer' + str(i))(x)
        x = layers.AlphaDropout(drop_out, name='alpha_layer' + str(i))(x)

    outputs = layers.Dense(output_dim, activation="softmax", name="softmax_layer0")(x)

    model = models.Model(inputs, outputs)
    return model


def build_rnn_cnn_model(
        voca_dim, time_steps, output_dim, rnn_dim, mlp_dim, num_filters, filter_sizes,
        item_embedding=None, rnn_depth=1, mlp_depth=1,
        drop_out=0.5, rnn_drop_out=0.5, rnn_state_drop_out=0.5, cnn_drop_out=0.5, pooling='max',
        trainable_embedding=False, gpu=False):
    """
    Create A Bidirectional CNN Model.

    :param voca_dim: vocabulary dimension size.
    :param time_steps: the length of input
    :param output_dim: the output dimension size
    :param rnn_dim: rrn dimension size
    :param num_filters: the number of filters
    :param filter_sizes: list of integers
        The kernel size.
    :param mlp_dim: the dimension size of fully connected layer
    :param item_embedding: integer, numpy 2D array, or None (default=None)
        If item_embedding is a integer, connect a randomly initialized embedding matrix to the input tensor.
        If item_embedding is a matrix, this matrix will be used as the embedding matrix.
        If item_embedding is None, then connect input tensor to RNN layer directly.
    :param rnn_depth: rnn depth
    :param mlp_depth: the depth of fully connected layers
    :param num_att_channel: the number of attention channels, this can be used to mimic multi-head attention mechanism
    :param drop_out: dropout rate of fully connected layers
    :param rnn_drop_out: dropout rate of rnn layers
    :param rnn_state_drop_out: dropout rate of rnn state tensor
    :param cnn_drop_out: dropout rate of between cnn layer and fully connected layers
    :param pooling: str, either 'max' or 'average'
        Pooling method.
    :param trainable_embedding: boolean
    :param gpu: boolean, default=False
        If True, CuDNNLSTM is used instead of LSTM for RNN layer.
    :return: keras model
    """

    if item_embedding is not None:
        inputs = models.Input(shape=(time_steps,), dtype='int32', name='input0')
        x = inputs

        # item embedding
        if isinstance(item_embedding, np.ndarray):
            assert voca_dim == item_embedding.shape[0]
            x = layers.Embedding(
                voca_dim, item_embedding.shape[1], input_length=time_steps,
                weights=[item_embedding, ], trainable=trainable_embedding,
                mask_zero=False, name='embedding_layer0'
            )(x)
        elif utils.is_integer(item_embedding):
            x = layers.Embedding(
                voca_dim, item_embedding, input_length=time_steps,
                trainable=trainable_embedding,
                mask_zero=False, name='embedding_layer0'
            )(x)
        else:
            raise ValueError("item_embedding must be either integer or numpy matrix")
    else:
        inputs = models.Input(shape=(time_steps, voca_dim), dtype='float32', name='input0')
        x = inputs

    if gpu:
        # rnn encoding
        for i in range(rnn_depth):
            x = layers.Bidirectional(
                layers.CuDNNLSTM(rnn_dim, return_sequences=True),
                name='bi_lstm_layer' + str(i))(x)
            x = layers.BatchNormalization(name='rnn_batch_norm_layer' + str(i))(x)
            x = layers.Dropout(name="rnn_dropout_layer" + str(i))(x)
    else:
        # rnn encoding
        for i in range(rnn_depth):
            x = layers.Bidirectional(
                layers.LSTM(rnn_dim, return_sequences=True, dropout=rnn_drop_out, recurrent_dropout=rnn_state_drop_out),
                name='bi_lstm_layer' + str(i))(x)
            x = layers.BatchNormalization(name='rnn_batch_norm_layer' + str(i))(x)

    pooled_outputs = []
    for i in range(len(filter_sizes)):
        conv = layers.Conv1D(num_filters, kernel_size=filter_sizes[i], padding='valid', activation='relu')(x)
        if pooling == 'max':
            conv = layers.MaxPooling1D(pool_size=time_steps - filter_sizes[i] + 1, strides=1, padding='valid')(conv)
        else:
            conv = layers.AveragePooling1D(pool_size=time_steps - filter_sizes[i] + 1, strides=1, padding='valid')(conv)
        pooled_outputs.append(conv)

    x = layers.Concatenate(name='concated_layer')(pooled_outputs)
    x = layers.Flatten()(x)
    x = layers.Dropout(cnn_drop_out, name='conv_dropout_layer')(x)
    x = layers.BatchNormalization(name="batch_norm_layer")(x)

    # MLP Layers
    for i in range(mlp_depth - 1):
        x = layers.Dense(mlp_dim, activation='selu', kernel_initializer='lecun_normal', name='selu_layer' + str(i))(x)
        x = layers.AlphaDropout(drop_out, name='alpha_layer' + str(i))(x)

    outputs = layers.Dense(output_dim, activation="softmax", name="softmax_layer0")(x)

    model = models.Model(inputs, outputs)
    return model