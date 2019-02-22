import numpy as np
from keras import backend
from keras import layers
from keras import models

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
        trainable_embedding=False, gpu=False, return_customized_layers=False):
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
    :param return_customized_layers: boolean, default=False
        If True, return model and customized object dictionary, otherwise return model only
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
            x = layers.Dropout(rnn_drop_out, name="rnn_dropout_layer" + str(i))(x)
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
        xx = layers.Dot([2, 1], name='focus_head' + str(h) + '_layer0')([x_per, attention])
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

    if return_customized_layers:
        return model, {'AttentionWeight': clayers.AttentionWeight}
    return model


def build_cnn_model(
        voca_dim, time_steps, output_dim, mlp_dim, num_filters, filter_sizes,
        item_embedding=None, mlp_depth=1,
        drop_out=0.5, cnn_drop_out=0.5, pooling='max',
        trainable_embedding=False, return_customized_layers=False):
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
    :param return_customized_layers: boolean, default=False
        If True, return model and customized object dictionary, otherwise return model only
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

    if return_customized_layers:
        return model, dict()

    return model


def build_birnn_cnn_model(
        voca_dim, time_steps, output_dim, rnn_dim, mlp_dim, num_filters, filter_sizes,
        item_embedding=None, rnn_depth=1, mlp_depth=1,
        drop_out=0.5, rnn_drop_out=0.5, rnn_state_drop_out=0.5, cnn_drop_out=0.5, pooling='max',
        trainable_embedding=False, gpu=False, return_customized_layers=False):
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
    :param return_customized_layers: boolean, default=False
        If True, return model and customized object dictionary, otherwise return model only
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
            x = layers.Dropout(rnn_drop_out, name="rnn_dropout_layer" + str(i))(x)
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

    if return_customized_layers:
        return model, dict()

    return model


def build_birnn_feature_coattention_cnn_model(
        voca_dim, time_steps, num_features, feature_dim, output_dim, model_dim, mlp_dim, num_filters, filter_sizes,
        item_embedding=None, rnn_depth=1, mlp_depth=1,
        drop_out=0.5, rnn_drop_out=0., rnn_state_drop_out=0., cnn_drop_out=0.5, pooling='max',
        trainable_embedding=False, gpu=False, return_customized_layers=False):
    """
    Create A Bidirectional Attention Model.

    :param voca_dim: vocabulary dimension size.
    :param time_steps: the length of input
    :param output_dim: the output dimension size
    :param model_dim: rrn dimension size
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
    :param return_customized_layers: boolean, default=False
        If True, return model and customized object dictionary, otherwise return model only
    :return: keras model
    """
    if model_dim % 2 == 1:
        model_dim += 1

    if item_embedding is not None:
        inputs = models.Input(shape=(time_steps,), dtype='int32', name='input0')
        x1 = inputs

        # item embedding
        if isinstance(item_embedding, np.ndarray):
            assert voca_dim == item_embedding.shape[0]
            x1 = layers.Embedding(
                voca_dim, item_embedding.shape[1], input_length=time_steps,
                weights=[item_embedding, ], trainable=trainable_embedding,
                mask_zero=False, name='embedding_layer0'
            )(x1)
        elif utils.is_integer(item_embedding):
            x1 = layers.Embedding(
                voca_dim, item_embedding, input_length=time_steps,
                trainable=trainable_embedding,
                mask_zero=False, name='embedding_layer0'
            )(x1)
        else:
            raise ValueError("item_embedding must be either integer or numpy matrix")
    else:
        inputs = models.Input(shape=(time_steps, voca_dim), dtype='float32', name='input0')
        x1 = inputs

    inputs1 = models.Input(shape=(num_features, feature_dim), dtype='float32', name='input1')
    x2 = layers.Dense(feature_dim, name="feature_map_layer", activation="relu")(inputs1)

    if gpu:
        # rnn encoding
        for i in range(rnn_depth):
            x1 = layers.Bidirectional(
                layers.CuDNNLSTM(int(model_dim / 2), return_sequences=True),
                name='bi_lstm_layer' + str(i))(x1)
            x1 = layers.BatchNormalization(name='rnn_batch_norm_layer' + str(i))(x1)
            x1 = layers.Dropout(rnn_drop_out, name="rnn_dropout_layer" + str(i))(x1)
    else:
        # rnn encoding
        for i in range(rnn_depth):
            x1 = layers.Bidirectional(
                layers.LSTM(int(model_dim / 2), return_sequences=True, dropout=rnn_drop_out,
                            recurrent_dropout=rnn_state_drop_out),
                name='bi_lstm_layer' + str(i))(x1)
            x1 = layers.BatchNormalization(name='rnn_batch_norm_layer' + str(i))(x1)

    # attention
    attens = clayers.CoAttentionWeight(name="coattention_weights_layer")([x1, x2])

    attens1 = clayers.FeatureNormalization(name="normalized_coattention_weights_layer1", axis=1)(attens)
    attens2 = clayers.FeatureNormalization(name="normalized_coattention_weights_layer2", axis=2)(attens)

    # compare
    focus1 = layers.Dot((1, 1), name="focus_layer1")([attens1, x1])
    focus2 = layers.Dot((2, 1), name="focus_layer2")([attens2, x2])

    pair1 = layers.Concatenate(axis=-1, name="pair_layer1")([x1, focus2])
    pair2 = layers.Concatenate(axis=-1, name="pair_layer2")([x2, focus1])

    x1 = layers.TimeDistributed(layers.Dense(model_dim, activation="relu"), name="compare_layer1")(pair1)
    x2 = layers.TimeDistributed(layers.Dense(model_dim, activation="relu"), name="compare_layer2")(pair2)

    # Multi-Channel CNN for x1
    pooled_outputs = []
    for i in range(len(filter_sizes)):
        conv = layers.Conv1D(num_filters, kernel_size=filter_sizes[i], padding='valid', activation='relu')(x1)
        if pooling == 'max':
            conv = layers.MaxPooling1D(pool_size=time_steps - filter_sizes[i] + 1, strides=1, padding='valid')(conv)
        else:
            conv = layers.AveragePooling1D(pool_size=time_steps - filter_sizes[i] + 1, strides=1, padding='valid')(conv)
        pooled_outputs.append(conv)

    x1 = layers.Concatenate(name='concated_layer')(pooled_outputs)
    x1 = layers.Flatten()(x1)
    x1 = layers.Dropout(cnn_drop_out, name='conv_dropout_layer')(x1)
    x1 = layers.BatchNormalization(name="batch_norm_layer")(x1)

    # Average Pool for x2
    x2 = layers.GlobalAveragePooling1D(name="average_pool_layer")(x2)

    x = layers.Concatenate(axis=1, name="concat_deep_feature_layer")([x1, x2])

    # MLP Layers
    for i in range(mlp_depth - 1):
        x = layers.Dense(mlp_dim, activation='selu', kernel_initializer='lecun_normal', name='selu_layer' + str(i))(x)
        x = layers.AlphaDropout(drop_out, name='alpha_layer' + str(i))(x)

    outputs = layers.Dense(output_dim, activation="softmax", name="softmax_layer0")(x)

    model = models.Model(inputs, outputs)

    if return_customized_layers:
        return model, {'CoAttentionWeight': clayers.CoAttentionWeight,
                       "FeatureNormalization": clayers.FeatureNormalization}

    return model


def build_birnn_multifeature_coattention_model(
        voca_dim, time_steps, num_feature_channels, num_features, feature_dim, output_dim, model_dim, atten_dim, mlp_dim,
        item_embedding=None, rnn_depth=1, mlp_depth=1,
        drop_out=0.5, rnn_drop_out=0., rnn_state_drop_out=0.,
        trainable_embedding=False, gpu=False, return_customized_layers=False):
    """
    Create A Bidirectional Attention Model.

    :param voca_dim: vocabulary dimension size.
    :param time_steps: the length of input
    :param output_dim: the output dimension size
    :param model_dim: rrn dimension size
    :param mlp_dim: the dimension size of fully connected layer
    :param item_embedding: integer, numpy 2D array, or None (default=None)
        If item_embedding is a integer, connect a randomly initialized embedding matrix to the input tensor.
        If item_embedding is a matrix, this matrix will be used as the embedding matrix.
        If item_embedding is None, then connect input tensor to RNN layer directly.
    :param rnn_depth: rnn depth
    :param mlp_depth: the depth of fully connected layers
    :param num_feature_channels: the number of attention channels, this can be used to mimic multi-head attention mechanism
    :param drop_out: dropout rate of fully connected layers
    :param rnn_drop_out: dropout rate of rnn layers
    :param rnn_state_drop_out: dropout rate of rnn state tensor
    :param trainable_embedding: boolean
    :param gpu: boolean, default=False
        If True, CuDNNLSTM is used instead of LSTM for RNN layer.
    :param return_customized_layers: boolean, default=False
        If True, return model and customized object dictionary, otherwise return model only
    :return: keras model
    """

    if model_dim % 2 == 1:
        model_dim += 1

    if item_embedding is not None:
        inputs = models.Input(shape=(time_steps,), dtype='int32', name='input0')
        x1 = inputs

        # item embedding
        if isinstance(item_embedding, np.ndarray):
            assert voca_dim == item_embedding.shape[0]
            x1 = layers.Embedding(
                voca_dim, item_embedding.shape[1], input_length=time_steps,
                weights=[item_embedding, ], trainable=trainable_embedding,
                mask_zero=False, name='embedding_layer0'
            )(x1)
        elif utils.is_integer(item_embedding):
            x1 = layers.Embedding(
                voca_dim, item_embedding, input_length=time_steps,
                trainable=trainable_embedding,
                mask_zero=False, name='embedding_layer0'
            )(x1)
        else:
            raise ValueError("item_embedding must be either integer or numpy matrix")
    else:
        inputs = models.Input(shape=(time_steps, voca_dim), dtype='float32', name='input0')
        x1 = inputs

    inputs1 = list()
    for fi in range(num_feature_channels):
        inputs1.append(models.Input(shape=(num_features, feature_dim), dtype='float32', name='input1' + str(fi)))

    feature_map_layer = layers.TimeDistributed(
        layers.Dense(model_dim, name="feature_map_layer", activation="sigmoid"), name="td_feature_map_layer"
    )
    x2s = list(map(
        lambda input_: feature_map_layer(input_),
        inputs1
    ))

    if gpu:
        # rnn encoding
        for i in range(rnn_depth):
            x1 = layers.Bidirectional(
                layers.CuDNNLSTM(int(model_dim / 2), return_sequences=True),
                name='bi_lstm_layer' + str(i))(x1)
            x1 = layers.BatchNormalization(name='rnn_batch_norm_layer' + str(i))(x1)
            x1 = layers.Dropout(rnn_drop_out, name="rnn_dropout_layer" + str(i))(x1)
    else:
        # rnn encoding
        for i in range(rnn_depth):
            x1 = layers.Bidirectional(
                layers.LSTM(int(model_dim / 2), return_sequences=True, dropout=rnn_drop_out,
                            recurrent_dropout=rnn_state_drop_out),
                name='bi_lstm_layer' + str(i))(x1)
            x1 = layers.BatchNormalization(name='rnn_batch_norm_layer' + str(i))(x1)

    coatten_layer = clayers.CoAttentionWeight(name="coattention_weights_layer")
    featnorm_layer1 = clayers.FeatureNormalization(name="normalized_coattention_weights_layer1", axis=1)
    featnorm_layer2 = clayers.FeatureNormalization(name="normalized_coattention_weights_layer2", axis=2)
    focus_layer1 = layers.Dot((1, 1), name="focus_layer1")
    focus_layer2 = layers.Dot((2, 1), name="focus_layer2")
    pair_layer1 = layers.Concatenate(axis=-1, name="pair_layer1")
    pair_layer2 = layers.Concatenate(axis=-1, name="pair_layer2")

    compare_layer1 = layers.TimeDistributed(layers.Dense(model_dim, activation="relu"), name="compare_layer1")
    compare_layer2 = layers.TimeDistributed(layers.Dense(model_dim, activation="relu"), name="compare_layer2")
    flatten_layer = layers.Flatten(name="flatten_layer")

    xs = list()
    for x2_ in x2s:
        xs += _coatten_compare_aggregate(
            coatten_layer, featnorm_layer1, featnorm_layer2,
            focus_layer1, focus_layer2, pair_layer1, pair_layer2,
            compare_layer1, compare_layer2,
            flatten_layer,
            x1, x2_)

    x = layers.Concatenate(axis=1, name="concat_feature_layer")(xs)

    # MLP Layers
    for i in range(mlp_depth - 1):
        x = layers.Dense(mlp_dim, activation='selu', kernel_initializer='lecun_normal', name='selu_layer' + str(i))(x)
        x = layers.AlphaDropout(drop_out, name='alpha_layer' + str(i))(x)

    outputs = layers.Dense(output_dim, activation="softmax", name="softmax_layer0")(x)

    model = models.Model([inputs] + inputs1, outputs)

    if return_customized_layers:
        return model, {'CoAttentionWeight': clayers.CoAttentionWeight,
                       "FeatureNormalization": clayers.FeatureNormalization}

    return model


def _coatten_compare_aggregate(
        coatten_layer, featnorm_layer1, featnorm_layer2,
        focus_layer1, focus_layer2, pair_layer1, pair_layer2,
        compare_layer1, compare_layer2,
        flatten_layer, x1, x2):
    # attention
    attens = coatten_layer([x1, x2])

    attens1 = featnorm_layer1(attens)
    attens2 = featnorm_layer2(attens)

    # compare
    focus1 = focus_layer1([attens1, x1])
    focus2 = focus_layer2([attens2, x2])

    pair1 = pair_layer1([x1, focus2])
    pair2 = pair_layer2([x2, focus1])

    x1 = compare_layer1(pair1)
    x2 = compare_layer2(pair2)

    # Average Pool for x1
    x1 = flatten_layer(x1)

    # Average Pool for x2
    x2 = flatten_layer(x2)

    return [x1, x2]
