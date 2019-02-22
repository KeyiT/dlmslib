from keras import initializers, regularizers, constraints, activations
from keras.engine import Layer
import keras.backend as K
from keras.layers import merge
import tensorflow as tf

class AttentionWeight(Layer):
    """
        This code is a modified version of cbaziotis implementation:  GithubGist cbaziotis/AttentionWithContext.py
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, steps)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWeight())
        """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWeight, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWeight, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = _dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = _dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        return a

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]

    def get_config(self):
        config = {
            'W_regularizer': regularizers.serialize(self.W_regularizer),
            'u_regularizer': regularizers.serialize(self.u_regularizer),
            'b_regularizer': regularizers.serialize(self.b_regularizer),
            'W_constraint': constraints.serialize(self.W_constraint),
            'u_constraint': constraints.serialize(self.u_constraint),
            'b_constraint': constraints.serialize(self.b_constraint),
            'bias': self.bias
        }
        base_config = super(AttentionWeight, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CoAttentionWeight(Layer):
    """
        Unnormalized Co-Attention operation for temporal data.
        Supports Masking.
        Follows the work of Ankur et al. [https://aclweb.org/anthology/D16-1244]
        "A Decomposable Attention Model for Natural Language Inference"
        # Input shape
            List of 2 3D tensor with shape: `(samples, steps1, features1)` and `(samples, steps2, features2)`.
        # Output shape
            3D tensor with shape: `(samples, steps1, step2)`.
        :param kwargs:
        """

    def __init__(self, W_regularizer=None, W_constraint=None, **kwargs):

        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.W_constraint = constraints.get(W_constraint)

        super(CoAttentionWeight, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CoAttentionWeight, self).build(input_shape)

        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Coattention` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])

        if shape1[-1] != shape2[-1]:
            raise ValueError("The last dimention of input tensors must be same. "
                             "Otherwise use RemappedCoattentionWeight instead")

        self.W = self.add_weight((shape1[-1], shape1[-1]),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)


    def compute_mask(self, input, input_mask=None):
        # pass the mask to the next layers
        return input_mask

    def call(self, inputs, **kwargs):
        if len(inputs) != 2:
            raise ValueError('A `Subtract` layer should be called '
                             'on exactly 2 inputs')

        x1, x2 = inputs[0], inputs[1]

        if x1.shape[-1] != x2.shape[-1]:
            raise ValueError("The last dimention of input tensors must be same. "
                             "Otherwise use RemappedCoattentionWeight instead")

        # atten = exp(u1 W u2^T)
        atten = _dot_product(x1, self.W)
        atten = K.batch_dot(atten, x2, axes=[2, 2])
        atten = K.exp(atten)

        return atten

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Coattention` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])

        if shape1[0] != shape2[0]:
            raise ValueError("batch size must be same")

        return shape1[0], shape1[1], shape2[1]

    def get_config(self):
        config = {
            'W_regularizer': regularizers.serialize(self.W_regularizer),
            'W_constraint': constraints.serialize(self.W_constraint),
        }
        base_config = super(CoAttentionWeight, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RemappedCoAttentionWeight(Layer):
    """
        Unnormalized Co-Attention operation for temporal data.
        Supports Masking.
        Follows the work of Ankur et al. [https://aclweb.org/anthology/D16-1244]
        "A Decomposable Attention Model for Natural Language Inference"
        # Input shape
            List of 2 3D tensor with shape: `(samples, steps1, features1)` and `(samples, steps2, features2)`.
        # Output shape
            3D tensor with shape: `(samples, steps1, step2)`.
        :param kwargs:
        """

    def __init__(self, model_size, activation='sigmoid',
                 W1_regularizer=None, W2_regularizer=None, b1_regularizer=None, b2_regularizer=None,
                 W1_constraint=None, W2_constraint=None, b1_constraint=None, b2_constraint=None,
                 bias1=True, bias2=True, **kwargs):

        self.model_size = model_size
        self.init = initializers.get('glorot_uniform')

        self.W1_regularizer = regularizers.get(W1_regularizer)
        self.W2_regularizer = regularizers.get(W2_regularizer)
        self.b1_regularizer = regularizers.get(b1_regularizer)
        self.b2_regularizer = regularizers.get(b2_regularizer)

        self.W1_constraint = constraints.get(W1_constraint)
        self.W2_constraint = constraints.get(W2_constraint)
        self.b1_constraint = constraints.get(b1_constraint)
        self.b2_constraint = constraints.get(b2_constraint)

        self.bias1 = bias1
        self.bias2 = bias2
        self.activation = activations.get(activation)
        super(RemappedCoAttentionWeight, self).__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) != 2:
            raise ValueError("input must be a size two list which contains two tensors")

        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])

        self.W1 = self.add_weight((self.model_size, shape1[-1]),
                                  initializer=self.init,
                                  name='{}_W1'.format(self.name),
                                  regularizer=self.W1_regularizer,
                                  constraint=self.W1_constraint)

        self.W2 = self.add_weight((self.model_size, shape2[-1]),
                                  initializer=self.init,
                                  name='{}_W2'.format(self.name),
                                  regularizer=self.W2_regularizer,
                                  constraint=self.W2_constraint)

        if self.bias1:
            self.b1 = self.add_weight((self.model_size,),
                                      initializer='zero',
                                      name='{}_b1'.format(self.name),
                                      regularizer=self.b1_regularizer,
                                      constraint=self.b1_constraint)

        if self.bias2:
            self.b2 = self.add_weight((self.model_size,),
                                      initializer='zero',
                                      name='{}_b2'.format(self.name),
                                      regularizer=self.b2_regularizer,
                                      constraint=self.b2_constraint)

    def compute_mask(self, input, input_mask=None):
        # pass the mask to the next layers
        return input_mask

    def call(self, inputs, **kwargs):
        if len(inputs) != 2:
            raise ValueError('A `Subtract` layer should be called '
                             'on exactly 2 inputs')

        x1, x2 = inputs[0], inputs[1]

        # u = Wx + b
        u1 = _dot_product(x1, self.W1)
        if self.bias1:
            u1 += self.b1

        u2 = _dot_product(x2, self.W2)
        if self.bias2:
            u2 += self.b2

        # u = Activation(Wx + b)
        u1 = self.activation(u1)
        u2 = self.activation(u2)

        # atten = exp(u1 u2^T)
        atten = K.batch_dot(u1, u2, axes=[2, 2])
        atten = K.exp(atten)

        return atten

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Dot` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])

        if shape1[0] != shape2[0]:
            raise ValueError("batch size must be same")

        return shape1[0], shape1[1], shape2[1]

    def get_config(self):
        config = {
            'activation': self.activation,
            'model_size': self.model_size,
            'W1_regularizer': regularizers.serialize(self.W1_regularizer),
            'W2_regularizer': regularizers.serialize(self.W2_regularizer),
            'b1_regularizer': regularizers.serialize(self.b1_regularizer),
            'b2_regularizer': regularizers.serialize(self.b2_regularizer),
            'W1_constraint': constraints.serialize(self.W1_constraint),
            'W2_constraint': constraints.serialize(self.W2_constraint),
            'b1_constraint': constraints.serialize(self.b1_constraint),
            'b2_constraint': constraints.serialize(self.b2_constraint),
            'bias1': self.bias1,
            'bias2': self.bias2
        }
        base_config = super(RemappedCoAttentionWeight, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FeatureNormalization(Layer):
    """
        Normalize feature along a specific axis.
        Supports Masking.

        # Input shape
            A ND tensor with shape: `(samples, feature1 ... featuresN).
        # Output shape
            ND tensor with shape: `(samples, feature1 ... featuresN)`.
        :param kwargs:
        """

    def __init__(self, axis=-1, **kwargs):

        self.axis = axis
        self.supports_masking = True
        super(FeatureNormalization, self).__init__(**kwargs)

    def build(self, input_shape):

        super(FeatureNormalization, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # don't pass the mask to the next layers
        return None

    def call(self, inputs, mask=None):
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a = K.cast(mask, K.floatx()) * inputs
        else:
            a = inputs

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=self.axis, keepdims=True) + K.epsilon(), K.floatx())

        return a

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'axis': self.axis
        }
        base_config = super(FeatureNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FeatureSelection1D(Layer):
    """
        Normalize feature along a specific axis.
        Supports Masking.

        # Input shape
            A ND tensor with shape: `(samples, timesteps, features)
            A 2D tensor with shape: [samples, num_selected_features]
        # Output shape
            ND tensor with shape: `(samples, num_selected_features, features)`.
        :param kwargs:
        """

    def __init__(self, num_selects, **kwargs):

        self.num_selects = num_selects
        self.supports_masking = True
        super(FeatureSelection1D, self).__init__(**kwargs)

    def build(self, input_shape):

        super(FeatureSelection1D, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # don't pass the mask to the next layers
        return None

    def call(self, inputs, mask=None):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError('FeatureSelection1D layer should be called '
                             'on a list of 2 inputs.')

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a = K.cast(mask, K.floatx()) * inputs[0]
        else:
            a = inputs[0]

        b = inputs[1]

        a = tf.batch_gather(
            a, b
        )

        return a

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `FeatureSelection1D` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])

        if shape2[0] != shape1[0]:
            raise ValueError("batch size must be same")

        if shape2[1] != self.num_selects:
            raise ValueError("must conform to the num_select")

        return (shape1[0], self.num_selects, shape1[2])

    def get_config(self):
        config = {
            'num_selects': self.num_selects
        }
        base_config = super(FeatureSelection1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)