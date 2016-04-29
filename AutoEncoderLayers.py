from keras.engine import Layer, InputSpec
from keras import activations, constraints, initializations, regularizers
from keras import backend as K
from theano.tensor import as_tensor_variable
from theano.tensor.nnet.neighbours import images2neibs, neibs2images
import theano.tensor as T
import theano

dtensor5 = T.TensorType('float32', (False,) * 5)

class DeConvAfterDownSampling(Layer):
    """
    upsampling layer
    """
    def __init__(self, nb_filter, nb_row, nb_col,
                 init='glorot_uniform', activation='linear', weights=None, dim_ordering='th',
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, **kwargs):
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.activation = activations.get(activation)
        self.dim_ordering = dim_ordering

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights
        super(DeConvAfterDownSampling, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            stack_size = input_shape[1]
            self.W_shape = (self.nb_filter, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            stack_size = input_shape[3]
            self.W_shape = (self.nb_row, self.nb_col, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
        self.b = K.zeros((self.nb_filter,), name='{}_b'.format(self.name))
        self.trainable_weights = [self.W, self.b]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        # rows = rows
        # cols = cols
        if self.dim_ordering == 'th':
            return (1, self.nb_filter, None, self.nb_row*self.nb_col)
        elif self.dim_ordering == 'tf':
            return (1, self.nb_filter, None, self.nb_row*self.nb_col)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        x = K.sum(x, axis=-1, keepdims=True)
        x = K.resize_images(x, self.nb_row, self.nb_col, self.dim_ordering)
        conv_out = self.deconv2d(x=x, kernel=self.W, dim_ordering=self.dim_ordering, filter_shape=self.W_shape)
        if self.dim_ordering == 'th':
            output = conv_out + K.reshape(self.b, (self.nb_filter, 1, 1))
        elif self.dim_ordering == 'tf':
            output = conv_out + K.reshape(self.b, (self.nb_filter, 1, 1))
            # output = conv_out + self.b
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        output = self.activation(output)
        # return output
        results_shape = output.shape
        return output.reshape((1, results_shape[0], results_shape[1], results_shape[2]))

    def deconv2d(self, x, kernel, dim_ordering='th',
           image_shape=None, filter_shape=None):
        if dim_ordering not in {'th', 'tf'}:
            raise Exception('Unknown dim_ordering ' + str(dim_ordering))
        if dim_ordering == 'tf':
            x = x.dimshuffle((0, 3, 1, 2))
            kernel = kernel.dimshuffle((2, 0, 1))
            if image_shape:
                image_shape = (image_shape[0], image_shape[3],
                               image_shape[1], image_shape[2])
            if filter_shape:
                filter_shape = (filter_shape[2],
                                filter_shape[0], filter_shape[1])

        def int_or_none(value):
            try:
                return int(value)
            except TypeError:
                return None

        if image_shape is not None:
            image_shape = tuple(int_or_none(v) for v in image_shape)

        if filter_shape is not None:
            filter_shape = tuple(int_or_none(v) for v in filter_shape)

        inputs = as_tensor_variable(x)
        filters = as_tensor_variable(kernel)

        # conv_out = get_conv_out(inputs, filters)
        # filters = K.resize_images(filters, 86, 86, dim_ordering="tf")

        conv_out = get_deconv_out_new(inputs, filters)

        # if dim_ordering == 'tf':
        #     conv_out = conv_out.dimshuffle((0, 2, 3, 1))
        return conv_out

def get_single_deconv_out(input, filter):

    # reshape inputs
    img = input.reshape((1, 1, input.shape[0], input.shape[1]))
    kernel = filter.reshape((1, K.prod(filter.shape)))

    # construct split function
    # image = T.tensor4("image")
    neibs = images2neibs(img, neib_shape=filter.shape, neib_step=filter.shape)
    # window_function = theano.function([image], neibs)
    #
    # neibs_val = window_function(img_val)

    neibs = neibs*kernel

    # construct merge function
    img_new = neibs2images(neibs, filter.shape, img.shape)

    return img_new[0][0]

def get_single_deconv_out_new(input, filter):
    output_shape = (1, 1, input.shape[0]*filter.shape[0], input.shape[1]*filter.shape[1])
    # reshape inputs
    image = input.reshape((1, K.prod(input.shape)))
    kernel = filter.reshape((1, K.prod(filter.shape)))

    # neibs = images2neibs(output, neib_shape=filter.shape, neib_step=filter.shape)

    def fn(i, k):
        return i*k
    results, updates = theano.scan(fn=fn, sequences=image[0], non_sequences=kernel[0])

    # neibs = neibs*results
    img_new = neibs2images(results, filter.shape, output_shape)

    return img_new[0][0]

def inner_loop(n, fil, r, b, ipt, deconv_out):
    patch = get_single_deconv_out_new(ipt, fil)
    r = r + T.set_subtensor(deconv_out[b, n, :, :], patch)
    return r

def outer_loop(b, ipt, r, filters, deconv_out):
    results, _ = theano.scan(fn=inner_loop, sequences=[T.arange(filters.shape[0]), filters],
                             outputs_info=r, non_sequences=[b, ipt, deconv_out])
    final_result = results[-1]
    return final_result

def get_deconv_out(inputs, filters):
    output_shape = (inputs.shape[0], filters.shape[0], inputs.shape[1]*filters.shape[1], inputs.shape[2]*filters.shape[2])

    deconv_out = T.zeros(output_shape)

    results, _ = theano.scan(fn=outer_loop, sequences=[T.arange(inputs.shape[0]), inputs],
                                   outputs_info=T.zeros_like(deconv_out),
                             non_sequences=[filters, deconv_out])

    return results[-1]

def get_deconv_out_new(inputs, filters):
    inputs_shape = inputs.shape
    filters_shape = filters.shape
    # img = K.reshape(inputs, (inputs_shape[0], 1, inputs_shape[1], inputs_shape[2]))
    kernel = K.reshape(filters, (filters_shape[0], 1, filters_shape[1]*filters_shape[2]))
    # def fn(i, k):
    #     i = K.reshape(i, (1, 1, i.shape[0], i.shape[1]))
    #     neibs = images2neibs(i, neib_shape=(filters_shape[-2], filters_shape[-1]),
    #                      neib_step=(filters_shape[-2], filters_shape[-1]))
    #     neibs = neibs*k
    #     return neibs
    # results,_ = theano.scan(fn=fn, sequences=inputs, non_sequences=kernel)

    neibs = images2neibs(inputs, neib_shape=(filters_shape[1], filters_shape[2]),
                         neib_step=(filters_shape[1], filters_shape[2]))

    def fn(k, n):
        return n*k
    results,_ = theano.scan(fn=fn, sequences=kernel, non_sequences=neibs)
    # results = neibs * kernel

    # results_shape = results.shape
    return results
