from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers import Layer
from keras import initializers
from keras import activations



class AttentionLayer(Layer):

    def __init__(self,
                 activation='tanh',
                 initializer='glorot_uniform',
                 return_attention=False,
                 W_regularizer=None,
                 u_regularizer=None,
                 b_regularizer=None,
                 W_constraint=None,
                 u_constraint=None,
                 b_constraint=None,
                 bias=True,
                 **kwargs):
        
        self.activation = activations.get(activation)
        self.initializer = initializers.get(initializer)
        
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
        
        self.bias = bias
        self.supports_masking = True
        self.return_attention = return_attention

        super().__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch, time, amount_features)
        
        # the attention size matches the feature dimension
        amount_features = input_shape[-1]
        attention_size  = input_shape[-1]

        self.W = self.add_weight((amount_features, attention_size),
                                 initializer=self.initializer,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint,
                                 name='attention_W')
        self.b = None
        if self.bias:
            self.b = self.add_weight((attention_size,),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint,
                                     name='attention_b')

        self.context = self.add_weight((attention_size,),
                                       initializer=self.initializer,
                                       regularizer=self.u_regularizer,
                                       constraint=self.u_constraint,
                                       name='attention_us')

        super().build(input_shape)

    def call(self, x, mask=None):        
        # U = tanh(H*W + b) (eq. 8)        
        ui = K.dot(x, self.W) # (b, t, a)
        if self.b is not None:
            ui += self.b
        ui = self.activation(ui)            # (b, t, a)

        # Z = U * us (eq. 9)
        us = K.expand_dims(self.context)   # (a, 1)
        ui_us = K.dot(ui, us)              # (b, t, a) * (a, 1) = (b, t, 1)
        ui_us = K.squeeze(ui_us, axis=-1)  # (b, t, 1) -> (b, t)
        
        # alpha = softmax(Z) (eq. 9)
        alpha = self._masked_softmax(ui_us, mask) # (b, t)
        alpha = K.expand_dims(alpha, axis=-1)     # (b, t, 1)
        
        if self.return_attention:
            return alpha
        else:
            # v = alpha_i * x_i (eq. 10)
            return K.sum(x * alpha, axis=1)
    
    def _masked_softmax(self, logits, mask):
        # softmax(x):
        #    b = max(x)
        #    s_i = exp(xi - b) / exp(xj - b)
        #    return s
        
        b = K.max(logits, axis=-1, keepdims=True)
        logits = logits - b

        exped = K.exp(logits)

        # ignoring masked inputs
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            exped *= mask

        partition = K.sum(exped, axis=-1, keepdims=True)
        return exped / partition

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return input_shape[:-1]
        else:
            return input_shape[:-2] + input_shape[-1:]

    def compute_mask(self, x, input_mask=None):
        return None