from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers import Layer

class AttentionLayer(Layer):
    """Attention layer implementation based on the code from Baziotis, Christos on
    Gist (https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2) and on
    the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification".
    
    The difference between these implementations are:
    - Compatibility with Keras 2.2;
    - Code annotations;
    - Easy way to retrieve attention weights;
    - Tested with TensorFlow backend only!
    """
    
    @staticmethod
    def dot_product(x, kernel):
        """
        Wrapper for dot product between a matrix and a vector x*u.
        The shapes are arranged as (batch_size, timesteps, features) \
        * (features,) = (batch_size, timesteps)
        Args:
            x: Matrix with shape (batch_size, timesteps, features)
            u: Vector with shape (features,)
        Returns:
            W = x*u with shape (batch_size, timesteps)
        """
        if K.backend() == 'tensorflow':
            return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
        else:
            return K.dot(x, kernel)
    
    def __init__(self, return_coefficients=False,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """If return_coefficients is True, this layer produces two outputs,
        the first is the weighted hidden tensor for the input sequence with
        shape (batch_size, n_features) and the second the attention weights
        with shape (batch_size, seq_len, 1). This layer supports masking."""
    
        self.supports_masking = True
        self.return_coefficients = return_coefficients
        self.init = initializers.get('glorot_uniform')
        
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        assert len(input_shape) == 3
        
        # the attention vector/matrix is equals to the RNN hidden dimension
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

        super(AttentionLayer, self).build(input_shape)
        
    @staticmethod
    def masked_softmax(alpha, mask):
        """Masks alpha and then performs softmax, as Keras's softmax doesn't support masking."""
        alpha = K.exp(alpha)
        if mask is not None:
            alpha *= K.cast(mask, K.floatx())

        partition = K.cast(K.sum(alpha, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        return alpha / partition
    
    def call(self, x, mask=None):        
        # V = tanh(Wx+b), but vectorized as V = tanh(X*W + b)
        v_att = K.dot(x, self.W)                   # > (batch, seq_len, attention_size = amount_features)
        if self.bias:
            v_att += self.b
        v_att = K.tanh(v_att)                      # > (batch, seq_len, attention_size)
        
        # alpha = softmax(V \dot U)). Each alpha is an attention scalar.
        alpha = self.dot_product(v_att, self.u)    # > (batch, seq_len)
        alpha = self.masked_softmax(alpha, mask)
        alpha = K.expand_dims(alpha)               # > (batch, seq_len, 1)
        
        # (batch, seq_len, features) * (batch, seq_len, 1) -> (batch, seq_len, features)
        attended_x = x * alpha
        
        # c_i = sum_i(x_i * alpha_i)
        c = K.sum(attended_x, axis=1)              # > (batch, amount_features)
        
        if self.return_coefficients:
            return [c, alpha]
        else:
            return c
    
    def compute_output_shape(self, input_shape):
        """The attention mechanism computes a weighted average between all
        hidden vectors generated by the previous sequential layer, hence the
        input is expected to be (batch_size, seq_len, amount_features) and
        after averaging each feature vector, the output it (batch_size, seq_len)."""
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return (input_shape[0], input_shape[-1])
    
    def compute_mask(self, x, input_mask=None):
        """This layer produces a single attended vector from a list of hidden vectors,
        hence it can't be masked as this means masking a single vector.
        """
        return None