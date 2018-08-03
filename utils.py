import numpy as np

from keras import layers
from keras import models
from attention.layers import AttentionLayer
from matplotlib import pyplot as plt


def build_copy_model(model, has_masking=False):
    """Copies the model up to the attention layer to get its coefficients."""
    
    sequence_in = layers.Input(shape=(None, 2), name='input')
    
    if has_masking:
        masked_in = layers.Masking(name='Mask')(sequence_in)
        offset = 1
    else:
        masked_in = sequence_in
        offset = 0
    
    lstm2 = layers.LSTM(5,
                        return_sequences=True,
                        name='LSTM',
                        weights=model.layers[offset + 1].get_weights()
                       )(masked_in)

    _, att = AttentionLayer(return_coefficients=True,
                            weights=model.layers[offset + 2].get_weights()
                           )(lstm2)
    model = models.Model(inputs=[sequence_in], outputs=[att])    

    model.summary()
    return model


def plot_attention(model, index, d):
    coefficients = model.predict(d[index:index+1])[0]

    plt.figure(figsize=(20, 3))
    plt.grid(True, alpha=0.4)
    plt.plot(range(coefficients.shape[0]), coefficients, '-o', alpha=0.5)

    real_position = d[index].argmax(0)[1]
    r = np.linspace(coefficients.min() - 0.001, coefficients.max() + 0.001)

    plt.title('Sample %d - correct @ %d' % (index, real_position))
    plt.plot([real_position] * len(r), r, c='red', alpha=0.6)
    
    plt.xlabel('Timestep')
    plt.ylabel('Attention')
    
    return coefficients