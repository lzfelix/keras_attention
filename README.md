# Keras Attention Layer
Dead-simple Attention layer implementation in Keras based on the work of Yang et al. ["Hierarchical Attention Networks
for Document Classification"](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)

**Notice:** the initial version of this repository was based on the implementation by [Christos Baziotis](https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2). However, recently this repository was rewritten from scratch with the following features:

- Compatibility with Keras 2.2 (tested with TensorFlow 1.8.0);
- Annotations showing dimension transformations and equations;
- Numerically stable softmax using the exp-normalize trick; **(New!)**
- Easy way to recover the attention weights applied to each sample to make nice visualizations (see
[neat-vision](https://github.com/cbaziotis/neat-vision)); **(Updated!)**
- Example showing differences between vanilla, attention model and attention with masking;
- Example on the sum toy task showing how attention weights can be distributed across timesteps in a sample; **(New!)**
- Example on sentiment analysis of movie reviews (but GitHub does not render notebook markup, you may want to download the notebook to see word highlights, as in the example below); **(New!)**
- Allows customizing the attention activation function, since removing it might be beneficial for some tasks, as shown in ["A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task"](https://arxiv.org/abs/1606.02858) by Chen et al. **(New!)**





> ![Attention example on a movie review](https://github.com/lzfelix/keras_attention/blob/master/movie_attention.png)

Example of attention on words for sentiment classification in a movie review in the Keras IMDb dataset. Darker colors mean larger weights and, consequently, more importance is given to those term.

> ![Attention example](https://github.com/lzfelix/keras_attention/blob/master/attention_example.png)

Example of attention weights across timesteps during the classification of a sequential sample.
