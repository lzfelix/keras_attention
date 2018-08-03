# Keras Attention Layer
Attention layer implementation in Keras based on the work of Yang et al. ["Hierarchical Attention Networks
for Document Classification"](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf) and on the
implementation by [Christos Baziotis](https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2). The differences
on this implementation include

- Compatibility with Keras 2.2 (tested with TensorFlow 1.8.0);
- Annotations on the code showing dimension transformations and equations;
- Easy way to recover the attention weights applied to each sample to make nice visualizations (see
[neat-vision](https://github.com/cbaziotis/neat-vision));
- Bundled with examples showing differences between vanilla and attention model and attention with masking.
