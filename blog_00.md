## Decoding Attention is What I Want to Learn

Welcome to **Decoding Attention is What I Want to Learn** - a blog series that demystifies modern attention mechanisms in neural networks. Through synthetic datasets inspired by geometric shapes, we'll explore how simple ideas evolve into complex models like Transformers. Our examples include both 1D time series data (with non-overlapping shapes and amplitude variations) and 2D images (with colored shapes and group-average color filtering).

**In this series, you'll encounter:**

- **Blog Post 1 – Foundations and Standard Attention:**
  - **Dataset Overview:** We introduce a synthetic time series dataset, where each sample is a vector (e.g., length 500) containing non-overlapping shapes (triangles, rectangles, semicircles) drawn with unique amplitudes. The target is formed by redrawing these shapes using the group-average amplitude.
  - **Baseline Models:** Explore Fully Connected Networks (FCNs), Convolutional Neural Networks (CNNs), and Recurrent Neural Networks (RNNs) as initial approaches.
  - **Attention Mechanisms:** Understand the basics of dot-product attention, multi-head attention, and positional encodings (linear, learnable, sinusoidal and rotary (RoPE)). Visual comparisons (e.g., full versus triangular masked attention matrices) help solidify the concepts.

- **Blog Post 2 – Efficient, Sparse, and Adaptive Attention:**
  - **Advanced Attention:** Delve into the computational challenges of full attention and learn how global, local, sparse, and adaptive attention variants can help.
  - **Linearized Approaches:** Get introduced to linearized attention methods such as Linformer, Performer, and Reformer, accompanied by diagrams and complexity comparisons.

- **Blog Post 3 (Proposed) – Transformer Architectures and Cross-Attention for Vision:**
  - **New Dataset for Vision:** We'll introduce a synthetic 2D shapes dataset where images contain multiple colored shapes with unique group-average color rules (with special filtering for filled circles).
  - **Tokenization & Positional Encoding:** Learn how vision Transformers tokenize an image into patches (e.g., 16×16) and encode positional information using sinusoidal, RoPE, or learnable methods. Diagrams will illustrate the patching process.
  - **Encoder-Decoder Transformers & Cross-Attention:** Explore how an encoder-decoder Transformer processes the tokenized patches - with a particular focus on cross-attention, where the decoder's queries attend to the encoder's outputs. Comparative illustrations (e.g., full attention vs. channel-specific attention matrices) will be included.
  - **Discussion:** While briefly comparing encoder-only and decoder-only variants, the main focus will remain on explaining and visualizing the encoder-decoder architecture.

Stay tuned as we decode attention - not just because "attention is all you need," but because understanding it is essential for unlocking the full potential of modern neural networks!
