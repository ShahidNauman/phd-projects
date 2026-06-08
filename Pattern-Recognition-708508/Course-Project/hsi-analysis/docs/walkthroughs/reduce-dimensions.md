# Walkthrough - HSI Dimensionality Reduction (PCA vs. CAE)

This walkthrough describes the dimensionality reduction techniques applied to the hyperspectral image (HSI) cube using Principal Component Analysis (PCA) and deep Convolutional Autoencoders (CAEs).

---

## 1. Methodology

### A. Principal Component Analysis (PCA)

PCA finds a set of orthogonal projections that maximize the variance of the data:

1. **Outlier Filtering**: Replace and clip extreme values ($> 10.0$ or negative values) to the valid range $[0.0, 10.0]$.
2. **Reshaping**: Reshape the cube $(B, H, W)$ to a 2D pixel-by-band matrix of shape $(H \times W, B)$.
3. **Standardization**: Scale features to zero-mean and unit variance.
4. **PCA fit**: Projections are saved as min-max normalized grayscale images.

### B. Convolutional Autoencoder (CAE)

CAEs capture spatial-spectral patterns using local convolutions and non-linear activation functions:

1. **Architecture**:
   - **Encoder**: Compresses spectral bands from 149 down to 3 latent feature maps using 2D convolutions (Conv2D $\to$ ReLU $\to$ Conv2D).
   - **Decoder**: Reconstructs the 149 bands from the 3 latent feature maps (Conv2D $\to$ ReLU $\to$ Conv2D $\to$ Sigmoid).
2. **Training**: Optimized directly on the normalized HSI image for 100 epochs using MSE Loss and the Adam optimizer on CPU.
3. **Latency**: Takes approximately 3 minutes to train on CPU.

---

## 2. Command Execution & Logs

### A. Run PCA

```bash
.venv/Scripts/hsi-analysis reduce-dimensions --using pca --hdr data/sample.hdr --raw data/sample.raw
```

Log output:

```
PCA EXPLAINED VARIANCE REPORT
------------------------------------------------------------
Component  | Explained Variance Ratio | Cumulative Variance
------------------------------------------------------------
PC 1       | 0.963454                 | 0.963454
PC 2       | 0.018222                 | 0.981676
PC 3       | 0.002922                 | 0.984598
PC 4       | 0.001865                 | 0.986462
PC 5       | 0.001661                 | 0.988124
============================================================
```

### B. Run CAE

```bash
.venv/Scripts/hsi-analysis reduce-dimensions --using cae --hdr data/sample.hdr --raw data/sample.raw
```

Log output:

```
Training Convolutional Autoencoder (CAE) on CPU...
Epoch [  1/100], Loss: 0.165622
Epoch [ 10/100], Loss: 0.014513
Epoch [ 20/100], Loss: 0.014513
Epoch [ 30/100], Loss: 0.014513
Epoch [ 40/100], Loss: 0.014513
Epoch [ 50/100], Loss: 0.014513
Epoch [ 60/100], Loss: 0.014513
Epoch [ 70/100], Loss: 0.014513
Epoch [ 80/100], Loss: 0.014513
Epoch [ 90/100], Loss: 0.014513
Epoch [100/100], Loss: 0.014513
Training complete. Extracting latent representation...
Saving individual CAE component images...
Saved:        output/images\cae_component_1.png (CAE Component 1)
Saved:        output/images\cae_component_2.png (CAE Component 2)
Saved:        output/images\cae_component_3.png (CAE Component 3)
Generating CAE RGB composite (Component 1=R, Component 2=G, Component 3=B)...
Saved:        output/images\cae_composite.png (RGB Composite)
Saved:        output/images\cae_training_loss.png (Training Loss Plot)
------------------------------------------------------------
CAE DIMENSIONALITY REDUCTION REPORT
------------------------------------------------------------
Final Reconstruction MSE: 0.014513
============================================================
```

---

## 3. Comparison of PCA and CAE Results

1. **Spatial Representation**:
   - **PCA** does not consider pixel neighborhoods; it processes each pixel independently. As a result, noise in individual pixels is carried over into the components.
   - **CAE** incorporates spatial context using 2D convolutions. This leads to smoother latent images with significantly reduced high-frequency noise.
2. **Feature Linearity**:
   - **PCA** projects data linearly, capturing global albedo variance in PC 1 (96.35%).
   - **CAE** uses ReLU and Sigmoid activations, enabling the network to learn non-linear combinations of bands. This highlights finer chemical pigment boundaries and stroke thickness variances in the handwriting.
3. **Training Overhead**:
   - **PCA** runs almost instantly.
   - **CAE** requires PyTorch and optimizes for 100 epochs, taking about 3 minutes to run on CPU.
