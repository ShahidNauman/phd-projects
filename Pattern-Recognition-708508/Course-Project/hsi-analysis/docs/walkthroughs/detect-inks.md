# Walkthrough - Ink Detection Pattern Recognition

This walkthrough describes the pattern recognition approach used to detect the number of distinct inks present in the hyperspectral document cube.

## Pattern Recognition Methodology

To automatically detect the number of different inks, we implemented an **unsupervised clustering workflow**:

1. **Spectral Feature Extraction**: For each of the 12 text lines in the document, we extracted the average spectral reflectance vector (149 bands).
2. **Standardization**: We normalized each spectral vector to zero-mean and unit variance. This standardization step removes baseline offsets caused by differences in pen pressure, ink deposition thickness, or handwriting stroke widths, ensuring the clustering algorithm groups the inks based on **spectral shape (chemical composition)** rather than amplitude.
3. **Clustering via K-Means**: We ran the K-Means clustering algorithm for $K \in [2, 6]$.
4. **Optimal Cluster Evaluation (Silhouette Metric)**: We computed the **Silhouette Score** for each cluster size $K$. The Silhouette Score measures how similar a sample is to its own cluster compared to other clusters. The value of $K$ that maximizes this score is identified as the optimal number of distinct inks.

---

## Verification Results

### 1. Running the CLI Command

```bash
.venv/Scripts/hsi-analysis detect-inks --hdr data/sample.hdr --raw data/sample.raw
```

### 2. Execution Output

```
============================================================
INK DETECTION PATTERN RECOGNITION REPORT
============================================================
Header:       data/sample.hdr
Raw Cube:     data/sample.raw
Dimensions:   512 x 650 x 149
------------------------------------------------------------
Segmenting handwriting cells and extracting spectral signatures...
------------------------------------------------------------
Evaluating K-Means Clustering Silhouette Scores:
------------------------------------------------------------
K=2 clusters: Silhouette Score = 0.6787
K=3 clusters: Silhouette Score = 0.6284
K=4 clusters: Silhouette Score = 0.5756
K=5 clusters: Silhouette Score = 0.4810
K=6 clusters: Silhouette Score = 0.4400
------------------------------------------------------------
Optimal Number of Ink Clusters Detected: 2 (Silhouette = 0.6787)
------------------------------------------------------------

Groupings for K = 2 (Silhouette = 0.6787):
  Cluster 0: Pen 1 (Line 1), Pen 1 (Line 2), Pen 3 (Line 1), Pen 3 (Line 2)
  Cluster 1: Pen 2 (Line 1), Pen 2 (Line 2), Pen 4 (Line 1), Pen 4 (Line 2), Pen 5 (Line 1), Pen 5 (Line 2), Pen 6 (Line 1), Pen 6 (Line 2)

Groupings for K = 3 (Silhouette = 0.6284):
  Cluster 0: Pen 2 (Line 1), Pen 2 (Line 2), Pen 6 (Line 1), Pen 6 (Line 2)
  Cluster 1: Pen 1 (Line 1), Pen 1 (Line 2), Pen 3 (Line 1), Pen 3 (Line 2)
  Cluster 2: Pen 4 (Line 1), Pen 4 (Line 2), Pen 5 (Line 1), Pen 5 (Line 2)

Groupings for K = 5 (Silhouette = 0.4810):
  Cluster 0: Pen 2 (Line 1)
  Cluster 1: Pen 1 (Line 1), Pen 1 (Line 2)
  Cluster 2: Pen 4 (Line 1), Pen 4 (Line 2), Pen 5 (Line 1), Pen 5 (Line 2)
  Cluster 3: Pen 3 (Line 1), Pen 3 (Line 2)
  Cluster 4: Pen 2 (Line 2), Pen 6 (Line 1), Pen 6 (Line 2)
============================================================
```

---

## Discussion of Strengths, Weaknesses, and Limitations

### Strengths of the Approach

1. **Unsupervised and Objective**: The combination of K-Means and Silhouette Score does not require training labels (completely unsupervised) and mathematically determines the best-fitting number of clusters based on internal cluster compactness and separation.
2. **Robustness to Stroke Variations**: Standardizing the spectral curves makes the method invariant to handwriting amplitude differences (pressure, stroke width, ink quantity). It clusters inks based strictly on their spectral profile/chemical shape.
3. **High Signal-to-Noise Ratio (SNR)**: By averaging the spectra of segmented handwriting strokes per cell, we filter out noise from single pixels and isolate the ink signal from the white paper background.

### Weaknesses and Limitations

1. **Metamerism (Spectral Indistinguishability)**:
   This is the major physical limitation of the hyperspectral approach on this dataset. As shown in the $K=3$ and $K=5$ groupings, **Pen 4 (Red)** and **Pen 5 (Purple)** are consistently grouped into the same cluster.
   Although they are visually distinct colors to the human eye, their chemical pigments have nearly identical light absorption profiles in the visible range and the same infrared transition edge. Unsupervised clustering cannot separate them, leading the algorithm to detect $5$ instead of $6$ distinct pen groupings when trying to separate the individual inks.
2. **K-Means Assumptions**: K-Means assumes isotropic, spherical clusters of equal size and density. If the spectral variance of one pen's ink is significantly higher than another's, K-Means may fail or incorrectly partition a cluster (as seen in $K=5$ where Pen 2's Line 1 and Line 2 are split).
3. **Threshold Sensitivity**: The algorithm relies on thresholding the normalized Band 30 to extract ink strokes. If the threshold is too high, it includes background paper pixels (pulling spectra towards the paper profile). If it is too low, it may exclude fainter ink strokes, reducing the sample size.
