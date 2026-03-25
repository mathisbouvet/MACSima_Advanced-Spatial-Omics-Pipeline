<div align="center">

# 🔬 Spatial-Omics Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![Biotech](https://img.shields.io/badge/Field-Spatial_Proteomics-blue?style=flat-square)](#)
[![Framework](https://img.shields.io/badge/Framework-Methodological_Showcase-brightgreen?style=flat-square)](#)

**A methodological framework for Python-based analysis of immunofluorescence spatial data.**  
*This repository focuses on the validation and benchmarking of analytical pipelines.*

<br>

**Mathis Bouvet**  
*Biologist — Reproduction & Development*  
March 2026

</div>



## Pipeline Architecture

The workflow is structured into two main components: segmentation quality control and clustering-based phenotypic characterization.



### I. Segmentation Quality Control

> **Problem:** How can the reliability of automated segmentation in biological tissues be quantitatively assessed?

This module, detailed in [`Test of segmentation.md`](Test%20of%20segmentation.md), includes:

- **Standardization**: Conversion of complex ROI structures (Fiji) into programmatically usable binary masks.  
- **Fidelity Analysis**: Comparative evaluation using the **Kolmogorov–Smirnov test** to identify the most accurate segmentation method.  
- **Isolation Forest**: Detection of artefacts and aberrant segmentations through multidimensional anomaly detection.  
- **Recommendation Engineering**: Application of **Mann–Whitney U tests** to guide parameter optimization (e.g., smoothing, sensitivity).

---

### II. Unsupervised Phenotypic Characterization

> **Problem:** How can clustering validity be ensured in large and heterogeneous cellular populations?

The protocol described in [`Comparison of clusters.md`](Comparison%20of%20clusters.md) includes:

- **Clusterability Assessment**: Use of the **Hopkins statistic** to validate the presence of inherent structure prior to clustering.  
- **Dimensionality Reduction (PCA)**: Projection preserving 90% of protein variance.  
- **Automatic K Selection**: Consensus strategy combining **Silhouette**, **Davies–Bouldin**, and **Calinski–Harabasz** indices.  
- **Stability Analysis (ARI)**: Robustness evaluation via 80% bootstrapping using the **Adjusted Rand Index**.

---

### III. Marker Co-expression & Correlation Analysis (MACSima)

> **Problem:** How to statistically validate biological signatures and distinguish true co-expression from background noise in cyclic imaging?

The protocol described in [`Marker correlation analysis.md`](Marker%20correlation%20analysis.md) focuses on:

* **Multi-Parametric Correlation**: Comparative analysis using **Pearson**, **Spearman**, and **Kendall** coefficients to evaluate both linear and non-linear relationships between markers.
* **Robustness to Artifacts**: Use of rank-based correlations to minimize the impact of fluorescence saturation and non-specific background noise frequently encountered in MACSima data.
* **Statistical Significance Mapping**: Implementation of a dynamic threshold ($r > 0.30$) combined with **p-value** filtering (from $*$ to $***$) to identify robust phenotypic signatures.
* **Sensitivity Assessment**: Evaluation of how segmentation precision and tissue heterogeneity influence correlation coefficients, ensuring that co-expression reflects true biology rather than segmentation leakage.

---

### IV. Cellular Heterogeneity & Intensity Analysis

> **Problem:** How to quantify marker expression levels and assess cellular heterogeneity across distinct biomarker populations?

The protocol described in [`Analysis of Cellular Heterogeneity by Cyclic Imaging (MACSima).md`](Analysis%20of%20Cellular%20Heterogeneity%20by%20Cyclic%20Imaging%20(MACSima).md) focuses on:

- **Population Filtering**: Identification of positive cells by establishing baseline expression thresholds (values > 0) to isolate expressing populations.
- **Intensity Quantification**: Measurement of the average signal intensity per cell to evaluate marker quality, overall expression levels, and signal homogeneity.
- **Data Stratification**: Exportation of population-specific datasets to facilitate independent clustering and targeted spatial downstream analysis.
- **Expression Visualization**: Utilization of combined boxplots and stripplots to map the distribution of cellular average intensities per biomarker.
- **Biomarker Correlation**: Computation and heatmap visualization of intensity correlations across different markers to reveal co-expression networks.



## Methodological Stack

| Category | Tools & Libraries |
| :--- | :--- |
| **Data Science** | `Scikit-Learn`, `Pandas`, `NumPy` |
| **Statistics** | `SciPy` (non-parametric tests, distributions) |
| **Bio-Imaging** | `OpenCV`, `Scikit-Image`, `Read-ROI` |
| **Visualization** | `Matplotlib`, `Seaborn` |

---

<br>

> **Ethical Note**: This repository is a methodological showcase and does not contain real biological data.
