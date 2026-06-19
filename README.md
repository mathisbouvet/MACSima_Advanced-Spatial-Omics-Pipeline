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

The workflow is structured into two main components: segmentation quality control and clustering-based phenotypic characterization. Each component is documented here as a step-by-step protocol, and packaged as a standalone, installable Python library.

## Table of Contents
- [Pipeline Architecture](#pipeline-architecture)
  - [I. Segmentation Quality Control](#i-segmentation-quality-control)
  - [II. Unsupervised Phenotypic Characterization](#ii-unsupervised-phenotypic-characterization)
- [Methodological Stack](#methodological-stack)



### I. Segmentation Quality Control

[![Tests](https://github.com/mathisbouvet/macsima-qc/actions/workflows/tests.yml/badge.svg)](https://github.com/mathisbouvet/macsima-qc/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/macsima-qc.svg)](https://pypi.org/project/macsima-qc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mathisbouvet/macsima-qc/blob/main/LICENSE)

> **Problem:** How can the reliability of automated segmentation in biological tissues be quantitatively assessed?

This module, detailed in [`01_segmentation_qc.md`](protocols/01_segmentation_qc.md), includes:

- **Standardization**: Conversion of complex ROI structures (Fiji) into programmatically usable binary masks.  
- **Fidelity Analysis**: Comparative evaluation using the **Kolmogorov–Smirnov test** to identify the most accurate segmentation method.  
- **Isolation Forest**: Detection of artefacts and aberrant segmentations through multidimensional anomaly detection.  
- **Recommendation Engineering**: Application of **Mann–Whitney U tests** to guide parameter optimization (e.g., smoothing, sensitivity).

📦 Packaged as **[`macsima-qc`](https://github.com/mathisbouvet/macsima-qc)**, available on PyPI:

```bash
pip install macsima-qc
```

---

### II. Unsupervised Phenotypic Characterization

[![PyPI](https://img.shields.io/pypi/v/spatial-cluster-compare.svg)](https://pypi.org/project/spatial-cluster-compare/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mathisbouvet/spatial-cluster-compare/blob/main/LICENSE)

> **Problem:** How can clustering validity be ensured in large and heterogeneous cellular populations?

The protocol described in [`02_comparaison_cluster.md`](protocols/02_comparaison_cluster.md) includes:

- **Clusterability Assessment**: Use of the **Hopkins statistic** to validate the presence of inherent structure prior to clustering.  
- **Dimensionality Reduction (PCA)**: Projection preserving 90% of protein variance.  
- **Automatic K Selection**: Consensus strategy combining **Silhouette**, **Davies–Bouldin**, and **Calinski–Harabasz** indices.  
- **Stability Analysis (ARI)**: Robustness evaluation via 80% bootstrapping using the **Adjusted Rand Index**.

📦 Packaged as **[`spatial-cluster-compare`](https://github.com/mathisbouvet/spatial-cluster-compare)**, available on PyPI:

```bash
pip install spatial-cluster-compare
```

---

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