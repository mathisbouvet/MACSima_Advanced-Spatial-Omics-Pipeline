<div align="center">

# 🔬 Spatial-Omics Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![Biotech](https://img.shields.io/badge/Field-Spatial_Proteomics-blue?style=flat-square)](#)
[![Framework](https://img.shields.io/badge/Framework-Methodological_Showcase-brightgreen?style=flat-square)](#)
[![Instrument](https://img.shields.io/badge/Instrument-MACiQView-orange?style=flat-square)](#)
[![Last Commit](https://img.shields.io/github/last-commit/mathisbouvet/MACSima_Spatial-Omics-Pipeline?style=flat-square)](https://github.com/mathisbouvet/MACSima_Spatial-Omics-Pipeline/commits/main/)
[![Citation](https://img.shields.io/badge/Cite-this_repository-9cf?style=flat-square)](#citation)

**A methodological framework for Python-based analysis of immunofluorescence spatial data.**  
*This repository focuses on the validation and benchmarking of analytical pipelines.*

<br>

**Mathis Bouvet**  
*Biologist — Reproduction & Development*  
March 2026

</div>



## Pipeline Architecture

The workflow is structured into two main components: segmentation quality control and clustering-based phenotypic characterization. Each component is documented here as a step-by-step protocol, and packaged as a standalone, installable Python library.

> **Data Origin**: Raw multiplex imaging data is acquired on the MACSima platform via **MACiQView**, then exported and processed through the Python pipelines described below.

## Table of Contents
- [Pipeline Architecture](#pipeline-architecture)
  - [I. Segmentation Quality Control](#i-segmentation-quality-control)
  - [II. Unsupervised Phenotypic Characterization](#ii-unsupervised-phenotypic-characterization)
- [Notebooks](#notebooks)
- [Methodological Stack](#methodological-stack)
- [Citation](#citation)



### I. Segmentation Quality Control

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

## Notebooks

The exploratory analyses behind both modules are available as Jupyter notebooks in [`/notebooks`](notebooks):

- Development notebook for the segmentation QC protocol, prior to packaging as `macsima-qc`.
- Development notebook for the clustering validation protocol, prior to packaging as `spatial-cluster-compare`.

---

## Methodological Stack

| Category | Tools & Libraries |
| :--- | :--- |
| **Acquisition** | MACSima — `MACiQView` |
| **Data Science** | `Scikit-Learn`, `Pandas`, `NumPy` |
| **Statistics** | `SciPy` (non-parametric tests, distributions) |
| **Bio-Imaging** | `OpenCV`, `Scikit-Image`, `Read-ROI` |
| **Visualization** | `Matplotlib`, `Seaborn` |

---

## Citation

If you use `macsima-qc` or `spatial-cluster-compare` in your research, please consider citing this repository:

```bibtex
@software{bouvet_spatial_omics_pipeline,
  author  = {Bouvet, Mathis},
  title   = {Spatial-Omics Pipeline: A methodological framework for Python-based analysis of immunofluorescence spatial data},
  year    = {2026},
  url     = {https://github.com/mathisbouvet/MACSima_Spatial-Omics-Pipeline}
}
```

---

<br>

> **Ethical Note**: This repository is a methodological showcase and does not contain real biological data.
