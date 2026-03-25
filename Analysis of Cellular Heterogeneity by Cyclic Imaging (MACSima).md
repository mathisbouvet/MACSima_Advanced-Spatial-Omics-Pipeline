<h1 style="color: #333; text-align: center; padding: 15px 0; border-top: 3px double #333; border-bottom: 3px double #333; font-family: Georgia, serif; font-style: italic; font-weight: normal; margin-top: 20px; margin-bottom: 30px;">
  Analysis of Cellular Heterogeneity by Cyclic Imaging (MACSima)
</h1>

<div style="text-align: center; font-family: Georgia, serif; font-size: 1.2em; color: #555; margin-top: -10px; margin-bottom: 30px;">
  By <strong>Mathis BOUVET</strong> — Biologist specializing in Reproduction and Development
  <br>
  <span style="font-size: 0.8em; font-style: italic;">March 2026</span>
</div>

> **Important note**
> : This document contains no real data

The objective of this pipeline is to analyze the different intensities between the markers.

<div style="border: 1px solid #d65323; border-radius: 10px; padding: 20px; background-color: rgba(213, 101, 45, 0.1); color: #000000;">
  <strong>Importing libraries</strong><br>

<details>
  <summary><b>Show/hide configuration code</b></summary>

```python
import importlib
import subprocess
import sys

required_packages = {
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn"
}

def install_and_import():
    print("Analysis of the work environment...")
    for module_name, package_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name} is already operational.")
        except ImportError:
            print(f"⚠️ {module_name} lack. Installation of {package_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                print(f"🚀 {package_name} installed successfully !")
            except Exception as e:
                print(f"❌ Error during installation of {package_name} : {e}")

install_and_import()

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance_matrix

sns.set_theme(style="whitegrid")
%matplotlib inline 

print("\n All data analysis libraries are ready !")
```
</details>
</div>

## 1 Importing data


### 1.a Loading and cleaning
We select the cells that are positive for each marker.

Here, if the value is greater than 0, the cells are considered to express the marker.

We then measure the average signal intensity per cell (only for those considered positive). This allows us to assess the marker quality, its expression level, and its homogeneity. We assume that low intensity may result from low expression or from noise. These data should therefore be compared with the visualization of results in MACSiQView.

Finally, we create separate files to analyze each population specifically, perform independent clustering, and conduct targeted spatial analysis.

```python
df = pd.read_csv("[Fill].csv")

# Filtering of positive cells
marker1_pos = df[df["marker1 Cell Exp"] > 0]
marker2_pos = df[df["marker2 Cell Exp"] > 0]
marker3_pos = df[df["marker3 Cell Exp"] > 0]

print("Medium intensity marker1 :", marker1_pos["marker1 Cell Intensity Average"].mean())
print("Medium intensity marker2 :", marker2_pos["marker2 Cell Intensity Average"].mean())

# The CSV files can be exported for further analysis.
marker1_pos.to_csv("marker1_positive_cells.csv", index=False)
marker2_pos.to_csv("marker2_positive_cells.csv", index=False)
marker3_pos.to_csv("marker3_positive_cells.csv", index=False)

print("Number of positive cells per marker :")
print({
    "marker1": len(marker1_pos),
    "marker2": len(marker2_pos),
    "marker3": len(marker3_pos),
})
```
## 2 Visualization of results
### 2.a The intensity of the markers

```python
data = {
    "marker1": marker1_pos["marker1 Cell Intensity Average"],
    "marker2": marker2_pos["marker2 Cell Intensity Average"],
    "marker3": marker3_pos["marker3 Cell Intensity Average"]
}

viz_df = pd.DataFrame(data)
viz_df_melt = viz_df.melt(var_name="marker", value_name="Intensity")

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x="marker", y="Intensity", data=viz_df_melt, palette="pastel")
sns.stripplot(x="marker", y="Intensity", data=viz_df_melt, color='black', alpha=0.3, jitter=0.2)
plt.title("Distribution of cellular intensities by marker")
plt.ylabel("Medium Intensity")
plt.xlabel("Biomarker")
plt.tight_layout()
plt.show()
```

### 2.b Correlation between biomarkers

```python
# The intensities are merged into a single table.
intensity_df = pd.DataFrame({
    "marker1": df["marker1 Cell Intensity Average"],
    "marker2": df["marker2 Cell Intensity Average"],
    "marker3": df["marker3 Cell Intensity Average"]
})

corr = intensity_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between biomarker intensities")
plt.show()
```