<h1 style="color: #333; text-align: center; padding: 15px 0; border-top: 3px double #333; border-bottom: 3px double #333; font-family: Georgia, serif; font-style: italic; font-weight: normal; margin-top: 20px; margin-bottom: 30px;">
Marker correlation analysis (MACSima)
</h1>

<div style="text-align: center; font-family: Georgia, serif; font-size: 1.2em; color: #555; margin-top: -10px; margin-bottom: 30px;">
  By <strong>Mathis BOUVET</strong> — Biologist specializing in Reproduction and Development
  <br>
  <span style="font-size: 0.8em; font-style: italic;">March 2026</span>
</div>
<br>

> **Important note**
> : This document contains no real data




<div style="border: 1px solid #569cd6; border-radius: 10px; padding: 20px; background-color: rgba(86, 156, 214, 0.1); color: #000000;">
  <strong>The correlation</strong><br>
Pearson correlation serves as a fundamental statistical indicator for quantifying the linear relationship between the expression intensities of two markers within a single cell segmentation unit. The choice of the correlation threshold fixed here at 0.30 is critical for discriminating relevant biological signals from instrumental background noise or non-specific staining. A high threshold, typically above 0.50, is preferred to confirm strict co-expression within the same cellular compartment or to validate a specific phenotype identity. Conversely, a more moderate threshold between 0.20 and 0.50 captures subtler interactions, such as functional proteins sharing a common microenvironment or markers of immediate proximity. However, it is crucial to consider that these coefficients can be influenced by the precision of the initial segmentation: overly permissive cell boundaries can artificially correlate signals from adjacent cells, while marked tissue heterogeneity may dilute a strong correlation that is localized within a specific tissue structure
</div>
<br>

<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
  Objective
</h2>
The goal is to statistically identify which markers share a similar expression profile at the cellular level, which allows us to define phenotypic signatures or cellular neighborhoods.
<br>


<br>

<div style="border: 1px solid #d65323; border-radius: 10px; padding: 20px; background-color: rgba(213, 101, 45, 0.1); color: #000000;">
  <strong>Importing libraries</strong><br>

<details>
  <summary><b>Show/hide configuration code</b></summary>

```python
import importlib
import subprocess
import sys

# Library Dictionary
required_packages = {
    "matplotlib": "matplotlib",
    "pandas": "pandas",
}
def install_and_import():
    print("Environmental analysis nlp_env...")
    for module_name, package_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name} is already ready.")
        except ImportError:
            print(f"⚠️ {module_name} lack. Installation of {package_name} in progress...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f" {package_name} installed successfully !")

install_and_import()

# Importing libraries after installation
import pandas as pd 
import matplotlib.pyplot as plt

print("All modules are imported and ready to use!")
```
</details>
</div>

## 1. Loading data

```python
df = pd.read_csv(r"[Fichier_analyse].csv")
pd.set_option("display.max_columns", None)
print(df.head())
```
## 2. Calculation of correlation

While Pearson correlation is the gold standard for linear relationships, other statistical approaches such as Spearman or Kendall correlations offer increased robustness against the heterogeneity of imaging data. Spearman's rank correlation, by focusing on ranks rather than absolute intensity values, captures non-linear monotonic relationships while minimizing the impact of outliers or fluorescence saturation artifacts. Similarly, Kendall's Tau provides a rigorous concordance analysis, particularly well suited for highly skewed marker distributions. The choice between these methods depends on the statistical distribution of the signals: a strictly proportional relationship is best described by Pearson, whereas a global biological trend potentially affected by non-uniform background noise is more accurately represented by a non-parametric approach like Spearman.

Here we use a comparison between 3 correlation coefficients used in biology `Pearson`, `Spearman`, `Kendal`.

What is particularly interesting when comparing a large number of markers for instance, during immune marker screening is that comparing correlation types can reveal specific data nuances. If a marker's ranking drops significantly when switching to Spearman, it likely suggests that a few cells with extremely high expression were artificially inflating the Pearson correlation.

```python
plt.style.use('dark_background')

target = "(marker to analyze) Cell Exp"
methods = {'pearson': pearsonr, 'spearman': spearmanr, 'kendall': kendalltau}
results = {m: [] for m in methods}

def get_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""
for col in df.columns:
    if col != target and df[col].dtype in ['float64', 'int64']:
        mask = df[col].notna() & df[target].notna()
        if mask.sum() > 2:
            for m_name, m_func in methods.items():
                corr, p_val = m_func(df[col][mask], df[target][mask])
                if corr > 0.30: # Ton seuil de force
                    results[m_name].append({
                        'Marker': col, 
                        'Corr': corr, 
                        'p': p_val, 
                        'Stars': get_stars(p_val)
                    })

fig, axes = plt.subplots(1, 3, figsize=(22, 12), sharey=False)
fig.suptitle(f"Multi-Statistical Analysis (Réf: {target})", fontsize=20, weight="bold", y=0.98)

for i, (m_name, data_list) in enumerate(results.items()):
    if data_list:
        res_df = pd.DataFrame(data_list).sort_values('Corr', ascending=True)
        labels = [f"{row['Marker']} {row['Stars']}" for _, row in res_df.iterrows()]
        color_map = plt.cm.viridis(res_df['Corr'])
        axes[i].barh(labels, res_df['Corr'], color=color_map, edgecolor="white", linewidth=0.6)
        axes[i].set_title(f"Méthode : {m_name.upper()}", fontsize=15, pad=15, color="cyan")
        axes[i].axvline(0.3, color="red", linestyle="--", alpha=0.6, label="Seuil r=0.30")
        axes[i].set_xlabel("Correlation coefficient")
        axes[i].grid(axis='x', linestyle=':', alpha=0.2)
    else:
        axes[i].text(0.5, 0.5, "No correlation > 0.30", ha='center', va='center', fontsize=14)
legend_text = (
    "Significance (p-value) :\n"
    "*** : p < 0.001 (Extremely significant)\n"
    "** : p < 0.01  (Very significant)\n"
    "* : p < 0.05  (Significant)\n"
    "Minimum correlation threshold : 0.30"
)

fig.text(0.85, 0.02, legend_text, fontsize=11, color="white",
         bbox=dict(facecolor='black', edgecolor='white', boxstyle='round,pad=1', alpha=0.8))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])

plt.show()
```