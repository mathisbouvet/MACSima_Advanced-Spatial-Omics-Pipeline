<h1 style="color: #333; text-align: center; padding: 15px 0; border-top: 3px double #333; border-bottom: 3px double #333; font-family: Georgia, serif; font-style: italic; font-weight: normal; margin-top: 20px; margin-bottom: 30px;">
  Protocole de Contrôle Qualité et d'Optimisation de la Segmentation Tissulaire
</h1>

<div style="text-align: center; font-family: Georgia, serif; font-size: 1.2em; color: #555; margin-top: -10px; margin-bottom: 30px;">
  Par <strong>Mathis BOUVET</strong> — Biologiste spécialiste en Reproduction et Développement
  <br>
  <span style="font-size: 0.8em; font-style: italic;">Mars 2026</span>
</div>

> **Note importante**
> : Ce document ne contient aucune donnée réel


<span style="background-color: #007acc; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: bold;">Issu d'une coupe tissulaire</span> <span style="background-color: #007acc; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: bold;">Code Python</span> <span style="background-color: #007acc; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: bold;">MACSima</span> <span style="background-color: #007acc; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: bold;">Segmentation</span>

<div style="border: 1px solid #569cd6; border-radius: 10px; padding: 20px; background-color: rgba(86, 156, 214, 0.1); color: #000000;">
  <strong>Le gatin</strong><br>
Le gating en immunofluorescence désigne le processus de sélection et de segmentation de populations cellulaires spécifiques au sein d'un échantillon complexe, agissant comme un filtre numérique pour isoler des événements biologiques d'intérêt selon des critères d'intensité ou de morphologie. Cette démarche repose sur des méthodes variées allant du seuil manuel, où l'expérimentateur définit visuellement une limite de fluorescence, aux approches plus robustes de normalisation par Z-score qui permettent de classer les cellules en quadrants (positifs ou négatifs) de manière statistique et reproductible.  On utilise également des algorithmes de Deep Learning comme Cellpose ou StarDist pour affiner cette sélection en intégrant des paramètres de forme et de texture tissulaire. Toutefois, la fiabilité du gating rencontre des limites importantes, notamment le risque de subjectivité lors d'un paramétrage manuel ou le phénomène de sur-apprentissage (overfitting) des modèles automatisés qui peuvent échouer à généraliser leurs prédictions face à la diversité des architectures biologiques, comme entre un testicule et un ovaire. De plus, un gating mal calibré peut entraîner l'inclusion d'artefacts de fluorescence ou de bruits de fond, faussant ainsi l'analyse de co-expression et la réalité de la "vérité terrain" biologique.
</div>
<br>

<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
  Objectif
</h2>
L'enjeu central de ce protocole est de quantifier avec précision la co-expression des marqueurs protéiques TH et HuCD au sein d'une architecture tissulaire complexe. En s'appuyant sur les intensités moyennes de fluorescence extraites après segmentation, l'objectif est de s'affranchir de la subjectivité visuelle par l'application d'un gating statistique rigoureux basé sur le Z-score. Cette approche permet de classifier chaque cellule dans un quadrant spécifique afin de déterminer si les protéines sont exprimées de manière isolée ou conjointe. Enfin, cette analyse numérique est corrélée à une cartographie spatiale (coordonnées X, Y) pour valider si les populations cellulaires ainsi identifiées suivent une organisation biologique cohérente ou des gradients d'expression particuliers au sein de l'échantillon.

<br>

<div style="border: 1px solid #d65323; border-radius: 10px; padding: 20px; background-color: rgba(213, 101, 45, 0.1); color: #000000;">
  <strong>Importation des librairies</strong><br>

<details>
  <summary><b>Afficher/masquer le code de configuration</b></summary>

```python
import importlib
import subprocess
import sys

required_packages = {
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "umap": "umap-learn",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "scipy": "scipy"
}

def install_and_import():
    print("Analyse de l'environnement en cours...")
    for module_name, package_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name} est déjà prêt.")
        except ImportError:
            print(f"⚠️ {module_name} manque. Installation de {package_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                print(f"🚀 {package_name} installé avec succès !")
            except Exception as e:
                print(f"❌ Erreur lors de l'installation de {package_name}: {e}")

install_and_import()

# Importations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

print("\n✨ Tous les modules sont importés et prêts pour l'analyse !")
```
</details>
</div>


## 1. Chargement des données 

Pour illustrer ce script, on prend deux marqueurs, TH et HUCD. 

On standardise et on supprime les valeurs extremes

```python
df = pd.read_csv("[Fichier].csv")

col_TH = "TH Cell Intensity Average"
col_HuCD = "HUCD Cell Intensity Average"

df_quad = df[[col_TH, col_HuCD]].dropna().copy()

scaler = StandardScaler()
df_quad[["TH_z", "HuCD_z"]] = scaler.fit_transform(df_quad[[col_TH, col_HuCD]])

df_quad = df_quad[
    (df_quad["TH_z"].abs() <= 3) &
    (df_quad["HuCD_z"].abs() <= 3)
].copy()
```
## Gating

Pour classer les cellules, pusiqu'on a deux marqueur, on fait une classifcation +/- par cadran définis par les axes à 0 (la moyenne). Une cellule est considérée "positive" pour un marqueur si  son intensité est supérieure ou égale à la moyenne globale (Z-score > 0), et négative (-) si elle est en dessous

```python
def quadrant(row):
    if row["TH_z"] >= 0 and row["HuCD_z"] >= 0:
        return "TH+ / HuCD+"
    elif row["TH_z"] >= 0 and row["HuCD_z"] < 0:
        return "TH+ / HuCD−"
    elif row["TH_z"] < 0 and row["HuCD_z"] >= 0:
        return "TH− / HuCD+"
    else:
        return "TH− / HuCD−"

df_quad["Quadrant"] = df_quad.apply(quadrant, axis=1)

colors = {
    "TH+ / HuCD+": "red",
    "TH+ / HuCD−": "orange",
    "TH− / HuCD+": "deepskyblue",
    "TH− / HuCD−": "gray"
}

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(7, 7), facecolor='black')

for label, color in colors.items():
    subset = df_quad[df_quad["Quadrant"] == label]
    ax.scatter(subset["HuCD_z"], subset["TH_z"], label=label, c=color, s=5)

ax.axhline(0, color='white', linestyle='--', linewidth=1)
ax.axvline(0, color='white', linestyle='--', linewidth=1)

ax.set_xlim(-2, 3)
ax.set_ylim(3, -2)
ax.set_aspect('equal', adjustable='box')

ax.set_xlabel("HuCD (z-score)", color='white')
ax.set_ylabel("TH (z-score)", color='white')
ax.set_title("TH vs HuCD (z-score) - Fond noir", color='white')

ax.tick_params(axis='both', colors='white')

legend = ax.legend(title="Quadrants", loc="upper left", markerscale=2, facecolor='black', edgecolor='white')
plt.setp(legend.get_texts(), color='white')
plt.setp(legend.get_title(), color='white')

plt.tight_layout()

plt.savefig("TH_HuCD_quadrants_dark.png", dpi=600, facecolor=fig.get_facecolor())
plt.show()
```
## 3. Analyse statistique 

On peut calculer la corrélation statistique par Pearson

```python
r, p_corr = pearsonr(df_quad["TH_z"], df_quad["HuCD_z"])
print(f"r = {r:.2f}, p = {p_corr:.4f}")
```

## 4. Cartographie spatiale des cellules

On peut projeter les cellules dans l'espace grâce à leur cordonnées X et Y

```python
# Exemple avec TH
col_TH = "TH Cell Intensity Average"

plt.figure(figsize=(7, 7))
plt.scatter(df["Nuc X"], df["Nuc Y"], c=df[col_TH], cmap="inferno", s=3)
plt.colorbar(label="Intensité TH")
plt.title("Expression de TH dans l'OZ (spatial)")
plt.xlabel("X")
plt.ylabel("Y")
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.savefig("TH_HuCD.png", dpi=600, facecolor=fig.get_facecolor())
plt.show()
```