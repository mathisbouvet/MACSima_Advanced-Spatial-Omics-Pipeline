<h1 style="color: #333; text-align: center; padding: 15px 0; border-top: 3px double #333; border-bottom: 3px double #333; font-family: Georgia, serif; font-style: italic; font-weight: normal; margin-top: 20px; margin-bottom: 30px;">
Protocole de Phénotypage Cellulaire et d’Analyse de Proximité Spatiale par Distances Euclidiennes
</h1>

<div style="text-align: center; font-family: Georgia, serif; font-size: 1.2em; color: #555; margin-top: -10px; margin-bottom: 30px;">
  Par <strong>Mathis BOUVET</strong> — Biologiste spécialiste en Reproduction et Développement
  <br>
  <span style="font-size: 0.8em; font-style: italic;">Mars 2026</span>
</div>

> **Note importante**
> : Ce document ne contient aucune donnée réel




<div style="border: 1px solid #569cd6; border-radius: 10px; padding: 20px; background-color: rgba(86, 156, 214, 0.1); color: #000000;">
  <strong>Le gatin</strong><br>
Le gating en immunofluorescence désigne le processus de sélection et de segmentation de populations cellulaires spécifiques au sein d'un échantillon complexe. Il agit comme un filtre numérique isolant des événements biologiques selon des critères d'intensité (via le Z-score) ou de morphologie.Au-delà du phénotype, ce protocole intègre la <strong>dimension spatiale</strong> en calculant la proximité des cellules par rapport aux structures environnantes. La mesure de référence ici est la <strong>distance euclidienne</strong>, qui calcule le segment le plus court entre deux points dans un plan 2D.Cependant, selon l'architecture biologique, d'autres métriques peuvent être envisagées :<ul><li><strong>Distance Euclidienne :</strong> La ligne droite classique $d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$. Idéale pour les tissus denses.</li><li><strong>Distance de Manhattan :</strong> Somme des valeurs absolues des différences de coordonnées. Utile pour des structures organisées en "grille" ou réseaux orthogonaux.</li><li><strong>Distance de Mahalanobis :</strong> Prend en compte la corrélation entre les variables et la distribution des données, utile pour identifier des "outliers" spatiaux.</li></ul>L'enjeu est de transformer une simple image en une base de données relationnelle où chaque cellule est définie par ce qu'elle exprime et <em>où</em> elle se situe.
</div>
<br>

<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
  Objectif
</h2>
L'enjeu central de ce protocole est de quantifier avec précision la co-expression de marqueurs protéiques au sein d'une architecture tissulaire complexe. En s'affranchissant de la subjectivité visuelle, cette méthode s'appuie sur un gating statistique rigoureux (Z-score) et un clustering non supervisé pour classifier les populations cellulaires de manière impartiale. L'analyse vise ensuite à corréler ces signatures moléculaires avec une cartographie spatiale. En automatisant le calcul des distances minimales entre les clusters cellulaires et les structures anatomiques d'intérêt, le protocole permet de valider statistiquement (via tests non paramétriques) l'existence de niches biologiques, de gradients d'expression ou d'organisations préférentielles au sein de l'échantillon.

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
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
    "scipy": "scipy"
}

def install_and_import():
    print("Analyse de l'environnement en cours...")
    for module_name, package_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            # Cas particulier pour sklearn qui ne renvoie pas d'ImportError parfois 
            # mais vide si mal installé
        except ImportError:
            print(f"⚠️ {module_name} manque. Installation de {package_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                print(f"🚀 {package_name} installé avec succès !")
            except Exception as e:
                print(f"❌ Erreur lors de l'installation de {package_name}: {e}")
        else:
            print(f"✅ {module_name} est déjà prêt.")

# Exécution de l'installation automatique
install_and_import()


# Importations finales (sans doublons)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Rectangle
from itertools import combinations
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("\n✨ Tous les modules sont importés et prêts pour l'analyse (Clustering, Stats, Viz) !")
```
</details>
</div>



## 1. Pré-traitement des données
### 1.a Importation

La lame présenté un artefact que l'on peut exclure pendant l'importation du CSV

```python
df = pd.read_csv(r"[Segmentation de la zone].csv")
df = df[~(
    (df["Nuc X"] >= 22800) & (df["Nuc X"] <= 26750) &
    (df["Nuc Y"] >= 32000) & (df["Nuc Y"] <= 35000)
)]
```
### 1.b Définition des seuils 

On définit un seuil vas pour éliminer le bruit de fond et un seuil haut pour éliminer les agrégats d'anticorps ou des poussières fluorescentes. On choisit ensuite les marqueurs que l'on veut discriminer

```python
# Seuils bas
seuils = {
    "marqueur1 Cell Intensity Average": 1000,
    "marqueur2 Cell Intensity Average": 400,
    "marqueur3 Cell Intensity Average": 30,
    "marqueur4 Cell Intensity Average": 100
}

# Seuils haut
plafonds = {
    "marqueur1 Cell Intensity Average": 2500, 
    "marqueur2 Cell Intensity Average": 1000,
    "marqueur3 Cell Intensity Average": 1000,
    "marqueur4 Cell Intensity Average": 1000
}

df["marqueur1_pos"] = (
    (df["marqueur1 Intensity Average"] > seuils["marqueur1 Intensity Average"]) &
    (df["marqueur1 Intensity Average"] < plafonds["marqueur1 Intensity Average"])
)

df["marqueur2_pos"] = (
    (df["marqueur2 Cell Intensity Average"] > seuils["marqueur2 Cell Intensity Average"]) &
    (df["marqueur2 Cell Intensity Average"] < plafonds["marqueur2 Cell Intensity Average"])
)

df["marqueur3_pos"] =  (
    (df["marqueur3 Cell Intensity Average"] > seuils["marqueur3 Cell Intensity Average"]) &
    (df["marqueur3 Cell Intensity Average"] < plafonds["marqueur3 Cell Intensity Average"])
)

df["marqueur4_pos"] = (
    (df["marqueur4 Cell Intensity Average"] > seuils["marqueur4 Cell Intensity Average"]) &
    (df["marqueur4 Cell Intensity Average"] < plafonds["marqueur4 Cell Intensity Average"])
)
```

### 1.c Création des sous-populations 

Ici on peut utiliser des masques booléens (Vrai/faux) pour identifier les cellules exprimant différentes combinaison de marqueur. 

```python
populations = {
    "marqueur1+": df["Syk_pos"],
    "marqueur1+/marqueur2-": df["marqueur1_pos"] & df["marqueur2_pos"],
}

# Couleur
couleurs = {
    "marqueur1": "gray",
    "marqueur1/marqueur2": "blue",

}
```
### 1.d Visualisation

```python
plt.figure(figsize=(12, 8))

plt.scatter(
    df["Nuc X"], df["Nuc Y"],
    color="lightgray",
    s=5,
    alpha=0.2,
    label="Toutes cellules"
)

for label, mask in populations.items():
    plt.scatter(
        df.loc[mask, "Nuc X"],
        df.loc[mask, "Nuc Y"],
        label=label,
        s=10,
        alpha=0.7,
        color=couleurs[label]
    )

plt.gca().invert_yaxis()
plt.xlabel("Nuc X")
plt.ylabel("Nuc Y")
plt.title("Distribution spatiale des sous-populations marqueur1")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
```
## 2. Clustering K-Means et analyse des distances

### 2.a Importation des différentes structures

Ici on va utiliser `StandarScaler`pour la normalisation avant d'appliquer un algorithme K-Means Clustering (avec son score de Silhouette). Ensuite, on calcule la distance Euclidienne pour chaque cellule. Le but est de déterminer si un cluster de cellules à tendance à se regrouper préférentiellement près d'une structure. Les statistiques sont issu d'un test de Mann-Whitney U

```python
df_cells = pd.read_csv(r"[Segmentation de la zone].csv")
df_cells = df_cells[~(
    (df_cells["Nuc X"] >= 22800) & (df_cells["Nuc X"] <= 26750) &
    (df_cells["Nuc Y"] >= 32000) & (df_cells["Nuc Y"] <= 35000)
)]


base_path = r""
structures = {
    "structure1": "structure1.csv",
    "structure2": "structure2.csv",
    "structure3": "structure3.csv",
    "structure4": "structure4.csv",
}
```
### 2.b Isolation des cellules Syk + 

On filtre à nouveau le jeu de données pour ne conserver que les cellules fortement positives pour Syk 

```python
df_marqueur1 = df_cells[
    (df_cells["marqueur1 Intensity Average"] > 1000) &
    (df_cells["marqueur1 Intensity Average"] < 2500)
].copy()

colonnes_markeurs = [
    "marqueur2 Cell Intensity Average",
    "marqueur4 Cell Intensity Average",
    "marqueur3 Cell Intensity Average",
]
```
### 2.c Normalisation et clustering

On utilise StandardScaler() pour la normalisation et l'algorithme KMeans pour regrouper les cellules en 3 clusters basés sur leur similarité d'expression. Le score de silhouette évalue la pertinnence de ces clusters

```python
scaler = StandardScaler()
X = scaler.fit_transform(df_syk[colonnes_markeurs])
kmeans = KMeans(n_clusters=3, random_state=42)
df_syk["cluster"] = kmeans.fit_predict(X)

silhouette_avg = silhouette_score(X, df_syk["cluster"])
print(f"Le score de silhouette pour les clusters est : {silhouette_avg:.2f}")
```
### 2.d Calcul des distance 

Pour chaque cellule Syk positive, on calcule la distance euclidienne minimale vers plusieurs structures anatomiques.

$$
d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
$$

```python
for nom, fichier in structures.items():
    df_struct = pd.read_csv(os.path.join(base_path, fichier))
    coords_struct = df_struct[["Nuc X", "Nuc Y"]].values
    distances_min = []
    for _, cell in df_syk.iterrows():
        dists = np.sqrt((coords_struct[:, 0] - cell["Nuc X"])**2 + (coords_struct[:, 1] - cell["Nuc Y"])**2)
        distances_min.append(dists.min())
    df_syk[f"dist_{nom}"] = distances_min
```

### 2.e Statistiques de Mann-Whitney U et visualisation

On compare les distances entre les différents clusters pour chaque structure. Les p-values sont traduites en étoiles et ajoutées sur les boxplotés généré via Seaborn

```python
clusters = sorted(df_syk["cluster"].unique())
structures_names = list(structures.keys())
stats_results = {}

for struct in structures_names:
    col_dist = f"dist_{struct}"
    stats_results[struct] = []
    for c1, c2 in combinations(clusters, 2):
        dist_c1 = df_syk[df_syk["cluster"] == c1][col_dist]
        dist_c2 = df_syk[df_syk["cluster"] == c2][col_dist]
        stat, pval = mannwhitneyu(dist_c1, dist_c2, alternative='two-sided')
        stats_results[struct].append({
            "cluster_1": c1,
            "cluster_2": c2,
            "p_value": pval
        })

def get_significance_symbol(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

df_distances = pd.DataFrame()
for nom in structures:
    df_temp = df_syk[["cluster", f"dist_{nom}"]].copy()
    df_temp = df_temp.rename(columns={f"dist_{nom}": "distance"})
    df_temp["structure"] = nom
    df_distances = pd.concat([df_distances, df_temp], axis=0)

for nom in structures:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df_distances[df_distances["structure"] == nom],
        x="cluster", y="distance", palette="Set2", ax=ax
    )
    titre_structure = nom.replace("_", " ").capitalize()
    ax.set_title(f"Distance au {titre_structure} selon le cluster marqueur1", fontsize=12)
    ax.set_xlabel("Cluster de profils marqueur1", fontsize=10)
    ax.set_ylabel("Distance minimale (μm)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    tests = stats_results[nom]
    y_max = df_distances[df_distances["structure"] == nom]["distance"].max()
    height = y_max * 0.05
    offset = 0
    paires_autorisees = [(0, 1), (0, 2), (1, 2)]

    for test in tests:
        c1, c2, p = test["cluster_1"], test["cluster_2"], test["p_value"]
        if (c1, c2) not in paires_autorisees and (c2, c1) not in paires_autorisees:
            continue
        if p < 0.05:
            x1, x2 = clusters.index(c1), clusters.index(c2)
            y = y_max + offset * height
            ax.plot([x1, x1, x2, x2], [y, y + height/2, y + height/2, y], lw=1.5, c='k')
            ax.text((x1 + x2) * 0.5, y + height / 2, get_significance_symbol(p),
                    ha='center', va='bottom', color='k', fontsize=10)
            offset += 1

    plt.tight_layout()
    nom_fichier = f"distance_boxplot_{nom}.png"
    plt.savefig(nom_fichier, dpi=600)
    plt.close()

# Dictionnaire
renommage = {
    # Ne pas hésiter à renommer les marqueurs, exemple : 
    '2-FoxP1 Cell Intensity Average': 'FoxP1',
}

moyennes_par_cluster = df_syk.groupby("cluster")[colonnes_markeurs].mean()

colonnes_renommees = [renommage.get(col, col) for col in moyennes_par_cluster.columns]
moyennes_par_cluster.columns = colonnes_renommees

plt.figure(figsize=(12, 6))
sns.heatmap(
    moyennes_par_cluster,
    cmap="vlag", center=0, annot=False,
    cbar_kws={"label": "Z-score"}
)
plt.title("Profils moyens des marqueurs dans chaque cluster marqueur1", fontsize=14)
plt.xlabel("Marqueurs", fontsize=11)
plt.ylabel("Cluster", fontsize=11)
plt.tight_layout()
plt.savefig("heatmap_profils_clusters.png", dpi=600)
plt.show()

print("\n--- Profils moyens par cluster (intensité normalisée) ---")
for cluster_id, row in moyennes_par_cluster.iterrows():
    print(f"\nCluster {cluster_id}")
    print("-" * 20)
    print(row.sort_values(ascending=False).head(10))

print("\n--- Nombre de cellules par cluster ---")
print(df_marqueur1["cluster"].value_counts().sort_index())
```
## 3. Cartographie spatiale avancée 

On créer une présentation montrant la localisation exacte des clusters par rapport aux structures anatomique. 

```python
df_seg = pd.read_csv(r"[Segmentation de la zone].csv")
df_clust = pd.read_csv(r"df_marqueur1_clusters.csv")

df_seg = df_seg[~(
    (df_seg["Nuc X"] >= 22800) & (df_seg["Nuc X"] <= 26750) &
    (df_seg["Nuc Y"] >= 32000) & (df_seg["Nuc Y"] <= 35000)
)]

df_seg["X_round"] = df_seg["Nuc X"].round(1)
df_seg["Y_round"] = df_seg["Nuc Y"].round(1)
df_clust["X_round"] = df_clust["Nuc X"].round(1)
df_clust["Y_round"] = df_clust["Nuc Y"].round(1)

df_merged = pd.merge(
    df_seg,
    df_clust[["X_round", "Y_round", "cluster"]],
    on=["X_round", "Y_round"],
    how="left"
)

base_path = r""
structures = {
    "structure1": "structure1.csv",
    "structure2": "structure2.csv",
    "structure3": "structure3.csv",
    "structure4": "structure4.csv",
    "structure5": "structure5.csv",
}

structure_colors = sns.color_palette("pastel", n_colors=len(structures))

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

for i, (nom, fichier) in enumerate(structures.items()):
    try:
        df_struct = pd.read_csv(os.path.join(base_path, fichier))
        ax.scatter(
            df_struct["Nuc X"],
            df_struct["Nuc Y"],
            s=5,
            color=structure_colors[i],
            alpha=0.8,
            label=nom.replace("_", " ").capitalize()
        )
    except Exception as e:
        print(f"Erreur lecture structure {nom} : {e}")

ax.scatter(
    df_merged["Nuc X"],
    df_merged["Nuc Y"],
    color="dimgray",
    s=5,
    alpha=0.1,
    label="Toutes cellules"
)

palette = sns.color_palette("Set2", n_colors=df_merged["cluster"].nunique())
for cluster in sorted(df_merged["cluster"].dropna().unique()):
    subset = df_merged[df_merged["cluster"] == cluster]
    ax.scatter(
        subset["Nuc X"],
        subset["Nuc Y"],
        s=10,
        alpha=1,
        label=f"Cluster {int(cluster)}",
        color=palette[int(cluster) % len(palette)]
    )

x_min, x_max = 22800, 26750
y_min, y_max = 32000, 35000
ax.add_patch(
    Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=1.5,
        edgecolor='red',
        facecolor='none',
        linestyle='--',
        label='Zone exclue'
    )
)

ax.invert_yaxis()
ax.set_xlabel("Nuc X", color='white')
ax.set_ylabel("Nuc Y", color='white')
ax.set_title("Distribution spatiale des clusters marqueur1 avec structures anatomiques", fontsize=14, color='white')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, linestyle="--", alpha=0.3, color="white")
plt.tight_layout()facecolor=fig.get_facecolor())
plt.show()
```

## 4. Boxplots horizontaux et Heatmap détaillée

```python
df_syk = pd.read_csv(r"")

base_path = r""
structures = {
    "structure1": "structure1.csv",
    "structure2": "structure2.csv",
    "structure3": "structure3.csv",
    "structure4": "structure4.csv",
    "structure5": "structure5.csv",
}
```

### 4.a Calcul des distances  et boxplots

On recalcul les distances mais cette fois on affiche sur un seul graphique

```python
df_syk = pd.read_csv(r"Hdf_syk_clusters.csv")

base_path = r"[Segmentation de la zone].csv"
structures = {
    "structure1": "structure1.csv",
    "structure2": "structure2.csv",
    "structure3": "structure3.csv",
    "structure4": "structure4.csv",
    "structure5": "structure5.csv",
}

for nom, fichier in structures.items():
    df_struct = pd.read_csv(os.path.join(base_path, fichier))
    coords_struct = df_struct[["Nuc X", "Nuc Y"]].values
    distances_min = []
    for _, cell in df_syk.iterrows():
        dists = np.sqrt((coords_struct[:, 0] - cell["Nuc X"])**2 + (coords_struct[:, 1] - cell["Nuc Y"])**2)
        distances_min.append(dists.min())
    df_syk[f"dist_{nom}"] = distances_min

df_distances = pd.DataFrame()
for nom in structures:
    df_temp = df_syk[["cluster", f"dist_{nom}"]].copy()
    df_temp = df_temp.rename(columns={f"dist_{nom}": "distance"})
    df_temp["structure"] = nom
    df_distances = pd.concat([df_distances, df_temp], axis=0)

structure_labels = {
    #On peut renommer les structures proprements, exemples : 
    "structure1": "Plexus Veineux",
}
df_distances["structure"] = df_distances["structure"].replace(structure_labels)

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

sns.boxplot(
    data=df_distances,
    x="structure",
    y="distance",
    hue="cluster",
    orient="v",
    palette="Set2",
    width=0.6,
    ax=ax
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=30,
    ha='right',
    fontsize=12,
    fontweight='bold',
    color='white'
)

ax.set_ylabel("Distance minimale (μm)", fontsize=12, fontweight='bold', color='white')
ax.set_xlabel("Structure", fontsize=12, fontweight='bold', color='white')
ax.set_title("Distances aux structures selon les clusters marqueur1", fontsize=14, fontweight='bold', color='white')

legend = ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title_fontsize=11)
plt.setp(legend.get_texts(), color='white')
plt.setp(legend.get_title(), color='white')

ax.grid(True, linestyle="--", alpha=0.5, color="white")
plt.tight_layout()get_facecolor())
plt.show()
```
### 4.b Heatmap détaillé

On calcule la moyenne d'expression standardisée de chaque marqueur pour chaque cluster

```python

df_syk = pd.read_csv(r"df_syk_clusters.csv")

# Liste des colonnes à supprimer
colonnes_a_supprimer = [
    "1-alpha_Tubulin Cell Intensity Average",
    "1-B3_tubulin Cell Intensity Average",
    "3-Synapsin Cell Intensity Average",
    "5-HUCD Cell Intensity Average",
    "DAPI Cell Intensity Average",
    "Gata3 ab210672 Cell Intensity Average",
    "Myosin_Smooth_Muscle REA1107 Cell Intensity Average",
    "CD31 REA1028 Cell Intensity Average",
    "SHIP_1 Cell Intensity Average",
    "Oct3_4 Cell Intensity Average"
]

# Supprimer les colonnes du DataFrame
df_syk = df_syk.drop(columns=colonnes_a_supprimer)

# Sélection des marqueurs d'expression cellulaire uniquement
colonnes_marqueurs = [col for col in df_syk.columns if col.endswith("Cell Intensity Average")]

# Normalisation (z-score)
df_heat = df_syk[["cluster"] + colonnes_marqueurs].copy()
df_heat[colonnes_marqueurs] = StandardScaler().fit_transform(df_heat[colonnes_marqueurs])

# Moyenne par cluster
moyennes = df_heat.groupby("cluster")[colonnes_marqueurs].mean()

# Renommage des colonnes pour affichage lisible
renommage = {
    '2-FoxP1 Cell Intensity Average': 'FoxP1',
    '2-Isl1 Cell Intensity Average': 'Isl1',
    'Ki_67 REA183 Cell Intensity Average': 'Ki-67',
    'Pro Collagen III alpha I AF488 Cell Intensity Average': 'Collagen III',
    '3-Synaptophysin Cell Intensity Average': 'Synaptophysin',
    'Vimentin REA409 Cell Intensity Average': 'Vimentin',
    '1-NeuF-H Cell Intensity Average': 'NeuF-H',
    'Sox2 Cell Intensity Average': 'Sox2',
    '6-VGLUT2 Cell Intensity Average': 'VGLUT2',
    'CD239 REA276 Cell Intensity Average': 'CD239',
    'Collagen IV AF488 Cell Intensity Average': 'Collagen IV',
    'FoxF1 AF488 Cell Intensity Average': 'FoxF1',
    '5-NeuN Cell Intensity Average': 'NeuN',
    '4-Sox10 Cell Intensity Average': 'Sox10',
    '2-FoxP2 Cell Intensity Average': 'FoxP2',
    'marqueur4 Cell Intensity Average': 'EpCAM',
    'marqueur2 Cell Intensity Average': 'Iba-1',
    '4-Phox2B Cell Intensity Average': 'Phox2B',
    'Myogenin REA930 Cell Intensity Average': 'Myogenin',
    '6-TH Cell Intensity Average': 'TH',
    'SOX9 Cell Intensity Average': 'Sox 9',
    'marqueur1 Intensity Average': 'Syk',
    'PAX_6 Cell Intensity Average': 'Pax 6',
    'TSPAN8 REA443 Cell Intensity Average': 'TSPAN8',
    'Notch3 Cell Intensity Average': 'Notch3',
    'Fetoprotein AF488 Cell Intensity Average': 'Fetoprotein',
    'S100A8 REA917 Cell Intensity Average': 'S100A8',
    '1-Neurofilament Cell Intensity Average': 'Neurofilament',
    'Sma ab202509 Cell Intensity Average': 'SMA',
    'marqueur3 Cell Intensity Average': 'Oct-2'
}
moyennes.columns = [renommage.get(col, col) for col in moyennes.columns]

# Supprimer certains marqueurs du DataFrame moyennes
colonnes_a_exclure = ["Myogenin", "S100A8", "SMA", "Syk"]
moyennes = moyennes.drop(columns=colonnes_a_exclure)

# === HEATMAP FOND NOIR — CLUSTERS EN LIGNES ===
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(16, 3))  # Format paysage
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

sns.heatmap(
    moyennes,
    cmap="RdBu_r",
    center=0,
    cbar_kws={"label": "Z-score"},
    linewidths=0.3,
    linecolor='gray',
    annot=True,
    fmt=".1f",
    annot_kws={"size": 7, "color": "black"},
    xticklabels=True,
    yticklabels=True,
    ax=ax
)

# Titres et axes
ax.set_title("Expression moyenne des marqueurs cellulaires par cluster (marqueur1)", fontsize=14, fontweight='bold', color='white')
ax.set_xlabel("Marqueurs", fontsize=12, fontweight='bold', color='white')
ax.set_ylabel("Clusters", fontsize=12, fontweight='bold', color='white')

# Ticks
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10, fontweight='bold', color='white')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig("heatmap_expression_cellulaire_clusters_fond_noir_horizontal.png", dpi=600, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
```