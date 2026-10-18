"""Exporta el PCA del modelo final con la paleta de perfiles y Montserrat Bold."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / 'pca_montserrat'
OUT.mkdir(exist_ok=True)
font_manager.fontManager.addfont(str(Path.home() / 'Library/Fonts/Montserrat-Bold.ttf'))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.weight': 'bold',
    'axes.labelweight': 'bold', 'axes.titleweight': 'bold', 'font.size': 12,
    'text.color': '#393939', 'axes.labelcolor': '#393939',
    'pdf.fonttype': 42, 'svg.fonttype': 'path'})
# Colores aproximados de la referencia proporcionada, ordenados por cluster.
COLORS = ['#3D5B6B', '#A85C39', '#8B93A8', '#8F5F80', '#6E8054', '#C39B35']
NAMES = ['Compradora independiente', 'Inversionista consolidado', 'Compradora residencial consolidada', 'Comprador familiar de mayor espacio', 'Comprador joven de vivienda', 'Inversionista emergente']
cols = ['edad', 'sexo', 'estado_civil', 'razon_de_compra', 'tipo_unidad']
df = pd.read_csv(ROOT / '01 Limpieza Código/data/features/leads_enriched_qualified_final.csv', low_memory=False)
df = df.loc[df[cols].notna().all(axis=1)].copy()
df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
df = df.dropna(subset=['edad']).reset_index(drop=True)
blocks = [StandardScaler().fit_transform(df[['edad']]) * .25]
for col, weight in zip(cols[1:], [1.2, 1.5, 1.4, 1.]):
    df[col] = df[col].astype(str)
    blocks.append(OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit_transform(df[[col]]) / np.sqrt(df[col].nunique()) * weight)
X = np.hstack(blocks)
model = joblib.load(ROOT / '02 K Means/Output/modelo_final_notebook/modelo_KMeans_R1.joblib')
labels = model.predict(X)
expected = pd.read_csv(ROOT / '02 K Means/Output/modelo_final_notebook/resumen_clusters_R1.csv').sort_values('cluster')['observaciones'].to_numpy()
assert np.array_equal(np.bincount(labels, minlength=6), expected), 'Los datos no coinciden con los perfiles del modelo guardado.'
original = pd.read_csv(ROOT / '02 K Means/Output/modelo_final_notebook/asignacion_clusters_R1.csv').sort_values('indice_original')
assert np.array_equal(original['indice_original'].to_numpy(), df.index.to_numpy())
assert np.array_equal(original['cluster_R1'].to_numpy(), labels), 'La numeración difiere del PCA original.'
pca = PCA(n_components=2, random_state=42)
xy = pca.fit_transform(X)
centers = pca.transform(model.cluster_centers_)
fig, ax = plt.subplots(figsize=(6, 10), facecolor='white')
for g, (color, name) in enumerate(zip(COLORS, NAMES)):
    ix = labels == g
    ax.scatter(xy[ix, 0], xy[ix, 1], s=13, alpha=.4, color=color,
               edgecolors='none', label=f'{g} · {name}', rasterized=True)
ax.scatter(centers[:, 0], centers[:, 1], marker='X', s=175,
           color='#292929', edgecolor='white', linewidth=1.2, label='Centroides', zorder=5)
ax.set(xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
       ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_axisbelow(True)
ax.grid(alpha=.16)
ax.spines[['top', 'right']].set_visible(False)
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontweight('bold')
fig.subplots_adjust(left=.19, right=.97, top=.98, bottom=.09)
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(OUT / f'pca_montserrat_bold_3x5.{ext}', dpi=300, facecolor='white')
plt.close(fig)
print(f'Validado: {len(labels):,} observaciones; tamaños de cluster idénticos al modelo final.')
print(OUT)
