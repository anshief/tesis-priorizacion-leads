"""Presentación limpia de la permutation importance existente del modelo final."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / 'permutation_importance_limpia'
OUT.mkdir(exist_ok=True)
font_manager.fontManager.addfont(font_manager.findfont('DejaVu Sans'))
plt.rcParams.update({'font.family':'Montserrat', 'font.weight':'bold',
    'axes.labelweight':'bold', 'font.size':12, 'text.color':'black',
    'pdf.fonttype':42, 'svg.fonttype':'path'})
source = ROOT / '03 Prediccion/qualified_run/results/modelo_final_importancia.csv'
data = pd.read_csv(source).sort_values('imp_media', ascending=False)
labels = {
 'es_referido':'¿Referido?', 'es_inversion':'¿Compra para inversión?',
 'canal_de_contacto':'Canal de contacto', 'canal_de_anuencia':'Canal de anuencia',
 'edad_x_ingreso':'Edad × ingreso', 'participantes_en_credito':'Participantes en el crédito',
 'tipo_unidad':'Tipo de vivienda', 'campania_audiencia':'Audiencia de la campaña',
 'edad_x_referido':'Edad × referido', 'lugar_trabajo':'Ubicación del trabajo',
 'opciones_visitadas':'Opciones visitadas', 'campania_tipo':'Tipo de campaña',
 'lugar_vivienda':'Ubicación de residencia', 'ingreso_x_inversion':'Ingreso × inversión',
 'estado_civil':'Estado civil', 'edad_n':'Edad', 'mayor_45':'¿Tiene 45 años o más?',
 'sexo':'Sexo', 'ingreso_familiar':'Rango de ingreso familiar',
 'ingreso_alto':'¿Ingreso medio ≥ Q32 mil?', 'ingreso_mid':'Ingreso familiar (punto medio)',
 'edad_x_inversion':'Edad × inversión', 'canal_familia':'Grupo del canal de anuencia',
 'razon_de_compra':'Motivo de compra'}
assert set(data.variable) == set(labels)
data['etiqueta'] = data.variable.map(labels)
data.to_csv(OUT / 'importancias_con_etiquetas.csv', index=False)
for n in (15, 24):
    subset = data.head(n)
    fig, ax = plt.subplots(figsize=(12.5, 8.5 if n == 15 else 12.5), facecolor='white')
    y = np.arange(n)
    means = subset.imp_media.to_numpy()
    stds = subset.imp_std.to_numpy()
    ax.barh(y, means, height=.56,
            color=['#455A62' if v >= 0 else '#68454F' for v in means])
    ax.errorbar(means, y, xerr=stds, fmt='none', ecolor='#252329',
                elinewidth=.9, capsize=2.5, capthick=.9)
    ax.axvline(0, color='#AAA7AD', lw=.7)
    ax.set_yticks(y, subset.etiqueta, color='black')
    ax.invert_yaxis()
    for yy, v, sd in zip(y, means, stds):
        ax.text(max(v + sd, 0) + .002, yy, f'{v:.4f}', va='center', fontsize=10.5)
    ax.set_xlim(min(-.008, float((means-stds).min())-.002), float((means+stds).max())+.016)
    ax.set_xticks([])
    ax.tick_params(axis='y', length=0, pad=16)
    ax.spines[:].set_visible(False)
    ax.set_xlabel('Caída de PR-AUC al permutar la variable', fontsize=11, color='black', labelpad=15)
    fig.subplots_adjust(left=.37, right=.98, top=.99, bottom=.075)
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT / f'permutation_importance_top{n}.{ext}', dpi=220,
                    facecolor='white', bbox_inches='tight', pad_inches=.25)
    plt.close(fig)
print('Exportadas versiones de 15 y 24 variables; medias y desviaciones originales conservadas.')
