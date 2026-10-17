"""Versión gráfica oscura de los aportes calculados, sin volver a estimarlos."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
font = Path(font_manager.findfont('DejaVu Sans'))
font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'font.weight': 'bold', 'axes.labelweight': 'bold',
                     'pdf.fonttype': 42, 'svg.fonttype': 'path'})
data = pd.read_csv(HERE / 'aportes_24_variables.csv')
top = data.head(12)
values = np.r_[top.aporte_log_odds, data.iloc[12:].aporte_log_odds.sum()]
labels = [f'{r.etiqueta} · {r.valor_lead}' for r in top.itertuples()]
labels = [s.replace('Casad@', 'Casado').replace('Q30k A 33.9k', 'Q30–33.9 mil') for s in labels]
labels += ['Otras variables']
background, foreground = '#FFFFFF', '#000000'
fig, ax = plt.subplots(figsize=(13, 8.2), facecolor=background)
ax.set_facecolor(background)
y = np.arange(len(values))
ax.barh(y, values, height=.55,
        color=['#455A62' if v >= 0 else '#68454F' for v in values])
ax.axvline(0, color='#64616B', lw=.8)
ax.set_yticks(y, labels, color=foreground)
ax.invert_yaxis()
ax.set_xlim(-.42, .79)
for yy, v in zip(y, values):
    ax.text(v + (.018 if v >= 0 else -.018), yy, f'{v:+.3f}',
            ha='left' if v >= 0 else 'right', va='center', color=foreground, fontsize=11)
ax.set_xticks([])
ax.tick_params(axis='y', length=0, pad=20)
ax.spines[:].set_visible(False)
ax.set_xlabel('Aporte al puntaje (log-odds)', color=foreground, fontsize=10, labelpad=14)
fig.subplots_adjust(left=.43, right=.98, top=.98, bottom=.07)
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(HERE / f'lead_percentil85_fondo_blanco.{ext}', dpi=220,
                facecolor=background, bbox_inches='tight', pad_inches=.3)
fig.savefig(HERE / 'lead_percentil85_texto_oscuro_transparente.png', dpi=220,
            transparent=True, bbox_inches='tight', pad_inches=.3)
print('Fuente:', font_manager.findfont(font_manager.FontProperties(family='Montserrat', weight='bold')))
