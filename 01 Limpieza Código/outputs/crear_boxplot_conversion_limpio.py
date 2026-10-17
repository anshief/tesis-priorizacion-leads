from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

out = Path(__file__).resolve().parent
with (out / 'datos_boxplot_489.csv').open() as f:
    rows = [r for r in csv.DictReader(f) if r['tarea'] == 'Conversión' and r['pr_auc']]
assert len(rows) == 392
font_manager.fontManager.addfont(font_manager.findfont('DejaVu Sans'))
plt.rcParams.update({'font.family':'Montserrat', 'font.weight':'bold',
                     'axes.labelweight':'bold', 'font.size':7, 'svg.fonttype':'path'})
order = ['Regresión logística','Random forest','Hist. gradient boosting','Extra trees',
         'Gradient boosting','Análisis discriminante','AdaBoost','Naive Bayes','Red neuronal']
fig, ax = plt.subplots(figsize=(14/2.54, 10/2.54), facecolor='white')
fig.subplots_adjust(left=.15, right=.975, bottom=.31, top=.94)
rng = np.random.default_rng(42)
for i, family in enumerate(order):
    v = np.array([float(r['pr_auc']) for r in rows if r['familia']==family])
    if len(v)>1:
        ax.boxplot([v], positions=[i], widths=.61, patch_artist=True, showfliers=False,
                   boxprops={'facecolor':'black', 'edgecolor':'black', 'linewidth':.8},
                   medianprops={'color':'white', 'linewidth':1.15},
                   whiskerprops={'color':'black','linewidth':.8},
                   capprops={'color':'black','linewidth':.8}, manage_ticks=False, zorder=2)
    ax.scatter(i+rng.uniform(-.24,.24,len(v)), v, s=9,
               facecolors=(.74,.74,.74,.55), edgecolors=(.4,.4,.4,.35), linewidths=.25, zorder=3)
ax.set_xlim(-.6,8.6)
ax.set_ylim(0,.26)
ax.set_yticks([0,.05,.10,.15,.20,.25], ['0','0.05','0.10','0.15','0.20','0.25'])
ax.set_xticks(range(9), order, rotation=43, ha='right', rotation_mode='anchor', fontsize=6.5)
ax.set_ylabel('PR-AUC', fontsize=8, labelpad=10)
ax.tick_params(axis='x', length=0, pad=9)
ax.tick_params(axis='y', length=3, width=.8, pad=7)
for spine in ax.spines.values(): spine.set_visible(False)
for ext in ['png','svg']:
    fig.savefig(out/f'boxplot_conversion_14x10cm.{ext}', dpi=508, facecolor='white')
print('392 puntos; PNG 2800 × 2000 px a 508 dpi; SVG 14 × 10 cm.')
