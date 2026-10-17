from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

out=Path(__file__).resolve().parent
with (out/'datos_boxplot_489.csv').open() as f:
    rows=[r for r in csv.DictReader(f) if r['tarea']=='Conversión' and r['pr_auc']]
metric_path=out.parents[1]/'03 Prediccion/qualified_run/results/modelo_final_metricas.csv'
with metric_path.open() as f:
    final=float(next(r for r in csv.DictReader(f) if r['CV']=='selección')['pr_auc'])
# Actualizar la referencia de LR final a la métrica vigente, sin agregar otro registro.
matches=[r for r in rows if r['lote']=='Universo H' and r['algoritmo']=='LR']
assert len(matches)==1
matches[0]['pr_auc']=final
font_manager.fontManager.addfont(font_manager.findfont('DejaVu Sans'))
plt.rcParams.update({'font.family':'Montserrat','font.weight':'bold','axes.labelweight':'bold',
                     'svg.fonttype':'path','font.size':10})
order=['Regresión logística','Random forest','Hist. gradient boosting','Extra trees',
       'Gradient boosting','Análisis discriminante','AdaBoost','Naive Bayes','Red neuronal']
fig,ax=plt.subplots(figsize=(12/2.54,14/2.54),dpi=300,facecolor='white')
fig.subplots_adjust(left=.34,right=.965,bottom=.12,top=.97)
rng=np.random.default_rng(42)
for i,family in enumerate(order):
    group=[r for r in rows if r['familia']==family]
    vals=np.array([float(r['pr_auc']) for r in group])
    color='#111111' if i==0 else '#BDBDBD'
    if len(vals)>1:
        ax.boxplot([vals],positions=[i],orientation='horizontal',widths=.56,
                   patch_artist=True,showfliers=False,manage_ticks=False,
                   boxprops={'facecolor':color,'edgecolor':color,'linewidth':.8},
                   whiskerprops={'color':color,'linewidth':.9},
                   capprops={'color':color,'linewidth':.9},
                   medianprops={'color':'white','linewidth':1.2},zorder=2)
    jit=rng.uniform(-.19,.19,len(vals))
    for j,r in enumerate(group):
        if r is matches[0]:
            ax.scatter(vals[j],i,s=34,c='#E56838',edgecolors='white',linewidths=.65,zorder=5)
        else:
            ax.scatter(vals[j],i+jit[j],s=7,facecolors=(.65,.65,.65,.45),
                       edgecolors=(.4,.4,.4,.3),linewidths=.2,zorder=3)
labels=['Regresión\nlogística','Random\nforest','Hist. gradient\nboosting','Extra\ntrees','Gradient\nboosting','Análisis\ndiscriminante','AdaBoost','Naive\nBayes','Red\nneuronal']
ax.set_yticks(range(9),labels,fontsize=11)
ax.set_ylim(8.65,-.65)
ax.set_xlim(0,.255)
ax.set_xticks([0,.05,.10,.15,.20,.25],['0','.05','.10','.15','.20','.25'])
ax.tick_params(axis='x',labelsize=10,length=3,width=.7,pad=5)
ax.tick_params(axis='y',length=0,pad=8)
ax.set_xlabel('PR-AUC',fontsize=11,labelpad=8)
for side in ['top','right','left']:ax.spines[side].set_visible(False)
ax.spines['bottom'].set_color('#BDBDBD')
ax.spines['bottom'].set_linewidth(.7)
for ext in ['png','svg']:
    fig.savefig(out/f'boxplot_conversion_horizontal_12x14cm.{ext}',dpi=300,facecolor='white')
print(f'Final PR-AUC: {final}; 392 puntos; 12 × 14 cm; 300 dpi')
