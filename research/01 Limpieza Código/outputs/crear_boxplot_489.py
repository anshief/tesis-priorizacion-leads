from pathlib import Path
import csv, json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / '03 Prediccion'
Q = P / 'qualified_run'
OUT = Path(__file__).resolve().parent
rows = []
def read(p):
    with p.open() as f:
        return list(csv.DictReader(f))
names = {'LR':'Regresión logística','LR_l1':'Regresión logística','LR_l2':'Regresión logística',
         'RF':'Random forest','EXT':'Extra trees','GBM':'Hist. gradient boosting',
         'HGB':'Hist. gradient boosting','GB':'Gradient boosting','LDA':'Análisis discriminante',
         'ADA':'AdaBoost','NB':'Naive Bayes','MLP':'Red neuronal'}
def add(batch, alg, metric, source, config='', task='Conversión', note=''):
    rows.append(dict(lote=batch,familia=names[alg],algoritmo=alg,tarea=task,
                     pr_auc='' if metric is None else float(metric),configuracion=config,
                     fuente=str(source),nota=note))
for folder, pop in [(P/'results','Población general'),(Q/'results','Población calificada')]:
    source=folder/'all_experiments_comparison.csv'
    for r in read(source):
        task='Conversión' if r['task']=='Conversion' else 'Abandono'
        add(pop+' / '+task,r['algorithm'],r['pr_auc'],source,r['feature_set']+' / '+r['experiment'],task)
for file,batch in [('completo_models_comparison.csv','I1–I5'),('u18_models_comparison.csv','J1–J5'),('recall_models_comparison.csv','A/B/C/E')]:
    source=Q/'results'/file
    for r in read(source):add(batch,r['algorithm'],r['pr_auc'],source,r['feature_set']+' / '+r['experiment'])
source=Q/'results/u18_comparacion_podado.csv'
r=read(source)[1]
add('J5 podado','LR',r['pr_auc'],source,r['modelo'])
source=P/'03_modelado_predictivo_leads_qualified.ipynb'
nb=json.loads(source.read_text())
for c in nb['cells']:
    if 'u20_results = run_u18_grid' in ''.join(c.get('source',[])):
        s=''.join(''.join(o.get('text',[])) for o in c.get('outputs',[]))
        matches=re.findall(r'ok\s+(\w+)\s+\| J6_Clean\s+\| PR=([0-9.]+)',s)
        assert len(matches)==8
        for i,(alg,value) in enumerate(matches):
            add('J6',alg,value,source,'sin SMOTE' if i<4 else 'con SMOTE',note='PR-AUC recuperado de salida de notebook, 4 decimales')
source=Q/'results/rna_vs_lr_metricas.csv'
for r in read(source):
    if r['CV']=='selección':add('Universo H','LR' if r['modelo'].startswith('LR') else 'MLP',r['pr_auc'],source,r['modelo'],note='Primera semilla; métrica histórica del archivo')
for file,batch in [('results_v1_grid.csv','v1'),('results_v2_tuning.csv','v2')]:
    source=Q/'exploracion'/file
    for r in read(source):add(batch,r['alg'],r['pr_auc'],source,r['pool']+' / '+r.get('balance',r.get('best_params','')))
source=Q/'exploracion/results_v3_univariate.csv'
for r in read(source):add('v3 univariado','LR',r['pr_auc_solo'],source,r['variable'])
source=Q/'exploracion/results_v3_forward.csv'
for r in read(source):add('v3 forward','LR',r['pr_auc'],source,'Paso '+r['paso']+' / '+r['variable'])
source=Q/'exploracion/results_v3_backward_pool.csv'
assert len(read(source))==11
add('v3 backward','LR',None,source,'11 variables',note='Archivo conserva variables, no PR-AUC; no se sustituye por candidato v5 con otros hiperparámetros')
source=Q/'exploracion/results_v4_ablation.csv'
for r in read(source):add('v4',r['modelo'],r['pr_auc'],source,r['ablacion'])
source=Q/'exploracion/results_v5_finalists.csv'
for r in read(source):add('v5',r['modelo'],r['pr_auc_sel'],source,r['candidato'],note='Primera semilla; segunda semilla no contada aparte')
source=Q/'exploracion/results_v6_universes.csv'
for r in read(source):add('v6','LR',r['pr_auc'],source,r['set'])
source=Q/'exploracion/results_v7_oof.csv'
r=read(source)
ap=average_precision_score([int(x['y']) for x in r],[float(x['prob']) for x in r])
add('v7','LR',ap,source,'Universo intermedio',note='Average precision calculada desde predicciones OOF guardadas')
assert len(rows)==489
assert sum(r['tarea']=='Abandono' for r in rows)==96
assert sum(r['pr_auc']!='' for r in rows)==488
with (OUT/'datos_boxplot_489.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

font_manager.fontManager.addfont(font_manager.findfont('DejaVu Sans'))
plt.rcParams.update({'font.family':'Montserrat','font.weight':'bold','axes.labelweight':'bold',
                     'axes.titleweight':'bold','font.size':11,'svg.fonttype':'path','pdf.fonttype':42})
order=['Regresión logística','Random forest','Hist. gradient boosting','Extra trees',
       'Gradient boosting','Análisis discriminante','AdaBoost','Naive Bayes','Red neuronal']
labels=['Regresión\nlogística','Random\nforest','Hist. gradient\nboosting','Extra\ntrees',
        'Gradient\nboosting','Análisis\ndiscriminante','AdaBoost','Naive\nBayes','Red\nneuronal']
fig,ax=plt.subplots(figsize=(17,9),facecolor='white')
fig.subplots_adjust(left=.075,right=.975,top=.75,bottom=.24)
rng=np.random.default_rng(42)
for i,fam in enumerate(order):
    subset=[r for r in rows if r['familia']==fam]
    values=np.array([r['pr_auc'] for r in subset if r['pr_auc']!=''])
    ax.scatter(i+rng.uniform(-.28,.28,len(values)),values,s=28,facecolors='#666666',
               edgecolors='white',linewidths=.4,alpha=.6,zorder=2)
    if len(values)>1:
        ax.boxplot([values],positions=[i],widths=.46,patch_artist=True,showfliers=False,
                   boxprops={'facecolor':(1,1,1,.15),'edgecolor':'black','linewidth':2},
                   medianprops={'color':'black','linewidth':3},
                   whiskerprops={'color':'black','linewidth':1.7},
                   capprops={'color':'black','linewidth':1.7},zorder=3,manage_ticks=False)
    ax.text(i,1.025,f'n = {len(subset)}'+('*' if i==0 else ''),
            transform=ax.get_xaxis_transform(),ha='center',fontsize=11,color='#333333')
ax.set_xticks(range(len(order)),labels,fontsize=11)
ax.set_xlim(-.6,len(order)-.4)
ax.set_ylim(0,1)
ax.set_yticks([0,.2,.4,.6,.8,1],['0','0.2','0.4','0.6','0.8','1.0'])
ax.set_ylabel('PR-AUC',labelpad=14,fontsize=13)
ax.grid(axis='y',color='#e6e6e6',lw=.8,zorder=0)
for spine in ax.spines.values():spine.set_visible(False)
ax.tick_params(axis='both',length=0,pad=12)
fig.text(.075,.945,'489',fontsize=50,va='top')
fig.text(.215,.931,'CONFIGURACIONES EVALUADAS',fontsize=24,va='top')
fig.text(.215,.86,'Algoritmos · conjuntos de variables · balanceo · hiperparámetros · universos',fontsize=11,color='#555555')
fig.text(.075,.125,'* 488 puntos con PR-AUC disponible + 1 registro de regresión logística sin métrica recuperable.',fontsize=10)
fig.text(.075,.084,'Cada punto representa un resultado; las cajas muestran mediana y rango intercuartílico.',fontsize=10,color='#555555')
fig.text(.075,.048,'Incluye conversión y abandono, distintas poblaciones y validaciones. Resume la exploración; no constituye un ranking comparable.',fontsize=9,color='#555555')
for ext in ['png','svg']:
    fig.savefig(OUT/f'boxplot_489_montserrat_bn.{ext}',dpi=220,facecolor='white')
print('Saved 489 records, 488 measured points')
print('Counts:',{f:sum(r['familia']==f for r in rows) for f in order})
