"""Ejecuta el modelo final guardado y explica un lead cercano al percentil 85.

Aportes exactos de la regresión logística: beta * (x - media de X en H),
agrupados por variable original. La suma más la base reproduce el logit.
"""
from pathlib import Path
import json
import pickle
import sys
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / 'ejemplo_percentil85'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / '05 Power BI'))
from scoring_leads.preparacion import INCOME_MID, CANAL_FAMILIA

model_path = ROOT / '03 Prediccion/qualified_run/models/modelo_final_busqueda.pkl'
with model_path.open('rb') as f:
    bundle = pickle.load(f)
pipe = bundle['pipeline']
df = pd.read_csv(ROOT / '01 Limpieza Código/data/features/leads_enriched_qualified_final.csv', low_memory=False)
# Misma ingeniería de variables que §21.4, sobre la base ya limpia.
df['ingreso_mid'] = df['ingreso_familiar'].map(INCOME_MID)
df['edad_n'] = pd.to_numeric(df['edad'], errors='coerce')
df['es_inversion'] = (df['razon_de_compra'] == 'Inversión').astype(int)
df['canal_familia'] = df['canal_de_anuencia'].map(CANAL_FAMILIA).fillna('otro')
df['es_referido'] = (df['canal_familia'] == 'referido').astype(int)
df['edad_x_inversion'] = df['edad_n'] * df['es_inversion']
df['edad_x_ingreso'] = df['edad_n'] * df['ingreso_mid'] / 100.0
df['edad_x_referido'] = df['edad_n'] * df['es_referido']
df['mayor_45'] = (df['edad_n'] >= 45).astype(int)
df['ingreso_x_inversion'] = df['ingreso_mid'] * df['es_inversion']
df['ingreso_alto'] = (df['ingreso_mid'] >= 32).astype(int)
h = df.loc[df[bundle['requisitos_universo']].notna().all(axis=1)].copy()
h['fila_base_csv'] = h.index + 2
h = h.reset_index(drop=True)
X = h[bundle['features']]
prob = pipe.predict_proba(X)[:, 1]
percentile = pd.Series(prob).rank(method='average', pct=True) * 100
i = int((percentile - 85).abs().idxmin())
prep = pipe.named_steps['prep']
clf = pipe.named_steps['clf']
xt = prep.transform(X)
mean = np.asarray(xt.mean(axis=0)).ravel()
coef = clf.coef_[0]
base = float(clf.intercept_[0] + mean @ coef)
contrib = (xt[i] - mean) * coef
names = prep.get_feature_names_out()
groups = []
for name in names:
    if name.startswith('num__'):
        groups.append(name[5:])
    else:
        candidates = [c for c in prep.transformers_[0][2] if name[5:].startswith(c + '_')]
        groups.append(max(candidates, key=len))
detail = pd.DataFrame({'termino': names, 'variable': groups, 'aporte_log_odds': contrib})
agg = detail.groupby('variable', sort=False)['aporte_log_odds'].sum().reindex(bundle['features'])
assert np.isclose(expit(base + agg.sum()), prob[i], atol=1e-12)
assert len(agg) == 24 and np.isfinite(contrib).all()

labels = {'edad_n':'Edad', 'sexo':'Sexo', 'estado_civil':'Estado civil',
 'ingreso_mid':'Ingreso (punto medio, miles Q)', 'ingreso_familiar':'Ingreso familiar',
 'razon_de_compra':'Razón de compra', 'canal_de_anuencia':'Canal de anuencia',
 'canal_de_contacto':'Canal de contacto', 'canal_familia':'Familia del canal',
 'tipo_unidad':'Tipo de unidad', 'opciones_visitadas':'Opciones visitadas',
 'participantes_en_credito':'Participantes en crédito', 'lugar_trabajo':'Lugar de trabajo',
 'lugar_vivienda':'Lugar de vivienda', 'campania_tipo':'Tipo de campaña',
 'campania_audiencia':'Audiencia de campaña', 'es_inversion':'Es inversión',
 'es_referido':'Es referido', 'edad_x_inversion':'Edad × inversión',
 'ingreso_x_inversion':'Ingreso × inversión', 'edad_x_ingreso':'Edad × ingreso / 100',
 'edad_x_referido':'Edad × referido', 'mayor_45':'Edad ≥ 45', 'ingreso_alto':'Ingreso medio ≥ Q32 mil'}
def value(v):
    if pd.isna(v): return 'Sin dato'
    if isinstance(v, (float, np.floating)): return f'{v:g}'
    return str(v)

table = agg.rename('aporte_log_odds').reset_index()
table['valor_lead'] = table['variable'].map(lambda c: value(h.loc[i,c]))
table['etiqueta'] = table['variable'].map(labels)
table = table.iloc[np.argsort(-table.aporte_log_odds.abs().to_numpy())]
table.to_csv(OUT / 'aportes_24_variables.csv', index=False)
detail.to_csv(OUT / 'aportes_terminos_transformados.csv', index=False)
top = table.head(12).copy()
rest = table.iloc[12:].aporte_log_odds.sum()
plot_labels = [f'{r.etiqueta}: {r.valor_lead}' for r in top.itertuples()] + ['Otras 12 variables (suma)']
vals = np.r_[top.aporte_log_odds.to_numpy(), rest]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
fig, ax = plt.subplots(figsize=(13.5,9))
fig.patch.set_facecolor('#fafbfe')
ax.set_facecolor('#fafbfe')
y = np.arange(len(vals))
ax.barh(y, vals, color=['#148777' if v >= 0 else '#d46659' for v in vals], height=.68)
ax.set_yticks(y, plot_labels)
ax.invert_yaxis()
ax.axvline(0, color='#7c8798', lw=1)
span = max(abs(vals))
ax.set_xlim(min(vals)-.34*span,max(vals)+.34*span)
for yy,v in zip(y,vals):
    ax.text(v + (.025*span if v >= 0 else -.025*span), yy, f'{v:+.3f}',
            va='center',ha='left' if v >= 0 else 'right',fontsize=10)
ax.set_xlabel('Aporte al puntaje logístico (log-odds) frente al perfil promedio de H',labelpad=14)
ax.spines[['top','right','left']].set_visible(False)
ax.tick_params(axis='y',length=0)
fig.suptitle('¿Por qué este lead está en el percentil 85?',x=.04,ha='left',y=.98,fontsize=21,fontweight='bold',color='#243149')
fig.text(.04,.925,f'Percentil {percentile[i]:.2f}  |  Probabilidad estimada: {prob[i]*100:.2f}%  |  Referencia: {len(h):,} leads',fontsize=13,color='#243149')
fig.text(.04,.885,'Verde: aumenta la predicción   ·   Rojo: disminuye la predicción',fontsize=11,color='#48566b')
fig.text(.04,.08,f'Base: {base:.4f} + aportes: {agg.sum():+.4f} = puntaje: {base+agg.sum():.4f} → probabilidad: {prob[i]*100:.2f}%',fontsize=11)
fig.text(.04,.025,'Aportes exactos del modelo guardado; no son puntos porcentuales ni efectos causales.\nPredicciones del modelo entrenado, distintas de las predicciones OOF de evaluación.',fontsize=10,color='#596579')
fig.subplots_adjust(left=.43,right=.97,top=.84,bottom=.19)
fig.savefig(OUT / 'lead_percentil85.png',dpi=180,facecolor=fig.get_facecolor())
fig.savefig(OUT / 'lead_percentil85.pdf',bbox_inches='tight',facecolor=fig.get_facecolor())
id_col = next((c for c in ['id_trato','id_Trato'] if c in h),None)
summary = {'modelo':str(model_path),'sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
 'universo':len(h),'indice_H':i,'fila_base_csv':int(h.loc[i,'fila_base_csv']),
 'id_trato':str(h.loc[i,id_col]) if id_col else None,
 'percentil':float(percentile[i]),'probabilidad':float(prob[i]),
 'base_logit':base,'suma_aportes':float(agg.sum()),'probabilidad_reconstruida':float(expit(base+agg.sum())),
 'metodo':'beta*(x-media_H), agrupado por variable; inferencia del artefacto guardado, no OOF',
 'perfil':{c:value(h.loc[i,c]) for c in bundle['features']}}
(OUT/'resumen.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
print(table.head(8).to_string(index=False))
