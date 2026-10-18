"""Fase 1d: ablación por familia de variables.

Para el mejor pool (P4_Eng) se mide la caída de PR-AUC (OOF, LDA y GB) al
retirar cada familia completa. Cuantifica:
  - cuánto aporta campania_* (familia hermana de campania_plataforma, que el
    usuario retiró por fuga) — si carga mucho, hay que discutirla;
  - cuánto aporta cada familia legítima (edad, ingreso, canal, lugar, ...).
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/private/tmp/claude-501/-Users-angieflores-Documents-MEGAPROYECTO-02-K-Means/32058bcb-a5eb-4597-8495-c05b18ef0058/scratchpad')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import average_precision_score

from explore_common import load_universe, POOLS, oof_metrics, SEED

BASE = '/private/tmp/claude-501/-Users-angieflores-Documents-MEGAPROYECTO-02-K-Means/32058bcb-a5eb-4597-8495-c05b18ef0058/scratchpad'
d = load_universe()
y = d['target_won'].values
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
POOL = [c for c in POOLS['P4_Eng'] if c in d.columns]

FAMILIES = {
    'edad':      ['edad_n', 'edad_x_inversion', 'edad_x_ingreso', 'edad_x_referido', 'mayor_45'],
    'ingreso':   ['ingreso_mid', 'ingreso_familiar', 'ingreso_per_capita',
                  'ingreso_x_inversion', 'ingreso_alto'],
    'demografia': ['sexo', 'estado_civil', 'tiene_hijos_bin', 'nucleo_n', 'etapa_familiar'],
    'razon':     ['razon_de_compra', 'es_inversion'],
    'canal':     ['canal_de_anuencia', 'canal_de_contacto', 'canal_familia',
                  'es_referido'],
    'campania':  ['campania_tipo', 'campania_audiencia'],
    'lugar':     ['lugar_trabajo', 'lugar_vivienda'],
    'comercial': ['tipo_unidad', 'opciones_visitadas', 'participantes_en_credito'],
}


def make_pipe(pool, model):
    cat = [c for c in pool if not pd.api.types.is_numeric_dtype(d[c])]
    num = [c for c in pool if c not in cat]
    linear = model == 'LDA'
    if linear:
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=30)
    else:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    steps = []
    if cat:
        steps.append(('cat', Pipeline([
            ('imp', SimpleImputer(strategy='constant', fill_value='_missing_')),
            ('enc', enc)]), cat))
    if num:
        steps.append(('num', Pipeline([
            ('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num))
    prep = ColumnTransformer(steps, remainder='drop')
    clf = (LinearDiscriminantAnalysis() if model == 'LDA'
           else GradientBoostingClassifier(n_estimators=150, learning_rate=0.05,
                                           max_depth=3, subsample=0.8, random_state=SEED))
    return Pipeline([('prep', prep), ('clf', clf)])


def pr_oof(pool, model):
    prob = cross_val_predict(make_pipe(pool, model), d[pool], y, cv=skf,
                             method='predict_proba', n_jobs=-1)[:, 1]
    return average_precision_score(y, prob)


rows = []
for model in ['LDA', 'GB']:
    pr_full = pr_oof(POOL, model)
    print(f"\n{model} — pool completo ({len(POOL)} vars): PR={pr_full:.4f}", flush=True)
    rows.append({'modelo': model, 'ablacion': '(pool completo)',
                 'n_vars': len(POOL), 'pr_auc': round(pr_full, 4), 'caida': 0.0})
    for fam, cols in FAMILIES.items():
        pool_wo = [c for c in POOL if c not in cols]
        pr = pr_oof(pool_wo, model)
        rows.append({'modelo': model, 'ablacion': f'sin_{fam}',
                     'n_vars': len(pool_wo), 'pr_auc': round(pr, 4),
                     'caida': round(pr_full - pr, 4)})
        print(f"  sin {fam:10s} ({len(cols)} vars fuera): PR={pr:.4f}  "
              f"caída={pr_full-pr:+.4f}", flush=True)
    # solo cada familia (aislada)
    for fam, cols in FAMILIES.items():
        pool_only = [c for c in cols if c in d.columns]
        try:
            pr = pr_oof(pool_only, model)
            rows.append({'modelo': model, 'ablacion': f'solo_{fam}',
                         'n_vars': len(pool_only), 'pr_auc': round(pr, 4),
                         'caida': np.nan})
            print(f"  SOLO {fam:10s}: PR={pr:.4f}", flush=True)
        except Exception as e:
            print(f"  SOLO {fam}: ERR {e}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(f'{BASE}/results_v4_ablation.csv', index=False)
print("\nOK fase 1d", flush=True)
