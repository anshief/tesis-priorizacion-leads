"""Fase 2: reconstrucción del universo con las variables ganadoras.

Para cada conjunto ganador S se define U(S) = leads calificados con los
componentes base de S completos, se reentrena dentro de U(S) (LR tuneada,
5-fold OOF) y se compara:
  a) métricas dentro de su propio universo (con su tasa base);
  b) métricas sobre las MISMAS filas del universo actual de 13K
     (comparación directa contra el campeón de fase 1).
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve

from explore_common import (load_universe, oof_metrics, SEED, INCOME_MID,
                            CANAL_FAMILIA, UNIVERSE_VARS, DATA, _clean_nucleo)

BASE = '/private/tmp/claude-501/-Users-angieflores-Documents-MEGAPROYECTO-02-K-Means/32058bcb-a5eb-4597-8495-c05b18ef0058/scratchpad'

df = pd.read_csv(DATA, low_memory=False)
print(f"Dataset calificado completo: {len(df):,} filas, "
      f"{int(df['target_won'].sum())} conversiones ({df['target_won'].mean()*100:.2f}%)", flush=True)


def engineer(d):
    d = d.copy()
    d['edad_n'] = pd.to_numeric(d['edad'], errors='coerce')
    d['ingreso_mid'] = d['ingreso_familiar'].map(INCOME_MID)
    d['nucleo_n'] = d['nucleo_familiar'].apply(_clean_nucleo)
    d['es_inversion'] = (d['razon_de_compra'] == 'Inversión').astype(int)
    d['canal_familia'] = d['canal_de_anuencia'].map(CANAL_FAMILIA).fillna('otro')
    d['es_referido'] = (d['canal_familia'] == 'referido').astype(int)
    d['edad_x_inversion'] = d['edad_n'] * d['es_inversion']
    d['edad_x_ingreso'] = d['edad_n'] * d['ingreso_mid'] / 100.0
    d['edad_x_referido'] = d['edad_n'] * d['es_referido']
    d['mayor_45'] = (d['edad_n'] >= 45).astype(int)
    return d


df = engineer(df)
mask_13k = df[UNIVERSE_VARS].notna().all(axis=1)
print(f"Universo fase 1 (10 vars completas): {mask_13k.sum():,} filas", flush=True)

# Conjuntos ganadores: (features del modelo, componentes base requeridos completos)
SETS = {
    'S_A_Canal': {
        'feats': ['canal_de_anuencia', 'canal_de_contacto', 'canal_familia', 'es_referido'],
        'req': ['canal_de_anuencia', 'canal_de_contacto'],
    },
    'S_B_Canal+Edad': {
        'feats': ['canal_de_anuencia', 'canal_de_contacto', 'canal_familia',
                  'es_referido', 'edad_n', 'mayor_45', 'edad_x_referido'],
        'req': ['canal_de_anuencia', 'canal_de_contacto', 'edad'],
    },
    'S_C_Canal+Edad+Lugar': {
        'feats': ['canal_de_anuencia', 'canal_de_contacto', 'canal_familia',
                  'es_referido', 'edad_n', 'mayor_45', 'edad_x_referido',
                  'lugar_trabajo'],
        'req': ['canal_de_anuencia', 'canal_de_contacto', 'edad', 'lugar_trabajo'],
    },
    'S_D_Backward11': {
        'feats': ['canal_de_anuencia', 'canal_de_contacto', 'canal_familia',
                  'participantes_en_credito', 'lugar_trabajo', 'campania_tipo',
                  'es_referido', 'edad_x_inversion', 'edad_x_ingreso',
                  'edad_x_referido', 'mayor_45'],
        'req': ['canal_de_anuencia', 'canal_de_contacto', 'edad', 'razon_de_compra',
                'ingreso_familiar', 'participantes_en_credito', 'lugar_trabajo',
                'campania_tipo'],
    },
}


def make_pipe(dd, pool):
    cat = [c for c in pool if not pd.api.types.is_numeric_dtype(dd[c])]
    num = [c for c in pool if c not in cat]
    steps = []
    if cat:
        steps.append(('cat', Pipeline([
            ('imp', SimpleImputer(strategy='constant', fill_value='_missing_')),
            ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                                  min_frequency=30))]), cat))
    if num:
        steps.append(('num', Pipeline([
            ('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num))
    prep = ColumnTransformer(steps, remainder='drop')
    return Pipeline([('prep', prep),
                     ('clf', LogisticRegression(C=0.55, penalty='l2', max_iter=2000,
                                                random_state=SEED))])


skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows = []
for name, spec in SETS.items():
    req = spec['req']
    feats = spec['feats']
    mask_u = df[req].notna().all(axis=1)
    du = df.loc[mask_u].reset_index(drop=True)
    yu = du['target_won'].values
    pipe = make_pipe(du, feats)
    prob = cross_val_predict(pipe, du[feats], yu, cv=skf,
                             method='predict_proba', n_jobs=-1)[:, 1]
    m = oof_metrics(yu, prob)
    # métricas restringidas a las filas del universo de 13K (comparación directa)
    in13 = mask_13k.loc[mask_u.index[mask_u]].values
    m13 = oof_metrics(yu[in13], prob[in13]) if in13.sum() > 100 else {}
    rows.append({'set': name, 'n_universo': len(du),
                 'conversiones': int(yu.sum()),
                 'tasa_base_%': round(yu.mean() * 100, 3),
                 'n_feats': len(feats),
                 'pr_auc': round(m['pr_auc'], 4),
                 'lift10': round(m['lift_10'], 2),
                 'pctleads_r70': round(m['pct_leads_r70'], 1),
                 'prec_r70': round(m['prec_r70'], 4),
                 'pr_auc_en13k': round(m13.get('pr_auc', np.nan), 4),
                 'lift10_en13k': round(m13.get('lift_10', np.nan), 2),
                 'pctleads_r70_en13k': round(m13.get('pct_leads_r70', np.nan), 1)})
    print(f"{name:22s} U={len(du):6,} conv={int(yu.sum()):4d} base={yu.mean()*100:.2f}% | "
          f"PR={m['pr_auc']:.4f} lift10={m['lift_10']:.2f} "
          f"%leads(r70)={m['pct_leads_r70']:.1f} | "
          f"en13K: PR={m13.get('pr_auc', float('nan')):.4f} "
          f"%leads(r70)={m13.get('pct_leads_r70', float('nan')):.1f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(f'{BASE}/results_v6_universes.csv', index=False)
print("\nOK fase 2", flush=True)
print(res.to_string(index=False), flush=True)
