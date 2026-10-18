"""Fase 2 (iteración 2): universo intermedio — relajar solo tiene_hijos y
nucleo_familiar (cuellos de botella del embudo §17.2, familia demografía con
aporte ~0 en la ablación). Universo esperado ≈ 18K (+38% vs 13K).

Modelo: LR tuneada (C=0.55) con el pool ingenierizado sin las variables que
dependen de nucleo/tiene_hijos.
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

from explore_common import (oof_metrics, SEED, INCOME_MID, CANAL_FAMILIA,
                            UNIVERSE_VARS, DATA, _clean_nucleo)

BASE = '/private/tmp/claude-501/-Users-angieflores-Documents-MEGAPROYECTO-02-K-Means/32058bcb-a5eb-4597-8495-c05b18ef0058/scratchpad'
df = pd.read_csv(DATA, low_memory=False)

df['edad_n'] = pd.to_numeric(df['edad'], errors='coerce')
df['ingreso_mid'] = df['ingreso_familiar'].map(INCOME_MID)
df['es_inversion'] = (df['razon_de_compra'] == 'Inversión').astype(int)
df['canal_familia'] = df['canal_de_anuencia'].map(CANAL_FAMILIA).fillna('otro')
df['es_referido'] = (df['canal_familia'] == 'referido').astype(int)
df['edad_x_inversion'] = df['edad_n'] * df['es_inversion']
df['edad_x_ingreso'] = df['edad_n'] * df['ingreso_mid'] / 100.0
df['edad_x_referido'] = df['edad_n'] * df['es_referido']
df['mayor_45'] = (df['edad_n'] >= 45).astype(int)
df['ingreso_x_inversion'] = df['ingreso_mid'] * df['es_inversion']
df['ingreso_alto'] = (df['ingreso_mid'] >= 32).astype(int)

REQ_E = ['fuente_pt2', 'canal_de_anuencia', 'canal_de_contacto', 'sexo',
         'razon_de_compra', 'estado_civil', 'ingreso_familiar', 'edad']
FEATS_E = ['edad_n', 'sexo', 'estado_civil', 'ingreso_mid', 'ingreso_familiar',
           'razon_de_compra', 'canal_de_anuencia', 'canal_de_contacto',
           'canal_familia', 'tipo_unidad', 'opciones_visitadas',
           'participantes_en_credito', 'lugar_trabajo', 'lugar_vivienda',
           'campania_tipo', 'campania_audiencia', 'es_inversion', 'es_referido',
           'edad_x_inversion', 'ingreso_x_inversion', 'edad_x_ingreso',
           'edad_x_referido', 'mayor_45', 'ingreso_alto',
           'fuente_pt2', 'proyecto']

mask_13k = df[UNIVERSE_VARS].notna().all(axis=1)
mask_e = df[REQ_E].notna().all(axis=1)
du = df.loc[mask_e].reset_index(drop=True)
yu = du['target_won'].values
in13 = mask_13k.loc[mask_e.index[mask_e]].values
print(f"Universo E (sin exigir tiene_hijos/nucleo): {len(du):,} filas, "
      f"{int(yu.sum())} conversiones ({yu.mean()*100:.2f}%)", flush=True)

cat = [c for c in FEATS_E if not pd.api.types.is_numeric_dtype(du[c])]
num = [c for c in FEATS_E if c not in cat]
prep = ColumnTransformer([
    ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='_missing_')),
                      ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                                            min_frequency=30))]), cat),
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc', StandardScaler())]), num)], remainder='drop')
pipe = Pipeline([('prep', prep),
                 ('clf', LogisticRegression(C=0.55, penalty='l2', max_iter=2000,
                                            random_state=SEED))])

for tag, seed in [('sel', SEED), ('conf', 2025)]:
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    prob = cross_val_predict(pipe, du[FEATS_E], yu, cv=skf,
                             method='predict_proba', n_jobs=-1)[:, 1]
    m = oof_metrics(yu, prob)
    m13 = oof_metrics(yu[in13], prob[in13])
    print(f"[{tag}] universo completo: PR={m['pr_auc']:.4f} lift10={m['lift_10']:.2f} "
          f"%leads(r70)={m['pct_leads_r70']:.1f} prec_r70={m['prec_r70']:.4f}", flush=True)
    print(f"[{tag}] restringido a 13K: PR={m13['pr_auc']:.4f} lift10={m13['lift_10']:.2f} "
          f"%leads(r70)={m13['pct_leads_r70']:.1f}", flush=True)
    if tag == 'sel':
        pd.DataFrame({'prob': prob, 'y': yu, 'in13k': in13}).to_csv(
            f'{BASE}/results_v7_oof.csv', index=False)
print("OK fase 2 iter 2", flush=True)
