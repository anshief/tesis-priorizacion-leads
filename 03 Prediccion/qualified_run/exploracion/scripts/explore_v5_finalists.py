"""Fase 1e: candidatos finalistas — confirmación con CV de otra semilla,
umbrales operativos, permutation importance y SHAP del ganador.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/private/tmp/claude-501/-Users-angieflores-Documents-MEGAPROYECTO-02-K-Means/32058bcb-a5eb-4597-8495-c05b18ef0058/scratchpad')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.inspection import permutation_importance

from explore_common import load_universe, POOLS, oof_metrics, SEED

BASE = '/private/tmp/claude-501/-Users-angieflores-Documents-MEGAPROYECTO-02-K-Means/32058bcb-a5eb-4597-8495-c05b18ef0058/scratchpad'
d = load_universe()
y = d['target_won'].values

POOL_BW11 = ['canal_de_anuencia', 'canal_de_contacto', 'canal_familia',
             'participantes_en_credito', 'lugar_trabajo', 'campania_tipo',
             'es_referido', 'edad_x_inversion', 'edad_x_ingreso',
             'edad_x_referido', 'mayor_45']
POOL_CANAL = ['canal_de_anuencia', 'canal_de_contacto', 'canal_familia', 'es_referido']
POOL_CANAL_EDAD = POOL_CANAL + ['edad_n', 'mayor_45', 'edad_x_referido']


def make_pipe(pool, model, **kw):
    cat = [c for c in pool if not pd.api.types.is_numeric_dtype(d[c])]
    num = [c for c in pool if c not in cat]
    linear = model == 'LR'
    enc = (OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=30)
           if linear else
           OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    steps = []
    if cat:
        steps.append(('cat', Pipeline([
            ('imp', SimpleImputer(strategy='constant', fill_value='_missing_')),
            ('enc', enc)]), cat))
    if num:
        steps.append(('num', Pipeline([
            ('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num))
    prep = ColumnTransformer(steps, remainder='drop')
    if model == 'LR':
        clf = LogisticRegression(C=kw.get('C', 0.55), penalty='l2', max_iter=2000,
                                 class_weight=kw.get('class_weight'), random_state=SEED)
    else:
        clf = GradientBoostingClassifier(n_estimators=326, learning_rate=0.0528,
                                         max_depth=5, min_samples_leaf=79,
                                         subsample=0.836, random_state=SEED)
    return Pipeline([('prep', prep), ('clf', clf)])


CANDIDATES = {
    'C1_LR_P5_Eng+Proj':  (POOLS['P5_Eng+Proj'], 'LR'),
    'C2_LR_P4_Eng':       (POOLS['P4_Eng'], 'LR'),
    'C3_LR_Backward11':   (POOL_BW11, 'LR'),
    'C4_GB_P5_Eng+Proj':  (POOLS['P5_Eng+Proj'], 'GB'),
    'C5_LR_SoloCanal':    (POOL_CANAL, 'LR'),
    'C6_LR_Canal+Edad':   (POOL_CANAL_EDAD, 'LR'),
}

rows = []
oof_store = {}
for name, (pool, model) in CANDIDATES.items():
    pool = [c for c in pool if c in d.columns]
    pipe = make_pipe(pool, model)
    row = {'candidato': name, 'modelo': model, 'n_vars': len(pool)}
    for tag, seed in [('sel', SEED), ('conf', 2025)]:
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        prob = cross_val_predict(pipe, d[pool], y, cv=skf,
                                 method='predict_proba', n_jobs=-1)[:, 1]
        m = oof_metrics(y, prob)
        row[f'pr_auc_{tag}'] = round(m['pr_auc'], 4)
        row[f'pctleads_r70_{tag}'] = round(m['pct_leads_r70'], 1)
        row[f'lift10_{tag}'] = round(m['lift_10'], 2)
        if tag == 'sel':
            oof_store[name] = prob
    rows.append(row)
    print(f"{name:22s} vars={row['n_vars']:2d} "
          f"PR sel={row['pr_auc_sel']:.4f} conf={row['pr_auc_conf']:.4f} | "
          f"%leads(r70) sel={row['pctleads_r70_sel']} conf={row['pctleads_r70_conf']} | "
          f"lift10={row['lift10_sel']}", flush=True)

fin = pd.DataFrame(rows)
fin.to_csv(f'{BASE}/results_v5_finalists.csv', index=False)

# ── umbrales operativos del mejor equilibrado ──
print("\n" + "=" * 70, flush=True)
print("UMBRALES OPERATIVOS (OOF sel) por candidato — recall 0.60/0.70/0.80", flush=True)
thr_rows = []
for name, prob in oof_store.items():
    prc, rcc, thr = precision_recall_curve(y, prob)
    for target in [0.60, 0.70, 0.80]:
        idx = np.where(rcc[:-1] >= target)[0]
        if len(idx):
            i = idx[-1]
            sel = prob >= thr[i]
            thr_rows.append({'candidato': name, 'objetivo': target,
                             'recall': round(rcc[i], 3), 'precision': round(prc[i], 4),
                             'pct_leads': round(sel.mean() * 100, 1),
                             'lift_vs_base': round(prc[i] / y.mean(), 2)})
thr_df = pd.DataFrame(thr_rows)
thr_df.to_csv(f'{BASE}/results_v5_thresholds.csv', index=False)
print(thr_df.to_string(index=False), flush=True)

# ── permutation importance + SHAP del mejor por PR conf ──
best_name = fin.sort_values('pr_auc_conf', ascending=False).iloc[0]['candidato']
pool, model = CANDIDATES[best_name]
pool = [c for c in pool if c in d.columns]
print(f"\nGANADOR (por PR conf): {best_name}", flush=True)

tr, te = train_test_split(np.arange(len(d)), test_size=0.20, random_state=SEED, stratify=y)
pipe = make_pipe(pool, model)
pipe.fit(d.iloc[tr][pool], y[tr])
pi = permutation_importance(pipe, d.iloc[te][pool], y[te],
                            scoring='average_precision', n_repeats=15,
                            random_state=SEED, n_jobs=-1)
imp = pd.DataFrame({'variable': pool, 'imp_media': pi.importances_mean,
                    'imp_std': pi.importances_std}).sort_values('imp_media', ascending=False)
imp.to_csv(f'{BASE}/results_v5_perm_importance.csv', index=False)
print("\nPERMUTATION IMPORTANCE (hold-out 20%):", flush=True)
print(imp.round(4).to_string(index=False), flush=True)

# SHAP
try:
    import shap
    Xt = pipe.named_steps['prep'].transform(d.iloc[te][pool])
    feat_names = pipe.named_steps['prep'].get_feature_names_out()
    if model == 'LR':
        expl = shap.LinearExplainer(pipe.named_steps['clf'], Xt)
        sv = expl.shap_values(Xt)
    else:
        expl = shap.TreeExplainer(pipe.named_steps['clf'])
        sv = expl.shap_values(Xt)
    mean_abs = np.abs(sv).mean(axis=0)
    shap_df = pd.DataFrame({'feature': feat_names, 'mean_abs_shap': mean_abs}) \
        .sort_values('mean_abs_shap', ascending=False)
    shap_df.to_csv(f'{BASE}/results_v5_shap.csv', index=False)
    print("\nTOP 20 SHAP (|valor| medio):", flush=True)
    print(shap_df.head(20).round(4).to_string(index=False), flush=True)
except Exception as e:
    print(f"SHAP falló: {e}", flush=True)

print("\nOK fase 1e", flush=True)
