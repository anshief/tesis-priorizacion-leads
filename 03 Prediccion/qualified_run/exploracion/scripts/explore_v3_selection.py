"""Fase 1c: estrategias de selección de variables sobre el pool ingenierizado.

1. Lift univariado / mutual information por variable.
2. Path de regularización L1 (qué variables sobreviven según C).
3. Forward selection greedy (variable a variable, CV 5-fold, PR-AUC).
4. Poda hacia atrás desde el pool completo.
"""
import sys, time, warnings
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
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score

from explore_common import load_universe, POOLS, oof_metrics, SEED

BASE = '/private/tmp/claude-501/-Users-angieflores-Documents-MEGAPROYECTO-02-K-Means/32058bcb-a5eb-4597-8495-c05b18ef0058/scratchpad'
d = load_universe()
y = d['target_won'].values
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
POOL = [c for c in POOLS['P4_Eng'] if c in d.columns]


def make_pipe(pool, C=1.0, penalty='l2'):
    cat = [c for c in pool if not pd.api.types.is_numeric_dtype(d[c])]
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
    solver = 'saga' if penalty == 'l1' else 'lbfgs'
    return Pipeline([('prep', prep),
                     ('clf', LogisticRegression(class_weight='balanced', C=C,
                                                penalty=penalty, solver=solver,
                                                max_iter=2000, random_state=SEED))])


def cv_pr(pool, C=1.0, penalty='l2'):
    prob = cross_val_predict(make_pipe(pool, C, penalty), d[pool], y, cv=skf,
                             method='predict_proba', n_jobs=-1)[:, 1]
    return average_precision_score(y, prob), prob


# ── 1. Mutual information + lift univariado (cada variable sola en un LR) ──
print("=" * 70, flush=True)
print("1. SEÑAL UNIVARIADA (LR con una sola variable, PR-AUC OOF)", flush=True)
uni_rows = []
for v in POOL:
    try:
        pr, _ = cv_pr([v])
        uni_rows.append({'variable': v, 'pr_auc_solo': round(pr, 4)})
        print(f"  {v:24s} PR={pr:.4f}", flush=True)
    except Exception as e:
        print(f"  {v:24s} ERR {e}", flush=True)
uni = pd.DataFrame(uni_rows).sort_values('pr_auc_solo', ascending=False)
uni.to_csv(f'{BASE}/results_v3_univariate.csv', index=False)

# ── 2. Path L1 ──
print("\n" + "=" * 70, flush=True)
print("2. PATH L1 — variables que sobreviven según C", flush=True)
pipe_full = make_pipe(POOL, C=1.0, penalty='l1')
for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
    p = make_pipe(POOL, C=C, penalty='l1')
    p.fit(d[POOL], y)
    names = p.named_steps['prep'].get_feature_names_out()
    coefs = p.named_steps['clf'].coef_[0]
    surv = [(n, round(c, 3)) for n, c in zip(names, coefs) if abs(c) > 1e-6]
    surv.sort(key=lambda t: -abs(t[1]))
    print(f"  C={C:6.3f}: {len(surv):3d} features | top: {surv[:8]}", flush=True)

# ── 3. Forward selection greedy ──
print("\n" + "=" * 70, flush=True)
print("3. FORWARD SELECTION (greedy, PR-AUC OOF)", flush=True)
remaining = list(POOL)
selected = []
best_hist = []
current_best = 0.0
for step in range(min(10, len(POOL))):
    scores = []
    for v in remaining:
        try:
            pr, _ = cv_pr(selected + [v])
            scores.append((pr, v))
        except Exception:
            pass
    scores.sort(reverse=True)
    pr_best, v_best = scores[0]
    gain = pr_best - current_best
    print(f"  paso {step+1}: +{v_best:24s} PR={pr_best:.4f} (ganancia {gain:+.4f})", flush=True)
    best_hist.append({'paso': step + 1, 'variable': v_best,
                      'pr_auc': round(pr_best, 4), 'ganancia': round(gain, 4)})
    if gain < 0.002 and step >= 2:
        print("  ganancia marginal < 0.002 — se detiene la selección", flush=True)
        break
    selected.append(v_best)
    remaining.remove(v_best)
    current_best = pr_best
pd.DataFrame(best_hist).to_csv(f'{BASE}/results_v3_forward.csv', index=False)
print(f"  SELECCIONADAS: {selected}", flush=True)

# ── 4. Poda hacia atrás desde el pool completo ──
print("\n" + "=" * 70, flush=True)
print("4. BACKWARD ELIMINATION (desde P4_Eng completo)", flush=True)
pool_bw = list(POOL)
pr_full, _ = cv_pr(pool_bw)
print(f"  pool completo ({len(pool_bw)} vars): PR={pr_full:.4f}", flush=True)
improved = True
while improved and len(pool_bw) > 3:
    improved = False
    scores = []
    for v in pool_bw:
        cand = [c for c in pool_bw if c != v]
        try:
            pr, _ = cv_pr(cand)
            scores.append((pr, v))
        except Exception:
            pass
    scores.sort(reverse=True)
    pr_best, v_drop = scores[0]
    if pr_best >= pr_full - 0.0005:  # quitar si no empeora (tolerancia)
        pool_bw.remove(v_drop)
        pr_full = pr_best
        improved = True
        print(f"  - quita {v_drop:24s} → PR={pr_best:.4f} ({len(pool_bw)} vars)", flush=True)
print(f"  POOL FINAL BACKWARD ({len(pool_bw)}): {pool_bw} | PR={pr_full:.4f}", flush=True)
pd.Series(pool_bw).to_csv(f'{BASE}/results_v3_backward_pool.csv', index=False)
print("\nOK fase 1c", flush=True)
