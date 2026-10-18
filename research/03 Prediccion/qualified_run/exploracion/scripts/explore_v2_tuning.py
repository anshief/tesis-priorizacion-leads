"""Fase 1b: tuning de hiperparámetros sobre las mejores familias del grid v1.

Evalúa con OOF anidado simple: RandomizedSearchCV (PR-AUC, 3-fold interno)
dentro de cada fold externo sería muy costoso; en su lugar se tunea con CV
5-fold sobre todo el universo y se reporta el mejor CV score — la validación
final honesta se hace en fase 1e con un hold-out reservado.
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/private/tmp/claude-501/-Users-angieflores-Documents-MEGAPROYECTO-02-K-Means/32058bcb-a5eb-4597-8495-c05b18ef0058/scratchpad')

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              GradientBoostingClassifier)

from explore_common import load_universe, POOLS, oof_metrics, SEED

d = load_universe()
y = d['target_won'].values
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)

def make_prep(pool, linear):
    cat = [c for c in pool if not pd.api.types.is_numeric_dtype(d[c])]
    num = [c for c in pool if c not in cat]
    if linear:
        cat_pipe = Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='_missing_')),
                             ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                                                   min_frequency=30))])
    else:
        cat_pipe = Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='_missing_')),
                             ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    steps = []
    if cat: steps.append(('cat', cat_pipe, cat))
    if num: steps.append(('num', num_pipe, num))
    return ColumnTransformer(steps, remainder='drop')

SEARCHES = {
    'LR': (True, LogisticRegression(class_weight='balanced', max_iter=2000,
                                    solver='saga', random_state=SEED),
           {'clf__C': loguniform(1e-3, 30),
            'clf__penalty': ['l1', 'l2'],
            'clf__class_weight': ['balanced', None, {0: 1, 1: 10}, {0: 1, 1: 25}]}),
    'HGB': (False, HistGradientBoostingClassifier(random_state=SEED),
            {'clf__learning_rate': loguniform(0.01, 0.3),
             'clf__max_iter': randint(100, 600),
             'clf__max_leaf_nodes': randint(7, 63),
             'clf__min_samples_leaf': randint(10, 100),
             'clf__l2_regularization': loguniform(1e-3, 10),
             'clf__max_depth': [None, 3, 5, 8],
             'clf__class_weight': ['balanced', None]}),
    'RF': (False, RandomForestClassifier(n_jobs=-1, random_state=SEED),
           {'clf__n_estimators': randint(200, 800),
            'clf__max_depth': [None, 5, 8, 12, 20],
            'clf__min_samples_leaf': randint(5, 80),
            'clf__max_features': ['sqrt', 0.3, 0.5, 0.8],
            'clf__class_weight': ['balanced', 'balanced_subsample', None]}),
    'EXT': (False, ExtraTreesClassifier(n_jobs=-1, random_state=SEED),
            {'clf__n_estimators': randint(200, 800),
             'clf__max_depth': [None, 5, 8, 12, 20],
             'clf__min_samples_leaf': randint(5, 80),
             'clf__max_features': ['sqrt', 0.3, 0.5, 0.8],
             'clf__class_weight': ['balanced', 'balanced_subsample', None]}),
    'GB': (False, GradientBoostingClassifier(random_state=SEED),
           {'clf__n_estimators': randint(80, 400),
            'clf__learning_rate': loguniform(0.01, 0.3),
            'clf__max_depth': randint(2, 6),
            'clf__subsample': uniform(0.5, 0.5),
            'clf__min_samples_leaf': randint(10, 100)}),
    'LDA': (True, LinearDiscriminantAnalysis(),
            {'clf__solver': ['lsqr', 'eigen'],
             'clf__shrinkage': [None, 'auto', 0.05, 0.1, 0.2, 0.4, 0.6]}),
}

POOLS_TO_TUNE = ['P2_Clean', 'P4_Eng', 'P5_Eng+Proj']
N_ITER = 40

rows = []
best_estimators = {}
t0 = time.time()
for pool_name in POOLS_TO_TUNE:
    pool = [c for c in POOLS[pool_name] if c in d.columns]
    for alg, (linear, est, dist) in SEARCHES.items():
        pipe = Pipeline([('prep', make_prep(pool, linear)), ('clf', est)])
        rs = RandomizedSearchCV(pipe, dist, n_iter=N_ITER, scoring='average_precision',
                                cv=skf, n_jobs=-1, random_state=SEED, refit=True, verbose=0)
        rs.fit(d[pool], y)
        # métricas OOF con la mejor config re-evaluada
        best = rs.best_estimator_
        prob = cross_val_predict(best, d[pool], y, cv=skf, method='predict_proba',
                                 n_jobs=-1)[:, 1]
        m = oof_metrics(y, prob)
        rows.append({'pool': pool_name, 'alg': alg, 'n_vars': len(pool),
                     'cv_pr_auc_search': round(rs.best_score_, 4),
                     **{k: round(v, 4) for k, v in m.items()},
                     'best_params': str(rs.best_params_)})
        key = f"{pool_name}|{alg}"
        best_estimators[key] = rs.best_params_
        print(f"{pool_name:12s} {alg:4s} PR(OOF)={m['pr_auc']:.4f} ROC={m['roc_auc']:.3f} "
              f"%leads(r70)={m['pct_leads_r70']:.1f} lift10={m['lift_10']:.2f} "
              f"| {rs.best_params_}", flush=True)

res = pd.DataFrame(rows).sort_values('pr_auc', ascending=False)
base = '/private/tmp/claude-501/-Users-angieflores-Documents-MEGAPROYECTO-02-K-Means/32058bcb-a5eb-4597-8495-c05b18ef0058/scratchpad'
res.to_csv(f'{base}/results_v2_tuning.csv', index=False)
print(f"\nOK tuning en {time.time()-t0:.0f}s", flush=True)
print(res[['pool','alg','pr_auc','roc_auc','pct_leads_r70','prec_r70','lift_10','lift_20']]
      .to_string(index=False), flush=True)
