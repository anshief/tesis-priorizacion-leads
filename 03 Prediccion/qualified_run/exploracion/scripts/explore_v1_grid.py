"""Fase 1a: grid amplio — pools × algoritmos × balanceo, CV 5-fold OOF."""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/private/tmp/claude-501/-Users-angieflores-Documents-MEGAPROYECTO-02-K-Means/32058bcb-a5eb-4597-8495-c05b18ef0058/scratchpad')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from explore_common import load_universe, POOLS, oof_metrics, SEED

d = load_universe()
y = d['target_won'].values
print(f"Universo: {len(d):,} filas, {y.sum()} conversiones ({y.mean()*100:.2f}%)", flush=True)

ALGS = {
    'LR_l2':   lambda: LogisticRegression(class_weight='balanced', max_iter=500, C=1.0, random_state=SEED),
    'LR_l1':   lambda: LogisticRegression(class_weight='balanced', max_iter=500, C=1.0,
                                          penalty='l1', solver='saga', random_state=SEED),
    'RF':      lambda: RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                              min_samples_leaf=20, n_jobs=-1, random_state=SEED),
    'EXT':     lambda: ExtraTreesClassifier(n_estimators=300, class_weight='balanced',
                                            min_samples_leaf=20, n_jobs=-1, random_state=SEED),
    'HGB':     lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                      max_leaf_nodes=31, min_samples_leaf=30,
                                                      class_weight='balanced', random_state=SEED),
    'GB':      lambda: GradientBoostingClassifier(n_estimators=150, learning_rate=0.05,
                                                  max_depth=3, subsample=0.8, random_state=SEED),
    'ADA':     lambda: AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=SEED),
    'NB':      lambda: GaussianNB(),
    'LDA':     lambda: LinearDiscriminantAnalysis(),
}

BALANCE = {
    'none':      None,
    'smote10':   ('smote', 0.10),
    'smote25':   ('smote', 0.25),
    'under50':   ('under', 0.50),
    'over25':    ('over', 0.25),
}


def build(pool, alg_name, alg_factory, balance):
    cat = [c for c in pool if not pd.api.types.is_numeric_dtype(d[c])]
    num = [c for c in pool if c not in cat]
    linear = alg_name.startswith(('LR', 'NB', 'LDA'))
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
    prep = ColumnTransformer(steps, remainder='drop')
    if balance is None:
        return Pipeline([('prep', prep), ('clf', alg_factory())])
    kind, strat = balance
    if kind == 'smote':
        sampler = SMOTE(sampling_strategy=strat, random_state=SEED, k_neighbors=5)
    elif kind == 'under':
        sampler = RandomUnderSampler(sampling_strategy=strat, random_state=SEED)
    else:
        sampler = RandomOverSampler(sampling_strategy=strat, random_state=SEED)
    return ImbPipeline([('prep', prep), ('bal', sampler), ('clf', alg_factory())])


skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows = []
t0 = time.time()
# Balanceos alternativos solo en pools grandes para acotar el grid
GRID = []
for pool_name in POOLS:
    for alg in ALGS:
        GRID.append((pool_name, alg, 'none'))
for pool_name in ['P2_Clean', 'P4_Eng', 'P5_Eng+Proj']:
    for alg in ['LR_l2', 'HGB', 'RF', 'GB']:
        for bal in ['smote10', 'smote25', 'under50', 'over25']:
            GRID.append((pool_name, alg, bal))

print(f"Configs totales: {len(GRID)}", flush=True)
for gi, (pool_name, alg, bal) in enumerate(GRID):
    pool = [c for c in POOLS[pool_name] if c in d.columns]
    try:
        pipe = build(pool, alg, ALGS[alg], BALANCE[bal])
        prob = np.zeros(len(d))
        for tr, te in skf.split(d, y):
            pipe.fit(d.iloc[tr][pool], y[tr])
            prob[te] = pipe.predict_proba(d.iloc[te][pool])[:, 1]
        m = oof_metrics(y, prob)
        rows.append({'pool': pool_name, 'alg': alg, 'balance': bal,
                     'n_vars': len(pool), **{k: round(v, 4) for k, v in m.items()}})
        print(f"[{gi+1}/{len(GRID)}] {pool_name:14s} {alg:6s} {bal:8s} "
              f"PR={m['pr_auc']:.4f} ROC={m['roc_auc']:.3f} "
              f"%leads(r70)={m['pct_leads_r70']:.1f} lift10={m['lift_10']:.2f}", flush=True)
    except Exception as e:
        print(f"[{gi+1}/{len(GRID)}] {pool_name} {alg} {bal} ERR {e}", flush=True)

res = pd.DataFrame(rows).sort_values('pr_auc', ascending=False)
out = '/private/tmp/claude-501/-Users-angieflores-Documents-MEGAPROYECTO-02-K-Means/32058bcb-a5eb-4597-8495-c05b18ef0058/scratchpad/results_v1_grid.csv'
res.to_csv(out, index=False)
print(f"\nOK {len(res)} configs en {time.time()-t0:.0f}s -> {out}", flush=True)
print("\nTOP 20 por PR-AUC (OOF 5-fold):", flush=True)
print(res.head(20).to_string(index=False), flush=True)
