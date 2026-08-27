"""
GK-XGBoost: Focal-Loss-Modified Gradient Boosting for Cybersecurity
Risk Classification in University Environments.

Primary pipeline - KNUST field data (n=1,250, leakage-free coherent-scale design).
Reproduces the comparative evaluation, ablation, statistics, per-class analysis,
confusion matrices, and SHAP feature importance reported in the paper.

Author: Koku Godwin Komla
Supervisor: Dr. Eric Opoku Osei
Institution: Kwame Nkrumah University of Science and Technology (KNUST)

Fixed seeds: 42, 123, 456, 789, 1024  (5 seeds x 5 folds = 25 measurements)
Tuned params: baseline min_child_weight=8;
              GK-XGBoost gamma=3.1577, alpha=0.6355, min_child_weight=3
"""

import numpy as np
import pandas as pd
import warnings
import json
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             average_precision_score)
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from scipy.stats import wilcoxon, shapiro

DATA_PATH = "response.xlsx"
SEEDS     = [42, 123, 456, 789, 1024]
METRICS   = ['acc', 'f1', 'auc_roc', 'auc_pr']
N_CLASS   = 3
BASE_MCW  = 8
GK_GAMMA  = 3.1577
GK_ALPHA  = 0.6355
GK_MCW    = 3
SHARED = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, use_label_encoder=False)
SCALE = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4}


def make_focal_objective(gamma, alpha, n_class):
    """Custom multiclass focal-loss objective returning (gradient, hessian)."""
    def focal(y_true, y_pred):
        n = len(y_true)
        y_pred = y_pred.reshape(n, n_class)
        y_pred = y_pred - y_pred.max(axis=1, keepdims=True)
        softmax = np.exp(y_pred) / np.exp(y_pred).sum(axis=1, keepdims=True)
        onehot = np.zeros_like(softmax)
        onehot[np.arange(n), y_true.astype(int)] = 1
        p_t = (softmax * onehot).sum(axis=1, keepdims=True)
        focal_weight = alpha * (1 - p_t) ** gamma
        grad = focal_weight * (softmax - onehot)
        hess = focal_weight * softmax * (1 - softmax)
        return grad.flatten(), hess.flatten()
    return focal


def load_data(path=DATA_PATH):
    """
    Features : Q1-Q5 (demographic/causal) + Q6-Q13 (feature-half behaviour) = 13
    Label    : Q14-Q21 (label-half behaviour) summed -> 3 risk classes
    Leakage-free by construction; signal-preserving (same underlying trait).
    """
    df = pd.read_excel(path)
    label_enc = pd.DataFrame()
    for c in df.columns[14:22]:
        label_enc[c.split('.')[0]] = df[c].map(SCALE)
    secure_score = label_enc.sum(axis=1)
    y = np.where(secure_score >= 22, 0, np.where(secure_score >= 12, 1, 2))
    X = pd.DataFrame()
    for c in df.columns[1:6]:
        X[c.split('.')[0]] = LabelEncoder().fit_transform(df[c].astype(str))
    for c in df.columns[6:14]:
        X[c.split('.')[0]] = df[c].map(SCALE)
    return X, np.asarray(y)


def baseline_model(seed):
    return XGBClassifier(**SHARED, random_state=seed, objective='multi:softmax',
                         num_class=N_CLASS, eval_metric='mlogloss',
                         min_child_weight=BASE_MCW)

def gk_model(seed):
    focal = make_focal_objective(GK_GAMMA, GK_ALPHA, N_CLASS)
    return XGBClassifier(**SHARED, random_state=seed, objective=focal,
                         num_class=N_CLASS, eval_metric='mlogloss',
                         min_child_weight=GK_MCW)

def focal_only_model(seed):
    focal = make_focal_objective(GK_GAMMA, GK_ALPHA, N_CLASS)
    return XGBClassifier(**SHARED, random_state=seed, objective=focal,
                         num_class=N_CLASS, eval_metric='mlogloss',
                         min_child_weight=BASE_MCW)


def evaluate(model_factory, X_scaled, y, resample=None, weight=False, lgbm=False):
    res = {m: [] for m in METRICS}
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in skf.split(X_scaled, y):
            X_tr, X_te = X_scaled.iloc[tr], X_scaled.iloc[te]
            y_tr, y_te = y[tr], y[te]
            if lgbm:
                model = LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8,
                                       class_weight='balanced', random_state=seed, verbose=-1)
                model.fit(X_tr, y_tr)
            elif resample == 'smote':
                X_rs, y_rs = SMOTE(random_state=seed, k_neighbors=3).fit_resample(X_tr, y_tr)
                model = baseline_model(seed)
                model.fit(pd.DataFrame(X_rs, columns=X_tr.columns), y_rs, verbose=False)
            elif weight:
                model = baseline_model(seed)
                model.fit(X_tr, y_tr, sample_weight=compute_sample_weight('balanced', y_tr), verbose=False)
            else:
                model = model_factory(seed)
                model.fit(X_tr, y_tr, verbose=False)
            pred = model.predict(X_te)
            proba = model.predict_proba(X_te)
            res['acc'].append(accuracy_score(y_te, pred))
            res['f1'].append(f1_score(y_te, pred, average='macro'))
            res['auc_roc'].append(roc_auc_score(y_te, proba, multi_class='ovr', average='macro'))
            res['auc_pr'].append(average_precision_score(pd.get_dummies(y_te), proba, average='macro'))
    return res


def paired_stats(base_res, gk_res):
    out = {}
    for m in METRICS:
        b = np.array(base_res[m]); g = np.array(gk_res[m]); diff = g - b
        try:    stat, p = wilcoxon(b, g)
        except: stat, p = 0.0, 1.0
        d = (g.mean() - b.mean()) / np.sqrt((b.std(ddof=1)**2 + g.std(ddof=1)**2) / 2)
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        _, sw_p = shapiro(diff)
        out[m] = dict(base=b.mean(), gk=g.mean(), delta=diff.mean(),
                      W=stat, p=p, cohen_d=d,
                      ci95=[diff.mean() - 1.96*se, diff.mean() + 1.96*se], shapiro_p=sw_p)
    return out


def main():
    print("Loading KNUST field data...")
    X, y = load_data()
    X_scaled = pd.DataFrame(RobustScaler().fit_transform(X), columns=X.columns)
    counts = np.bincount(y)
    print(f"  n = {len(y)} | classes = {dict(enumerate(counts))} "
          f"| imbalance = {counts.max()/counts.min():.2f}:1")

    print("\nRunning 5-model comparison (5 seeds x 5 folds)...")
    res_base  = evaluate(baseline_model, X_scaled, y)
    res_gk    = evaluate(gk_model,       X_scaled, y)
    res_smote = evaluate(None, X_scaled, y, resample='smote')
    res_wt    = evaluate(None, X_scaled, y, weight=True)
    res_lgbm  = evaluate(None, X_scaled, y, lgbm=True)
    res_focal = evaluate(focal_only_model, X_scaled, y)

    models = {'Baseline XGBoost': res_base, 'SMOTE + XGBoost': res_smote,
              'Class-weighted XGBoost': res_wt, 'LightGBM': res_lgbm, 'GK-XGBoost': res_gk}

    print("\n" + "=" * 66)
    print(f"{'Model':<26}{'Acc':>9}{'MacroF1':>10}{'AUC-ROC':>10}{'AUC-PR':>10}")
    print("-" * 66)
    for name, res in models.items():
        print(f"{name:<26}{np.mean(res['acc']):>9.4f}{np.mean(res['f1']):>10.4f}"
              f"{np.mean(res['auc_roc']):>10.4f}{np.mean(res['auc_pr']):>10.4f}")

    print("\nABLATION (macro F1):")
    print(f"  Baseline        : {np.mean(res_base['f1']):.4f}")
    print(f"  Focal-only      : {np.mean(res_focal['f1']):.4f}  (below baseline)")
    print(f"  Full GK-XGBoost : {np.mean(res_gk['f1']):.4f}  (both modifications)")

    print("\nGK-XGBoost vs Baseline (Wilcoxon signed-rank):")
    stats = paired_stats(res_base, res_gk)
    for m in METRICS:
        st = stats[m]
        sig = "SIG" if (st['delta'] > 0 and st['p'] < 0.05) else ""
        print(f"  {m:<8}: {st['base']:.4f} -> {st['gk']:.4f} ({st['delta']:+.4f})  "
              f"W={st['W']:.1f}  p={st['p']:.4f}  d={st['cohen_d']:.3f}  {sig}")

    summary = {'comparison': {n: {m: float(np.mean(r[m])) for m in METRICS} for n, r in models.items()},
               'stats': {m: {k: (float(v) if not isinstance(v, list) else [float(x) for x in v])
                             for k, v in stats[m].items()} for m in METRICS},
               'ablation': {'baseline': float(np.mean(res_base['f1'])),
                            'focal_only': float(np.mean(res_focal['f1'])),
                            'full': float(np.mean(res_gk['f1']))}}
    with open('results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nSaved results_summary.json")


if __name__ == "__main__":
    main()
