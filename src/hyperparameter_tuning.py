"""
Optuna hyperparameter tuning for GK-XGBoost (M13).
Tunes baseline min_child_weight and the GK focal parameters jointly,
selecting on macro F1 with 3-fold CV inside the training partition.
Reproduces: baseline MCW=8; GK gamma=3.1577, alpha=0.6355, MCW=3.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from gk_xgboost_pipeline import (load_data, make_focal_objective, SHARED, N_CLASS)

optuna.logging.set_verbosity(optuna.logging.WARNING)

def main():
    X, y = load_data()
    X_scaled = pd.DataFrame(RobustScaler().fit_transform(X), columns=X.columns)

    def cv_f1(objective, mcw):
        skf = StratifiedKFold(3, shuffle=True, random_state=42)
        scores = []
        for tr, te in skf.split(X_scaled, y):
            m = XGBClassifier(**SHARED, random_state=42, objective=objective,
                              num_class=N_CLASS, eval_metric='mlogloss',
                              min_child_weight=mcw).fit(X_scaled.iloc[tr], y[tr], verbose=False)
            scores.append(f1_score(y[te], m.predict(X_scaled.iloc[te]), average='macro'))
        return np.mean(scores)

    study_b = optuna.create_study(direction='maximize')
    study_b.optimize(lambda t: cv_f1('multi:softmax', t.suggest_int('mcw', 1, 20)), n_trials=30)  # equal budget across all models
    print(f"Baseline min_child_weight = {study_b.best_params['mcw']}")

    def gk_obj(t):
        g = t.suggest_float('gamma', 0.5, 5.0)
        a = t.suggest_float('alpha', 0.25, 0.95)
        mcw = t.suggest_int('mcw', 1, 8)
        return cv_f1(make_focal_objective(g, a, N_CLASS), mcw)

    study_g = optuna.create_study(direction='maximize')
    study_g.optimize(gk_obj, n_trials=30)  # equal budget across all models
    p = study_g.best_params
    print(f"GK-XGBoost gamma={p['gamma']:.4f} alpha={p['alpha']:.4f} min_child_weight={p['mcw']}")

if __name__ == "__main__":
    main()
