"""
SHAP interpretability for GK-XGBoost (RO2, RO3).
Computes mean absolute SHAP values for the High-Risk class and designs
three role-differentiated explanation views (student, educator, IT admin).
"""
import numpy as np, pandas as pd, warnings, json
warnings.filterwarnings('ignore')
import shap
from sklearn.preprocessing import LabelEncoder, RobustScaler
from gk_xgboost_pipeline import (load_data, baseline_model, gk_model, SCALE)

def mean_abs_shap(values, n_feat):
    a = np.array(values)
    return (np.abs(a).mean(axis=(0, 2)) if a.ndim == 3 else np.abs(a).mean(axis=0))[:n_feat]

def main():
    X, y = load_data()
    X_scaled = pd.DataFrame(RobustScaler().fit_transform(X), columns=X.columns)

    base = baseline_model(42).fit(X_scaled, y, verbose=False)
    gk   = gk_model(42).fit(X_scaled, y, verbose=False)

    shap_base = mean_abs_shap(shap.TreeExplainer(base).shap_values(X_scaled), X.shape[1])
    shap_gk   = mean_abs_shap(shap.TreeExplainer(gk).shap_values(X_scaled), X.shape[1])

    table = pd.DataFrame({'Feature': list(X.columns),
                          'Baseline': shap_base, 'GK_XGBoost': shap_gk})
    table = table.sort_values('GK_XGBoost', ascending=False)
    table.to_csv('shap_importance.csv', index=False)
    print(table.to_string(index=False))

    ranked = table['Feature'].tolist()
    # Role views differ in WHICH attributions are surfaced and at WHAT granularity.
    role_views = {
        'student':  {'granularity': 'individual, top-3 personal drivers',
                     'surfaced': ranked[:3],
                     'message': 'The three habits most raising your personal risk, with a fix for each.'},
        'educator': {'granularity': 'cohort mean over behaviour features only',
                     'surfaced': [f for f in ranked if f.startswith('Q')][:5],
                     'message': 'Behaviour patterns aggregated across a class group; demographics suppressed.'},
        'it_admin': {'granularity': 'population-level, all 13 features ranked',
                     'surfaced': ranked,
                     'message': 'Full ranked driver list for prioritising institution-wide campaigns.'}}
    with open('role_views.json', 'w') as f:
        json.dump(role_views, f, indent=2)
    print("\nSaved shap_importance.csv and role_views.json")
    print("NOTE: role views are a design artefact; not empirically user-evaluated (future work).")

if __name__ == "__main__":
    main()
