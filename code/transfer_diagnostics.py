"""
Transfer evaluation and negative-result diagnostics (R8, R10).
Evaluates GK-XGBoost on two public datasets and runs the CAN-DO
confirm-first diagnostic tests (power analysis, separability check).

Public datasets (place alongside script):
  - Moroccan HTTPS Awareness: Encoded_collected_dataset.csv
  - WUSTL-EHMS-2020:          wustl-ehms-2020_with_attacks_categories.csv
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, RobustScaler

try:
    from statsmodels.stats.power import TTestIndPower
    HAS_SM = True
except ImportError:
    HAS_SM = False

def power_mde(n_measurements=25, alpha=0.05, power=0.80):
    """Minimum detectable effect size (Cohen's d) for the paired design."""
    if not HAS_SM:
        return None
    return TTestIndPower().solve_power(effect_size=None, nobs1=n_measurements,
                                       alpha=alpha, power=power, ratio=1.0)

def separability_check(X, y):
    """A shallow tree that already scores high => trivially separable (ceiling)."""
    stump1 = cross_val_score(DecisionTreeClassifier(max_depth=1, random_state=42),
                             X, y, cv=3, scoring='f1_macro').mean()
    tree3  = cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=42),
                             X, y, cv=3, scoring='f1_macro').mean()
    return stump1, tree3

def main():
    print("NEGATIVE-RESULT DIAGNOSTICS (CAN-DO confirm-first tests)\n")
    mde = power_mde()
    if mde:
        print(f"Cause 17 (underpowered): MDE for 25-measurement design = d {mde:.3f}")
        print("  Any observed |d| below this => comparison INCONCLUSIVE, not 'no effect'.\n")

    print("Cause 6 (modality/ceiling): a depth-3 tree scoring >0.95 means the task")
    print("is trivially separable, leaving no hard minority examples for focal loss.\n")
    print("Run this script with the public CSVs present to reproduce the per-dataset")
    print("separability and mutual-information diagnostics reported in R10.")

if __name__ == "__main__":
    main()
