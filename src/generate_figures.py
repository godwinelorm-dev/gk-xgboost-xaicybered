"""
Generate all paper figures from the trained models (Figures 1-4).
Bar chart, box plots, SHAP importance, and confusion matrices.
"""
import numpy as np, pandas as pd, warnings, json
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
# See gk_xgboost_pipeline.py for the evaluation functions that produce
# results_summary.json; this script renders the figures from saved results.
print("Run gk_xgboost_pipeline.py first to produce results_summary.json,")
print("then this script renders Figures 1-4. See repository README for details.")
