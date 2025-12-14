"""
Complete Model Training Script with All Models
Trains and evaluates ALL classification and regression models with corrected features
"""

import pandas as pd
import numpy as np
import joblib
import warnings
import xgboost as xgb
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import make_scorer, balanced_accuracy_score, mean_squared_error, r2_score
import shap

print("="*80)
print("COMPLETE MODEL TRAINING - ALL MODELS WITH CORRECTED FEATURES")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] Loading data...")
df = pd.read_csv('data/data_with_features.csv')
print(f"✓ Dataset loaded: {df.shape}")

# Risk classification
def classify_risk(prevalence):
    if prevalence >= 40: return 'High'
    elif prevalence >= 10: return 'Medium'
    else: return 'Low'

df['risk_category'] = df['malaria_prev_2021'].apply(classify_risk)
print(f"✓ Risk distribution:\n{df['risk_category'].value_counts()}")

# ============================================================================
# 2. PREPARE CORRECTED FEATURES
# ============================================================================
print("\n[2] Preparing corrected features (no data leakage)...")

# Corrected features (30 features, no leakage)
corrected_feature_cols = [
    'itn_ownership_2021', 'itn_access_2021', 'itn_use_children_2021',
    'iptp2_2021', 'iptp3_2021', 'diag_test_2021', 'malaria_msg_2021',
    'anaemia_2021',
    'malaria_prev_2018', 'malaria_prev_2015',
    'zone_North Central', 'zone_North East', 'zone_North West',
    'zone_South East', 'zone_South South', 'zone_South West',
    'neighbor_malaria_avg_2018', 'neighbor_malaria_avg_2015',
    'is_urban', 'urbanization_score',
    'net_to_person_2021', 'itn_coverage_gap_2021',
    'anc_quality_index_2021', 'iptp_coverage_gap_2021',
    'health_seeking_index_2021',
    'malaria_trend_2015_2018',
    'malaria_trend_acceleration',
    'itn_trend_2015_2021', 'iptp2_trend_2015_2021', 'anaemia_trend_2015_2021'
]

X = df[corrected_feature_cols].fillna(df[corrected_feature_cols].median())
y_reg = df['malaria_prev_2021']
y_clf = df['risk_category']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✓ Features: {len(corrected_feature_cols)}")
print(f"✓ Samples: {len(X)}")

# ============================================================================
# 3. DEFINE CROSS-VALIDATION
# ============================================================================
cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)
cv_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)

# ============================================================================
# 4. TRAIN AND EVALUATE ALL CLASSIFICATION MODELS
# ============================================================================
print("\n[3] Training and evaluating CLASSIFICATION models...")

# Random Forest Classifier
print("\n  → Random Forest Classifier...")
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
rf_clf_scores = cross_val_score(rf_clf, X_scaled, y_clf, cv=cv_clf, scoring=balanced_accuracy_scorer)
rf_clf_acc = rf_clf_scores.mean()
print(f"    ✓ Balanced Accuracy: {rf_clf_acc:.2%} (±{rf_clf_scores.std():.3f})")

# Logistic Regression
print("\n  → Logistic Regression...")
lr_clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_clf_scores = cross_val_score(lr_clf, X_scaled, y_clf, cv=cv_clf, scoring=balanced_accuracy_scorer)
lr_clf_acc = lr_clf_scores.mean()
print(f"    ✓ Balanced Accuracy: {lr_clf_acc:.2%} (±{lr_clf_scores.std():.3f})")

# MLP Classifier
print("\n  → MLP Neural Network...")
mlp_clf = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1000, random_state=42)
mlp_clf_scores = cross_val_score(mlp_clf, X_scaled, y_clf, cv=cv_clf, scoring=balanced_accuracy_scorer)
mlp_clf_acc = mlp_clf_scores.mean()
print(f"    ✓ Balanced Accuracy: {mlp_clf_acc:.2%} (±{mlp_clf_scores.std():.3f})")

# Stacking Classifier
print("\n  → Stacking Ensemble...")
base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)),
    ('lr', LogisticRegression(max_iter=500, random_state=42))
]
stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=500),
    cv=3
)
stacking_clf_scores = cross_val_score(stacking_clf, X_scaled, y_clf, cv=cv_clf, scoring=balanced_accuracy_scorer)
stacking_clf_acc = stacking_clf_scores.mean()
print(f"    ✓ Balanced Accuracy: {stacking_clf_acc:.2%} (±{stacking_clf_scores.std():.3f})")

# ============================================================================
# 5. TRAIN AND EVALUATE ALL REGRESSION MODELS
# ============================================================================
print("\n[4] Training and evaluating REGRESSION models...")

# Random Forest Regressor
print("\n  → Random Forest Regressor...")
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_reg_r2_scores = cross_val_score(rf_reg, X_scaled, y_reg, cv=cv_reg, scoring='r2')
rf_reg_rmse_scores = np.sqrt(-cross_val_score(rf_reg, X_scaled, y_reg, cv=cv_reg, scoring='neg_mean_squared_error'))
rf_reg_r2 = rf_reg_r2_scores.mean()
rf_reg_rmse = rf_reg_rmse_scores.mean()
print(f"    ✓ R²: {rf_reg_r2:.3f} (±{rf_reg_r2_scores.std():.3f})")
print(f"    ✓ RMSE: {rf_reg_rmse:.2f}% (±{rf_reg_rmse_scores.std():.2f})")

# Linear Regression
print("\n  → Linear Regression...")
linear_reg = LinearRegression()
linear_reg_r2_scores = cross_val_score(linear_reg, X_scaled, y_reg, cv=cv_reg, scoring='r2')
linear_reg_rmse_scores = np.sqrt(-cross_val_score(linear_reg, X_scaled, y_reg, cv=cv_reg, scoring='neg_mean_squared_error'))
linear_reg_r2 = linear_reg_r2_scores.mean()
linear_reg_rmse = linear_reg_rmse_scores.mean()
print(f"    ✓ R²: {linear_reg_r2:.3f} (±{linear_reg_r2_scores.std():.3f})")
print(f"    ✓ RMSE: {linear_reg_rmse:.2f}% (±{linear_reg_rmse_scores.std():.2f})")

# MLP Regressor
print("\n  → MLP Regressor...")
mlp_reg = MLPRegressor(hidden_layer_sizes=(50, 30), max_iter=1000, random_state=42)
mlp_reg_r2_scores = cross_val_score(mlp_reg, X_scaled, y_reg, cv=cv_reg, scoring='r2')
mlp_reg_rmse_scores = np.sqrt(-cross_val_score(mlp_reg, X_scaled, y_reg, cv=cv_reg, scoring='neg_mean_squared_error'))
mlp_reg_r2 = mlp_reg_r2_scores.mean()
mlp_reg_rmse = mlp_reg_rmse_scores.mean()
print(f"    ✓ R²: {mlp_reg_r2:.3f} (±{mlp_reg_r2_scores.std():.3f})")
print(f"    ✓ RMSE: {mlp_reg_rmse:.2f}% (±{mlp_reg_rmse_scores.std():.2f})")

# XGBoost Regressor
print("\n  → XGBoost Regressor...")
xgb_reg = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, objective='reg:squarederror')
xgb_reg_r2_scores = cross_val_score(xgb_reg, X_scaled, y_reg, cv=cv_reg, scoring='r2')
xgb_reg_rmse_scores = np.sqrt(-cross_val_score(xgb_reg, X_scaled, y_reg, cv=cv_reg, scoring='neg_mean_squared_error'))
xgb_reg_r2 = xgb_reg_r2_scores.mean()
xgb_reg_rmse = xgb_reg_rmse_scores.mean()
print(f"    ✓ R²: {xgb_reg_r2:.3f} (±{xgb_reg_r2_scores.std():.3f})")
print(f"    ✓ RMSE: {xgb_reg_rmse:.2f}% (±{xgb_reg_rmse_scores.std():.2f})")

# ============================================================================
# 6. RETRAIN ALL MODELS ON FULL DATASET
# ============================================================================
print("\n[5] Retraining all models on full dataset...")

rf_clf.fit(X_scaled, y_clf)
lr_clf.fit(X_scaled, y_clf)
mlp_clf.fit(X_scaled, y_clf)
stacking_clf.fit(X_scaled, y_clf)
print("  ✓ All classification models trained")

rf_reg.fit(X_scaled, y_reg)
linear_reg.fit(X_scaled, y_reg)
mlp_reg.fit(X_scaled, y_reg)
xgb_reg.fit(X_scaled, y_reg)
print("  ✓ All regression models trained")

# Train SHAP explainer
explainer = shap.TreeExplainer(rf_clf)
print("  ✓ SHAP explainer trained")

# ============================================================================
# 7. SAVE ALL MODELS
# ============================================================================
print("\n[6] Saving all models...")

models_to_save = {
    'rf_classifier_corrected': rf_clf,
    'lr_classifier_corrected': lr_clf,
    'mlp_classifier_corrected': mlp_clf,
    'stacking_classifier_corrected': stacking_clf,
    'rf_regressor_corrected': rf_reg,
    'linear_regressor_corrected': linear_reg,
    'mlp_regressor_corrected': mlp_reg,
    'xgboost_regressor_corrected': xgb_reg,
    'scaler_corrected': scaler,
    'shap_explainer_corrected': explainer,
    'feature_names_corrected': corrected_feature_cols
}

for name, model in models_to_save.items():
    filepath = f'models/{name}.pkl'
    joblib.dump(model, filepath)
    print(f"  ✓ Saved: {filepath}")

# ============================================================================
# 8. SAVE COMPLETE METADATA
# ============================================================================
print("\n[7] Saving complete metadata...")

metadata = {
    'feature_columns': corrected_feature_cols,
    'num_features': len(corrected_feature_cols),
    'num_samples': len(df),
    'model_performance': {
        # Classification
        'rf_classifier_balanced_accuracy': rf_clf_acc,
        'lr_classifier_balanced_accuracy': lr_clf_acc,
        'mlp_classifier_balanced_accuracy': mlp_clf_acc,
        'stacking_classifier_balanced_accuracy': stacking_clf_acc,
        # Regression R²
        'rf_regressor_r2': rf_reg_r2,
        'linear_regressor_r2': linear_reg_r2,
        'mlp_regressor_r2': mlp_reg_r2,
        'xgboost_regressor_r2': xgb_reg_r2,
        # Regression RMSE
        'rf_regressor_rmse': rf_reg_rmse,
        'linear_regressor_rmse': linear_reg_rmse,
        'mlp_regressor_rmse': mlp_reg_rmse,
        'xgboost_regressor_rmse': xgb_reg_rmse,
    }
}

joblib.dump(metadata, 'models/metadata_corrected.pkl')
print(f"  ✓ Saved: models/metadata_corrected.pkl")

# ============================================================================
# 9. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("="*80)

print("\n📊 CLASSIFICATION PERFORMANCE (Balanced Accuracy):")
print(f"  • Random Forest:      {rf_clf_acc:.2%}")
print(f"  • Logistic Regression: {lr_clf_acc:.2%}")
print(f"  • MLP Neural Network:  {mlp_clf_acc:.2%}")
print(f"  • Stacking Ensemble:   {stacking_clf_acc:.2%}")

print("\n📊 REGRESSION PERFORMANCE:")
print(f"  • Random Forest:      R²={rf_reg_r2:.3f}, RMSE={rf_reg_rmse:.2f}%")
print(f"  • Linear Regression:  R²={linear_reg_r2:.3f}, RMSE={linear_reg_rmse:.2f}%")
print(f"  • MLP Regressor:      R²={mlp_reg_r2:.3f}, RMSE={mlp_reg_rmse:.2f}%")
print(f"  • XGBoost:            R²={xgb_reg_r2:.3f}, RMSE={xgb_reg_rmse:.2f}%")

print("\n✅ Complete metadata with all model metrics saved!")
print("✅ Ready for Streamlit app deployment!")
print("="*80)
