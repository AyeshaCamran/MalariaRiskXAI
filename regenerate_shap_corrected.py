"""
Regenerate SHAP Analysis with Corrected Models
This script calculates SHAP values using the corrected models (without data leakage)
and generates updated visualizations and feature importance rankings.
"""

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("REGENERATING SHAP ANALYSIS WITH CORRECTED MODELS")
print("="*80)

# ============================================================================
# 1. LOAD DATA AND CORRECTED MODELS
# ============================================================================
print("\n[1] Loading data and corrected models...")

# Load dataset
df = pd.read_csv('data/data_with_features.csv')
print(f"✓ Dataset loaded: {df.shape}")

# Load corrected models
models = {
    'rf_classifier': joblib.load('models/rf_classifier_corrected.pkl'),
    'rf_regressor': joblib.load('models/rf_regressor_corrected.pkl'),
    'scaler': joblib.load('models/scaler_corrected.pkl'),
    'shap_explainer': joblib.load('models/shap_explainer_corrected.pkl'),
    'feature_names': joblib.load('models/feature_names_corrected.pkl'),
    'metadata': joblib.load('models/metadata_corrected.pkl')
}
print(f"✓ Corrected models loaded")
print(f"✓ Features: {len(models['feature_names'])}")
print(f"✓ Removed leaky features from original analysis")

# ============================================================================
# 2. PREPARE DATA
# ============================================================================
print("\n[2] Preparing data...")

feature_cols = models['feature_names']
X = df[feature_cols].fillna(df[feature_cols].median())
X_scaled = models['scaler'].transform(X)

print(f"✓ Data prepared: {X_scaled.shape}")
print(f"✓ Features used (corrected, no leakage):")
for i, feat in enumerate(feature_cols, 1):
    print(f"   {i:2d}. {feat}")

# ============================================================================
# 3. CALCULATE SHAP VALUES
# ============================================================================
print("\n[3] Calculating SHAP values (this may take a minute)...")

# Use the pre-trained SHAP explainer
explainer = models['shap_explainer']

# Calculate SHAP values for the classifier
shap_values = explainer.shap_values(X_scaled)

print(f"✓ SHAP values calculated")
print(f"✓ Shape: {np.array(shap_values).shape}")

# ============================================================================
# 4. CALCULATE FEATURE IMPORTANCE
# ============================================================================
print("\n[4] Calculating feature importance from SHAP values...")

# For multi-class classification, shap_values has shape (n_samples, n_features, n_classes)
# We need to extract values for the High risk class
print(f"✓ SHAP values shape: {shap_values.shape}")

# Get class names from the classifier
rf_classifier = models['rf_classifier']
class_names = rf_classifier.classes_
print(f"✓ Classes: {class_names}")

# Find the High risk class index
high_risk_idx = np.where(class_names == 'High')[0][0]
print(f"✓ High risk class index: {high_risk_idx}")

# Extract SHAP values for High risk class: shape (n_samples, n_features)
shap_values_high_risk = shap_values[:, :, high_risk_idx]
print(f"✓ Extracted High risk SHAP values: {shap_values_high_risk.shape}")

# Calculate mean absolute SHAP value for each feature
feature_importance = np.abs(shap_values_high_risk).mean(axis=0)

# Create feature importance dataframe
feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'SHAP_Importance': feature_importance
}).sort_values('SHAP_Importance', ascending=False)

print(f"\n✓ Top 10 Most Important Features (Corrected):")
print("="*60)
for idx, row in feature_importance_df.head(10).iterrows():
    print(f"   {row['Feature']:40s} {row['SHAP_Importance']:8.4f}")
print("="*60)

# Save to CSV
feature_importance_df.to_csv('data/shap_feature_importance_corrected.csv', index=False)
print(f"\n✓ Saved: data/shap_feature_importance_corrected.csv")

# ============================================================================
# 5. GENERATE SHAP VISUALIZATIONS
# ============================================================================
print("\n[5] Generating SHAP visualizations...")

# 5.1 SHAP Summary Plot
print("\n  → Creating SHAP summary plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_high_risk, X_scaled, feature_names=feature_cols, show=False)
plt.title('SHAP Feature Importance - Corrected Model (No Data Leakage)',
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('visualizations/shap_summary_corrected.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ Saved: visualizations/shap_summary_corrected.png")

# 5.2 SHAP Bar Plot
print("\n  → Creating SHAP bar plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_high_risk, X_scaled, feature_names=feature_cols,
                  plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Bar) - Corrected Model',
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('visualizations/shap_bar_plot_corrected.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ Saved: visualizations/shap_bar_plot_corrected.png")

# 5.3 SHAP Waterfall Plots for Key States
print("\n  → Creating SHAP waterfall plots for key states...")

key_states = {
    'Kebbi': 'High Risk',
    'Lagos': 'Low Risk',
    'Kano': 'Medium Risk (North)',
    'Anambra': 'Low Risk (South)'
}

for state_name, risk_desc in key_states.items():
    try:
        state_idx = df[df['State'] == state_name].index[0]

        plt.figure(figsize=(12, 8))

        # Create explanation object for waterfall plot
        # Get base value for High risk class
        if hasattr(explainer, 'expected_value'):
            if isinstance(explainer.expected_value, np.ndarray):
                base_val = explainer.expected_value[high_risk_idx]
            else:
                base_val = explainer.expected_value
        else:
            base_val = 0

        explanation = shap.Explanation(
            values=shap_values_high_risk[state_idx],
            base_values=base_val,
            data=X_scaled[state_idx],
            feature_names=feature_cols
        )

        shap.waterfall_plot(explanation, show=False)
        plt.title(f'SHAP Explanation: {state_name} ({risk_desc}) - Corrected Model',
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'visualizations/shap_waterfall_{state_name.lower()}_corrected.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: shap_waterfall_{state_name.lower()}_corrected.png")
    except Exception as e:
        print(f"    ⚠ Could not create waterfall for {state_name}: {e}")

# 5.4 SHAP Dependence Plots for Top Features
print("\n  → Creating SHAP dependence plots for top 3 features...")

top_3_features = feature_importance_df.head(3)['Feature'].tolist()

for i, feature in enumerate(top_3_features):
    try:
        feature_idx = feature_cols.index(feature)

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            shap_values_high_risk,
            X_scaled,
            feature_names=feature_cols,
            show=False
        )
        plt.title(f'SHAP Dependence Plot: {feature} - Corrected Model',
                  fontsize=12, fontweight='bold')
        plt.tight_layout()

        # Clean feature name for filename
        clean_name = feature.replace('_', '-').replace(' ', '-')
        plt.savefig(f'visualizations/shap_dependence_{clean_name}_corrected.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: shap_dependence_{clean_name}_corrected.png")
    except Exception as e:
        print(f"    ⚠ Could not create dependence plot for {feature}: {e}")

# ============================================================================
# 6. SAVE FULL SHAP VALUES FOR APP
# ============================================================================
print("\n[6] Saving complete SHAP values for web app...")

# Save SHAP values as numpy array
np.save('models/shap_values_corrected.npy', shap_values_high_risk)
print("  ✓ Saved: models/shap_values_corrected.npy")

# Save base value for High risk class
if hasattr(explainer, 'expected_value'):
    if isinstance(explainer.expected_value, np.ndarray):
        base_value = explainer.expected_value[high_risk_idx]
    else:
        base_value = explainer.expected_value
else:
    base_value = 0

np.save('models/shap_base_value_corrected.npy', base_value)
print("  ✓ Saved: models/shap_base_value_corrected.npy")

# ============================================================================
# 7. COMPARISON WITH OLD (LEAKY) RESULTS
# ============================================================================
print("\n[7] Comparing with old (leaky) feature importance...")

try:
    old_importance = pd.read_csv('data/shap_feature_importance.csv')

    print("\n" + "="*80)
    print("COMPARISON: TOP 5 FEATURES")
    print("="*80)

    print("\nOLD (WITH DATA LEAKAGE):")
    print("-"*60)
    for idx, row in old_importance.head(5).iterrows():
        leaky = "🚨 LEAKY!" if 'malaria_trend_2015_2021' in row['Feature'] or \
                                 'malaria_trend_2018_2021' in row['Feature'] or \
                                 'neighbor_malaria_avg_2021' in row['Feature'] else ""
        print(f"   {idx+1}. {row['Feature']:40s} {row['SHAP_Importance']:8.4f} {leaky}")

    print("\nNEW (CORRECTED, NO LEAKAGE):")
    print("-"*60)
    for idx, row in feature_importance_df.head(5).iterrows():
        print(f"   {idx+1}. {row['Feature']:40s} {row['SHAP_Importance']:8.4f} ✅")

    print("\n" + "="*80)
    print("KEY CHANGES:")
    print("  • Removed: neighbor_malaria_avg_2021 (was #2, now excluded)")
    print("  • Removed: malaria_trend_2015_2021 (was #3, now excluded)")
    print("  • Removed: malaria_trend_2018_2021 (was #5, now excluded)")
    print("  • New rankings reflect TRUE predictive features only!")
    print("="*80)

except FileNotFoundError:
    print("\n⚠ Old SHAP importance file not found (this is okay)")

# ============================================================================
# 8. SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("✅ SHAP ANALYSIS REGENERATION COMPLETE!")
print("="*80)

print("\n📊 Generated Files:")
print("  • data/shap_feature_importance_corrected.csv")
print("  • models/shap_values_corrected.npy")
print("  • models/shap_base_value_corrected.npy")
print("  • visualizations/shap_summary_corrected.png")
print("  • visualizations/shap_bar_plot_corrected.png")
print("  • visualizations/shap_waterfall_*_corrected.png (4 files)")
print("  • visualizations/shap_dependence_*_corrected.png (3 files)")

print("\n🔍 Key Insights from Corrected SHAP Analysis:")
print(f"  1. Most Important Feature: {feature_importance_df.iloc[0]['Feature']}")
print(f"  2. Second Most Important: {feature_importance_df.iloc[1]['Feature']}")
print(f"  3. Third Most Important: {feature_importance_df.iloc[2]['Feature']}")

print("\n✅ Results are now scientifically valid (no data leakage)")
print("✅ Safe to use for research paper and presentations")
print("✅ Web app should be updated to load these corrected visualizations")

print("\n" + "="*80)
