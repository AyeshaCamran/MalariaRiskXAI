"""
generate_visualizations.py

This script generates professional, publication-ready visualizations for the
MalariaRiskXAI project using the corrected models.
"""

import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import shap

# ============================================================================
# 0. STYLE CONFIGURATION
# ============================================================================
# Use a clean, academic style
plt.style.use('default')
sns.set_style("white")

# Fonts
FONT_FAMILY = "DejaVu Sans" # A common sans-serif alternative to Arial
TITLE_FONT_SIZE = 14
AXIS_FONT_SIZE = 12

# Palettes
CONTINUOUS_PALETTE = "magma"
CATEGORICAL_PALETTE = {"Low": "#3498DB", "Medium": "#F1C40F", "High": "#E74C3C"}
SINGLE_COLOR_GRADIENT = "Blues"

# ============================================================================
# 1. LOAD DATA AND MODELS
# ============================================================================
print("[1] Loading data and corrected models...")

# Load data
df = pd.read_csv('data/data_with_features.csv')
def classify_risk(prevalence):
    if prevalence >= 40: return 'High'
    if prevalence >= 10: return 'Medium'
    return 'Low'
df['risk_category'] = df['malaria_prev_2021'].apply(classify_risk)


# Load models and related objects
try:
    rf_clf = joblib.load('models/rf_classifier_corrected.pkl')
    rf_reg = joblib.load('models/rf_regressor_corrected.pkl')
    scaler = joblib.load('models/scaler_corrected.pkl')
    feature_names = joblib.load('models/feature_names_corrected.pkl')
    explainer = joblib.load('models/shap_explainer_corrected.pkl')
    print("✓ Models loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: Model file not found - {e}. Please run 'train_models_corrected.py' first.")
    exit()

# Prepare data for predictions
X = df[feature_names].fillna(df[feature_names].median())
X_scaled = scaler.transform(X)

y_true_clf = df['risk_category']
y_pred_clf = rf_clf.predict(X_scaled)

y_true_reg = df['malaria_prev_2021']
y_pred_reg = rf_reg.predict(X_scaled)


# ============================================================================
# 2. VISUALIZATION FUNCTIONS
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, labels=['Low', 'Medium', 'High']):
    """
    Generates a clean heatmap of the confusion matrix with percentages.
    """
    print("\n[2.1] Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap=SINGLE_COLOR_GRADIENT,
                xticklabels=labels, yticklabels=labels, cbar=False, annot_kws={"size": AXIS_FONT_SIZE})

    ax.set_title("Confusion Matrix of Risk Categories (%)", fontsize=TITLE_FONT_SIZE, fontfamily=FONT_FAMILY, weight='bold')
    ax.set_xlabel("Predicted Label", fontsize=AXIS_FONT_SIZE, fontfamily=FONT_FAMILY)
    ax.set_ylabel("True Label", fontsize=AXIS_FONT_SIZE, fontfamily=FONT_FAMILY)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    plt.tight_layout()
    plt.savefig("visualizations/corrected_confusion_matrix.png", dpi=300)
    print("✓ Saved: visualizations/corrected_confusion_matrix.png")
    plt.close()

def plot_feature_importance(feature_names):
    """
    Generates a horizontal bar chart of the top 10 SHAP feature importances.
    Uses pre-calculated SHAP importance from CSV for efficiency.
    """
    print("\n[2.2] Generating SHAP Feature Importance Plot...")

    # Try to load pre-calculated SHAP importance
    try:
        importance_df = pd.read_csv('data/shap_feature_importance_corrected.csv')
        print("  ✓ Using pre-calculated SHAP values")
    except FileNotFoundError:
        print("  ⚠ Pre-calculated SHAP values not found. Please run regenerate_shap_corrected.py first.")
        return

    top_10_features = importance_df.sort_values('SHAP_Importance', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create bars with color gradient
    bars = ax.barh(range(len(top_10_features)), top_10_features['SHAP_Importance'].values)

    # Color gradient
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_10_features)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_yticks(range(len(top_10_features)))
    ax.set_yticklabels(top_10_features['Feature'].values, fontsize=AXIS_FONT_SIZE)
    ax.invert_yaxis()  # Highest at top

    ax.set_title("Top 10 Feature Importances (SHAP) - Corrected Model",
                 fontsize=TITLE_FONT_SIZE, fontfamily=FONT_FAMILY, weight='bold')
    ax.set_xlabel("Mean Absolute SHAP Value", fontsize=AXIS_FONT_SIZE, fontfamily=FONT_FAMILY)
    ax.set_ylabel("")

    # Add value labels on bars
    for i, (idx, row) in enumerate(top_10_features.iterrows()):
        ax.text(row['SHAP_Importance'], i, f" {row['SHAP_Importance']:.4f}",
                va='center', fontsize=10)

    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.savefig("visualizations/corrected_shap_feature_importance.png", dpi=300, bbox_inches='tight')
    print("  ✓ Saved: visualizations/corrected_shap_feature_importance.png")
    plt.close()


def plot_geographic_risk_map(df):
    """
    Creates a sorted horizontal bar chart representing states, grouped by Zone,
    colored by Risk Level.
    """
    print("\n[2.3] Generating Geographic Risk Map (Bar Chart)...")
    
    # Sort data for plotting
    df_sorted = df.sort_values(by=['Zone', 'malaria_prev_2021'], ascending=[True, False])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create the bar plot
    bars = sns.barplot(x='malaria_prev_2021', y='State', data=df_sorted,
                       hue='risk_category', palette=CATEGORICAL_PALETTE, dodge=False)

    ax.set_title("Geographic Malaria Risk Distribution by State and Zone (2021)", fontsize=TITLE_FONT_SIZE, fontfamily=FONT_FAMILY, weight='bold')
    ax.set_xlabel("Malaria Prevalence (%)", fontsize=AXIS_FONT_SIZE, fontfamily=FONT_FAMILY)
    ax.set_ylabel("State", fontsize=AXIS_FONT_SIZE, fontfamily=FONT_FAMILY)

    # Style legend
    legend = ax.get_legend()
    legend.set_title("Risk Level")
    
    # Add vertical lines for zone separators
    zone_boundaries = df_sorted['Zone'].drop_duplicates(keep='last').index
    for i, boundary in enumerate(zone_boundaries[:-1]):
        ax.axhline(y=boundary + 0.5, color='grey', linestyle='--', linewidth=0.8)

    sns.despine()
    plt.tight_layout()
    plt.savefig("visualizations/corrected_geographic_risk_map.png", dpi=300)
    print("✓ Saved: visualizations/corrected_geographic_risk_map.png")
    plt.close()


def plot_actual_vs_predicted(y_true, y_pred):
    """
    Generates a scatter plot of actual vs. predicted values with a y=x reference line.
    """
    print("\n[2.4] Generating Actual vs. Predicted Scatter Plot...")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, s=100, color=CATEGORICAL_PALETTE['Low'])
    
    # y=x reference line
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'k--', lw=2, color='grey', label='y=x line')
    
    ax.set_title("Actual vs. Predicted Malaria Prevalence (2021)", fontsize=TITLE_FONT_SIZE, fontfamily=FONT_FAMILY, weight='bold')
    ax.set_xlabel("Actual Prevalence (%)", fontsize=AXIS_FONT_SIZE, fontfamily=FONT_FAMILY)
    ax.set_ylabel("Predicted Prevalence (%)", fontsize=AXIS_FONT_SIZE, fontfamily=FONT_FAMILY)
    ax.set_aspect('equal', adjustable='box')
    
    sns.despine()
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualizations/corrected_actual_vs_predicted.png", dpi=300)
    print("✓ Saved: visualizations/corrected_actual_vs_predicted.png")
    plt.close()


# ============================================================================
# 3. EXECUTE PLOTTING
# ============================================================================
if __name__ == "__main__":
    print("\n[3] Starting visualization generation...")
    
    # Create visualizations directory if it doesn't exist
    import os
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')

    plot_confusion_matrix(y_true_clf, y_pred_clf)
    plot_feature_importance(feature_names)
    plot_geographic_risk_map(df)
    plot_actual_vs_predicted(y_true_reg, y_pred_reg)
    
    print("\n" + "="*80)
    print("✅ All visualizations generated successfully!")
    print("="*80)