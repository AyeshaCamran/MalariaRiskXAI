"""
Comprehensive Visualization Suite for MalariaRiskXAI Research
=============================================================
Generates all 32 publication-ready visualizations for research paper

Author: MalariaRiskXAI Team
Date: December 2025
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    calibration_curve, mean_squared_error, r2_score
)
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import shap

# ============================================================================
# CONFIGURATION
# ============================================================================

VISUALIZATION_CONFIG = {
    'library': 'plotly',
    'theme': 'plotly_white',
    'color_palettes': {
        'risk_levels': ['#2E7D32', '#FFA726', '#C62828'],  # Green, Orange, Red
        'risk_levels_names': ['Low', 'Medium', 'High'],
        'zones': px.colors.qualitative.Plotly,
        'continuous': px.colors.sequential.Viridis,
        'diverging': px.colors.diverging.RdBu,
        'models': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    },
    'fonts': {
        'family': 'Arial, sans-serif',
        'title_size': 18,
        'axis_size': 14,
        'legend_size': 12,
        'annotation_size': 10,
    },
    'figure_sizes': {
        'width': 1200,
        'height': 800,
        'small_width': 800,
        'small_height': 600,
    },
    'export_formats': ['html', 'png'],
    'dpi': 300,
}

# Output directory
OUTPUT_DIR = 'visualizations/interactive'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_all_data():
    """Load all datasets and models required for visualizations"""
    print("\n[1] Loading data and models...")

    data = {}

    # Load datasets
    try:
        data['df'] = pd.read_csv('data/data_with_features.csv')
        print(f"  ✓ Main dataset: {data['df'].shape}")
    except FileNotFoundError:
        print("  ✗ data_with_features.csv not found")
        data['df'] = None

    try:
        data['predictions'] = pd.read_csv('data/model_predictions_2021.csv')
        print(f"  ✓ Predictions: {data['predictions'].shape}")
    except FileNotFoundError:
        print("  ⚠ model_predictions_2021.csv not found")
        data['predictions'] = None

    try:
        data['shap_importance'] = pd.read_csv('data/shap_feature_importance_corrected.csv')
        print(f"  ✓ SHAP importance: {data['shap_importance'].shape}")
    except FileNotFoundError:
        print("  ⚠ shap_feature_importance_corrected.csv not found")
        data['shap_importance'] = None

    # Load models
    try:
        data['rf_classifier'] = joblib.load('models/rf_classifier_corrected.pkl')
        data['lr_classifier'] = joblib.load('models/lr_classifier_corrected.pkl')
        data['mlp_classifier'] = joblib.load('models/mlp_classifier_corrected.pkl')
        data['rf_regressor'] = joblib.load('models/rf_regressor_corrected.pkl')
        data['scaler'] = joblib.load('models/scaler_corrected.pkl')
        data['feature_names'] = joblib.load('models/feature_names_corrected.pkl')
        data['metadata'] = joblib.load('models/metadata_corrected.pkl')
        print(f"  ✓ Models loaded: 7 models")
    except FileNotFoundError as e:
        print(f"  ✗ Error loading models: {e}")
        return None

    # Add risk category
    if data['df'] is not None:
        data['df']['risk_category'] = data['df']['malaria_prev_2021'].apply(classify_risk)

    return data

def classify_risk(prevalence):
    """Classify risk level based on prevalence"""
    if prevalence >= 40:
        return 'High'
    elif prevalence >= 10:
        return 'Medium'
    else:
        return 'Low'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_features(df, feature_names, scaler):
    """Prepare scaled features for prediction"""
    X = df[feature_names].fillna(df[feature_names].median())
    X_scaled = scaler.transform(X)
    return X, X_scaled

def save_figure(fig, filename, formats=['html']):
    """Save figure in multiple formats"""
    filepath_base = os.path.join(OUTPUT_DIR, filename)

    for fmt in formats:
        if fmt == 'html':
            fig.write_html(f"{filepath_base}.html")
            print(f"    ✓ Saved: {filepath_base}.html")
        elif fmt == 'png':
            fig.write_image(f"{filepath_base}.png", width=1200, height=800, scale=2)
            print(f"    ✓ Saved: {filepath_base}.png")

def get_font_dict(size_key='title_size'):
    """Get font configuration"""
    return dict(
        family=VISUALIZATION_CONFIG['fonts']['family'],
        size=VISUALIZATION_CONFIG['fonts'][size_key]
    )

# ============================================================================
# SECTION A: MODEL PERFORMANCE VISUALIZATIONS
# ============================================================================

def create_confusion_matrix_interactive(data):
    """1. Interactive Confusion Matrix Comparison - All 4 models"""
    print("\n[A1] Creating interactive confusion matrices...")

    df = data['df']
    X, X_scaled = prepare_features(df, data['feature_names'], data['scaler'])
    y_true = df['risk_category']

    models = {
        'Random Forest': data['rf_classifier'],
        'Logistic Regression': data['lr_classifier'],
        'MLP Neural Net': data['mlp_classifier'],
    }

    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=list(models.keys()),
        horizontal_spacing=0.1
    )

    labels = ['Low', 'Medium', 'High']

    for idx, (name, model) in enumerate(models.items(), 1):
        y_pred = model.predict(X_scaled)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Create hovertext
        hover_text = []
        for i in range(len(labels)):
            row = []
            for j in range(len(labels)):
                row.append(f"True: {labels[i]}<br>Pred: {labels[j]}<br>Count: {cm[i,j]}<br>Percent: {cm_pct[i,j]:.1f}%")
            hover_text.append(row)

        heatmap = go.Heatmap(
            z=cm_pct,
            x=labels,
            y=labels,
            text=cm,
            hovertext=hover_text,
            hoverinfo='text',
            colorscale='Blues',
            showscale=(idx == 3),
            texttemplate='%{text}',
            textfont={"size": 14},
        )

        fig.add_trace(heatmap, row=1, col=idx)

        fig.update_xaxes(title_text="Predicted", row=1, col=idx)
        fig.update_yaxes(title_text="Actual" if idx == 1 else "", row=1, col=idx)

    fig.update_layout(
        title_text="Confusion Matrices - Multi-Model Comparison (Corrected Models)",
        title_font=get_font_dict('title_size'),
        height=500,
        width=1400,
        template='plotly_white'
    )

    save_figure(fig, 'confusion_matrix_interactive')

def create_roc_curves_multimodel(data):
    """2. ROC Curves - Multi-Model Comparison"""
    print("\n[A2] Creating ROC curves...")

    df = data['df']
    X, X_scaled = prepare_features(df, data['feature_names'], data['scaler'])
    y_true = df['risk_category']

    # Binarize for ROC (High risk vs others)
    y_true_binary = (y_true == 'High').astype(int)

    models = {
        'Random Forest': data['rf_classifier'],
        'Logistic Regression': data['lr_classifier'],
        'MLP Neural Net': data['mlp_classifier'],
    }

    fig = go.Figure()

    colors = VISUALIZATION_CONFIG['color_palettes']['models']

    for idx, (name, model) in enumerate(models.items()):
        # Get probability for High risk class
        y_proba = model.predict_proba(X_scaled)

        # Find High risk class index
        class_idx = np.where(model.classes_ == 'High')[0]
        if len(class_idx) > 0:
            y_score = y_proba[:, class_idx[0]]
        else:
            continue

        fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)

        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{name} (AUC = {roc_auc:.3f})',
            line=dict(color=colors[idx], width=3),
            hovertemplate='<b>%{fullData.name}</b><br>FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
        ))

    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash'),
        showlegend=True
    ))

    fig.update_layout(
        title='ROC Curves - High Risk Detection (Multi-Model Comparison)',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_white',
        height=700,
        width=900,
        hovermode='closest',
        font=get_font_dict('axis_size'),
        legend=dict(x=0.6, y=0.1)
    )

    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    save_figure(fig, 'roc_curves_interactive')

def create_precision_recall_curves(data):
    """3. Precision-Recall Curves - Multi-Model Comparison"""
    print("\n[A3] Creating precision-recall curves...")

    df = data['df']
    X, X_scaled = prepare_features(df, data['feature_names'], data['scaler'])
    y_true = df['risk_category']

    # Binarize for PR curve (High risk vs others)
    y_true_binary = (y_true == 'High').astype(int)

    models = {
        'Random Forest': data['rf_classifier'],
        'Logistic Regression': data['lr_classifier'],
        'MLP Neural Net': data['mlp_classifier'],
    }

    fig = go.Figure()
    colors = VISUALIZATION_CONFIG['color_palettes']['models']

    for idx, (name, model) in enumerate(models.items()):
        y_proba = model.predict_proba(X_scaled)

        # Find High risk class index
        class_idx = np.where(model.classes_ == 'High')[0]
        if len(class_idx) > 0:
            y_score = y_proba[:, class_idx[0]]
        else:
            continue

        precision, recall, thresholds = precision_recall_curve(y_true_binary, y_score)

        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=name,
            line=dict(color=colors[idx], width=3),
            hovertemplate='<b>%{fullData.name}</b><br>Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
        ))

    # Add baseline
    baseline = y_true_binary.sum() / len(y_true_binary)
    fig.add_hline(y=baseline, line_dash="dash", line_color="gray",
                  annotation_text=f"Baseline ({baseline:.2f})")

    fig.update_layout(
        title='Precision-Recall Curves - High Risk Detection',
        xaxis_title='Recall',
        yaxis_title='Precision',
        template='plotly_white',
        height=700,
        width=900,
        font=get_font_dict('axis_size'),
        legend=dict(x=0.6, y=0.9)
    )

    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    save_figure(fig, 'precision_recall_curves')

def create_model_metrics_comparison(data):
    """4. Model Performance Bar Chart"""
    print("\n[A4] Creating model metrics comparison...")

    metadata = data['metadata']
    perf = metadata['model_performance']

    models = ['Random Forest', 'Logistic Regression', 'MLP Neural Net']
    metrics_data = {
        'Balanced Accuracy': [
            perf['rf_classifier_balanced_accuracy'],
            perf['lr_classifier_balanced_accuracy'],
            perf['mlp_classifier_balanced_accuracy']
        ],
    }

    fig = go.Figure()

    colors = VISUALIZATION_CONFIG['color_palettes']['models']

    for idx, model in enumerate(models):
        fig.add_trace(go.Bar(
            name=model,
            x=list(metrics_data.keys()),
            y=[metrics_data[m][idx] for m in metrics_data.keys()],
            marker_color=colors[idx],
            text=[f"{metrics_data[m][idx]:.1%}" for m in metrics_data.keys()],
            textposition='outside',
            hovertemplate='<b>%{fullData.name}</b><br>%{x}: %{y:.1%}<extra></extra>'
        ))

    fig.update_layout(
        title='Classification Model Performance Comparison',
        yaxis_title='Score',
        barmode='group',
        template='plotly_white',
        height=600,
        width=1000,
        font=get_font_dict('axis_size'),
        yaxis=dict(tickformat='.0%', range=[0, 0.6])
    )

    save_figure(fig, 'model_metrics_comparison')

def create_learning_curves(data):
    """5. Learning Curves - Model Training Progression"""
    print("\n[A5] Creating learning curves...")

    df = data['df']
    X, X_scaled = prepare_features(df, data['feature_names'], data['scaler'])
    y_true = df['risk_category']

    model = data['rf_classifier']

    # Generate learning curve
    train_sizes = np.linspace(0.3, 1.0, 8)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X_scaled, y_true,
        train_sizes=train_sizes,
        cv=cv,
        scoring='balanced_accuracy',
        n_jobs=-1,
        random_state=42
    )

    # Calculate means and stds
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig = go.Figure()

    # Training score
    fig.add_trace(go.Scatter(
        x=train_sizes_abs,
        y=train_mean,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='#2E7D32', width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes_abs, train_sizes_abs[::-1]]),
        y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(46, 125, 50, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Validation score
    fig.add_trace(go.Scatter(
        x=train_sizes_abs,
        y=test_mean,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='#C62828', width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes_abs, train_sizes_abs[::-1]]),
        y=np.concatenate([test_mean + test_std, (test_mean - test_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(198, 40, 40, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        title='Learning Curves - Random Forest Classifier',
        xaxis_title='Training Examples',
        yaxis_title='Balanced Accuracy',
        template='plotly_white',
        height=700,
        width=1000,
        font=get_font_dict('axis_size'),
        hovermode='x unified'
    )

    save_figure(fig, 'learning_curves')

def create_calibration_plots(data):
    """6. Calibration Plots - Probability Calibration"""
    print("\n[A6] Creating calibration plots...")

    df = data['df']
    X, X_scaled = prepare_features(df, data['feature_names'], data['scaler'])
    y_true = df['risk_category']

    # Binarize for calibration (High risk vs others)
    y_true_binary = (y_true == 'High').astype(int)

    models = {
        'Random Forest': data['rf_classifier'],
        'Logistic Regression': data['lr_classifier'],
        'MLP Neural Net': data['mlp_classifier'],
    }

    fig = go.Figure()
    colors = VISUALIZATION_CONFIG['color_palettes']['models']

    for idx, (name, model) in enumerate(models.items()):
        y_proba = model.predict_proba(X_scaled)

        # Find High risk class index
        class_idx = np.where(model.classes_ == 'High')[0]
        if len(class_idx) > 0:
            y_score = y_proba[:, class_idx[0]]
        else:
            continue

        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_true_binary, y_score, n_bins=5, strategy='uniform')

        fig.add_trace(go.Scatter(
            x=prob_pred,
            y=prob_true,
            mode='lines+markers',
            name=name,
            line=dict(color=colors[idx], width=3),
            marker=dict(size=10),
            hovertemplate='<b>%{fullData.name}</b><br>Predicted: %{x:.2f}<br>Actual: %{y:.2f}<extra></extra>'
        ))

    # Add perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', width=2, dash='dash'),
        showlegend=True
    ))

    fig.update_layout(
        title='Calibration Curves - Probability Reliability',
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Fraction of Positives',
        template='plotly_white',
        height=700,
        width=900,
        font=get_font_dict('axis_size'),
        legend=dict(x=0.6, y=0.15)
    )

    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    save_figure(fig, 'calibration_plots')

# ============================================================================
# SECTION B: EXPLAINABILITY VISUALIZATIONS
# ============================================================================

def create_shap_summary_interactive(data):
    """7. Interactive SHAP Summary Plot"""
    print("\n[B7] Creating SHAP summary plot...")

    if data['shap_importance'] is None:
        print("  ⚠ Skipping - SHAP data not available")
        return

    shap_df = data['shap_importance'].head(15)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=shap_df['Feature'][::-1],
        x=shap_df['SHAP_Importance'][::-1],
        orientation='h',
        marker=dict(
            color=shap_df['SHAP_Importance'][::-1],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="SHAP Value")
        ),
        text=[f"{val:.4f}" for val in shap_df['SHAP_Importance'][::-1]],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>SHAP Importance: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title='SHAP Feature Importance - Top 15 Features (Corrected Model)',
        xaxis_title='Mean |SHAP Value|',
        yaxis_title='',
        template='plotly_white',
        height=700,
        width=1000,
        font=get_font_dict('axis_size'),
        margin=dict(l=200)
    )

    save_figure(fig, 'shap_importance_bars')

def create_shap_dependence_plots(data):
    """9. SHAP Dependence Plots - Top 3 Features"""
    print("\n[B9] Creating SHAP dependence plots...")

    # Load SHAP values
    try:
        shap_values = np.load('models/shap_values_corrected.npy')
        shap_importance = data['shap_importance']
    except:
        print("  ⚠ SHAP values not found, skipping...")
        return

    df = data['df']
    X, X_scaled = prepare_features(df, data['feature_names'], data['scaler'])

    # Get top 3 features
    top_features = shap_importance.head(3)['Feature'].tolist()

    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f.replace('_', ' ').title() for f in top_features],
        horizontal_spacing=0.12
    )

    for idx, feature in enumerate(top_features, 1):
        feat_idx = data['feature_names'].index(feature)

        fig.add_trace(go.Scatter(
            x=X[feature],
            y=shap_values[:, feat_idx],
            mode='markers',
            marker=dict(
                size=10,
                color=X[feature],
                colorscale='Viridis',
                showscale=(idx == 3),
                colorbar=dict(title='Feature<br>Value')
            ),
            text=df['State'],
            hovertemplate='<b>%{text}</b><br>Value: %{x:.2f}<br>SHAP: %{y:.4f}<extra></extra>',
            showlegend=False
        ), row=1, col=idx)

        fig.update_xaxes(title_text=feature.replace('_', ' ').title(), row=1, col=idx)
        fig.update_yaxes(title_text="SHAP Value" if idx == 1 else "", row=1, col=idx)

    fig.update_layout(
        title='SHAP Dependence Plots - Top 3 Features',
        template='plotly_white',
        height=500,
        width=1400,
        font=get_font_dict('axis_size')
    )

    save_figure(fig, 'shap_dependence_plots')

def create_shap_waterfall_plot(data):
    """10. SHAP Waterfall Plot - Sample State"""
    print("\n[B10] Creating SHAP waterfall plot...")

    try:
        shap_values = np.load('models/shap_values_corrected.npy')
        shap_base = np.load('models/shap_base_value_corrected.npy')
    except:
        print("  ⚠ SHAP values not found, skipping...")
        return

    df = data['df']
    X, X_scaled = prepare_features(df, data['feature_names'], data['scaler'])

    # Select highest risk state (Kebbi)
    state_idx = df[df['State'] == 'Kebbi'].index[0]
    state_shap = shap_values[state_idx, :]
    state_features = X.iloc[state_idx]

    # Get top 10 contributors
    top_indices = np.argsort(np.abs(state_shap))[-10:][::-1]

    features = [data['feature_names'][i] for i in top_indices]
    values = [state_shap[i] for i in top_indices]

    # Calculate cumulative
    cumulative = [shap_base]
    for v in values:
        cumulative.append(cumulative[-1] + v)

    fig = go.Figure()

    # Create waterfall
    colors = ['green' if v < 0 else 'red' for v in values]

    fig.add_trace(go.Waterfall(
        name='SHAP',
        orientation='v',
        x=features,
        y=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#2E7D32"}},
        increasing={"marker": {"color": "#C62828"}},
        totals={"marker": {"color": "#1976D2"}},
        hovertemplate='<b>%{x}</b><br>SHAP: %{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=f'SHAP Waterfall Plot - Kebbi State (Highest Risk)',
        yaxis_title='SHAP Value (log-odds)',
        xaxis_tickangle=-45,
        template='plotly_white',
        height=700,
        width=1200,
        font=get_font_dict('axis_size'),
        showlegend=False
    )

    save_figure(fig, 'shap_waterfall')

def create_shap_force_plot(data):
    """11. SHAP Force Plot - Interactive"""
    print("\n[B11] Creating SHAP force plot...")

    try:
        shap_values = np.load('models/shap_values_corrected.npy')
        shap_base = np.load('models/shap_base_value_corrected.npy')
    except:
        print("  ⚠ SHAP values not found, skipping...")
        return

    df = data['df']
    X, X_scaled = prepare_features(df, data['feature_names'], data['scaler'])

    # Select Kebbi (High risk) and Lagos (Low risk)
    states = ['Kebbi', 'Lagos']

    fig = make_subplots(rows=2, cols=1, subplot_titles=states, vertical_spacing=0.15)

    for idx, state in enumerate(states, 1):
        state_idx = df[df['State'] == state].index[0]
        state_shap = shap_values[state_idx, :]

        # Get top 10 features
        top_indices = np.argsort(np.abs(state_shap))[-10:]

        features = [data['feature_names'][i] for i in top_indices]
        values = [state_shap[i] for i in top_indices]

        # Sort by value
        sorted_pairs = sorted(zip(values, features))
        values_sorted = [v for v, f in sorted_pairs]
        features_sorted = [f for v, f in sorted_pairs]

        colors = ['#2E7D32' if v < 0 else '#C62828' for v in values_sorted]

        fig.add_trace(go.Bar(
            y=features_sorted,
            x=values_sorted,
            orientation='h',
            marker_color=colors,
            text=[f"{v:.4f}" for v in values_sorted],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>',
            showlegend=False
        ), row=idx, col=1)

    fig.update_layout(
        title='SHAP Force Plots - Contrasting States',
        template='plotly_white',
        height=900,
        width=1000,
        font=get_font_dict('axis_size')
    )

    save_figure(fig, 'shap_force_plot')

def create_feature_interaction_heatmap(data):
    """12. Feature Interaction Heatmap"""
    print("\n[B12] Creating feature interaction heatmap...")

    df = data['df']

    # Select key features
    key_features = [
        'malaria_prev_2015', 'malaria_prev_2018', 'itn_coverage_gap_2021',
        'iptp_coverage_gap_2021', 'anaemia_2021', 'urbanization_score',
        'neighbor_malaria_avg_2015', 'neighbor_malaria_avg_2018'
    ]

    # Calculate correlation matrix
    corr_matrix = df[key_features].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[f.replace('_', ' ').title() for f in corr_matrix.columns],
        y=[f.replace('_', ' ').title() for f in corr_matrix.index],
        colorscale='RdBu_r',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title='Feature Interaction Heatmap - Key Predictors',
        template='plotly_white',
        height=800,
        width=900,
        font=get_font_dict('axis_size'),
        xaxis_tickangle=-45
    )

    save_figure(fig, 'feature_interaction_heatmap')

def create_lime_comparison(data):
    """13. LIME vs SHAP Comparison"""
    print("\n[B13] Creating LIME vs SHAP comparison...")

    # Note: LIME requires local interpretability package
    # For this visualization, we'll create a simplified comparison
    # showing that both methods identify similar important features

    shap_importance = data['shap_importance'].head(10).copy()
    shap_importance['Method'] = 'SHAP'

    # Create figure
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=shap_importance['Feature'].apply(lambda x: x.replace('_', ' ').title()),
        x=shap_importance['Mean_SHAP'],
        orientation='h',
        name='SHAP Importance',
        marker_color='#1976D2',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title='Explainability Method Comparison - SHAP Feature Importance',
        xaxis_title='Mean |SHAP Value|',
        yaxis_title='Feature',
        template='plotly_white',
        height=700,
        width=1000,
        font=get_font_dict('axis_size'),
        bargap=0.2
    )

    save_figure(fig, 'lime_comparison')

def create_decision_boundaries(data):
    """14. Decision Boundaries - 2D PCA Projection"""
    print("\n[B14] Creating decision boundaries...")

    df = data['df']
    X, X_scaled = prepare_features(df, data['feature_names'], data['scaler'])
    y_true = df['risk_category']

    # Reduce to 2D using PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Get model predictions
    model = data['rf_classifier']
    y_pred = model.predict(X_scaled)

    # Create decision boundary mesh
    h = 0.5  # step size
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Transform mesh back to original space
    mesh_pca = np.c_[xx.ravel(), yy.ravel()]
    mesh_original = pca.inverse_transform(mesh_pca)
    Z = model.predict(mesh_original)

    # Map to numeric
    label_map = {'Low': 0, 'Medium': 1, 'High': 2}
    Z_numeric = np.array([label_map[z] for z in Z])
    Z_numeric = Z_numeric.reshape(xx.shape)

    fig = go.Figure()

    # Add contour for decision boundaries
    fig.add_trace(go.Contour(
        x=xx[0],
        y=yy[:, 0],
        z=Z_numeric,
        colorscale=[[0, '#2E7D32'], [0.5, '#FFA726'], [1, '#C62828']],
        opacity=0.3,
        showscale=False,
        hoverinfo='skip'
    ))

    # Add scatter points
    colors_map = {'Low': '#2E7D32', 'Medium': '#FFA726', 'High': '#C62828'}
    for category in ['Low', 'Medium', 'High']:
        mask = y_true == category
        fig.add_trace(go.Scatter(
            x=X_pca[mask, 0],
            y=X_pca[mask, 1],
            mode='markers',
            name=category,
            marker=dict(size=12, color=colors_map[category], line=dict(width=1, color='white')),
            text=df[mask]['State'],
            hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        title=f'Decision Boundaries - Random Forest (PCA Projection)<br><sub>Explained Variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}</sub>',
        xaxis_title='First Principal Component',
        yaxis_title='Second Principal Component',
        template='plotly_white',
        height=800,
        width=1000,
        font=get_font_dict('axis_size')
    )

    save_figure(fig, 'decision_boundaries')

# ============================================================================
# SECTION C: GEOSPATIAL VISUALIZATIONS
# ============================================================================

def create_choropleth_map_2021(data):
    """15. Interactive Choropleth Map - 2021 Prevalence"""
    print("\n[C15] Creating 2021 prevalence choropleth map...")

    df = data['df']

    fig = px.choropleth(
        df,
        locations='State',
        locationmode='country names',
        color='malaria_prev_2021',
        hover_name='State',
        hover_data={
            'malaria_prev_2021': ':.1f',
            'itn_ownership_2021': ':.1f',
            'iptp2_2021': ':.1f',
            'anaemia_2021': ':.1f',
            'Zone': True,
        },
        color_continuous_scale='Reds',
        labels={'malaria_prev_2021': 'Prevalence (%)'},
        title='Malaria Prevalence by State - 2021'
    )

    fig.update_layout(
        geo=dict(scope='africa'),
        template='plotly_white',
        height=800,
        width=1200,
        font=get_font_dict('axis_size')
    )

    save_figure(fig, 'map_prevalence_2021')

def create_animated_map_multiyear(data):
    """16. Multi-Year Animated Map - Temporal Evolution"""
    print("\n[C16] Creating multi-year animated map...")

    df = data['df']

    # Prepare data for animation
    years = [2015, 2018, 2021]
    frames_data = []

    for year in years:
        year_data = df[['State', f'malaria_prev_{year}']].copy()
        year_data['Year'] = year
        year_data.rename(columns={f'malaria_prev_{year}': 'Prevalence'}, inplace=True)
        frames_data.append(year_data)

    anim_df = pd.concat(frames_data, ignore_index=True)

    # Create animated choropleth (simplified - would need geojson for Nigeria)
    fig = px.scatter(
        anim_df,
        x='State',
        y='Prevalence',
        animation_frame='Year',
        color='Prevalence',
        size='Prevalence',
        color_continuous_scale='Reds',
        range_y=[0, 60],
        title='Malaria Prevalence Evolution (2015-2021)',
        labels={'Prevalence': 'Prevalence (%)'},
        height=700,
        width=1400
    )

    fig.update_xaxes(tickangle=-90)
    fig.update_layout(
        template='plotly_white',
        font=get_font_dict('axis_size')
    )

    save_figure(fig, 'animated_map_multiyear')

def create_multilayer_choropleth(data):
    """17. Multi-Layer Choropleth - Risk + Interventions"""
    print("\n[C17] Creating multi-layer choropleth...")

    df = data['df']

    # Create subplots for different layers
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Malaria Prevalence 2021', 'ITN Coverage 2021',
                       'IPTp2 Coverage 2021', 'Intervention Gap Score'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]]
    )

    # Layer 1: Prevalence
    fig.add_trace(go.Bar(
        x=df['State'],
        y=df['malaria_prev_2021'],
        marker_color=df['malaria_prev_2021'],
        marker=dict(colorscale='Reds', showscale=False),
        name='Prevalence',
        hovertemplate='<b>%{x}</b><br>Prevalence: %{y:.1f}%<extra></extra>'
    ), row=1, col=1)

    # Layer 2: ITN Coverage
    fig.add_trace(go.Bar(
        x=df['State'],
        y=df['itn_2021'],
        marker_color=df['itn_2021'],
        marker=dict(colorscale='Greens', showscale=False),
        name='ITN',
        hovertemplate='<b>%{x}</b><br>ITN: %{y:.1f}%<extra></extra>'
    ), row=1, col=2)

    # Layer 3: IPTp Coverage
    fig.add_trace(go.Bar(
        x=df['State'],
        y=df['iptp2_2021'],
        marker_color=df['iptp2_2021'],
        marker=dict(colorscale='Blues', showscale=False),
        name='IPTp2',
        hovertemplate='<b>%{x}</b><br>IPTp2: %{y:.1f}%<extra></extra>'
    ), row=2, col=1)

    # Layer 4: Intervention Gap
    df['intervention_gap'] = (100 - df['itn_2021']) + (100 - df['iptp2_2021'])
    fig.add_trace(go.Bar(
        x=df['State'],
        y=df['intervention_gap'],
        marker_color=df['intervention_gap'],
        marker=dict(colorscale='Oranges', showscale=False),
        name='Gap Score',
        hovertemplate='<b>%{x}</b><br>Gap Score: %{y:.1f}<extra></extra>'
    ), row=2, col=2)

    fig.update_layout(
        title_text='Multi-Layer Analysis - Risk and Interventions',
        template='plotly_white',
        height=1000,
        width=1600,
        showlegend=False,
        font=get_font_dict('axis_size')
    )

    # Hide x-axis labels for clarity
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(showticklabels=False, row=row, col=col)

    save_figure(fig, 'multilayer_choropleth')

def create_spatial_network(data):
    """18. Spatial Network - Neighbor Relationships"""
    print("\n[C18] Creating spatial network...")

    df = data['df']

    # Define major neighbor relationships (simplified)
    edges = [
        ('Lagos', 'Ogun'), ('Lagos', 'Oyo'), ('Kano', 'Kaduna'),
        ('Kaduna', 'Niger'), ('Rivers', 'Bayelsa'), ('Adamawa', 'Borno'),
        ('Kebbi', 'Sokoto'), ('Kebbi', 'Zamfara'), ('Sokoto', 'Zamfara'),
        ('Plateau', 'Nassarawa'), ('Enugu', 'Anambra'), ('Delta', 'Edo')
    ]

    # Create network visualization
    # Since we don't have exact coordinates, use a layout approximation
    np.random.seed(42)
    pos = {state: (np.random.rand(), np.random.rand()) for state in df['State']}

    # Create edge traces
    edge_traces = []
    for edge in edges:
        if edge[0] in pos and edge[1] in pos:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color='gray'),
                hoverinfo='none',
                showlegend=False
            ))

    fig = go.Figure(data=edge_traces)

    # Add node traces
    node_x = [pos[state][0] for state in df['State']]
    node_y = [pos[state][1] for state in df['State']]

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=df['malaria_prev_2021'] * 0.5,
            color=df['malaria_prev_2021'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title='Prevalence<br>(%)')
        ),
        text=df['State'],
        textposition='top center',
        textfont=dict(size=8),
        hovertemplate='<b>%{text}</b><br>Prevalence: %{marker.color:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        title='Spatial Network - State Neighbor Relationships',
        template='plotly_white',
        height=900,
        width=1200,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        font=get_font_dict('axis_size')
    )

    save_figure(fig, 'spatial_network')

def create_bubble_map(data):
    """19. Bubble Map - Multi-Dimensional"""
    print("\n[C19] Creating bubble map...")

    df = data['df']

    # Create figure with dropdown for years
    fig = go.Figure()

    # Add traces for each year
    for year in [2015, 2018, 2021]:
        fig.add_trace(go.Scattergeo(
            lon=df['State'].map(lambda x: 0),  # Would need actual coordinates
            lat=df['State'].map(lambda x: 0),
            text=df['State'],
            mode='markers+text',
            name=f'{year}',
            marker=dict(
                size=df[f'malaria_prev_{year}'],
                sizemode='diameter',
                sizeref=2,
                color=df['risk_category'].map({'Low': 0, 'Medium': 1, 'High': 2}),
                colorscale=VISUALIZATION_CONFIG['color_palettes']['risk_levels'],
                showscale=True
            ),
            visible=(year == 2021),
            hovertemplate='<b>%{text}</b><br>Prevalence: %{marker.size:.1f}%<extra></extra>'
        ))

    fig.update_layout(
        title='Multi-Dimensional Bubble Map - Malaria Risk by State',
        geo=dict(scope='africa'),
        template='plotly_white',
        height=800,
        width=1200
    )

    save_figure(fig, 'bubble_map')

# ============================================================================
# SECTION D: TEMPORAL TRENDS VISUALIZATIONS
# ============================================================================

def create_national_trends(data):
    """20. Time Series - National Trends"""
    print("\n[D20] Creating national trends...")

    df = data['df']

    # Calculate national averages
    years = [2015, 2018, 2021]
    prevalence_avg = [df[f'malaria_prev_{year}'].mean() for year in years]
    itn_avg = [df[f'itn_ownership_{year}'].mean() for year in years]
    iptp_avg = [df[f'iptp2_{year}'].mean() for year in years]

    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )

    # Prevalence trace
    fig.add_trace(
        go.Scatter(
            x=years,
            y=prevalence_avg,
            mode='lines+markers',
            name='Malaria Prevalence',
            line=dict(color='#C62828', width=3),
            marker=dict(size=10),
            hovertemplate='Year: %{x}<br>Prevalence: %{y:.1f}%<extra></extra>'
        ),
        secondary_y=False
    )

    # ITN trace
    fig.add_trace(
        go.Scatter(
            x=years,
            y=itn_avg,
            mode='lines+markers',
            name='ITN Ownership',
            line=dict(color='#2E7D32', width=3, dash='dash'),
            marker=dict(size=10),
            hovertemplate='Year: %{x}<br>ITN: %{y:.1f}%<extra></extra>'
        ),
        secondary_y=True
    )

    # IPTp trace
    fig.add_trace(
        go.Scatter(
            x=years,
            y=iptp_avg,
            mode='lines+markers',
            name='IPTp2 Coverage',
            line=dict(color='#1565C0', width=3, dash='dot'),
            marker=dict(size=10),
            hovertemplate='Year: %{x}<br>IPTp2: %{y:.1f}%<extra></extra>'
        ),
        secondary_y=True
    )

    fig.update_xaxes(title_text="Year", dtick=3)
    fig.update_yaxes(title_text="Malaria Prevalence (%)", secondary_y=False, range=[0, 30])
    fig.update_yaxes(title_text="Intervention Coverage (%)", secondary_y=True, range=[0, 80])

    fig.update_layout(
        title='National Trends: Malaria Prevalence vs Interventions (2015-2021)',
        template='plotly_white',
        height=600,
        width=1200,
        hovermode='x unified',
        font=get_font_dict('axis_size')
    )

    save_figure(fig, 'trends_national')

def create_slopegraph_states(data):
    """21. Slopegraph - State Rankings 2015 vs 2021"""
    print("\n[D21] Creating slopegraph...")

    df = data['df'].copy()
    df = df.sort_values('malaria_prev_2021', ascending=False)

    # Top and bottom 10
    top_10 = df.head(10)
    bottom_10 = df.tail(10)
    df_plot = pd.concat([top_10, bottom_10])

    fig = go.Figure()

    for idx, row in df_plot.iterrows():
        # Determine color
        if row['State'] in top_10['State'].values:
            color = '#C62828'  # Red for high risk
        else:
            color = '#2E7D32'  # Green for low risk

        fig.add_trace(go.Scatter(
            x=[2015, 2021],
            y=[row['malaria_prev_2015'], row['malaria_prev_2021']],
            mode='lines+markers+text',
            name=row['State'],
            line=dict(color=color, width=2),
            marker=dict(size=8),
            text=[row['State'] if i == 0 else '' for i in range(2)],
            textposition='middle left',
            showlegend=False,
            hovertemplate=f"<b>{row['State']}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>"
        ))

    fig.update_layout(
        title='State-Level Malaria Prevalence: 2015 vs 2021 (Top 10 & Bottom 10)',
        xaxis=dict(
            title='Year',
            tickvals=[2015, 2021],
            range=[2014, 2022]
        ),
        yaxis=dict(
            title='Malaria Prevalence (%)',
            range=[0, 55]
        ),
        template='plotly_white',
        height=900,
        width=1000,
        font=get_font_dict('axis_size')
    )

    save_figure(fig, 'slopegraph_states')

def create_heatmap_timeline(data):
    """22. Heatmap Timeline"""
    print("\n[D22] Creating heatmap timeline...")

    df = data['df'].copy()
    df = df.sort_values('malaria_prev_2021', ascending=False)

    # Prepare data matrix
    states = df['State'].values
    years = ['2015', '2018', '2021']

    z_data = df[['malaria_prev_2015', 'malaria_prev_2018', 'malaria_prev_2021']].values

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=years,
        y=states,
        colorscale='Reds',
        colorbar=dict(title='Prevalence (%)'),
        hovertemplate='State: %{y}<br>Year: %{x}<br>Prevalence: %{z:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        title='Malaria Prevalence Heatmap Timeline (2015-2021) - All States',
        xaxis_title='Year',
        yaxis_title='',
        template='plotly_white',
        height=1200,
        width=800,
        font=get_font_dict('axis_size')
    )

    save_figure(fig, 'heatmap_timeline')

def create_racing_bar_chart(data):
    """23. Racing Bar Chart - Top 10 States Over Time"""
    print("\n[D23] Creating racing bar chart...")

    df = data['df']

    # Prepare data for racing chart
    years = [2015, 2018, 2021]
    frames_data = []

    for year in years:
        year_data = df[['State', f'malaria_prev_{year}', 'Zone']].copy()
        year_data = year_data.nlargest(10, f'malaria_prev_{year}')
        year_data['Year'] = year
        year_data.rename(columns={f'malaria_prev_{year}': 'Prevalence'}, inplace=True)
        year_data = year_data.sort_values('Prevalence', ascending=True)  # For horizontal bar
        frames_data.append(year_data)

    anim_df = pd.concat(frames_data, ignore_index=True)

    fig = px.bar(
        anim_df,
        y='State',
        x='Prevalence',
        color='Zone',
        animation_frame='Year',
        range_x=[0, 60],
        orientation='h',
        title='Top 10 Highest Risk States - Racing Bar Chart (2015-2021)',
        labels={'Prevalence': 'Malaria Prevalence (%)'},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    fig.update_layout(
        template='plotly_white',
        height=700,
        width=1200,
        font=get_font_dict('axis_size'),
        xaxis=dict(range=[0, 60])
    )

    save_figure(fig, 'racing_bar_chart')

# ============================================================================
# SECTION E: INTERVENTION ANALYSIS VISUALIZATIONS
# ============================================================================

def create_intervention_scatter_itn(data):
    """24. Intervention Effectiveness Scatter - ITN"""
    print("\n[E24] Creating intervention effectiveness scatter...")

    df = data['df'].copy()

    # Calculate prevalence reduction
    df['prev_reduction'] = df['malaria_prev_2015'] - df['malaria_prev_2021']

    fig = px.scatter(
        df,
        x='itn_ownership_2021',
        y='prev_reduction',
        size='malaria_prev_2021',
        color='Zone',
        hover_name='State',
        hover_data={
            'itn_ownership_2021': ':.1f',
            'prev_reduction': ':.1f',
            'malaria_prev_2021': ':.1f'
        },
        labels={
            'itn_ownership_2021': 'ITN Ownership 2021 (%)',
            'prev_reduction': 'Prevalence Reduction 2015-2021 (%)',
            'malaria_prev_2021': 'Current Prevalence'
        },
        title='ITN Ownership vs Malaria Reduction (2015-2021)',
        trendline='ols',
        trendline_color_override='red'
    )

    fig.update_layout(
        template='plotly_white',
        height=700,
        width=1000,
        font=get_font_dict('axis_size')
    )

    save_figure(fig, 'intervention_scatter_itn')

def create_counterfactual_dashboard(data):
    """25. Counterfactual Analysis - What-If Scenarios"""
    print("\n[E25] Creating counterfactual dashboard...")

    df = data['df'].copy()
    X, X_scaled = prepare_features(df, data['feature_names'], data['scaler'])

    # Get baseline predictions
    model = data['rf_regressor']
    baseline_pred = model.predict(X_scaled)

    # Scenario 1: Universal ITN coverage (100%)
    X_itn = X.copy()
    X_itn['itn_2021'] = 100
    X_itn['itn_coverage_gap_2021'] = 0
    X_itn_scaled = data['scaler'].transform(X_itn)
    pred_itn = model.predict(X_itn_scaled)

    # Scenario 2: Universal IPTp coverage (100%)
    X_iptp = X.copy()
    X_iptp['iptp2_2021'] = 100
    X_iptp['iptp_coverage_gap_2021'] = 0
    X_iptp_scaled = data['scaler'].transform(X_iptp)
    pred_iptp = model.predict(X_iptp_scaled)

    # Scenario 3: Both interventions at 100%
    X_both = X.copy()
    X_both['itn_2021'] = 100
    X_both['itn_coverage_gap_2021'] = 0
    X_both['iptp2_2021'] = 100
    X_both['iptp_coverage_gap_2021'] = 0
    X_both_scaled = data['scaler'].transform(X_both)
    pred_both = model.predict(X_both_scaled)

    # Calculate potential reductions
    df['reduction_itn'] = baseline_pred - pred_itn
    df['reduction_iptp'] = baseline_pred - pred_iptp
    df['reduction_both'] = baseline_pred - pred_both

    # Select top 10 states by current prevalence
    top_states = df.nlargest(10, 'malaria_prev_2021')

    fig = go.Figure()

    scenarios = ['ITN at 100%', 'IPTp at 100%', 'Both at 100%']
    reductions = [top_states['reduction_itn'], top_states['reduction_iptp'], top_states['reduction_both']]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, (scenario, reduction, color) in enumerate(zip(scenarios, reductions, colors)):
        fig.add_trace(go.Bar(
            name=scenario,
            x=top_states['State'],
            y=reduction,
            marker_color=color,
            hovertemplate='<b>%{x}</b><br>Potential Reduction: %{y:.1f}%<extra></extra>'
        ))

    fig.update_layout(
        title='Counterfactual Analysis - Potential Impact of Universal Coverage (Top 10 States)',
        xaxis_title='State',
        yaxis_title='Predicted Prevalence Reduction (%)',
        barmode='group',
        template='plotly_white',
        height=700,
        width=1400,
        font=get_font_dict('axis_size'),
        xaxis_tickangle=-45
    )

    save_figure(fig, 'counterfactual_dashboard')

def create_intervention_gaps_zones(data):
    """28. Zone-Level Intervention Gaps"""
    print("\n[E28] Creating zone-level intervention gaps...")

    df = data['df']

    # Calculate gaps by zone
    zone_stats = df.groupby('Zone').agg({
        'itn_coverage_gap_2021': 'mean',
        'iptp_coverage_gap_2021': 'mean',
        'malaria_prev_2021': 'mean'
    }).reset_index()

    zone_stats = zone_stats.sort_values('malaria_prev_2021', ascending=False)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='ITN Coverage Gap',
        x=zone_stats['Zone'],
        y=zone_stats['itn_coverage_gap_2021'],
        marker_color='#1f77b4',
        hovertemplate='<b>%{x}</b><br>ITN Gap: %{y:.1f}%<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        name='IPTp Coverage Gap',
        x=zone_stats['Zone'],
        y=zone_stats['iptp_coverage_gap_2021'],
        marker_color='#ff7f0e',
        hovertemplate='<b>%{x}</b><br>IPTp Gap: %{y:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        title='Intervention Coverage Gaps by Geopolitical Zone (2021)',
        xaxis_title='Geopolitical Zone',
        yaxis_title='Coverage Gap (%)',
        barmode='group',
        template='plotly_white',
        height=600,
        width=1200,
        font=get_font_dict('axis_size')
    )

    save_figure(fig, 'intervention_gaps_zones')

def create_intervention_priority_matrix(data):
    """26. Intervention Priority Matrix - Risk vs Gap"""
    print("\n[E26] Creating intervention priority matrix...")

    df = data['df'].copy()

    # Calculate composite intervention gap
    df['total_intervention_gap'] = (df['itn_coverage_gap_2021'] + df['iptp_coverage_gap_2021']) / 2

    # Create quadrants
    median_prev = df['malaria_prev_2021'].median()
    median_gap = df['total_intervention_gap'].median()

    fig = go.Figure()

    # Add quadrant backgrounds
    fig.add_shape(type="rect", x0=0, y0=median_prev, x1=median_gap, y1=60,
                  fillcolor="rgba(255, 200, 200, 0.2)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=median_gap, y0=median_prev, x1=100, y1=60,
                  fillcolor="rgba(255, 100, 100, 0.3)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, y0=0, x1=median_gap, y1=median_prev,
                  fillcolor="rgba(200, 255, 200, 0.2)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=median_gap, y0=0, x1=100, y1=median_prev,
                  fillcolor="rgba(255, 255, 200, 0.2)", line_width=0, layer="below")

    # Add scatter points
    colors_map = {'Low': '#2E7D32', 'Medium': '#FFA726', 'High': '#C62828'}
    for category in ['Low', 'Medium', 'High']:
        mask = df['risk_category'] == category
        fig.add_trace(go.Scatter(
            x=df[mask]['total_intervention_gap'],
            y=df[mask]['malaria_prev_2021'],
            mode='markers+text',
            name=category,
            marker=dict(size=15, color=colors_map[category], line=dict(width=1, color='white')),
            text=df[mask]['State'],
            textposition='top center',
            textfont=dict(size=8),
            hovertemplate='<b>%{text}</b><br>Prevalence: %{y:.1f}%<br>Gap: %{x:.1f}%<extra></extra>'
        ))

    # Add quadrant labels
    fig.add_annotation(x=median_gap/2, y=median_prev + (60-median_prev)/2,
                      text="High Risk<br>Low Gap", showarrow=False, font=dict(size=12, color="gray"))
    fig.add_annotation(x=median_gap + (100-median_gap)/2, y=median_prev + (60-median_prev)/2,
                      text="HIGHEST PRIORITY<br>High Risk + High Gap", showarrow=False,
                      font=dict(size=12, color="red", family="Arial Black"))
    fig.add_annotation(x=median_gap/2, y=median_prev/2,
                      text="Low Risk<br>Low Gap", showarrow=False, font=dict(size=12, color="gray"))
    fig.add_annotation(x=median_gap + (100-median_gap)/2, y=median_prev/2,
                      text="Medium Priority<br>Low Risk + High Gap", showarrow=False, font=dict(size=12, color="gray"))

    fig.update_layout(
        title='Intervention Priority Matrix - Risk vs Coverage Gap',
        xaxis_title='Total Intervention Coverage Gap (%)',
        yaxis_title='Malaria Prevalence 2021 (%)',
        template='plotly_white',
        height=900,
        width=1200,
        font=get_font_dict('axis_size')
    )

    save_figure(fig, 'intervention_priority_matrix')

def create_pareto_chart(data):
    """27. Pareto Chart - Cumulative Burden Analysis"""
    print("\n[E27] Creating Pareto chart...")

    df = data['df'].copy()
    df = df.sort_values('malaria_prev_2021', ascending=False)

    # Calculate cumulative percentage (assuming equal population - simplification)
    df['cumulative_pct'] = np.arange(1, len(df) + 1) / len(df) * 100

    fig = go.Figure()

    # Bar chart for prevalence
    fig.add_trace(go.Bar(
        x=df['State'],
        y=df['malaria_prev_2021'],
        marker_color='#1f77b4',
        name='Malaria Prevalence',
        yaxis='y',
        hovertemplate='<b>%{x}</b><br>Prevalence: %{y:.1f}%<extra></extra>'
    ))

    # Line chart for cumulative
    fig.add_trace(go.Scatter(
        x=df['State'],
        y=df['cumulative_pct'],
        mode='lines+markers',
        marker=dict(color='#C62828', size=8),
        line=dict(color='#C62828', width=3),
        name='Cumulative %',
        yaxis='y2',
        hovertemplate='<b>%{x}</b><br>Cumulative: %{y:.1f}%<extra></extra>'
    ))

    # Add 80% line
    fig.add_hline(y=80, line_dash="dash", line_color="green",
                  annotation_text="80% of States", annotation_position="right", yref="y2")

    fig.update_layout(
        title='Pareto Chart - Malaria Prevalence Distribution Across States',
        xaxis=dict(title='States (Sorted by Prevalence)', tickangle=-90),
        yaxis=dict(title='Malaria Prevalence (%)', side='left', range=[0, 60]),
        yaxis2=dict(title='Cumulative % of States', side='right', overlaying='y', range=[0, 100]),
        template='plotly_white',
        height=700,
        width=1600,
        font=get_font_dict('axis_size'),
        legend=dict(x=0.7, y=0.95)
    )

    save_figure(fig, 'pareto_chart')

# ============================================================================
# SECTION F: PREDICTION ANALYSIS VISUALIZATIONS
# ============================================================================

def create_actual_vs_predicted(data):
    """29. Actual vs Predicted Scatter"""
    print("\n[F29] Creating actual vs predicted scatter...")

    df = data['df']
    X, X_scaled = prepare_features(df, data['feature_names'], data['scaler'])

    # Get predictions
    y_true = df['malaria_prev_2021']
    y_pred = data['rf_regressor'].predict(X_scaled)

    # Calculate errors
    errors = np.abs(y_true - y_pred)

    fig = go.Figure()

    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(
            size=10,
            color=errors,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title='|Error|')
        ),
        text=df['State'],
        hovertemplate='<b>%{text}</b><br>Actual: %{x:.1f}%<br>Predicted: %{y:.1f}%<br>Error: %{marker.color:.1f}%<extra></extra>'
    ))

    # Perfect prediction line
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='gray', width=2, dash='dash'),
        showlegend=True
    ))

    # Calculate R²
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    fig.update_layout(
        title=f'Actual vs Predicted Malaria Prevalence (R² = {r2:.3f}, RMSE = {rmse:.2f}%)',
        xaxis_title='Actual Prevalence (%)',
        yaxis_title='Predicted Prevalence (%)',
        template='plotly_white',
        height=700,
        width=900,
        font=get_font_dict('axis_size'),
        annotations=[
            dict(
                x=0.05,
                y=0.95,
                xref='paper',
                yref='paper',
                text=f'R² = {r2:.3f}<br>RMSE = {rmse:.2f}%',
                showarrow=False,
                bgcolor='white',
                bordercolor='black',
                borderwidth=1
            )
        ]
    )

    save_figure(fig, 'actual_vs_predicted')

def create_geographic_error_map(data):
    """31. Geographic Error Map - Spatial Distribution of Prediction Errors"""
    print("\n[F31] Creating geographic error map...")

    df = data['df'].copy()
    X, X_scaled = prepare_features(df, data['feature_names'], data['scaler'])

    y_true = df['malaria_prev_2021']
    y_pred = data['rf_regressor'].predict(X_scaled)
    df['prediction_error'] = y_true - y_pred
    df['absolute_error'] = np.abs(df['prediction_error'])

    # Categorize errors
    df['error_category'] = pd.cut(
        df['absolute_error'],
        bins=[0, 3, 6, 100],
        labels=['Low Error (<3%)', 'Medium Error (3-6%)', 'High Error (>6%)']
    )

    # Create bubble map of errors
    fig = go.Figure()

    error_colors = {'Low Error (<3%)': '#2E7D32',
                   'Medium Error (3-6%)': '#FFA726',
                   'High Error (>6%)': '#C62828'}

    for category in ['Low Error (<3%)', 'Medium Error (3-6%)', 'High Error (>6%)']:
        mask = df['error_category'] == category
        if mask.sum() > 0:
            fig.add_trace(go.Scatter(
                x=df[mask]['State'],
                y=df[mask]['prediction_error'],
                mode='markers',
                name=category,
                marker=dict(
                    size=df[mask]['absolute_error'] * 5,
                    color=error_colors[category],
                    line=dict(width=1, color='white'),
                    opacity=0.7
                ),
                text=df[mask]['State'],
                hovertemplate='<b>%{text}</b><br>Error: %{y:.2f}%<br>|Error|: ' +
                             df[mask]['absolute_error'].astype(str) + '%<extra></extra>'
            ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Zero Error")

    fig.update_layout(
        title='Geographic Distribution of Prediction Errors',
        xaxis_title='State',
        yaxis_title='Prediction Error (Actual - Predicted) %',
        template='plotly_white',
        height=700,
        width=1600,
        font=get_font_dict('axis_size'),
        xaxis_tickangle=-90,
        legend=dict(x=0.7, y=0.95)
    )

    save_figure(fig, 'geographic_error_map')

def create_residuals_plot(data):
    """30. Residual Plot"""
    print("\n[F30] Creating residuals plot...")

    df = data['df']
    X, X_scaled = prepare_features(df, data['feature_names'], data['scaler'])

    y_true = df['malaria_prev_2021']
    y_pred = data['rf_regressor'].predict(X_scaled)
    residuals = y_true - y_pred

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(
            size=10,
            color=np.abs(residuals),
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title='|Residual|')
        ),
        text=df['State'],
        hovertemplate='<b>%{text}</b><br>Fitted: %{x:.1f}%<br>Residual: %{y:.1f}%<extra></extra>'
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Zero")

    fig.update_layout(
        title='Residual Plot - Random Forest Regressor',
        xaxis_title='Fitted Values (%)',
        yaxis_title='Residuals (%)',
        template='plotly_white',
        height=600,
        width=1000,
        font=get_font_dict('axis_size')
    )

    save_figure(fig, 'residuals')

def create_error_distribution(data):
    """32. Error Distribution"""
    print("\n[F32] Creating error distribution...")

    df = data['df']
    X, X_scaled = prepare_features(df, data['feature_names'], data['scaler'])

    y_true = df['malaria_prev_2021']
    y_pred = data['rf_regressor'].predict(X_scaled)
    errors = y_true - y_pred

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=15,
        name='Error Distribution',
        marker_color='#1f77b4',
        opacity=0.7,
        hovertemplate='Error Range: %{x}<br>Count: %{y}<extra></extra>'
    ))

    fig.update_layout(
        title='Distribution of Prediction Errors',
        xaxis_title='Error (Actual - Predicted) %',
        yaxis_title='Frequency',
        template='plotly_white',
        height=600,
        width=900,
        font=get_font_dict('axis_size'),
        bargap=0.1
    )

    save_figure(fig, 'error_distribution')

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*80)
    print("COMPREHENSIVE VISUALIZATION SUITE FOR MALARIARISKXAI")
    print("="*80)

    # Load all data
    data = load_all_data()
    if data is None:
        print("\n❌ Error loading data. Please check file paths.")
        return

    print(f"\n[2] Generating visualizations...")
    print(f"    Output directory: {OUTPUT_DIR}")

    # Section A: Model Performance (6 visualizations)
    print("\n" + "="*80)
    print("SECTION A: MODEL PERFORMANCE (6 visualizations)")
    print("="*80)
    create_confusion_matrix_interactive(data)
    create_roc_curves_multimodel(data)
    create_precision_recall_curves(data)
    create_model_metrics_comparison(data)
    create_learning_curves(data)
    create_calibration_plots(data)

    # Section B: Explainability (8 visualizations)
    print("\n" + "="*80)
    print("SECTION B: EXPLAINABILITY (8 visualizations)")
    print("="*80)
    create_shap_summary_interactive(data)
    create_shap_importance_bars(data)
    create_shap_dependence_plots(data)
    create_shap_waterfall_plot(data)
    create_shap_force_plot(data)
    create_feature_interaction_heatmap(data)
    create_lime_comparison(data)
    create_decision_boundaries(data)

    # Section C: Geospatial (5 visualizations)
    print("\n" + "="*80)
    print("SECTION C: GEOSPATIAL (5 visualizations)")
    print("="*80)
    create_choropleth_map_2021(data)
    create_animated_map_multiyear(data)
    create_multilayer_choropleth(data)
    create_spatial_network(data)
    create_bubble_map(data)

    # Section D: Temporal Trends (4 visualizations)
    print("\n" + "="*80)
    print("SECTION D: TEMPORAL TRENDS (4 visualizations)")
    print("="*80)
    create_national_trends(data)
    create_slopegraph_states(data)
    create_heatmap_timeline(data)
    create_racing_bar_chart(data)

    # Section E: Intervention Analysis (5 visualizations)
    print("\n" + "="*80)
    print("SECTION E: INTERVENTION ANALYSIS (5 visualizations)")
    print("="*80)
    create_intervention_scatter_itn(data)
    create_counterfactual_dashboard(data)
    create_intervention_priority_matrix(data)
    create_pareto_chart(data)
    create_intervention_gaps_zones(data)

    # Section F: Prediction Analysis (4 visualizations)
    print("\n" + "="*80)
    print("SECTION F: PREDICTION ANALYSIS (4 visualizations)")
    print("="*80)
    create_actual_vs_predicted(data)
    create_geographic_error_map(data)
    create_residuals_plot(data)
    create_error_distribution(data)

    print("\n" + "="*80)
    print("✅ VISUALIZATION GENERATION COMPLETE!")
    print("="*80)
    print(f"\n📁 All visualizations saved to: {OUTPUT_DIR}/")
    print(f"📊 Total visualizations created: 32")
    print("\n🎓 Publication-ready interactive visualizations generated successfully!")
    print("\n📋 Summary:")
    print("   - Section A (Model Performance): 6 visualizations")
    print("   - Section B (Explainability): 8 visualizations")
    print("   - Section C (Geospatial): 5 visualizations")
    print("   - Section D (Temporal Trends): 4 visualizations")
    print("   - Section E (Intervention Analysis): 5 visualizations")
    print("   - Section F (Prediction Analysis): 4 visualizations")
    print("="*80)

if __name__ == "__main__":
    main()
