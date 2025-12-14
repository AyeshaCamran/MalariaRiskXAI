"""
MalariaRiskXAI - Interactive Streamlit Dashboard
Explainable AI Framework for Malaria Risk Prediction in Nigeria
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from PIL import Image
import os
import matplotlib.pyplot as plt

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="MalariaRiskXAI",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DARK THEME STYLING
# ============================================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1E1E1E;
    }

    /* Metric containers */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #00D9FF;
    }

    /* Headers */
    h1 {
        color: #00D9FF;
        font-weight: 700;
        padding-top: 1rem;
    }

    h2 {
        color: #4ECDC4;
        font-weight: 600;
    }

    h3 {
        color: #95E1D3;
    }

    /* Cards */
    .css-1r6slb0 {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #00D9FF;
        color: #0E1117;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #4ECDC4;
        transform: scale(1.05);
    }

    /* Info boxes */
    .stAlert {
        background-color: #1E1E1E;
        border-left: 4px solid #00D9FF;
    }

    /* Tables */
    .dataframe {
        background-color: #1E1E1E;
    }

    /* Remove gradients */
    .css-1d391kg, .css-1v0mbdj {
        background: none !important;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .element-container {
        animation: fadeIn 0.5s ease-in;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #00D9FF !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA AND MODELS
# ============================================================================
@st.cache_data
def load_data():
    """Load all datasets"""
    df = pd.read_csv('data/data_with_features.csv')
    predictions = pd.read_csv('data/model_predictions_2021.csv')

    # Merge predictions into main dataframe
    df = df.merge(
        predictions[['State', 'risk_class_2021', 'predicted_prevalence', 'prediction_error']],
        on='State',
        how='left'
    )

    return df, predictions

@st.cache_resource
def load_models():
    """Load corrected trained models"""
    models = {
        'rf_classifier': joblib.load('models/rf_classifier_corrected.pkl'),
        'lr_classifier': joblib.load('models/lr_classifier_corrected.pkl'),
        'mlp_classifier': joblib.load('models/mlp_classifier_corrected.pkl'),
        'stacking_classifier': joblib.load('models/stacking_classifier_corrected.pkl'),
        'rf_regressor': joblib.load('models/rf_regressor_corrected.pkl'),
        'linear_regressor': joblib.load('models/linear_regressor_corrected.pkl'),
        'mlp_regressor': joblib.load('models/mlp_regressor_corrected.pkl'),
        'scaler': joblib.load('models/scaler_corrected.pkl'),
        'shap_explainer': joblib.load('models/shap_explainer_corrected.pkl'),
        'feature_names': joblib.load('models/feature_names_corrected.pkl'),
        'metadata': joblib.load('models/metadata_corrected.pkl')
    }
    return models

# Load data
df, predictions = load_data()
models = load_models()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.title("🦟 MalariaRiskXAI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Data Explorer", "🤖 Model Performance",
     "🔍 XAI Insights", "🎯 Risk Predictor"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    **MalariaRiskXAI** is an Explainable AI framework for predicting
    malaria risk hotspots in Nigeria using NMIS data.

    **Key Features:**
    - 37 Nigerian states analyzed
    - Multi-temporal data (2015-2021)
    - Validated ML models
    - Advanced XAI explanations
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Source:** Nigeria Malaria Indicator Survey (NMIS)")
st.sidebar.markdown("**Models:** RF, LR, MLP, Stacking")
st.sidebar.markdown("**XAI Methods:** SHAP, LIME, PDPs")

# ============================================================================
# PAGE 1: HOME
# ============================================================================
if page == "🏠 Home":
    # Header with animation
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 0;'>🦟 MalariaRiskXAI</h1>
        <p style='font-size: 1.2rem; color: #95E1D3;'>
            Explainable AI Framework for Predicting Malaria Risk Hotspots in Nigeria
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="States Analyzed",
            value="37",
            delta="100% Coverage"
        )

    with col2:
        st.metric(
            label="Model Balanced Accuracy",
            value=f"{models['metadata']['model_performance']['rf_classifier_balanced_accuracy']:.1%}",
            delta="Random Forest"
        )

    with col3:
        avg_prev = df['malaria_prev_2021'].mean()
        prev_2018 = df['malaria_prev_2018'].mean()
        delta = avg_prev - prev_2018
        st.metric(
            label="Avg Prevalence 2021",
            value=f"{avg_prev:.1f}%",
            delta=f"{delta:.1f}%"
        )

    with col4:
        high_risk = len(df[df['malaria_prev_2021'] >= 40])
        st.metric(
            label="High Risk States",
            value=high_risk,
            delta="Kebbi (49%)"
        )

    st.markdown("---")

    # Two-column layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📈 National Trends")

        # Temporal trend chart
        trends_data = pd.DataFrame({
            'Year': [2015, 2018, 2021],
            'Prevalence': [
                df['malaria_prev_2015'].mean(),
                df['malaria_prev_2018'].mean(),
                df['malaria_prev_2021'].mean()
            ]
        })

        fig = px.line(
            trends_data,
            x='Year',
            y='Prevalence',
            markers=True,
            template='plotly_dark'
        )
        fig.update_traces(
            line_color='#00D9FF',
            marker=dict(size=12, color='#4ECDC4'),
            line=dict(width=3)
        )
        fig.update_layout(
            height=300,
            paper_bgcolor='#1E1E1E',
            plot_bgcolor='#1E1E1E',
            font=dict(color='white'),
            yaxis_title="Average Prevalence (%)",
            xaxis_title="Year"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"""
        **Key Finding:** National malaria prevalence declined from
        **{trends_data['Prevalence'].iloc[0]:.1f}%** in 2015 to
        **{trends_data['Prevalence'].iloc[2]:.1f}%** in 2021,
        a **{((trends_data['Prevalence'].iloc[0] - trends_data['Prevalence'].iloc[2]) / trends_data['Prevalence'].iloc[0] * 100):.1f}%** reduction.
        """)

    with col2:
        st.markdown("### 🗺️ Risk Distribution 2021")

        # Risk categories
        risk_counts = df['malaria_prev_2021'].apply(lambda x:
            'High (≥40%)' if x >= 40 else
            'Medium (10-40%)' if x >= 10 else
            'Low (<10%)'
        ).value_counts()

        fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.4,
            marker=dict(colors=['#FF6B6B', '#FFD93D', '#6BCB77']),
            textfont=dict(size=14, color='white')
        )])

        fig.update_layout(
            height=300,
            paper_bgcolor='#1E1E1E',
            plot_bgcolor='#1E1E1E',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.1
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(f"""
        **Distribution:**
        - 🔴 High Risk: **{risk_counts.get('High (≥40%)', 0)}** states
        - 🟡 Medium Risk: **{risk_counts.get('Medium (10-40%)', 0)}** states
        - 🟢 Low Risk: **{risk_counts.get('Low (<10%)', 0)}** states
        """)

    st.markdown("---")

    # Project Highlights
    st.markdown("### 🌟 Project Highlights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### 📊 Comprehensive Data
        - **37 states** across 6 geopolitical zones
        - **Multi-temporal** analysis (2015, 2018, 2021)
        - **30 engineered features**
        - **21 baseline indicators** from NMIS
        """)

    with col2:
        st.markdown("""
        #### 🤖 Advanced ML Models
        - **Random Forest**
        - **MLP Neural Network**
        - **Logistic Regression**
        - **Stacking Ensemble**
        """)

    with col3:
        st.markdown("""
        #### 🔍 Explainable AI
        - **SHAP** analysis (global + local)
        - **LIME** explanations (6 states)
        - **Partial Dependence Plots**
        - **Feature interactions**
        """)

    st.markdown("---")

    # Top Risk States
    st.markdown("### 🎯 Top 10 High-Risk States (2021)")

    top_10 = df.nlargest(10, 'malaria_prev_2021')[['State', 'Zone', 'malaria_prev_2021', 'anaemia_2021', 'itn_ownership_2021']]
    top_10.columns = ['State', 'Zone', 'Malaria Prevalence (%)', 'Anaemia (%)', 'ITN Ownership (%)']

    # Color code by risk
    def color_risk(val):
        if val >= 40:
            return 'background-color: #FF6B6B; color: white'
        elif val >= 10:
            return 'background-color: #FFD93D; color: black'
        else:
            return 'background-color: #6BCB77; color: white'

    st.dataframe(
        top_10.style.applymap(color_risk, subset=['Malaria Prevalence (%)']),
        use_container_width=True,
        height=400
    )

# ============================================================================
# PAGE 2: DATA EXPLORER
# ============================================================================
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    st.markdown("Explore malaria indicators across Nigerian states")
    st.markdown("---")

    # Visualization selector
    col1, col2 = st.columns([2, 1])

    with col1:
        viz_type = st.selectbox(
            "Select Visualization",
            [
                "Geographic Risk Distribution",
                "Temporal Evolution (2015-2021)",
                "Temporal Risk Animation",
                "Correlation Heatmap",
                "Risk Distribution",
                "Top 10 High-Risk States",
                "Zone-Level Comparison",
                "Intervention Coverage"
            ]
        )

    with col2:
        show_data = st.checkbox("Show raw data", value=False)

    st.markdown("---")

    # Display visualizations
    if viz_type == "Geographic Risk Distribution":
        st.warning("Interactive map is currently unavailable. Displaying static plot.")
        if os.path.exists('visualizations/geographic_malaria_risk_distribution.png'):
            st.image('visualizations/geographic_malaria_risk_distribution.png', use_container_width=True)
        else:
            # Create interactive map
            fig = px.scatter(
                df,
                x='Zone',
                y='malaria_prev_2021',
                size='anaemia_2021',
                color='malaria_prev_2021',
                hover_name='State',
                color_continuous_scale='Reds',
                template='plotly_dark',
                title='Malaria Prevalence by State and Zone'
            )
            fig.update_layout(
                paper_bgcolor='#1E1E1E',
                plot_bgcolor='#1E1E1E',
                font=dict(color='white'),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **Insight:** Northern zones (North West, North East) show higher malaria
        prevalence compared to Southern zones. Kebbi state in the North West has
        the highest prevalence at 49%.
        """)

    elif viz_type == "Temporal Evolution (2015-2021)":
        if os.path.exists('visualizations/temporal_evolution_2015_2021.png'):
            st.image('visualizations/temporal_evolution_2015_2021.png', use_container_width=True)
        else:
            # Create temporal chart
            temporal_data = []
            for _, row in df.iterrows():
                temporal_data.extend([
                    {'State': row['State'], 'Year': 2015, 'Prevalence': row['malaria_prev_2015']},
                    {'State': row['State'], 'Year': 2018, 'Prevalence': row['malaria_prev_2018']},
                    {'State': row['State'], 'Year': 2021, 'Prevalence': row['malaria_prev_2021']}
                ])

            temporal_df = pd.DataFrame(temporal_data)

            fig = px.line(
                temporal_df,
                x='Year',
                y='Prevalence',
                color='State',
                template='plotly_dark',
                title='Malaria Prevalence Trends by State (2015-2021)'
            )
            fig.update_layout(
                paper_bgcolor='#1E1E1E',
                plot_bgcolor='#1E1E1E',
                font=dict(color='white'),
                height=600,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        st.success("""
        **Key Trend:** Overall declining trend from 2015 to 2021.
        The national average decreased by 6.6 percentage points (24% reduction).
        """)

    elif viz_type == "Temporal Risk Animation":
        st.markdown("### 📈 Temporal Risk Animation")
        st.info("Watch the evolution of malaria prevalence across states from 2015 to 2021.")

        # Prepare data for animation
        temporal_data = []
        for _, row in df.iterrows():
            temporal_data.extend([
                {'State': row['State'], 'Year': 2015, 'Prevalence': row['malaria_prev_2015']},
                {'State': row['State'], 'Year': 2018, 'Prevalence': row['malaria_prev_2018']},
                {'State': row['State'], 'Year': 2021, 'Prevalence': row['malaria_prev_2021']}
            ])
        temporal_df = pd.DataFrame(temporal_data)

        # Bar chart race
        fig = px.bar(
            temporal_df,
            x="State",
            y="Prevalence",
            color="Prevalence",
            animation_frame="Year",
            animation_group="State",
            range_y=[0, temporal_df['Prevalence'].max() * 1.1],
            color_continuous_scale='Reds',
            template='plotly_dark',
            title="Malaria Prevalence by State (2015-2021)"
        )
        fig.update_layout(
            paper_bgcolor='#1E1E1E',
            plot_bgcolor='#1E1E1E',
            font=dict(color='white'),
            height=600,
            xaxis={'tickangle': -90}
        )
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Correlation Heatmap":
        if os.path.exists('visualizations/correlation_heatmap_2021.png'):
            st.image('visualizations/correlation_heatmap_2021.png', use_container_width=True)
        else:
            # Create correlation heatmap
            corr_cols = ['malaria_prev_2021', 'itn_ownership_2021', 'anaemia_2021',
                        'iptp2_2021', 'diag_test_2021']
            corr_matrix = df[corr_cols].corr()

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu_r',
                zmid=0
            ))
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='#1E1E1E',
                plot_bgcolor='#1E1E1E',
                font=dict(color='white'),
                height=600,
                title='Feature Correlations'
            )
            st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **Insight:** Strong correlations between anaemia and malaria prevalence.
        ITN ownership shows negative correlation with prevalence.
        """)

    elif viz_type == "Risk Distribution":
        if os.path.exists('visualizations/risk_distribution.png'):
            st.image('visualizations/risk_distribution.png', use_container_width=True)

    elif viz_type == "Top 10 High-Risk States":
        if os.path.exists('visualizations/top10_high_risk_heatmap.png'):
            st.image('visualizations/top10_high_risk_heatmap.png', use_container_width=True)

    elif viz_type == "Zone-Level Comparison":
        st.markdown("### 📊 Malaria Prevalence by Geopolitical Zone")

        # Calculate zone statistics
        zone_stats = df.groupby('Zone').agg({
            'malaria_prev_2021': ['mean', 'min', 'max', 'count'],
            'itn_ownership_2021': 'mean',
            'iptp2_2021': 'mean'
        }).round(1)

        zone_stats.columns = ['Avg_Prevalence', 'Min_Prevalence', 'Max_Prevalence', 'State_Count', 'Avg_ITN', 'Avg_IPTp']
        zone_stats = zone_stats.reset_index().sort_values('Avg_Prevalence', ascending=False)

        # Create two columns for visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Clean bar chart for prevalence
            fig1 = go.Figure()

            # Add bars with error ranges
            fig1.add_trace(go.Bar(
                x=zone_stats['Zone'],
                y=zone_stats['Avg_Prevalence'],
                name='Average Prevalence',
                marker_color='#C62828',
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=zone_stats['Max_Prevalence'] - zone_stats['Avg_Prevalence'],
                    arrayminus=zone_stats['Avg_Prevalence'] - zone_stats['Min_Prevalence']
                ),
                hovertemplate='<b>%{x}</b><br>Average: %{y:.1f}%<br><extra></extra>'
            ))

            fig1.update_layout(
                title='Average Malaria Prevalence by Zone (2021)',
                xaxis_title='Geopolitical Zone',
                yaxis_title='Prevalence (%)',
                template='plotly_white',
                height=400,
                showlegend=False,
                xaxis_tickangle=-45
            )

            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # Grouped bar chart for interventions
            fig2 = go.Figure()

            fig2.add_trace(go.Bar(
                name='ITN Ownership',
                x=zone_stats['Zone'],
                y=zone_stats['Avg_ITN'],
                marker_color='#2E7D32',
                hovertemplate='<b>%{x}</b><br>ITN: %{y:.1f}%<extra></extra>'
            ))

            fig2.add_trace(go.Bar(
                name='IPTp Coverage',
                x=zone_stats['Zone'],
                y=zone_stats['Avg_IPTp'],
                marker_color='#1976D2',
                hovertemplate='<b>%{x}</b><br>IPTp: %{y:.1f}%<extra></extra>'
            ))

            fig2.update_layout(
                title='Intervention Coverage by Zone (2021)',
                xaxis_title='Geopolitical Zone',
                yaxis_title='Coverage (%)',
                template='plotly_white',
                height=400,
                barmode='group',
                xaxis_tickangle=-45
            )

            st.plotly_chart(fig2, use_container_width=True)

        # Add summary table
        st.markdown("### 📋 Zone Statistics Summary")
        summary_df = zone_stats.copy()
        summary_df.columns = ['Zone', 'Avg Prevalence (%)', 'Min (%)', 'Max (%)', 'States', 'Avg ITN (%)', 'Avg IPTp (%)']
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    elif viz_type == "Intervention Coverage":
        if os.path.exists('visualizations/intervention_recommendation_dashboard.png'):
            st.image('visualizations/intervention_recommendation_dashboard.png', use_container_width=True)

    # Show raw data
    if show_data:
        st.markdown("---")
        st.markdown("### 📋 Raw Data")
        st.dataframe(df, use_container_width=True, height=400)

# ============================================================================
# PAGE 3: MODEL PERFORMANCE
# ============================================================================
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    st.markdown("Comprehensive evaluation of all machine learning models")
    st.markdown("---")

    # Model metrics
    metadata = models['metadata']
    perf = metadata['model_performance']

    # Classification Models
    st.markdown("### 🎯 Classification Models (Balanced Accuracy)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Random Forest",
            value=f"{perf['rf_classifier_balanced_accuracy']:.1%}",
            delta="Best Model"
        )

    with col2:
        st.metric(
            label="Logistic Regression",
            value=f"{perf['lr_classifier_balanced_accuracy']:.1%}",
            delta="Interpretable"
        )

    with col3:
        st.metric(
            label="MLP Neural Network",
            value=f"{perf['mlp_classifier_balanced_accuracy']:.1%}",
            delta="Deep Learning"
        )

    with col4:
        st.metric(
            label="Stacking Ensemble",
            value=f"{perf['stacking_classifier_balanced_accuracy']:.1%}",
            delta="Ensemble"
        )

    st.markdown("---")

    # Regression Models
    st.markdown("### 📊 Regression Models")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Random Forest RMSE",
            value=f"{perf['rf_regressor_rmse']:.2f}%",
            delta=f"R² = {perf['rf_regressor_r2']:.3f}"
        )

    with col2:
        st.metric(
            label="Linear Regression RMSE",
            value=f"{perf['linear_regressor_rmse']:.2f}%",
            delta=f"R² = {perf['linear_regressor_r2']:.3f}"
        )

    with col3:
        st.metric(
            label="MLP Regressor RMSE",
            value=f"{perf['mlp_regressor_rmse']:.2f}%",
            delta=f"R² = {perf['mlp_regressor_r2']:.3f}"
        )

    st.markdown("---")

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎭 Confusion Matrices")
        if os.path.exists('visualizations/confusion_matrices_all_models.png'):
            st.image('visualizations/confusion_matrices_all_models.png', use_container_width=True)
        else:
            st.warning("Confusion matrix visualization not found.")


    with col2:
        st.markdown("### 📈 ROC Curves")
        if os.path.exists('visualizations/roc_curves_multiclass.png'):
            st.image('visualizations/roc_curves_multiclass.png', use_container_width=True)
        else:
            st.warning("ROC curve visualization not found.")

    st.success("""
    **Key Findings:**
    - The models now show realistic performance after correcting for data leakage.
    - Balanced accuracy is used to account for the imbalanced nature of the dataset.
    - Random Forest is the best performing classification model.
    - The regression models provide a reasonable estimate of malaria prevalence.
    """)

    # Model comparison table
    st.markdown("---")
    st.markdown("### 📋 Detailed Model Comparison")

    comparison_data = {
        'Model': ['Random Forest', 'Logistic Regression', 'MLP Neural Net', 'Stacking'],
        'Balanced Accuracy': [
            f"{perf['rf_classifier_balanced_accuracy']:.1%}",
            f"{perf['lr_classifier_balanced_accuracy']:.1%}",
            f"{perf['mlp_classifier_balanced_accuracy']:.1%}",
            f"{perf['stacking_classifier_balanced_accuracy']:.1%}"
        ],
        'Type': ['Tree-based', 'Linear', 'Deep Learning', 'Ensemble'],
        'Interpretability': ['High', 'Very High', 'Low', 'Medium'],
        'Training Time': ['Fast', 'Very Fast', 'Moderate', 'Slow']
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE 4: XAI INSIGHTS
# ============================================================================
elif page == "🔍 XAI Insights":
    st.title("🔍 Explainable AI Insights")
    st.markdown("Understand what drives malaria risk predictions")
    st.markdown("---")

    # XAI method selector
    xai_method = st.selectbox(
        "Select XAI Method",
        ["SHAP - Global Feature Importance",
         "SHAP - Waterfall Plot",
         "SHAP - Feature Interactions",
         "LIME - Individual State Explanations",
         "Partial Dependence Plots"]
    )

    st.markdown("---")

    if xai_method == "SHAP - Global Feature Importance":
        st.markdown("### 🎯 SHAP Global Feature Importance")

        # Try to load pre-calculated SHAP values first (faster)
        if os.path.exists('models/shap_values_corrected.npy'):
            try:
                with st.spinner("Loading SHAP values..."):
                    feature_cols = models['feature_names']
                    X = df[feature_cols].fillna(df[feature_cols].median())
                    X_scaled = models['scaler'].transform(X)

                    # Load pre-calculated SHAP values
                    shap_values_high_risk = np.load('models/shap_values_corrected.npy')

                    # Create the plot
                    st.markdown("#### Summary Plot - Corrected Model")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    shap.summary_plot(shap_values_high_risk, X_scaled, feature_names=feature_cols, show=False)
                    st.pyplot(fig)
                    plt.close()
            except Exception as e:
                st.error(f"Error loading pre-calculated SHAP values: {e}")
                st.info("Please run regenerate_shap_corrected.py to generate SHAP values.")
        else:
            with st.spinner("Calculating SHAP values... (this may take a moment)"):
                try:
                    # Prepare the data
                    feature_cols = models['feature_names']
                    X = df[feature_cols].fillna(df[feature_cols].median())
                    X_scaled = models['scaler'].transform(X)

                    # Calculate SHAP values
                    shap_values = models['shap_explainer'].shap_values(X_scaled)

                    # Handle multi-class SHAP values (shape: n_samples, n_features, n_classes)
                    rf_classifier = models['rf_classifier']
                    class_names = rf_classifier.classes_
                    high_risk_idx = np.where(class_names == 'High')[0][0]
                    shap_values_high_risk = shap_values[:, :, high_risk_idx]

                    # Create the plot
                    st.markdown("#### Summary Plot")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    shap.summary_plot(shap_values_high_risk, X_scaled, feature_names=feature_cols, show=False)
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Error calculating SHAP values: {e}")

        st.info("""
        **SHAP Interpretation (Corrected Model):**
        - Red indicates high feature values
        - Blue indicates low feature values
        - Position on x-axis shows impact on prediction
        - **Historical prevalence (2015, 2018)** is the strongest predictor
        - **Intervention gaps (ITN, IPTp)** are critical factors
        - **No data leakage** - only valid predictors used
        """)

    elif xai_method == "SHAP - Waterfall Plot":
        st.markdown("### 📍 SHAP Waterfall Plot")
        selected_state = st.selectbox("Select State", sorted(df['State'].unique()))

        if selected_state:
            # Check if pre-generated waterfall exists
            waterfall_file = f'visualizations/shap_waterfall_{selected_state.lower()}_corrected.png'
            if os.path.exists(waterfall_file):
                st.image(waterfall_file, use_container_width=True, caption=f"SHAP Explanation for {selected_state}")
            else:
                with st.spinner(f"Generating SHAP plot for {selected_state}..."):
                    try:
                        # Prepare the data for the selected state
                        state_index = df[df['State'] == selected_state].index[0]
                        feature_cols = models['feature_names']
                        X = df[feature_cols].fillna(df[feature_cols].median())
                        X_scaled = models['scaler'].transform(X)

                        # Get SHAP values
                        if os.path.exists('models/shap_values_corrected.npy'):
                            shap_values_high_risk = np.load('models/shap_values_corrected.npy')
                            base_value = np.load('models/shap_base_value_corrected.npy')
                            # Ensure base_value is a scalar
                            if isinstance(base_value, np.ndarray):
                                base_value = float(base_value.item() if base_value.size == 1 else base_value.mean())
                        else:
                            shap_values = models['shap_explainer'].shap_values(X_scaled)
                            rf_classifier = models['rf_classifier']
                            class_names = rf_classifier.classes_
                            high_risk_idx = np.where(class_names == 'High')[0][0]
                            shap_values_high_risk = shap_values[:, :, high_risk_idx]

                            if hasattr(models['shap_explainer'], 'expected_value'):
                                if isinstance(models['shap_explainer'].expected_value, np.ndarray):
                                    base_value = models['shap_explainer'].expected_value[high_risk_idx]
                                else:
                                    base_value = models['shap_explainer'].expected_value
                            else:
                                base_value = 0

                        # Create the waterfall plot
                        st.set_option('deprecation.showPyplotGlobalUse', False)

                        explanation = shap.Explanation(
                            values=shap_values_high_risk[state_index],
                            base_values=float(base_value),
                            data=X_scaled[state_index],
                            feature_names=feature_cols
                        )

                        # waterfall_plot creates its own figure
                        shap.waterfall_plot(explanation, max_display=15, show=False)
                        st.pyplot(plt.gcf(), clear_figure=True)
                        plt.close()
                    except Exception as e:
                        st.error(f"Error generating waterfall plot: {e}")
                        st.info("Try selecting a different state or run regenerate_shap_corrected.py")

    elif xai_method == "SHAP - Feature Interactions":
        st.markdown("### 🔗 SHAP Feature Interactions")

        if os.path.exists('visualizations/shap_interaction_plot.png'):
            st.image('visualizations/shap_interaction_plot.png', use_container_width=True)

        st.success("""
        **Key Interaction:**
        Strong interaction between `malaria_prev_2018` and `anaemia_2021`.
        States with both high historical prevalence AND high anaemia are at
        significantly higher risk.
        """)

        # State-specific SHAP examples
        st.markdown("---")
        st.markdown("### 📍 State-Specific SHAP Explanations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### High Risk: Kebbi")
            if os.path.exists('visualizations/shap_waterfall_kebbi.png'):
                st.image('visualizations/shap_waterfall_kebbi.png', use_container_width=True)

        with col2:
            st.markdown("#### Low Risk: Lagos")
            if os.path.exists('visualizations/shap_waterfall_lagos.png'):
                st.image('visualizations/shap_waterfall_lagos.png', use_container_width=True)

    elif xai_method == "LIME - Individual State Explanations":
        st.markdown("### 🔍 LIME Explanations")

        st.info("""
        **LIME (Local Interpretable Model-agnostic Explanations)** shows which
        features contributed most to the prediction for individual states.
        """)

        # State selector
        lime_state = st.selectbox(
            "Select State",
            ["Kebbi (High Risk)", "Zamfara (Medium Risk)", "Sokoto (Medium Risk)",
             "Bauchi (Medium Risk)", "Lagos (Low Risk)", "Anambra (Low Risk)"]
        )

        state_map = {
            "Kebbi (High Risk)": "kebbi",
            "Zamfara (Medium Risk)": "zamfara",
            "Sokoto (Medium Risk)": "sokoto",
            "Bauchi (Medium Risk)": "bauchi",
            "Lagos (Low Risk)": "lagos",
            "Anambra (Low Risk)": "anambra"
        }

        selected_state = state_map[lime_state]
        img_path = f'visualizations/lime_explanation_{selected_state}.png'

        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.warning(f"LIME explanation for {lime_state} not found.")

        st.markdown("---")

        # LIME summary
        if os.path.exists('data/lime_explanations_summary.csv'):
            st.markdown("### 📊 LIME Summary - All States")
            lime_summary = pd.read_csv('data/lime_explanations_summary.csv')
            st.dataframe(lime_summary, use_container_width=True, height=300)

    elif xai_method == "Partial Dependence Plots":
        st.markdown("### 📈 Partial Dependence Plots")

        st.info("""
        **Partial Dependence Plots** show how each feature affects predictions
        while holding other features constant.
        """)

        if os.path.exists('visualizations/partial_dependence_plots.png'):
            st.image('visualizations/partial_dependence_plots.png', use_container_width=True)

        st.success("""
        **Insights:**
        - Higher historical prevalence → Higher predicted risk (positive relationship)
        - Higher ITN ownership → Lower predicted risk (negative relationship)
        - Anaemia shows strong positive correlation with malaria risk
        - Spatial factors (neighbor prevalence) have moderate impact
        """)

# ============================================================================
# PAGE 5: RISK PREDICTOR
# ============================================================================
elif page == "🎯 Risk Predictor":
    st.title("🎯 Malaria Risk Prediction Simulator")
    st.markdown("Input state-level indicators to predict malaria risk")
    st.markdown("---")

    # Two modes
    mode = st.radio(
        "Select Mode",
        ["🔍 Explore Existing States", "🧪 Custom Prediction Simulator"],
        horizontal=True
    )

    st.markdown("---")

    if mode == "🔍 Explore Existing States":
        st.markdown("### 📍 State-by-State Predictions")

        # State selector
        selected_state = st.selectbox("Select State", sorted(df['State'].unique()))

        if selected_state:
            state_data = df[df['State'] == selected_state].iloc[0]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="Predicted Risk Category",
                    value=state_data['risk_class_2021'] if 'risk_class_2021' in state_data else
                          classify_risk(state_data['malaria_prev_2021'])
                )

            with col2:
                st.metric(
                    label="Malaria Prevalence 2021",
                    value=f"{state_data['malaria_prev_2021']:.1f}%"
                )

            with col3:
                st.metric(
                    label="Zone",
                    value=state_data['Zone']
                )

            st.markdown("---")

            # Key indicators
            st.markdown("### 📊 Key Indicators")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("ITN Ownership", f"{state_data['itn_ownership_2021']:.0f}%")
                st.metric("ITN Access", f"{state_data['itn_access_2021']:.0f}%")

            with col2:
                st.metric("IPTp2 Coverage", f"{state_data['iptp2_2021']:.0f}%")
                st.metric("IPTp3 Coverage", f"{state_data['iptp3_2021']:.0f}%")

            with col3:
                st.metric("Anaemia Prevalence", f"{state_data['anaemia_2021']:.0f}%")
                st.metric("Diagnostic Testing", f"{state_data['diag_test_2021']:.0f}%")

            with col4:
                st.metric("Malaria Messages", f"{state_data['malaria_msg_2021']:.0f}%")
                st.metric("2018 Prevalence", f"{state_data['malaria_prev_2018']:.1f}%")

            # Trend visualization
            st.markdown("---")
            st.markdown("### 📈 Temporal Trend")

            trend_data = pd.DataFrame({
                'Year': [2015, 2018, 2021],
                'Prevalence': [
                    state_data['malaria_prev_2015'],
                    state_data['malaria_prev_2018'],
                    state_data['malaria_prev_2021']
                ]
            })

            fig = px.line(
                trend_data,
                x='Year',
                y='Prevalence',
                markers=True,
                template='plotly_dark',
                title=f'Malaria Prevalence Trend - {selected_state}'
            )
            fig.update_traces(
                line_color='#00D9FF',
                marker=dict(size=12, color='#4ECDC4'),
                line=dict(width=3)
            )
            fig.update_layout(
                paper_bgcolor='#1E1E1E',
                plot_bgcolor='#1E1E1E',
                font=dict(color='white'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    else:  # Custom Prediction Simulator
        st.markdown("### 🧪 Custom Risk Prediction")

        st.info("""
        Adjust the sliders below to simulate different scenarios and see
        how malaria risk changes with different intervention coverage levels.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🛏️ Intervention Coverage")

            itn_ownership = st.slider("ITN Ownership (%)", 0, 100, 50)
            itn_access = st.slider("ITN Access (%)", 0, 100, 40)
            itn_use = st.slider("ITN Use (Children) (%)", 0, 100, 35)

            iptp2 = st.slider("IPTp2 Coverage (%)", 0, 100, 60)
            iptp3 = st.slider("IPTp3 Coverage (%)", 0, 100, 40)

        with col2:
            st.markdown("#### 📊 Health Indicators")

            anaemia = st.slider("Anaemia Prevalence (%)", 0, 100, 30)
            diag_test = st.slider("Diagnostic Testing (%)", 0, 100, 20)
            malaria_msg = st.slider("Malaria Messages (%)", 0, 100, 50)

            prev_2018 = st.slider("Historical Prevalence 2018 (%)", 0, 100, 25)
            prev_2015 = st.slider("Historical Prevalence 2015 (%)", 0, 100, 30)

        # Zone selection
        zone_options = ['North Central', 'North East', 'North West',
                       'South East', 'South South', 'South West']
        selected_zone = st.selectbox("Geopolitical Zone", zone_options)

        st.markdown("---")

        # Predict button
        if st.button("🎯 Predict Malaria Risk", type="primary"):
            with st.spinner("Analyzing indicators..."):
                # Prepare input data
                input_data = np.zeros(len(models['feature_names']))
                feature_names = models['feature_names']

                # Map input values
                input_dict = {
                    'itn_ownership_2021': itn_ownership,
                    'itn_access_2021': itn_access,
                    'itn_use_children_2021': itn_use,
                    'iptp2_2021': iptp2,
                    'iptp3_2021': iptp3,
                    'anaemia_2021': anaemia,
                    'diag_test_2021': diag_test,
                    'malaria_msg_2021': malaria_msg,
                    'malaria_prev_2018': prev_2018,
                    'malaria_prev_2015': prev_2015,
                    f'zone_{selected_zone}': 1,
                    'neighbor_malaria_avg_2018': prev_2018, # Using slider value
                    'neighbor_malaria_avg_2015': prev_2015, # Using slider value
                    'is_urban': 0,
                    'urbanization_score': 15,
                    'net_to_person_2021': itn_access,
                    'itn_coverage_gap_2021': 100 - itn_ownership,
                    'anc_quality_index_2021': (iptp2 + iptp3) / 200,
                    'iptp_coverage_gap_2021': 100 - iptp2,
                    'health_seeking_index_2021': diag_test / 100,
                    'malaria_trend_2015_2018': (prev_2018 - prev_2015) / 3,
                    'itn_trend_2015_2021': (itn_ownership - 50) / 6,
                    'iptp2_trend_2015_2021': (iptp2 - 40) / 6,
                    'anaemia_trend_2015_2021': (anaemia - 50) / 6
                }

                for i, feat in enumerate(feature_names):
                    if feat in input_dict:
                        input_data[i] = input_dict[feat]

                # Scale input
                input_scaled = models['scaler'].transform([input_data])

                # Make predictions
                rf_pred = models['rf_classifier'].predict(input_scaled)[0]
                rf_pred_proba = models['rf_classifier'].predict_proba(input_scaled)[0]

                reg_pred = models['rf_regressor'].predict(input_scaled)[0]

                # Display results with animation
                st.markdown("---")
                st.markdown("### 🎯 Prediction Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    # Color code by risk
                    risk_color = {
                        'High': '#FF6B6B',
                        'Medium': '#FFD93D',
                        'Low': '#6BCB77'
                    }

                    st.markdown(f"""
                    <div style='text-align: center; padding: 2rem; background-color: {risk_color[rf_pred]}; border-radius: 10px;'>
                        <h2 style='color: #0E1117; margin: 0;'>Risk Category</h2>
                        <h1 style='color: #0E1117; margin: 0; font-size: 3rem;'>{rf_pred}</h1>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.metric(
                        label="Predicted Prevalence",
                        value=f"{max(0, reg_pred):.1f}%"
                    )

                with col3:
                    st.metric(
                        label="Prediction Confidence",
                        value=f"{max(rf_pred_proba)*100:.1f}%"
                    )

                # Probability distribution
                st.markdown("---")
                st.markdown("### 📊 Risk Probability Distribution")

                prob_df = pd.DataFrame({
                    'Risk Category': ['Low', 'Medium', 'High'],
                    'Probability': rf_pred_proba
                })

                fig = px.bar(
                    prob_df,
                    x='Risk Category',
                    y='Probability',
                    color='Probability',
                    color_continuous_scale=['#6BCB77', '#FFD93D', '#FF6B6B'],
                    template='plotly_dark'
                )
                fig.update_layout(
                    paper_bgcolor='#1E1E1E',
                    plot_bgcolor='#1E1E1E',
                    font=dict(color='white'),
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

                # Recommendations
                st.markdown("---")
                st.markdown("### 💡 Recommendations")

                if rf_pred == 'High':
                    st.error("""
                    **High Risk Detected!** Immediate interventions recommended:
                    - Increase ITN ownership to >80%
                    - Close IPTp coverage gaps
                    - Enhance diagnostic testing capacity
                    - Target anaemia reduction programs
                    - Increase malaria awareness messaging
                    """)
                elif rf_pred == 'Medium':
                    st.warning("""
                    **Medium Risk.** Preventive measures advised:
                    - Maintain and improve ITN coverage
                    - Ensure consistent IPTp delivery
                    - Monitor prevalence trends closely
                    - Strengthen surveillance systems
                    """)
                else:
                    st.success("""
                    **Low Risk.** Maintain current interventions:
                    - Continue ITN distribution programs
                    - Sustain IPTp coverage
                    - Monitor for any increases in prevalence
                    - Share best practices with neighboring states
                    """)

def classify_risk(prevalence):
    """Helper function for risk classification"""
    if prevalence >= 40:
        return 'High'
    elif prevalence >= 10:
        return 'Medium'
    else:
        return 'Low'
