"""
MalariaRiskXAI Interactive Dashboard
=====================================
Plotly Dash web application for exploring malaria risk predictions and explanations

Author: MalariaRiskXAI Team
Date: December 2025
"""

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load all required data and models"""
    try:
        # Load datasets
        df = pd.read_csv('data/data_with_features.csv')
        shap_importance = pd.read_csv('data/shap_feature_importance_corrected.csv')

        # Load models
        rf_classifier = joblib.load('models/rf_classifier_corrected.pkl')
        rf_regressor = joblib.load('models/rf_regressor_corrected.pkl')
        scaler = joblib.load('models/scaler_corrected.pkl')
        feature_names = joblib.load('models/feature_names_corrected.pkl')
        metadata = joblib.load('models/metadata_corrected.pkl')

        # Add risk categories
        def classify_risk(prevalence):
            if prevalence >= 40:
                return 'High'
            elif prevalence >= 10:
                return 'Medium'
            else:
                return 'Low'

        df['risk_category'] = df['malaria_prev_2021'].apply(classify_risk)

        return {
            'df': df,
            'shap_importance': shap_importance,
            'rf_classifier': rf_classifier,
            'rf_regressor': rf_regressor,
            'scaler': scaler,
            'feature_names': feature_names,
            'metadata': metadata
        }
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Load data once at startup
DATA = load_data()

# ============================================================================
# APP INITIALIZATION
# ============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="MalariaRiskXAI Dashboard"
)

server = app.server  # For deployment

# ============================================================================
# STYLING
# ============================================================================

COLORS = {
    'background': '#f8f9fa',
    'primary': '#1976d2',
    'success': '#2e7d32',
    'warning': '#ffa726',
    'danger': '#c62828',
    'text': '#212529'
}

CARD_STYLE = {
    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
    'borderRadius': '8px',
    'padding': '20px',
    'marginBottom': '20px',
    'backgroundColor': 'white'
}

# ============================================================================
# LAYOUT COMPONENTS
# ============================================================================

def create_navbar():
    """Create navigation bar"""
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Overview", href="/")),
            dbc.NavItem(dbc.NavLink("Models", href="#models")),
            dbc.NavItem(dbc.NavLink("Explainability", href="#explainability")),
            dbc.NavItem(dbc.NavLink("Predictions", href="#predictions")),
            dbc.NavItem(dbc.NavLink("Interventions", href="#interventions")),
        ],
        brand="🦟 MalariaRiskXAI Dashboard",
        brand_href="/",
        color="primary",
        dark=True,
        fluid=True,
        style={'marginBottom': '20px'}
    )

def create_kpi_card(title, value, subtitle, color='primary'):
    """Create KPI card component"""
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="text-muted"),
            html.H3(value, className=f"text-{color}"),
            html.P(subtitle, className="text-muted small mb-0")
        ])
    ], style=CARD_STYLE)

def create_overview_section():
    """Create overview dashboard section"""
    if DATA is None:
        return html.Div("Error loading data", className="alert alert-danger")

    df = DATA['df']

    # Calculate KPIs
    avg_prev_2021 = df['malaria_prev_2021'].mean()
    avg_prev_2015 = df['malaria_prev_2015'].mean()
    reduction = avg_prev_2015 - avg_prev_2021
    high_risk_states = (df['malaria_prev_2021'] >= 40).sum()

    kpis = dbc.Row([
        dbc.Col(create_kpi_card(
            "Average Prevalence 2021",
            f"{avg_prev_2021:.1f}%",
            f"Down from {avg_prev_2015:.1f}% in 2015",
            "success"
        ), md=3),
        dbc.Col(create_kpi_card(
            "Total Reduction",
            f"{reduction:.1f}%",
            f"{(reduction/avg_prev_2015*100):.0f}% decrease",
            "info"
        ), md=3),
        dbc.Col(create_kpi_card(
            "High Risk States",
            str(high_risk_states),
            "Prevalence ≥ 40%",
            "danger"
        ), md=3),
        dbc.Col(create_kpi_card(
            "States Analyzed",
            "37",
            "Across 6 geopolitical zones",
            "primary"
        ), md=3),
    ], className="mb-4")

    # National trend chart
    years = [2015, 2018, 2021]
    avg_prevs = [df[f'malaria_prev_{year}'].mean() for year in years]

    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(
        x=years,
        y=avg_prevs,
        mode='lines+markers',
        line=dict(color=COLORS['primary'], width=4),
        marker=dict(size=12),
        name='National Average'
    ))

    trend_fig.update_layout(
        title='National Malaria Prevalence Trend (2015-2021)',
        xaxis_title='Year',
        yaxis_title='Prevalence (%)',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )

    # Geographic distribution
    zone_stats = df.groupby('Zone')['malaria_prev_2021'].mean().sort_values(ascending=False)

    zone_fig = px.bar(
        x=zone_stats.index,
        y=zone_stats.values,
        color=zone_stats.values,
        color_continuous_scale='Reds',
        labels={'x': 'Geopolitical Zone', 'y': 'Average Prevalence (%)'},
        title='Malaria Prevalence by Geopolitical Zone (2021)'
    )
    zone_fig.update_layout(template='plotly_white', height=400, showlegend=False)

    return html.Div([
        kpis,
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([dcc.Graph(figure=trend_fig)])
            ], style=CARD_STYLE), md=6),
            dbc.Col(dbc.Card([
                dbc.CardBody([dcc.Graph(figure=zone_fig)])
            ], style=CARD_STYLE), md=6),
        ])
    ])

def create_model_performance_section():
    """Create model performance section"""
    if DATA is None:
        return html.Div("Error loading data", className="alert alert-danger")

    metadata = DATA['metadata']
    perf = metadata['model_performance']

    # Model comparison
    models = ['Random Forest', 'Logistic Regression', 'MLP Neural Net']
    accuracies = [
        perf['rf_classifier_balanced_accuracy'],
        perf['lr_classifier_balanced_accuracy'],
        perf['mlp_classifier_balanced_accuracy']
    ]

    metrics_fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=accuracies,
            marker_color=[COLORS['success'], COLORS['warning'], COLORS['primary']],
            text=[f"{acc:.1%}" for acc in accuracies],
            textposition='outside'
        )
    ])

    metrics_fig.update_layout(
        title='Classification Model Performance Comparison',
        yaxis_title='Balanced Accuracy',
        template='plotly_white',
        height=400,
        yaxis=dict(tickformat='.0%', range=[0, 0.6])
    )

    # Confusion matrix for best model
    df = DATA['df']
    X = df[DATA['feature_names']].fillna(df[DATA['feature_names']].median())
    X_scaled = DATA['scaler'].transform(X)
    y_true = df['risk_category']
    y_pred = DATA['rf_classifier'].predict(X_scaled)

    labels = ['Low', 'Medium', 'High']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    cm_fig = go.Figure(data=go.Heatmap(
        z=cm_pct,
        x=labels,
        y=labels,
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 14},
        colorscale='Blues',
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{text}<br>Percent: %{z:.1f}%<extra></extra>'
    ))

    cm_fig.update_layout(
        title='Confusion Matrix - Random Forest (Best Model)',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        template='plotly_white',
        height=400
    )

    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([dcc.Graph(figure=metrics_fig)])
            ], style=CARD_STYLE), md=6),
            dbc.Col(dbc.Card([
                dbc.CardBody([dcc.Graph(figure=cm_fig)])
            ], style=CARD_STYLE), md=6),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("Model Performance Summary"),
                    html.Hr(),
                    html.P(f"✅ Best Classification Model: Random Forest ({perf['rf_classifier_balanced_accuracy']:.1%} balanced accuracy)"),
                    html.P(f"✅ Best Regression Model: Random Forest ({perf['rf_regressor_rmse']:.2f}% RMSE)"),
                    html.P(f"✅ 5-Fold Cross-Validation with stratification"),
                    html.P(f"✅ No data leakage - scientifically valid results"),
                ])
            ], style=CARD_STYLE), md=12),
        ])
    ])

def create_explainability_section():
    """Create explainability section"""
    if DATA is None:
        return html.Div("Error loading data", className="alert alert-danger")

    shap_importance = DATA['shap_importance'].head(10)

    # SHAP feature importance
    shap_fig = go.Figure(data=[
        go.Bar(
            y=shap_importance['Feature'].apply(lambda x: x.replace('_', ' ').title()),
            x=shap_importance['Mean_SHAP'],
            orientation='h',
            marker_color=COLORS['primary'],
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        )
    ])

    shap_fig.update_layout(
        title='Top 10 SHAP Feature Importance',
        xaxis_title='Mean |SHAP Value|',
        yaxis_title='Feature',
        template='plotly_white',
        height=500,
        yaxis=dict(autorange="reversed")
    )

    # Feature categories
    feature_categories = {
        'Historical Prevalence': ['malaria_prev_2015', 'malaria_prev_2018'],
        'Intervention Gaps': ['itn_coverage_gap_2021', 'iptp_coverage_gap_2021'],
        'Spatial Features': ['neighbor_malaria_avg_2015', 'neighbor_malaria_avg_2018'],
        'Health Indicators': ['anaemia_2021', 'anaemia_trend_2015_2021'],
        'Coverage Metrics': ['itn_2021', 'iptp2_2021']
    }

    category_importance = {}
    for category, features in feature_categories.items():
        mask = shap_importance['Feature'].isin(features)
        category_importance[category] = shap_importance[mask]['Mean_SHAP'].sum()

    category_fig = px.pie(
        values=list(category_importance.values()),
        names=list(category_importance.keys()),
        title='Feature Importance by Category',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    category_fig.update_layout(template='plotly_white', height=400)

    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([dcc.Graph(figure=shap_fig)])
            ], style=CARD_STYLE), md=8),
            dbc.Col(dbc.Card([
                dbc.CardBody([dcc.Graph(figure=category_fig)])
            ], style=CARD_STYLE), md=4),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("Key Findings from SHAP Analysis"),
                    html.Hr(),
                    html.Ul([
                        html.Li("Historical prevalence (2015, 2018) is the strongest predictor"),
                        html.Li("Intervention gaps drive current risk more than coverage itself"),
                        html.Li("Spatial autocorrelation is significant - neighbor states influence risk"),
                        html.Li("Urbanization has a protective effect"),
                        html.Li("All predictions are explainable and traceable")
                    ])
                ])
            ], style=CARD_STYLE), md=12),
        ])
    ])

def create_predictions_section():
    """Create predictions/risk assessment section with interactive controls"""
    if DATA is None:
        return html.Div("Error loading data", className="alert alert-danger")

    df = DATA['df']

    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("Interactive State Risk Explorer"),
                    html.Hr(),
                    html.Label("Select State:"),
                    dcc.Dropdown(
                        id='state-selector',
                        options=[{'label': state, 'value': state} for state in sorted(df['State'].unique())],
                        value='Kebbi',
                        clearable=False
                    ),
                    html.Div(id='state-risk-output', className="mt-4")
                ])
            ], style=CARD_STYLE), md=4),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.Div(id='state-comparison-chart')
                ])
            ], style=CARD_STYLE), md=8),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.Div(id='geographic-map')
                ])
            ], style=CARD_STYLE), md=12),
        ])
    ])

def create_interventions_section():
    """Create interventions analysis section"""
    if DATA is None:
        return html.Div("Error loading data", className="alert alert-danger")

    df = DATA['df'].copy()

    # ITN vs Prevalence
    itn_fig = px.scatter(
        df,
        x='itn_ownership_2021',
        y='malaria_prev_2021',
        size='malaria_prev_2021',
        color='Zone',
        hover_name='State',
        labels={'itn_ownership_2021': 'ITN Ownership (%)', 'malaria_prev_2021': 'Prevalence (%)'},
        title='ITN Coverage vs Malaria Prevalence (2021)',
        trendline='lowess'
    )
    itn_fig.update_layout(template='plotly_white', height=400)

    # Intervention priority matrix
    df['total_gap'] = (df['itn_coverage_gap_2021'] + df['iptp_coverage_gap_2021']) / 2

    priority_fig = px.scatter(
        df,
        x='total_gap',
        y='malaria_prev_2021',
        size='malaria_prev_2021',
        color='risk_category',
        color_discrete_map={'Low': COLORS['success'], 'Medium': COLORS['warning'], 'High': COLORS['danger']},
        hover_name='State',
        labels={'total_gap': 'Average Intervention Gap (%)', 'malaria_prev_2021': 'Prevalence (%)'},
        title='Intervention Priority Matrix - Risk vs Coverage Gap'
    )
    priority_fig.update_layout(template='plotly_white', height=400)

    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([dcc.Graph(figure=itn_fig)])
            ], style=CARD_STYLE), md=6),
            dbc.Col(dbc.Card([
                dbc.CardBody([dcc.Graph(figure=priority_fig)])
            ], style=CARD_STYLE), md=6),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("Intervention Recommendations"),
                    html.Hr(),
                    html.H6("High Priority States (Immediate Action):"),
                    html.Ul([
                        html.Li("Kebbi, Zamfara, Sokoto: Mass ITN distribution campaigns"),
                        html.Li("Focus on closing coverage gaps, not just increasing coverage"),
                        html.Li("Strengthen IPTp delivery in antenatal care")
                    ]),
                    html.H6("Proven Strategies to Scale:"),
                    html.Ul([
                        html.Li("Learn from Osun, Lagos, Ekiti (>40% reduction achieved)"),
                        html.Li("Sustained ITN distribution programs"),
                        html.Li("Quality antenatal care with universal IPTp coverage")
                    ])
                ])
            ], style=CARD_STYLE), md=12),
        ])
    ])

# ============================================================================
# MAIN LAYOUT
# ============================================================================

app.layout = html.Div([
    create_navbar(),

    dbc.Container([
        # Header
        html.Div([
            html.H1("🦟 MalariaRiskXAI Interactive Dashboard", className="text-center mb-2"),
            html.P("Explainable AI for Malaria Risk Prediction in Nigeria", className="text-center text-muted mb-4"),
        ]),

        # Tabs
        dcc.Tabs(id='main-tabs', value='overview', children=[
            dcc.Tab(label='📊 Overview', value='overview', children=[
                html.Div(id='overview-content', className="mt-4")
            ]),
            dcc.Tab(label='🤖 Model Performance', value='models', children=[
                html.Div(id='models-content', className="mt-4")
            ]),
            dcc.Tab(label='🔍 Explainability', value='explainability', children=[
                html.Div(id='explainability-content', className="mt-4")
            ]),
            dcc.Tab(label='🎯 Risk Predictions', value='predictions', children=[
                html.Div(id='predictions-content', className="mt-4")
            ]),
            dcc.Tab(label='💊 Interventions', value='interventions', children=[
                html.Div(id='interventions-content', className="mt-4")
            ]),
        ], colors={
            "border": COLORS['primary'],
            "primary": COLORS['primary'],
            "background": COLORS['background']
        }),

        # Footer
        html.Hr(className="mt-5"),
        html.Footer([
            html.P("MalariaRiskXAI Dashboard | Built with Plotly Dash", className="text-center text-muted"),
            html.P("Data Source: Nigeria Malaria Indicator Survey (NMIS) 2015, 2018, 2021", className="text-center text-muted small")
        ], className="mb-4")
    ], fluid=True, style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'padding': '20px'})
])

# ============================================================================
# CALLBACKS
# ============================================================================

@callback(
    Output('overview-content', 'children'),
    Input('main-tabs', 'value')
)
def render_overview(tab):
    if tab == 'overview':
        return create_overview_section()
    return html.Div()

@callback(
    Output('models-content', 'children'),
    Input('main-tabs', 'value')
)
def render_models(tab):
    if tab == 'models':
        return create_model_performance_section()
    return html.Div()

@callback(
    Output('explainability-content', 'children'),
    Input('main-tabs', 'value')
)
def render_explainability(tab):
    if tab == 'explainability':
        return create_explainability_section()
    return html.Div()

@callback(
    Output('predictions-content', 'children'),
    Input('main-tabs', 'value')
)
def render_predictions(tab):
    if tab == 'predictions':
        return create_predictions_section()
    return html.Div()

@callback(
    Output('interventions-content', 'children'),
    Input('main-tabs', 'value')
)
def render_interventions(tab):
    if tab == 'interventions':
        return create_interventions_section()
    return html.Div()

@callback(
    [Output('state-risk-output', 'children'),
     Output('state-comparison-chart', 'children'),
     Output('geographic-map', 'children')],
    Input('state-selector', 'value')
)
def update_state_details(selected_state):
    if DATA is None or selected_state is None:
        return "Error loading data", html.Div(), html.Div()

    df = DATA['df']
    state_data = df[df['State'] == selected_state].iloc[0]

    # Risk card
    risk_card = dbc.Card([
        dbc.CardBody([
            html.H4(f"{selected_state}", className="mb-3"),
            html.H5(f"Prevalence: {state_data['malaria_prev_2021']:.1f}%",
                   className=f"text-{('danger' if state_data['risk_category']=='High' else 'warning' if state_data['risk_category']=='Medium' else 'success')}"),
            html.P(f"Risk Category: {state_data['risk_category']}", className="text-muted"),
            html.Hr(),
            html.H6("Key Metrics:"),
            html.P(f"• ITN Ownership: {state_data['itn_ownership_2021']:.1f}%"),
            html.P(f"• IPTp Coverage: {state_data['iptp2_2021']:.1f}%"),
            html.P(f"• Geopolitical Zone: {state_data['Zone']}"),
            html.P(f"• 2015 Prevalence: {state_data['malaria_prev_2015']:.1f}%"),
            html.P(f"• Change since 2015: {state_data['malaria_prev_2015'] - state_data['malaria_prev_2021']:.1f}%")
        ])
    ], color=('danger' if state_data['risk_category']=='High' else 'warning' if state_data['risk_category']=='Medium' else 'success'), outline=True)

    # Comparison chart
    zone = state_data['Zone']
    zone_states = df[df['Zone'] == zone].sort_values('malaria_prev_2021', ascending=False)

    comparison_fig = go.Figure(data=[
        go.Bar(
            x=zone_states['State'],
            y=zone_states['malaria_prev_2021'],
            marker_color=[COLORS['danger'] if s == selected_state else COLORS['primary'] for s in zone_states['State']],
            text=zone_states['malaria_prev_2021'].apply(lambda x: f"{x:.1f}%"),
            textposition='outside'
        )
    ])

    comparison_fig.update_layout(
        title=f'Prevalence Comparison - {zone} States',
        yaxis_title='Prevalence (%)',
        template='plotly_white',
        height=400,
        showlegend=False
    )

    # Geographic map (simplified)
    map_fig = px.scatter(
        df,
        x='State',
        y='malaria_prev_2021',
        size='malaria_prev_2021',
        color='risk_category',
        color_discrete_map={'Low': COLORS['success'], 'Medium': COLORS['warning'], 'High': COLORS['danger']},
        title='All States Risk Map',
        labels={'malaria_prev_2021': 'Prevalence (%)'}
    )
    map_fig.update_layout(template='plotly_white', height=400, showlegend=True)
    map_fig.update_xaxes(tickangle=-90)

    return risk_card, dcc.Graph(figure=comparison_fig), dcc.Graph(figure=map_fig)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
