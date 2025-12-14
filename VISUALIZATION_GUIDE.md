# 📊 MalariaRiskXAI Visualization Suite Guide

Complete guide to using the comprehensive visualization suite and interactive dashboard.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Comprehensive Visualization Script](#comprehensive-visualization-script)
4. [Interactive Dashboard](#interactive-dashboard)
5. [Visualization Catalog](#visualization-catalog)
6. [Customization](#customization)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

The MalariaRiskXAI visualization suite includes:

- **32 Publication-Ready Visualizations** (`visualizations_comprehensive.py`)
- **Interactive Plotly Dash Dashboard** (`dashboard_app.py`)
- **Organized Output Structure** (`visualizations/interactive/`)

### Features

✅ **Interactive Plotly visualizations** with hover effects and zoom
✅ **Publication-ready quality** (300 DPI, professional styling)
✅ **Multiple export formats** (HTML, PNG)
✅ **Organized by research category**
✅ **Fully reproducible** and customizable

---

## 🔧 Installation

### Step 1: Install Dependencies

Update your `requirements.txt` to include visualization dependencies:

```bash
pip install plotly kaleido dash dash-bootstrap-components
```

Or install directly:

```bash
pip install plotly==5.18.0
pip install kaleido==0.2.1
pip install dash==2.14.2
pip install dash-bootstrap-components==1.5.0
```

### Step 2: Verify Data Files

Ensure these files exist:

```
data/
├── data_with_features.csv
├── model_predictions_2021.csv
└── shap_feature_importance_corrected.csv

models/
├── rf_classifier_corrected.pkl
├── rf_regressor_corrected.pkl
├── scaler_corrected.pkl
├── feature_names_corrected.pkl
├── metadata_corrected.pkl
├── shap_values_corrected.npy
└── shap_base_value_corrected.npy
```

---

## 📊 Comprehensive Visualization Script

### Quick Start

Generate all 32 visualizations:

```bash
python visualizations_comprehensive.py
```

**Output:** All visualizations saved to `visualizations/interactive/`

**Duration:** ~5-10 minutes (depending on system)

### What Gets Generated

The script creates 32 interactive HTML visualizations organized into 6 sections:

#### Section A: Model Performance (6 visualizations)

1. **Confusion Matrix** - Multi-model comparison
2. **ROC Curves** - High risk detection performance
3. **Precision-Recall Curves** - Imbalanced data handling
4. **Model Metrics Comparison** - Balanced accuracy bars
5. **Learning Curves** - Training progression
6. **Calibration Plots** - Probability reliability

#### Section B: Explainability (8 visualizations)

7. **SHAP Summary** - Top 15 feature importance
8. **SHAP Bar Plot** - Importance rankings
9. **SHAP Dependence Plots** - Feature effects (top 3)
10. **SHAP Waterfall** - Kebbi state example
11. **SHAP Force Plots** - Contrasting states
12. **Feature Interaction Heatmap** - Correlation matrix
13. **LIME Comparison** - Method validation
14. **Decision Boundaries** - 2D PCA projection

#### Section C: Geospatial (5 visualizations)

15. **Choropleth Map 2021** - Prevalence by state
16. **Multi-Year Animated Map** - Temporal evolution
17. **Multi-Layer Choropleth** - Risk + interventions
18. **Spatial Network** - Neighbor relationships
19. **Bubble Map** - Multi-dimensional view

#### Section D: Temporal Trends (4 visualizations)

20. **National Trends** - Time series 2015-2021
21. **Slopegraph** - State-level changes
22. **Heatmap Timeline** - All states, all years
23. **Racing Bar Chart** - Top 10 states evolution

#### Section E: Intervention Analysis (5 visualizations)

24. **ITN Scatter** - Coverage effectiveness
25. **Counterfactual Dashboard** - What-if scenarios
26. **Intervention Priority Matrix** - Risk vs gap
27. **Pareto Chart** - Cumulative burden
28. **Intervention Gaps by Zone** - Regional analysis

#### Section F: Prediction Analysis (4 visualizations)

29. **Actual vs Predicted** - R² scatter plot
30. **Residuals Plot** - Error analysis
31. **Geographic Error Map** - Spatial errors
32. **Error Distribution** - Histogram of residuals

### Configuration

Edit the `VISUALIZATION_CONFIG` dictionary in `visualizations_comprehensive.py`:

```python
VISUALIZATION_CONFIG = {
    'library': 'plotly',
    'theme': 'plotly_white',
    'color_palettes': {
        'risk_levels': ['#2E7D32', '#FFA726', '#C62828'],  # Green, Orange, Red
        'zones': px.colors.qualitative.Plotly,
        'continuous': px.colors.sequential.Viridis,
    },
    'fonts': {
        'title_size': 18,
        'axis_size': 14,
        'legend_size': 12,
    },
    'figure_sizes': {
        'width': 1200,
        'height': 800,
    },
    'export_formats': ['html', 'png'],  # Choose formats
    'dpi': 300,  # PNG resolution
}
```

### Advanced Usage

Generate specific sections only:

```python
# In main() function, comment out unwanted sections
def main():
    data = load_all_data()

    # Only generate explainability visualizations
    create_shap_summary_interactive(data)
    create_shap_importance_bars(data)
    # ... etc
```

---

## 🌐 Interactive Dashboard

### Launch Dashboard

Start the Plotly Dash web application:

```bash
python dashboard_app.py
```

**Access:** Open browser to `http://localhost:8050`

### Dashboard Features

#### 5 Interactive Tabs

1. **📊 Overview**
   - National KPIs (prevalence, reduction, high-risk states)
   - National trend line (2015-2021)
   - Geopolitical zone comparison

2. **🤖 Model Performance**
   - Model accuracy comparison
   - Confusion matrix (best model)
   - Performance summary cards

3. **🔍 Explainability**
   - Top 10 SHAP features
   - Feature importance by category (pie chart)
   - Key findings summary

4. **🎯 Risk Predictions**
   - **Interactive state selector**
   - State-specific risk cards
   - Zone comparison charts
   - Geographic risk map

5. **💊 Interventions**
   - ITN vs prevalence scatter
   - Intervention priority matrix
   - Evidence-based recommendations

### Dashboard Deployment

#### Option 1: Local Development

```bash
python dashboard_app.py
```

#### Option 2: Deploy to Cloud

**Render.com (Free):**

1. Push code to GitHub
2. Create new Web Service on Render
3. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn dashboard_app:server`
4. Deploy

**Heroku:**

```bash
# Create Procfile
echo "web: gunicorn dashboard_app:server" > Procfile

# Deploy
heroku create malaria-xai-dashboard
git push heroku main
```

### Customization

Edit `dashboard_app.py` to customize:

**Colors:**

```python
COLORS = {
    'background': '#f8f9fa',
    'primary': '#1976d2',
    'success': '#2e7d32',
    'warning': '#ffa726',
    'danger': '#c62828',
}
```

**Add New Tabs:**

```python
dcc.Tab(label='📈 New Section', value='newsection', children=[
    html.Div(id='newsection-content', className="mt-4")
])

# Add callback
@callback(Output('newsection-content', 'children'), Input('main-tabs', 'value'))
def render_newsection(tab):
    if tab == 'newsection':
        return create_newsection()
    return html.Div()
```

---

## 📁 Output Structure

After running `visualizations_comprehensive.py`:

```
visualizations/
└── interactive/
    ├── confusion_matrix_interactive.html
    ├── roc_curves_interactive.html
    ├── precision_recall_curves.html
    ├── model_metrics_comparison.html
    ├── learning_curves.html
    ├── calibration_plots.html
    ├── shap_summary_interactive.html
    ├── shap_importance_bars.html
    ├── shap_dependence_plots.html
    ├── shap_waterfall.html
    ├── shap_force_plot.html
    ├── feature_interaction_heatmap.html
    ├── lime_comparison.html
    ├── decision_boundaries.html
    ├── map_prevalence_2021.html
    ├── animated_map_multiyear.html
    ├── multilayer_choropleth.html
    ├── spatial_network.html
    ├── bubble_map.html
    ├── national_trends.html
    ├── slopegraph_states.html
    ├── heatmap_timeline.html
    ├── racing_bar_chart.html
    ├── intervention_scatter_itn.html
    ├── counterfactual_dashboard.html
    ├── intervention_priority_matrix.html
    ├── pareto_chart.html
    ├── intervention_gaps_zones.html
    ├── actual_vs_predicted.html
    ├── geographic_error_map.html
    ├── residuals.html
    └── error_distribution.html
```

**Optional PNG Export:**

Set `'export_formats': ['html', 'png']` in configuration to also generate PNG files for all visualizations.

---

## 🎓 Usage in Research Papers

### Recommended Figures for Publication

**Essential Figures (8):**

1. `confusion_matrix_interactive.html` - Model performance
2. `roc_curves_interactive.html` - Classification quality
3. `shap_summary_interactive.html` - Feature importance
4. `shap_waterfall.html` - Individual explanation
5. `map_prevalence_2021.html` - Geographic patterns
6. `national_trends.html` - Temporal evolution
7. `intervention_priority_matrix.html` - Policy implications
8. `actual_vs_predicted.html` - Regression quality

**Supplementary Figures (24):**

All remaining visualizations for comprehensive documentation.

### Citation

If using these visualizations in publications:

```bibtex
@software{malariariskxai_viz2025,
  title = {MalariaRiskXAI Visualization Suite},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/MalariaRiskXAI}
}
```

---

## 🔧 Troubleshooting

### Issue 1: Import Errors

**Error:** `ModuleNotFoundError: No module named 'plotly'`

**Fix:**

```bash
pip install plotly kaleido dash dash-bootstrap-components
```

### Issue 2: Missing Data Files

**Error:** `FileNotFoundError: data_with_features.csv not found`

**Fix:** Ensure you're running from the project root directory:

```bash
cd /path/to/MalariaRiskXAI
python visualizations_comprehensive.py
```

### Issue 3: SHAP Values Not Found

**Error:** `⚠ SHAP values not found, skipping...`

**Fix:** Generate SHAP values first:

```bash
python regenerate_shap_corrected.py
```

### Issue 4: Dashboard Not Loading

**Error:** Dashboard shows blank page

**Fix:**

1. Check data loaded correctly:

```python
# In dashboard_app.py, check DATA variable
if DATA is None:
    print("Data loading failed - check file paths")
```

2. Verify all dependencies:

```bash
pip list | grep -E "dash|plotly|pandas|joblib"
```

3. Check console for errors (press F12 in browser)

### Issue 5: Slow Visualization Generation

**Observation:** Script takes >15 minutes

**Optimization:**

1. Use faster model (reduce SHAP calculations)
2. Generate fewer visualizations (comment out sections)
3. Reduce figure resolution:

```python
'dpi': 150,  # Lower DPI for faster generation
```

### Issue 6: PNG Export Fails

**Error:** `ValueError: Image export using the "kaleido" engine requires...`

**Fix:**

```bash
pip install -U kaleido
```

If still fails, disable PNG export:

```python
'export_formats': ['html'],  # HTML only
```

---

## 📞 Support

For issues or questions:

1. Check [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for methodology details
2. Review [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for deployment help
3. Create issue on GitHub: `https://github.com/YOUR_USERNAME/MalariaRiskXAI/issues`

---

## 🚀 Quick Reference

### Generate All Visualizations

```bash
python visualizations_comprehensive.py
```

### Launch Dashboard

```bash
python dashboard_app.py
```

### View Visualizations

Open any HTML file in `visualizations/interactive/` with a web browser.

### Share Visualizations

1. **Local:** Share HTML files directly
2. **Web:** Host on GitHub Pages, Netlify, or Vercel
3. **Dashboard:** Deploy to Render, Heroku, or Streamlit Cloud

---

**Built with ❤️ for malaria elimination in Nigeria**

*Last Updated: December 14, 2025*
