# 🦟 MalariaRiskXAI

**Explainable AI Framework for Malaria Risk Prediction in Nigeria**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive web application that uses machine learning and explainable AI to predict and understand malaria risk across Nigerian states.

---

## 🎯 Overview

This project analyzes malaria risk across **37 Nigerian states** using:
- **Machine Learning Models:** Random Forest, XGBoost, MLP, Logistic Regression
- **Explainable AI:** SHAP, LIME, Partial Dependence Plots
- **Multi-temporal Data:** 2015, 2018, 2021 Nigeria Malaria Indicator Survey (NMIS)
- **Interactive Dashboard:** Streamlit web app

**Key Features:**
- ✅ Scientifically valid predictions (no data leakage)
- ✅ Complete explainability through SHAP analysis
- ✅ Geographic and temporal pattern analysis
- ✅ State-specific risk predictions and recommendations
- ✅ Interactive visualizations

---

## 📊 Results Summary

### Model Performance

| Task | Model | Performance |
|------|-------|------------|
| **Classification** | Random Forest | 46.67% balanced accuracy |
| **Regression** | Random Forest | RMSE: 6.53% (±7pp error) |

### Top Risk Drivers (SHAP Analysis)

1. 🏥 **Historical Prevalence** (2015, 2018) - Past burden predicts future risk
2. 🛏️ **ITN Coverage Gaps** - Intervention gaps drive risk
3. 🗺️ **Spatial Patterns** - Neighboring states influence each other
4. 🤰 **IPTp Gaps** - Pregnancy intervention coverage matters
5. 🏙️ **Urbanization** - Urban areas have lower risk

### Geographic Patterns

| Zone | Avg Prevalence | Highest Risk State |
|------|---------------|-------------------|
| North West | 38.2% | Kebbi (49%) |
| North East | 31.5% | Adamawa (42%) |
| South West | 8.5% | Lagos (4%) |

---

## 🚀 Quick Start

### Run Locally

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/MalariaRiskXAI.git
cd MalariaRiskXAI

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

App opens at `http://localhost:8501`

### Deploy to Cloud

See **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** for complete instructions.

---

## 📁 Project Structure

```
MalariaRiskXAI/
├── app.py                              # Main Streamlit app
├── requirements.txt                    # Dependencies
├── .streamlit/config.toml             # Configuration
│
├── models/                            # Trained models
│   ├── rf_classifier_corrected.pkl
│   ├── xgboost_regressor_corrected.pkl
│   └── ... (12 models total)
│
├── data/                              # Datasets
│   ├── data_with_features.csv
│   ├── shap_feature_importance_corrected.csv
│   └── ...
│
├── visualizations/                    # Generated plots
│   └── ... (PNG files)
│
├── train_all_models_corrected.py     # Model training script
├── regenerate_shap_corrected.py       # SHAP analysis script
├── generate_visualizations.py         # Visualization script
│
├── README.md                          # This file
├── DEPLOYMENT_GUIDE.md                # Deployment instructions
└── TECHNICAL_REPORT.md                # Complete technical docs
```

---

## 🎓 Methodology

### Data
- **Source:** Nigeria Malaria Indicator Survey (NMIS) 2015, 2018, 2021
- **Coverage:** 37 states across 6 geopolitical zones
- **Features:** 30 features (no data leakage)

### Models
- Random Forest (best: 46.67% balanced accuracy)
- XGBoost (strong regression: 6.90% RMSE)
- MLP Neural Network
- Logistic Regression
- Stacking Ensemble

### Explainability
- SHAP: Global and local feature importance
- LIME: Instance-level explanations
- Partial Dependence Plots: Feature effects

---

## 🔬 Key Findings

**National Trends:**
- 24% reduction in prevalence (2015-2021)
- North-South divide: Northern states 3-4x higher

**Critical Interventions:**
1. Close ITN coverage gaps (most modifiable)
2. Strengthen IPTp delivery
3. Target Kebbi, Zamfara, Sokoto (highest risk)
4. Scale successful strategies from Osun, Lagos

**Success Stories:**
- Osun: 54% reduction (best improvement)
- Lagos: 50% reduction (4% prevalence)

---

## 📊 App Features

### Pages:
1. **🏠 Home** - National statistics and trends
2. **📊 Data Explorer** - Geographic maps, temporal evolution
3. **🤖 Model Performance** - All model comparisons
4. **🔍 XAI Insights** - SHAP, LIME, PDPs
5. **🎯 Risk Predictor** - Custom predictions and recommendations

---

## 🛠️ Scripts

```bash
# Train all models
python train_all_models_corrected.py

# Regenerate SHAP analysis
python regenerate_shap_corrected.py

# Generate all 32 publication-ready visualizations
python visualizations_comprehensive.py

# Launch interactive dashboard
python dashboard_app.py
```

---

## 📖 Documentation

- **[README.md](README.md)** - Project overview (this file)
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Deploy to Streamlit Cloud
- **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)** - Complete technical documentation
- **[VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)** - Comprehensive visualization suite guide
- **[VISUALIZATION_SUITE_SUMMARY.md](VISUALIZATION_SUITE_SUMMARY.md)** - Visualization implementation summary

---

## 🎯 Use Cases

**For Policymakers:**
- Target resources to highest-risk states
- Identify intervention gaps
- Monitor progress over time

**For Researchers:**
- Understand risk drivers through explainable AI
- Validate findings with SHAP analysis
- Generate hypotheses for causal studies

**For Health Workers:**
- Get state-specific recommendations
- Evidence-based planning
- Track temporal trends

---

## 📚 Citation

```bibtex
@software{malariariskxai2025,
  title = {MalariaRiskXAI: Explainable AI for Malaria Risk Prediction in Nigeria},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/MalariaRiskXAI}
}
```

---

## 🙏 Acknowledgments

- Nigeria Malaria Indicator Survey (NMIS) for data
- National Malaria Elimination Programme (NMEP)
- Streamlit for the amazing framework
- SHAP/LIME developers for explainability tools

---

## 📞 Contact

- GitHub: [YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- Email: your.email@example.com

---

## 🌟 Status

- ✅ Data Processing Complete
- ✅ Models Trained (corrected for data leakage)
- ✅ SHAP Analysis Complete
- ✅ Web App Functional
- ✅ All Tests Passed
- 🚀 Ready for Deployment
- 📝 Ready for Publication

---

**Built with ❤️ for malaria elimination in Nigeria**

*Last Updated: December 14, 2025*
