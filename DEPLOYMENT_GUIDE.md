# 🚀 MalariaRiskXAI Deployment Guide

Complete guide to deploying your app to Streamlit Cloud.

---

## ✅ Pre-Deployment Checklist

**Your app is ready!** All tests passed:
- ✓ Repository size: 13MB (well under limits)
- ✓ All models loaded successfully
- ✓ All pages functional
- ✓ No Git LFS needed

---

## 🎯 Quick Deploy (5 Steps - 15 Minutes)

### Step 1: Initialize Git Repository

```bash
cd /Users/fuzailakhtar/Documents/MalariaRiskXAI
git init
git status
```

### Step 2: Commit Your Code

```bash
git add .
git commit -m "MalariaRiskXAI: Explainable AI for malaria risk prediction in Nigeria

- Corrected models (no data leakage)
- Classification: RF 46.67% balanced accuracy
- Regression: RF RMSE 6.53%
- Complete SHAP analysis with 30 features
- All app pages tested and functional"
```

### Step 3: Create GitHub Repository

Go to: **https://github.com/new**

- **Name:** `MalariaRiskXAI`
- **Description:** `Explainable AI Framework for Malaria Risk Prediction in Nigeria`
- **Public** ✓ (required for free Streamlit hosting)
- **DO NOT** check: README, .gitignore, or license
- Click **"Create repository"**

### Step 4: Push to GitHub

```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/MalariaRiskXAI.git
git branch -M main
git push -u origin main
```

**Authentication:**
- Username: Your GitHub username
- Password: Personal Access Token from https://github.com/settings/tokens

### Step 5: Deploy to Streamlit Cloud

1. Go to: **https://share.streamlit.io**
2. **Sign in** with GitHub
3. Click **"New app"**
4. Fill in:
   - Repository: `YOUR_USERNAME/MalariaRiskXAI`
   - Branch: `main`
   - Main file: `app.py`
   - App URL: `malaria-risk-xai` (or your choice)
5. Click **"Deploy!"**
6. Wait 10-15 minutes

Your app will be live at: `https://your-app-name.streamlit.app`

---

## 🔧 Troubleshooting

### Authentication Failed

**Use Personal Access Token:**
1. Go to https://github.com/settings/tokens
2. Generate new token (classic)
3. Select scope: `repo`
4. Copy token
5. Use as password when pushing

### Permission Denied (SSH)

**Switch to HTTPS:**
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/MalariaRiskXAI.git
git push -u origin main
```

### App Crashes on Deployment

**Check logs:**
1. Streamlit Cloud dashboard
2. Click "Manage app" → "Logs"
3. Look for errors

**Common fixes:**
```bash
# Verify all files in GitHub
git add .
git commit -m "Add missing files"
git push
```

---

## 📊 What Gets Deployed

### Required Files:
- ✅ `app.py` - Main application
- ✅ `requirements.txt` - Python dependencies
- ✅ `.streamlit/config.toml` - App configuration
- ✅ `models/*_corrected.pkl` - All 12 model files
- ✅ `data/*.csv` - All data files
- ✅ `visualizations/*.png` - All visualizations

### File Structure:
```
MalariaRiskXAI/
├── app.py
├── requirements.txt
├── .streamlit/config.toml
├── models/ (12 .pkl files + 2 .npy files)
├── data/ (CSV files)
├── visualizations/ (PNG files)
├── generate_visualizations.py
├── regenerate_shap_corrected.py
├── train_all_models_corrected.py
└── README.md
```

---

## 🎯 Testing Your Deployed App

Visit your app URL and test:

- [ ] **Home page** loads with metrics
- [ ] **Data Explorer** shows visualizations
- [ ] **Model Performance** displays all 8 models
- [ ] **XAI Insights** loads SHAP plots
- [ ] **Risk Predictor** makes predictions

---

## 🔄 Updating Your App

After making changes:

```bash
# Make changes to files
# ... edit code ...

# Commit and push
git add .
git commit -m "Description of changes"
git push

# Streamlit Cloud auto-deploys!
```

Changes deploy automatically within 1-2 minutes.

---

## 💰 Costs

**Streamlit Cloud Free Tier:**
- ✅ Unlimited public apps
- ✅ 1 GB RAM per app
- ✅ Auto-sleep after inactivity
- ✅ HTTPS SSL included
- ❌ No private apps

**Perfect for this project!**

---

## 📱 Custom Domain (Optional)

To use `malaria-xai.yoursite.com`:

1. Streamlit Cloud settings → Custom domain
2. Add your domain
3. Update DNS:
   - Type: CNAME
   - Name: malaria-xai
   - Value: (provided by Streamlit)
4. Wait for SSL (automatic)

---

## 🎓 Best Practices

### For Research:
- Keep repo public for reproducibility
- Add citation information in README
- Archive on Zenodo for DOI
- Link to published paper

### For Maintenance:
- Test locally before pushing
- Document changes in commits
- Monitor app dashboard
- Update dependencies regularly

### For Sharing:
- Use memorable app URL
- Add to presentations/papers
- Create QR code for posters
- Share on social media

---

## 📈 Monitoring

Streamlit Cloud dashboard shows:
- Number of visitors
- App uptime
- Resource usage
- Error logs
- Last deployment time

---

## ⚠️ Important Notes

### Cannot Deploy to Vercel
- Streamlit requires Python runtime
- Vercel is for Next.js/Node.js
- **Use Streamlit Cloud instead**

### File Size Limits
- Individual file: < 100MB
- Total repo: < 1GB
- Your repo: 13MB ✓

### Performance
- First load: 2-5 seconds (cold start)
- Subsequent: <1 second (cached)
- SHAP pre-calculated for speed

---

## 🚀 You're Ready!

Follow the 5 steps above and your app will be live in 15 minutes!

**Questions?** See README.md or TECHNICAL_REPORT.md for more details.

---

*Last Updated: December 14, 2025*
*All systems ready for deployment!*
