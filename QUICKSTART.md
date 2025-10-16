# 🚀 Quick Start Guide

Get started with the SpaceX Launch Prediction project in minutes!

## Option 1: Automated Pipeline (Fastest)

Run the entire analysis pipeline with a single command:

```bash
# Activate virtual environment
venv\Scripts\activate

# Run automated pipeline
python run_analysis.py
```

This will:
- ✅ Clean and prepare the data
- ✅ Train multiple ML models
- ✅ Perform hyperparameter tuning
- ✅ Save the best model
- ✅ Generate performance metrics

**Time:** ~5-10 minutes

---

## Option 2: Step-by-Step Notebooks (Recommended for Learning)

### Step 1: Install Dependencies
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Run EDA Notebook
```bash
jupyter notebook notebooks/01_Comprehensive_EDA.ipynb
```

**What you'll do:**
- Load and explore the dataset
- Clean and engineer features
- Create beautiful visualizations
- Discover insights and patterns

### Step 3: Train Models
```bash
jupyter notebook notebooks/02_Model_Training.ipynb
```

**What you'll do:**
- Prepare data for ML
- Train 3 baseline models
- Tune hyperparameters
- Evaluate performance
- Save the best model

### Step 4: Analyze Explainability
```bash
jupyter notebook notebooks/03_Model_Explainability.ipynb
```

**What you'll do:**
- Generate SHAP explanations
- Visualize feature importance
- Understand model decisions
- Analyze individual predictions

---

## Option 3: Launch Web App Directly

If models are already trained:

```bash
venv\Scripts\activate
streamlit run app/streamlit_app.py
```

**Features:**
- 🎯 Interactive prediction interface
- 📊 Real-time success probability
- 🔍 SHAP explanations
- 📈 Visual confidence gauge

The app opens at: `http://localhost:8501`

---

## 📋 Project Checklist

- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Data files present in `data/` folder
- [ ] Run Option 1 OR Option 2
- [ ] Models saved in `models/` folder
- [ ] Launch Streamlit app
- [ ] Make predictions!

---

## 🆘 Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### "No model found" in Streamlit
Run the automated pipeline first:
```bash
python run_analysis.py
```

### Jupyter not starting
```bash
pip install jupyter notebook
jupyter notebook
```

### SHAP visualization errors
Restart the Jupyter kernel and run all cells again.

---

## 📊 Expected Results

After training, you should see:

- **Models trained:** Logistic Regression, Random Forest, XGBoost
- **Best model accuracy:** ~85-95%
- **ROC-AUC score:** ~0.90-0.98
- **Files generated:**
  - `data/spacex_cleaned.csv`
  - `models/best_model.pkl`
  - `models/model_metadata.pkl`
  - `models/feature_names.pkl`

---

## 🎯 Next Steps

1. **Experiment:** Try different features in the Streamlit app
2. **Improve:** Add more features or try different algorithms
3. **Deploy:** Host the app on Streamlit Cloud
4. **Learn:** Explore SHAP explanations to understand predictions

---

## 💡 Tips

- **Run notebooks in order** (01 → 02 → 03)
- **Check outputs** of each cell before moving to the next
- **Save your work** regularly
- **Experiment** with different parameters in the app

---

**Happy Predicting! 🚀**
