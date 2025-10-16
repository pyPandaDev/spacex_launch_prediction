# 🚀 SpaceX Launch Success Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20RandomForest-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

An end-to-end Machine Learning project that predicts the success of SpaceX Falcon 9 rocket launches using historical launch data. This project includes comprehensive EDA, feature engineering, model training, explainability analysis with SHAP, and an interactive Streamlit web application.

## 🎯 Project Objectives

- Perform **Exploratory Data Analysis (EDA)** on historical SpaceX launch data
- Understand key factors influencing launch success (payload mass, booster type, orbit, launch site)
- Engineer relevant features to improve model performance
- Build, train, and evaluate **Machine Learning models** to predict launch success
- Interpret model predictions using **SHAP** explainability
- Deploy the final model as an **interactive Streamlit web application**

## 📁 Project Structure

```
spacex_launch_prediction/
│
├── data/
│   ├── spacex_launch_data.csv          # Raw data
│   └── spacex_cleaned.csv              # Cleaned data (generated)
│
├── notebooks/
│   ├── 01_Comprehensive_EDA.ipynb      # Exploratory Data Analysis
│   ├── 02_Model_Training.ipynb         # Model training & evaluation
│   └── 03_Model_Explainability.ipynb   # SHAP analysis
│
├── src/
│   ├── data_preprocessing.py           # Data cleaning & feature engineering
│   ├── model_training.py               # Model training utilities
│   └── visualizations.py               # Visualization functions
│
├── models/
│   ├── best_model.pkl                  # Trained model (generated)
│   ├── model_metadata.pkl              # Model metrics (generated)
│   ├── feature_names.pkl               # Feature list (generated)
│   └── shap_values.pkl                 # SHAP values (generated)
│
├── app/
│   └── streamlit_app.py                # Streamlit web application
│
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd spacex_launch_prediction
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment
**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Step 1: Run EDA Notebook
Open and run `notebooks/01_Comprehensive_EDA.ipynb` to:
- Load and explore the data
- Clean and preprocess features
- Create visualizations
- Generate insights

### Step 2: Train Models
Open and run `notebooks/02_Model_Training.ipynb` to:
- Prepare data for modeling
- Train baseline models (Logistic Regression, Random Forest, XGBoost)
- Perform hyperparameter tuning
- Evaluate model performance
- Save the best model

### Step 3: Analyze Explainability
Open and run `notebooks/03_Model_Explainability.ipynb` to:
- Generate SHAP explanations
- Visualize feature importance
- Analyze individual predictions

### Step 4: Launch Web Application
```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Dataset

The dataset contains **57 SpaceX Falcon 9 launches** from 2010-2018 with the following features:

| Feature | Description |
|---------|-------------|
| Flight Number | Sequential launch number |
| Date | Launch date |
| Time (UTC) | Launch time |
| Booster Version | Falcon 9 booster version |
| Launch Site | Launch location (CCAFS, VAFB, KSC) |
| Payload | Payload name/description |
| Payload Mass (kg) | Mass of payload |
| Orbit | Target orbit (LEO, GTO, etc.) |
| Customer | Customer/organization |
| Mission Outcome | Success or failure |
| Landing Outcome | Booster landing result |

## 🧪 Feature Engineering

### Created Features:
- **Temporal Features**: Year, Month, Day, Quarter, Day of Week
- **Booster Features**: Booster Type, Reused (Yes/No), Flight Number
- **Orbit Features**: Simplified orbit categories, Orbit Difficulty Score
- **Cumulative Features**: Cumulative launches, Success rate over time
- **Derived Features**: Payload-Orbit Ratio, Days since first launch
- **Categorical Features**: Is NASA mission, Is Commercial mission

## 🤖 Models

### Baseline Models:
1. **Logistic Regression** - Interpretable baseline
2. **Random Forest Classifier** - Ensemble method
3. **XGBoost Classifier** - Gradient boosting

### Optimization:
- **GridSearchCV** for Random Forest
- **RandomizedSearchCV** for XGBoost
- **StratifiedKFold** Cross-Validation (5-fold)

### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## 📈 Results

The best model achieves:
- **High accuracy** on test data
- **Strong ROC-AUC** score
- **Balanced precision and recall**

*(Exact metrics will be displayed after training)*

## 🔍 Model Explainability

Using **SHAP (SHapley Additive exPlanations)**:
- Feature importance rankings
- Feature impact on individual predictions
- Interaction effects between features
- Transparent, interpretable AI

## 🌐 Web Application Features

The Streamlit app provides:
- **Interactive prediction interface**
- **Real-time success probability**
- **Visual confidence gauge**
- **Mission analysis dashboard**
- **SHAP explanations** for transparency
- **Feature influence visualization**

## 📚 Key Libraries

- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **ML:** scikit-learn, xgboost
- **Explainability:** shap
- **Web App:** streamlit
- **Notebooks:** jupyter

## 🎓 Learning Outcomes

This project demonstrates:
- End-to-end ML pipeline development
- Data cleaning and feature engineering
- Multiple ML algorithm comparison
- Hyperparameter optimization
- Model evaluation best practices
- Explainable AI with SHAP
- Web application deployment

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is open-source and available for educational purposes.

## 🙏 Acknowledgments

- **SpaceX** for providing inspiration and historical launch data
- The open-source community for amazing ML libraries
- Kaggle and data science community for datasets

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ and 🚀 by aspiring data scientists**

*Last Updated: October 2025*
