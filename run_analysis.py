"""
Quick start script to run the entire analysis pipeline
"""
import sys
import os

# Add src to path
sys.path.append('src')

import pandas as pd
from data_preprocessing import load_data, clean_data, create_features, prepare_for_modeling, get_feature_target_split
from model_training import split_data, train_baseline_models, evaluate_models, tune_random_forest, tune_xgboost, save_model
from visualizations import create_summary_stats
import joblib

def main():
    print("=" * 80)
    print("🚀 SPACEX LAUNCH SUCCESS PREDICTION - AUTOMATED PIPELINE")
    print("=" * 80)
    
    # Step 1: Load and clean data
    print("\n[1/5] Loading and cleaning data...")
    df_raw = load_data('data/spacex_launch_data.csv')
    print(f"  ✓ Loaded {len(df_raw)} launches")
    
    df = clean_data(df_raw)
    print(f"  ✓ Data cleaned")
    
    df = create_features(df)
    print(f"  ✓ Features engineered ({df.shape[1]} features)")
    
    # Save cleaned data
    df.to_csv('data/spacex_cleaned.csv', index=False)
    print("  ✓ Cleaned data saved")
    
    # Display summary
    print("\n📊 DATASET SUMMARY:")
    summary = create_summary_stats(df)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Step 2: Prepare for modeling
    print("\n[2/5] Preparing data for modeling...")
    df_model = prepare_for_modeling(df)
    X, y = get_feature_target_split(df_model)
    print(f"  ✓ Features prepared: {X.shape}")
    print(f"  ✓ Target distribution: {y.value_counts().to_dict()}")
    
    # Step 3: Split data
    print("\n[3/5] Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print(f"  ✓ Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Step 4: Train models
    print("\n[4/5] Training baseline models...")
    models = train_baseline_models(X_train, y_train)
    
    # Evaluate baseline models
    print("\n📈 BASELINE MODEL PERFORMANCE:")
    results = evaluate_models(models, X_test, y_test)
    print(results.to_string(index=False))
    
    # Hyperparameter tuning
    print("\n[5/5] Hyperparameter tuning (this may take a few minutes)...")
    
    print("  Tuning Random Forest...")
    rf_best, rf_params = tune_random_forest(X_train, y_train)
    print(f"  ✓ Best RF params: {rf_params}")
    
    print("  Tuning XGBoost...")
    xgb_best, xgb_params = tune_xgboost(X_train, y_train)
    print(f"  ✓ Best XGB params: {xgb_params}")
    
    # Evaluate tuned models
    tuned_models = {
        'Random Forest (Tuned)': rf_best,
        'XGBoost (Tuned)': xgb_best
    }
    
    print("\n📈 TUNED MODEL PERFORMANCE:")
    tuned_results = evaluate_models(tuned_models, X_test, y_test)
    print(tuned_results.to_string(index=False))
    
    # Select best model
    all_results = pd.concat([results, tuned_results], ignore_index=True)
    best_idx = all_results['ROC-AUC'].idxmax()
    best_model_name = all_results.loc[best_idx, 'Model']
    best_roc_auc = all_results.loc[best_idx, 'ROC-AUC']
    
    if 'Tuned' in best_model_name:
        best_model = tuned_models[best_model_name]
    else:
        best_model = models[best_model_name]
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   ROC-AUC: {best_roc_auc:.4f}")
    
    # Save model
    print("\n💾 Saving model...")
    save_model(best_model, 'models/best_model.pkl')
    
    joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')
    print("  ✓ Feature names saved")
    
    metadata = {
        'model_name': best_model_name,
        'accuracy': all_results.loc[best_idx, 'Accuracy'],
        'precision': all_results.loc[best_idx, 'Precision'],
        'recall': all_results.loc[best_idx, 'Recall'],
        'f1': all_results.loc[best_idx, 'F1 Score'],
        'roc_auc': best_roc_auc,
        'features': X.columns.tolist()
    }
    joblib.dump(metadata, 'models/model_metadata.pkl')
    print("  ✓ Model metadata saved")
    
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Explore the notebooks for detailed analysis")
    print("  2. Run the Streamlit app: streamlit run app/streamlit_app.py")
    print("  3. Check notebooks/03_Model_Explainability.ipynb for SHAP analysis")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
