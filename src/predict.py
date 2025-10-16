"""
Standalone prediction script for command-line usage
"""
import pandas as pd
import joblib
import argparse
import os

def load_model():
    """Load the trained model and metadata"""
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
    feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
    metadata = joblib.load(os.path.join(model_dir, 'model_metadata.pkl'))
    
    return model, feature_names, metadata

def create_input_features(payload_mass, booster_type, launch_site, orbit_type, 
                         booster_reused, booster_flight_num, year, month,
                         is_nasa=False, is_commercial=True):
    """Create feature dictionary from input parameters"""
    
    # Calculate derived features
    orbit_difficulty_map = {'LEO': 1, 'Polar': 2, 'HEO': 3, 'GTO': 4, 'Other': 2.5}
    orbit_difficulty = orbit_difficulty_map.get(orbit_type, 2.5)
    payload_orbit_ratio = payload_mass / orbit_difficulty
    quarter = (month - 1) // 3 + 1
    cumulative_launches = max(1, (year - 2010) * 10)
    cumulative_success_rate = 0.95 if year >= 2017 else 0.85
    days_since_first = (year - 2010) * 365 + month * 30
    
    features = {
        'Payload Mass (kg)': payload_mass,
        'Year': year,
        'Month': month,
        'Quarter': quarter,
        'Booster_Reused': booster_reused,
        'Booster_Flight_Number': booster_flight_num,
        'Cumulative_Launches': cumulative_launches,
        'Cumulative_Success_Rate': cumulative_success_rate,
        'Orbit_Difficulty': orbit_difficulty,
        'Payload_Orbit_Ratio': payload_orbit_ratio,
        'Days_Since_First_Launch': days_since_first,
        'Is_NASA': 1 if is_nasa else 0,
        'Is_Commercial': 1 if is_commercial else 0
    }
    
    return features

def predict_launch_success(payload_mass=3000, booster_type='F9 Block 5', launch_site='CCAFS',
                          orbit_type='LEO', booster_reused=1, booster_flight_num=2,
                          year=2024, month=6, is_nasa=False, is_commercial=True):
    """
    Predict launch success probability
    
    Parameters:
    -----------
    payload_mass : float
        Payload mass in kg
    booster_type : str
        One of: 'F9 v1.0', 'F9 v1.1', 'F9 Full Thrust', 'F9 Block 4', 'F9 Block 5', 'Falcon Heavy'
    launch_site : str
        One of: 'CCAFS', 'VAFB', 'KSC'
    orbit_type : str
        One of: 'LEO', 'GTO', 'Polar', 'HEO', 'Other'
    booster_reused : int
        0 for new, 1 for reused
    booster_flight_num : int
        Number of flights for this booster
    year : int
        Launch year
    month : int
        Launch month (1-12)
    is_nasa : bool
        True if NASA mission
    is_commercial : bool
        True if commercial mission
    
    Returns:
    --------
    dict with prediction, probability, and details
    """
    
    # Load model
    model, feature_names, metadata = load_model()
    
    # Create features
    features = create_input_features(
        payload_mass, booster_type, launch_site, orbit_type,
        booster_reused, booster_flight_num, year, month,
        is_nasa, is_commercial
    )
    
    # Add one-hot encoded features
    for feat in feature_names:
        if feat not in features:
            if 'Booster_Type_' in feat:
                features[feat] = 1 if feat == f'Booster_Type_{booster_type}' else 0
            elif 'Launch_Site_Simplified_' in feat:
                features[feat] = 1 if feat == f'Launch_Site_Simplified_{launch_site}' else 0
            elif 'Orbit_Simplified_' in feat:
                features[feat] = 1 if feat == f'Orbit_Simplified_{orbit_type}' else 0
            else:
                features[feat] = 0
    
    # Create DataFrame
    input_df = pd.DataFrame([features])[feature_names]
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    result = {
        'prediction': 'Success' if prediction == 1 else 'Failure',
        'success_probability': probability[1],
        'failure_probability': probability[0],
        'model_name': metadata['model_name'],
        'model_accuracy': metadata['accuracy'],
        'input_parameters': {
            'payload_mass_kg': payload_mass,
            'booster_type': booster_type,
            'launch_site': launch_site,
            'orbit': orbit_type,
            'booster_reused': 'Yes' if booster_reused else 'No',
            'booster_flights': booster_flight_num,
            'year': year,
            'month': month
        }
    }
    
    return result

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='Predict SpaceX launch success')
    
    parser.add_argument('--payload', type=float, default=3000, help='Payload mass in kg')
    parser.add_argument('--booster', type=str, default='F9 Block 5', 
                       choices=['F9 v1.0', 'F9 v1.1', 'F9 Full Thrust', 'F9 Block 4', 'F9 Block 5', 'Falcon Heavy'],
                       help='Booster type')
    parser.add_argument('--site', type=str, default='CCAFS', 
                       choices=['CCAFS', 'VAFB', 'KSC'],
                       help='Launch site')
    parser.add_argument('--orbit', type=str, default='LEO',
                       choices=['LEO', 'GTO', 'Polar', 'HEO', 'Other'],
                       help='Target orbit')
    parser.add_argument('--reused', type=int, default=1, choices=[0, 1],
                       help='Booster reused (0=No, 1=Yes)')
    parser.add_argument('--flights', type=int, default=2, help='Booster flight number')
    parser.add_argument('--year', type=int, default=2024, help='Launch year')
    parser.add_argument('--month', type=int, default=6, help='Launch month')
    parser.add_argument('--nasa', action='store_true', help='NASA mission flag')
    parser.add_argument('--commercial', action='store_true', default=True, help='Commercial mission flag')
    
    args = parser.parse_args()
    
    # Make prediction
    result = predict_launch_success(
        payload_mass=args.payload,
        booster_type=args.booster,
        launch_site=args.site,
        orbit_type=args.orbit,
        booster_reused=args.reused,
        booster_flight_num=args.flights,
        year=args.year,
        month=args.month,
        is_nasa=args.nasa,
        is_commercial=args.commercial
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("🚀 SPACEX LAUNCH SUCCESS PREDICTION")
    print("=" * 70)
    print(f"\nModel: {result['model_name']}")
    print(f"Model Accuracy: {result['model_accuracy']:.2%}")
    print("\nInput Parameters:")
    for key, value in result['input_parameters'].items():
        print(f"  {key}: {value}")
    
    print("\n" + "-" * 70)
    print(f"\n{'✅ PREDICTION: SUCCESS' if result['prediction'] == 'Success' else '❌ PREDICTION: FAILURE'}")
    print(f"\nSuccess Probability: {result['success_probability']:.1%}")
    print(f"Failure Probability: {result['failure_probability']:.1%}")
    print("\n" + "=" * 70)
    
    # Confidence level
    if result['success_probability'] > 0.9:
        print("Confidence: Very High ⭐⭐⭐")
    elif result['success_probability'] > 0.75:
        print("Confidence: High ⭐⭐")
    else:
        print("Confidence: Moderate ⭐")
    
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
