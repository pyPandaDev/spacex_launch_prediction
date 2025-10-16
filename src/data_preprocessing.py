"""
Data preprocessing utilities for SpaceX launch prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re

def load_data(filepath):
    """Load SpaceX launch data from CSV"""
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """Clean and preprocess the raw data"""
    df = df.copy()
    
    # Standardize column names
    df.columns = df.columns.str.strip()
    
    # Create datetime column
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time (UTC)'])
    
    # Extract temporal features
    df['Year'] = df['DateTime'].dt.year
    df['Month'] = df['DateTime'].dt.month
    df['Day'] = df['DateTime'].dt.day
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['Quarter'] = df['DateTime'].dt.quarter
    
    # Clean Payload Mass
    df['Payload Mass (kg)'] = df['Payload Mass (kg)'].astype(str).str.replace(',', '').str.strip()
    df['Payload Mass (kg)'] = pd.to_numeric(df['Payload Mass (kg)'], errors='coerce')
    
    # Fill missing payload mass with median
    df['Payload Mass (kg)'].fillna(df['Payload Mass (kg)'].median(), inplace=True)
    
    # Clean Booster Version
    df['Booster Version'] = df['Booster Version'].str.strip()
    
    # Extract booster reuse information
    df['Booster_Reused'] = df['Booster Version'].str.contains(r'\.\d', regex=True).astype(int)
    
    # Extract flight number from booster version
    df['Booster_Flight_Number'] = df['Booster Version'].str.extract(r'\.(\d+)')[0]
    df['Booster_Flight_Number'] = pd.to_numeric(df['Booster_Flight_Number'], errors='coerce').fillna(1).astype(int)
    
    # Simplify booster type
    def simplify_booster(version):
        if 'FH' in str(version) or 'Heavy' in str(version):
            return 'Falcon Heavy'
        elif 'B5' in str(version):
            return 'F9 Block 5'
        elif 'B4' in str(version):
            return 'F9 Block 4'
        elif 'FT' in str(version):
            return 'F9 Full Thrust'
        elif 'v1.1' in str(version):
            return 'F9 v1.1'
        elif 'v1.0' in str(version):
            return 'F9 v1.0'
        else:
            return 'Other'
    
    df['Booster_Type'] = df['Booster Version'].apply(simplify_booster)
    
    # Simplify launch site
    df['Launch_Site_Simplified'] = df['Launch Site'].str.split().str[0]
    
    # Simplify orbit types
    def simplify_orbit(orbit):
        orbit = str(orbit).upper()
        if 'LEO' in orbit:
            return 'LEO'
        elif 'GTO' in orbit:
            return 'GTO'
        elif 'SSO' in orbit or 'POLAR' in orbit:
            return 'Polar'
        elif 'HEO' in orbit:
            return 'HEO'
        else:
            return 'Other'
    
    df['Orbit_Simplified'] = df['Orbit'].apply(simplify_orbit)
    
    # Create binary success target
    df['Mission_Success'] = df['Mission Outcome'].str.contains('Success', case=False, na=False).astype(int)
    
    # Create landing success target
    df['Landing_Success'] = df['Landing Outcome'].str.contains('Success', case=False, na=False).astype(int)
    
    # Create cumulative launch count (experience over time)
    df['Cumulative_Launches'] = range(1, len(df) + 1)
    
    # Create cumulative success rate
    df['Cumulative_Success_Rate'] = df['Mission_Success'].expanding().mean()
    
    return df

def create_features(df):
    """Create additional features for modeling"""
    df = df.copy()
    
    # Payload to orbit difficulty score (heavier payloads to higher orbits are harder)
    orbit_difficulty = {
        'LEO': 1,
        'Polar': 2,
        'HEO': 3,
        'GTO': 4,
        'Other': 2.5
    }
    df['Orbit_Difficulty'] = df['Orbit_Simplified'].map(orbit_difficulty)
    
    # Payload-orbit interaction
    df['Payload_Orbit_Ratio'] = df['Payload Mass (kg)'] / df['Orbit_Difficulty']
    
    # Days since first launch
    first_launch = df['DateTime'].min()
    df['Days_Since_First_Launch'] = (df['DateTime'] - first_launch).dt.days
    
    # Is this a NASA mission?
    df['Is_NASA'] = df['Customer'].str.contains('NASA', case=False, na=False).astype(int)
    
    # Is this a commercial mission?
    df['Is_Commercial'] = (~df['Customer'].str.contains('NASA|Air Force|NRO|NOAA', case=False, na=False)).astype(int)
    
    return df

def encode_categorical(df, categorical_cols):
    """One-hot encode categorical features"""
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded

def prepare_for_modeling(df):
    """Prepare the final dataset for modeling"""
    # Select features for modeling
    feature_cols = ['Payload Mass (kg)', 'Year', 'Month', 'Quarter', 
                    'Booster_Reused', 'Booster_Flight_Number', 
                    'Cumulative_Launches', 'Cumulative_Success_Rate',
                    'Orbit_Difficulty', 'Payload_Orbit_Ratio', 
                    'Days_Since_First_Launch', 'Is_NASA', 'Is_Commercial',
                    'Booster_Type', 'Launch_Site_Simplified', 'Orbit_Simplified']
    
    target_col = 'Mission_Success'
    
    df_model = df[feature_cols + [target_col]].copy()
    
    # Encode categorical variables
    categorical_cols = ['Booster_Type', 'Launch_Site_Simplified', 'Orbit_Simplified']
    df_model = encode_categorical(df_model, categorical_cols)
    
    return df_model

def get_feature_target_split(df_model, target_col='Mission_Success'):
    """Split features and target"""
    X = df_model.drop(columns=[target_col])
    y = df_model[target_col]
    return X, y
