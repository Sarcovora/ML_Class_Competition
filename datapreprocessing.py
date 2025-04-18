import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os
import re

PROCESS_TRAIN_DATA = False  # Set to False to process test.csv
OUTPUT_DIR = 'processed_data'
DATA_DIR = 'data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

if PROCESS_TRAIN_DATA:
    input_filename = os.path.join(DATA_DIR, 'train.csv')
    output_data_filename = os.path.join(OUTPUT_DIR, 'proc_train_data.pkl')
    output_mappings_filename = os.path.join(OUTPUT_DIR, 'proc_mappings.pkl')
    print("Processing TRAINING data...")
else:
    input_filename = os.path.join(DATA_DIR, 'test.csv')
    # Test data uses mappings FROM training data
    input_mappings_filename = os.path.join(OUTPUT_DIR, 'proc_mappings.pkl')
    output_data_filename = os.path.join(OUTPUT_DIR, 'proc_test_data.pkl')
    print("Processing TEST data...")


def parse_age_to_days(age_str):
    """Converts age string (e.g., '2 years', '3 weeks') to days."""
    if pd.isna(age_str):
        return np.nan
    age_str = str(age_str).lower()
    num = re.findall(r'\d+', age_str)
    if not num:
        return 0
    num = int(num[0])

    if 'year' in age_str:
        return num * 365
    elif 'month' in age_str:
        return num * 30
    elif 'week' in age_str:
        return num * 7
    elif 'day' in age_str:
        return num
    return 0


def clean_text_feature(text):
    """Basic cleaning for breed/color."""
    if pd.isna(text):
        return "unknown"
    text = str(text).lower()
    text = re.sub(r'\s+mix$', '', text)
    text = re.sub(r'\/.*', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = text.strip()
    if not text:
        return "unknown"
    return text


# --- Load Data ---
print(f"Loading {input_filename}...")
df = pd.read_csv(input_filename, header=0)
print("Columns before processing:", df.columns.tolist())


# --- Feature Engineering & Selection ---
# Keep ID for test set submission mapping later
ids = df['Id'].copy() if 'Id' in df.columns else None

# Drop columns
cols_to_drop = ['Id', 'Name', 'Found Location', 'Date of Birth']
if PROCESS_TRAIN_DATA:
    cols_to_drop.extend(['Outcome Time'])  # Not available at test time

# Ensure Id is not dropped if it exists and we are processing test data
if not PROCESS_TRAIN_DATA and 'Id' in cols_to_drop:
    cols_to_drop.remove('Id')

df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# --- Feature Processing ---
# Target Variable (Train only)
if PROCESS_TRAIN_DATA:
    outcome_mapping = {
        'Return to Owner': 0, 'Transfer': 1, 'Adoption': 2,
        'Died': 3, 'Euthanasia': 4
    }
    # Handle potential missing target values before mapping
    df = df.dropna(subset=['Outcome Type'])
    df['Outcome Type'] = df['Outcome Type'].map(outcome_mapping)
    target = df['Outcome Type'].astype(np.int64)
    df = df.drop(columns=['Outcome Type'])
else:
    target = None  # No target in test data

# Age
df['AgeInDays'] = df['Age upon Intake'].apply(parse_age_to_days)
df = df.drop(columns=['Age upon Intake'])

# Intake Time
time_col = 'Intake Time'
if PROCESS_TRAIN_DATA:
    df[time_col] = pd.to_datetime(df[time_col], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
else:  # Test data has different format
    df[time_col] = pd.to_datetime(df[time_col], format='%m/%d/%y %H:%M', errors='coerce')

df['IntakeMonth'] = df[time_col].dt.month.fillna(0).astype(int)
df['IntakeYear'] = df[time_col].dt.year.fillna(0).astype(int)
df['IntakeHour'] = df[time_col].dt.hour.fillna(0).astype(int)
df['IntakeDayOfWeek'] = df[time_col].dt.dayofweek.fillna(0).astype(int)  # Monday=0, Sunday=6
df['IntakeIsWeekend'] = df['IntakeDayOfWeek'].isin([5, 6]).astype(int)
df = df.drop(columns=[time_col])

# Categorical Features for Embedding
cat_embed_cols = ['Breed', 'Color']
if PROCESS_TRAIN_DATA:
    label_encoders = {}
    vocab_sizes = {}
    for col in cat_embed_cols:
        df[col] = df[col].apply(clean_text_feature)
        df[col] = df[col].fillna('unknown')
        le = LabelEncoder()
        # Fit on unique values including 'unknown'
        unique_vals = df[col].unique()
        le.fit(list(unique_vals))
        df[col + '_encoded'] = le.transform(df[col])
        label_encoders[col] = le
        vocab_sizes[col] = len(le.classes_)  # Size of vocabulary
        print(f"'{col}' vocab size: {vocab_sizes[col]}")
    df = df.drop(columns=cat_embed_cols)
    mappings = {'label_encoders': label_encoders, 'vocab_sizes': vocab_sizes}
else:  # Test Data
    print(f"Loading mappings from {input_mappings_filename}")
    with open(input_mappings_filename, 'rb') as f:
        mappings = pickle.load(f)
    label_encoders = mappings['label_encoders']
    for col in cat_embed_cols:
        df[col] = df[col].apply(clean_text_feature)
        df[col] = df[col].fillna('unknown')
        le = label_encoders[col]
        df[col + '_encoded'] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else le.transform(['unknown'])[0])
    df = df.drop(columns=cat_embed_cols)


# One-Hot Encoding
cat_ohe_cols = ['Intake Condition', 'Intake Type', 'Animal Type', 'Sex upon Intake']
# Clean up potential NaN before OHE
for col in cat_ohe_cols:
    df[col] = df[col].fillna('Unknown')
df = pd.get_dummies(df, columns=cat_ohe_cols, prefix=cat_ohe_cols, dummy_na=False)  # Already handled NaN

# --- Handle Missing Numerical Values ---
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
# Remove encoded categoricals from numerical list if they slipped in
numerical_cols = [col for col in numerical_cols if '_encoded' not in col]
# Remove Id if present in test df
if not PROCESS_TRAIN_DATA and 'Id' in numerical_cols:
    numerical_cols.remove('Id')

if PROCESS_TRAIN_DATA:
    imputation_values = df[numerical_cols].median()  # median for robustness to outliers
    mappings['imputation_values'] = imputation_values
    mappings['numerical_cols'] = numerical_cols
else:
    imputation_values = mappings['imputation_values']

df[numerical_cols] = df[numerical_cols].fillna(imputation_values)


# --- Scale Numerical Features ---
if PROCESS_TRAIN_DATA:
    scaler = StandardScaler()
    # Fit only on numerical cols identified
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    mappings['scaler'] = scaler
    # Save mappings and scaler (crucial for test set)
    print(f"Saving mappings and scaler to {output_mappings_filename}")
    with open(output_mappings_filename, 'wb') as f:
        pickle.dump(mappings, f)
else:  # Test Data
    scaler = mappings['scaler']
    numerical_cols = mappings['numerical_cols']
    df[numerical_cols] = scaler.transform(df[numerical_cols])


# --- Final Check & Save ---
# Align columns between train/test (important after OHE)... suffered from many bugs from this
if PROCESS_TRAIN_DATA:
    mappings['final_columns'] = df.columns.tolist()
    # Save mappings again with final columns
    with open(output_mappings_filename, 'wb') as f:
        pickle.dump(mappings, f)
else:
    # Ensure test has same columns as train, adding missing OHE cols with 0, removing extra
    train_cols = mappings['final_columns']
    # Add missing columns
    for col in train_cols:
        if col not in df.columns:
            df[col] = 0
    # Remove extra columns (excluding Id)
    extra_cols = [col for col in df.columns if col not in train_cols and col != 'Id']
    df = df.drop(columns=extra_cols)
    # Reorder to match training data exactly (excluding Id)
    df = df[train_cols + (['Id'] if 'Id' in df.columns else [])]  # Keep Id at the end for test

print("Columns after processing:", df.columns.tolist())
print(f"Data shape: {df.shape}")

# Separate features and target (if training)
if PROCESS_TRAIN_DATA:
    processed_data = {'X': df, 'y': target}
else:
    processed_data = {'X_test': df}  # Keep Id column in X_test

print(f"Saving processed data to {output_data_filename}")
with open(output_data_filename, 'wb') as f:
    pickle.dump(processed_data, f)

print("Preprocessing finished.")
