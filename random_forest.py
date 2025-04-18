import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report


def convert_to_weeks(value):
    value = value.lower()  # Make it case-insensitive
    if 'week' in value:
        # Extract number of weeks
        return int(value.split()[0])
    elif 'month' in value:
        # Convert months to weeks (1 month ≈ 4.345 weeks)
        return int(value.split()[0]) * 4.345
    elif 'year' in value:
        # Convert years to weeks (1 year ≈ 52.1775 weeks)
        return int(value.split()[0]) * 52.1775
    elif 'day' in value:
        # Convert days to 0 weeks
        return 0
    return 0  # In case of unexpected values

def extract_month_year(df, column='Intake Time'):
    """
    Convert the specified datetime column in the DataFrame to two new columns: Month and Year.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the datetime column.
        column (str): Name of the column to convert (default is 'Intake Time').

    Returns:
        pd.DataFrame: The original DataFrame with additional 'Month' and 'Year' columns.
    """
    # Convert the column to datetime objects. Adjust the format if needed.
    df[column] = pd.to_datetime(df[column], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

    # Extract the month and year from the datetime column
    df['Intake Month'] = df[column].dt.month
    df['Intake Year'] = df[column].dt.year

    return df


def process_data(data):    
    data = data.drop(columns=['Id', 'Name', 'Outcome Time'])
    data = data.drop(columns=['Found Location', 'Date of Birth'])
    # data = data.drop(columns=['Breed', 'Color'])
    data = data.dropna()
    print(data.columns)
    data = pd.get_dummies(data, columns=['Intake Condition', 'Intake Type', 'Animal Type', 'Sex upon Intake', 'Breed', 'Color'])
    data['Age upon Intake'] = data['Age upon Intake'].apply(convert_to_weeks)
    data = extract_month_year(data, column='Intake Time')
    data = data.drop(columns=['Intake Time'])
    print(data.columns)
    return data

def process_test_data(data):    
    data = data.drop(columns=['Id'])
    data = data.drop(columns=['Found Location', 'Date of Birth'])
    # data = data.drop(columns=['Breed', 'Color'])
    data = data.dropna()
    print(data.columns)
    data = pd.get_dummies(data, columns=['Intake Condition', 'Intake Type', 'Animal Type', 'Sex upon Intake', 'Breed', 'Color'])
    data['Age upon Intake'] = data['Age upon Intake'].apply(convert_to_weeks)
    data = extract_month_year(data, column='Intake Time')
    data = data.drop(columns=['Intake Time'])
    print(data.columns)
    return data

train_data = pd.read_csv('train.csv', header=0)
train_data = process_data(train_data)
print(train_data.head())

train_x = train_data.drop('Outcome Type', axis=1)
train_y = train_data['Outcome Type']

test_data = pd.read_csv('test.csv', header=0)
test_data = process_test_data(test_data)
test_data = test_data.reindex(columns=train_data.columns, fill_value=0)
test_data = test_data.drop(columns=['Outcome Type'])
print(test_data.head())

forest = RandomForestClassifier(n_estimators=150, min_samples_leaf=125, max_features=0.45, class_weight="balanced")
accuracy = cross_val_score(forest, train_x, train_y, cv=5, scoring='balanced_accuracy')
print(f"Accuracy: {accuracy}")

model = forest.fit(train_x, train_y)

print("=== PREDICTIONS ===")

predictions = forest.predict(test_data)

df = pd.DataFrame({
    "Id": np.arange(1, len(predictions) + 1),
    "Outcome Type": predictions
})
print(df['Outcome Type'].value_counts())

df.to_csv("predictions.csv", index=False)
