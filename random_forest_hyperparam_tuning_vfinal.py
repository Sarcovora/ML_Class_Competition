'''
Implement city found location
'''
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, make_scorer
import re

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

def extract_city(location):
    if location.lower().strip() == 'outside jurisdiction':
        return 'Outside Jurisdiction'
    
    # Match ' in City (ST)' pattern near the end
    match = re.search(r'\b in ([A-Za-z\s]+) \([A-Z]{2}\)$', location)
    if match:
        return match.group(1).strip()
    
    # Match 'City (ST)' without an address
    match = re.search(r'^([A-Za-z\s]+) \([A-Z]{2}\)$', location)
    if match:
        return match.group(1).strip()
    else:
        return 'Other'
    
def process_data(data):    
    data = data.drop(columns=['Id', 'Name', 'Outcome Time'])
    data = data.drop(columns=['Date of Birth', 'Found Location'])

    data = data.dropna()

    data = pd.get_dummies(data, columns=['Intake Condition', 'Intake Type', 'Animal Type', 'Sex upon Intake', 'Breed', 'Color', "City"])
    data['Age upon Intake'] = data['Age upon Intake'].apply(convert_to_weeks)
    data = extract_month_year(data, column='Intake Time')
    data = data.drop(columns=['Intake Time'])
    print(data.columns)
    # data.head()
    return data


process_count = 4

train_data = pd.read_csv('train.csv', header=0)
train_data = process_data(train_data)
train_data.head()

train_x = train_data.drop('Outcome Type', axis=1)
train_y = train_data['Outcome Type']
forest = RandomForestClassifier(n_jobs=process_count, class_weight="balanced", oob_score=make_scorer(balanced_accuracy_score))

param_grid = {
    'n_estimators': [75, 100, 125, 150, 200, 250, 275],
    'min_samples_leaf': [10, 25, 50, 75, 100, 125, 150],
    'max_features': [0.25, 0.55, 0.75, 0.85, 0.90, 0.95],
}



grid_search = RandomizedSearchCV(forest, param_grid, cv=5, scoring=make_scorer(balanced_accuracy_score), n_jobs=process_count, verbose=3, return_train_score=True)
grid_search.fit(train_x, train_y)

print(f"best params: {grid_search.best_params_}")
print(f"best score: {grid_search.best_score_}")
