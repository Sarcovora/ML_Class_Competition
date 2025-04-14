import pandas as pd

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


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
    data.head()
    return data


process_count = 12

train_data = pd.read_csv('train.csv', header=0)
train_data = process_data(train_data)
train_data.head()

train_x = train_data.drop('Outcome Type', axis=1)
train_y = train_data['Outcome Type']
forest = RandomForestClassifier(n_jobs=process_count)

param_grid = {
    'n_estimators': [50, 100, 150, 250, 500],
    'min_samples_leaf': [25, 50, 100, 500, 1000],
    'max_features': [0.5, 0.7, 0.85, 0.95, 1.0],
}

# param_grid = {
#     'n_estimators': [100, 250, 500],
#     'min_samples_leaf': [1, 5, 10, 25, 50],
#     'max_features': ['sqrt', 0.6, 0.8, 1.0],
# }

grid_search = GridSearchCV(forest, param_grid, cv=5, scoring='accuracy', n_jobs=process_count, verbose=3)
grid_search.fit(train_x, train_y)

print(f"best params: {grid_search.best_params_}")
print(f"best score: {grid_search.best_score_}")
