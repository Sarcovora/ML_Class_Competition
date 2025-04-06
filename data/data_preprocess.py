import pandas as pd

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

def assign_breed_frequency(df, column='Breed'):
    # Calculate the frequency of each breed
    breed_counts = df[column].value_counts()

    # Assign the actual frequency to the 'Breed_Popularity' column
    df['Breed'] = df[column].map(breed_counts)

    return df

train_data = pd.read_csv('train.csv', header=0)
print(('-' * 20) + "Columns before processing" + ('-' * 20))
print(train_data.columns)
train_data = train_data.drop(columns=['Id', 'Name', 'Outcome Time'])
train_data = train_data.drop(columns=['Found Location', 'Date of Birth'])
train_data = train_data.dropna()
train_data = pd.get_dummies(train_data, columns=['Intake Condition', 'Intake Type', 'Animal Type', 'Sex upon Intake'])
train_data['Age upon Intake'] = train_data['Age upon Intake'].apply(convert_to_weeks)
train_data = extract_month_year(train_data, column='Intake Time')
train_data = train_data.drop(columns=['Intake Time'])
train_data = assign_breed_frequency(train_data, column='Breed')
print(('-' * 20) + "Columns after processing" + ('-' * 20))
print(train_data.columns)

# save to disk as a serialized file
train_data.to_pickle('processed_train_data.pkl')
print("Processed dataset saved as 'processed_train_data.pkl'")

"""
can load data as:

train_data = pd.read_pickle('processed_train_data.pkl')
print("Processed dataset loaded.")
"""
