import openml
import pandas as pd

def fetchDataList():
    """
    Fetches a list of datasets from OpenML.

    Returns:
        pandas.DataFrame: A DataFrame containing information about datasets.
    """
    datasets_list = openml.datasets.list_datasets(output_format='dataframe')
    return datasets_list

def calcRangeDatasets(df):
    """
    Calculates the range (min and max) for various dataset attributes from a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing dataset attributes.

    Returns:
        dict: A dictionary containing the ranges for different dataset attributes.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a Pandas DataFrame")

    # List of columns for which minima and maxima are to be calculated
    columns_to_calculate = [
        'NumberOfInstances', 'NumberOfFeatures', 'NumberOfClasses',
        'NumberOfMissingValues', 'NumberOfInstancesWithMissingValues',
        'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures'
    ]

    # Check if all required columns are present
    for col in columns_to_calculate:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing in the DataFrame")

    # Calculate the minima and maxima for the relevant columns
    ranges = {}
    for col in columns_to_calculate:
        ranges[col] = [df[col].min(), df[col].max()]

    return ranges

def findDatasetNameWithMostFeatures(df, feature_column):
    """
    Finds the name of the dataset with the most features of a specific type.

    Args:
        df (pandas.DataFrame): The DataFrame containing dataset attributes.
        feature_column (str): The column indicating the number of features of the desired type.

    Returns:
        str: The name of the dataset with the most features of the specified type.
    """
    if feature_column not in df.columns or 'name' not in df.columns:
        raise ValueError(f"Required columns '{feature_column}' or 'name' are missing in the DataFrame")

    # Find the name of the dataset with the most features of the specified type
    max_features = df[feature_column].max()
    dataset_name = df[df[feature_column] == max_features]['name'].iloc[0]

    return dataset_name
