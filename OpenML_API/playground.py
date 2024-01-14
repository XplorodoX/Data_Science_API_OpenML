import openml
import pandas as pd

def calcRangeDatasets(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df muss ein Pandas DataFrame sein")

    # Liste der Spalten, für die Minima und Maxima berechnet werden sollen
    columns_to_calculate = [
        'NumberOfInstances', 'NumberOfFeatures', 'NumberOfClasses',
        'NumberOfMissingValues', 'NumberOfInstancesWithMissingValues',
        'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures'
    ]

    # Überprüfen, ob alle benötigten Spalten vorhanden sind
    for col in columns_to_calculate:
        if col not in df.columns:
            raise ValueError(f"Benötigte Spalte '{col}' fehlt im DataFrame")

    # Berechnen der Maxima und Minima für die relevanten Spalten
    ranges = {}
    for col in columns_to_calculate:
        ranges[col] = [df[col].min(), df[col].max()]

    return ranges

def findDatasetNameWithMostFeatures(df, feature_column):
    if feature_column not in df.columns or 'name' not in df.columns:
        raise ValueError(f"Benötigte Spalten '{feature_column}' oder 'name' fehlen im DataFrame")

    # Finden des Namens des Datensatzes mit den meisten Features der angegebenen Art
    max_features = df[feature_column].max()
    dataset_name = df[df[feature_column] == max_features]['name'].iloc[0]

    return dataset_name

def fetchDataList():
    datasets_list = openml.datasets.list_datasets(output_format='dataframe')
    return datasets_list

if __name__ == '__main__':
    datasets_list = fetchDataList()
    ranges = calcRangeDatasets(datasets_list)
    dataset_name_most_categorical = findDatasetNameWithMostFeatures(datasets_list, 'NumberOfSymbolicFeatures')
    print("Name des Datensatzes mit den meisten kategorialen Features:", dataset_name_most_categorical)

    dataset_name_most_numeric = findDatasetNameWithMostFeatures(datasets_list, 'NumberOfNumericFeatures')
    print("Name des Datensatzes mit den meisten numerischen Features:", dataset_name_most_numeric)