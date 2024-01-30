import openml
import pandas as pd
import dash
from dash import dcc, html, dash_table
import plotly.graph_objs as go
import numpy as np
import os

app = dash.Dash(__name__)

# Versuch, die dataset_id aus einer externen Konfigurationsdatei zu importieren
try:
    from config import dataset_id
except ImportError:
    dataset_id = None  # Keine dataset_id definiert

# Definition der test_file (kann überschrieben werden, wenn eine dataset_id vorhanden ist)
test_file = 'Data_Science_API_OpenML/Downloads/heart-statlog-gaps.csv'  # Beispiel: 'test.csv'

def load_dataset_from_file(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Datei erfolgreich geladen: {file_path}")
        return df
    except Exception as e:
        print(f"Fehler beim Laden der Datei {file_path}: {e}")
        return None

def download_dataset(dataset_id=None):
    # Priorisiere dataset_id über test_file, wenn vorhanden
    if dataset_id:
        try:
            dataset = openml.datasets.get_dataset(dataset_id)
            X, y, _, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
            df = pd.DataFrame(X, columns=attribute_names)
            if y is not None:
                df['target'] = y
            return df
        except Exception as e:
            print(f"Fehler beim Herunterladen des Datensatzes: {e}")
            return None
    elif test_file:
        return load_dataset_from_file(test_file)
    else:
        print("Keine Datenquelle angegeben.")
        return None

# Entscheide, welche Datenquelle verwendet wird basierend auf der Verfügbarkeit der dataset_id
initial_df = download_dataset(dataset_id)

def create_data_completeness_graph(df):
    total_values = np.product(df.shape)
    missing_values = df.isnull().sum().sum()
    complete_values = total_values - missing_values
    fig = go.Figure(data=[go.Pie(labels=['Vollständige Daten', 'Fehlende Datenfelder'], values=[complete_values, missing_values], hole=.3)])
    fig.update_layout(title_text="Vollständigkeit des Datensets")
    return fig

def create_feature_summary_table(df):
    summary = df.describe().transpose().reset_index()
    summary.rename(columns={'index': 'Feature'}, inplace=True)
    return summary.to_dict('records'), [{"name": i, "id": i} for i in summary.columns]


completeness_graph = create_data_completeness_graph(initial_df)
summary_records, columns = create_feature_summary_table(initial_df)

app.layout = html.Div([
    dcc.Graph(figure=completeness_graph),
    dash_table.DataTable(
        columns=columns,
        data=summary_records,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px'},
        style_header={'fontWeight': 'bold'}
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
