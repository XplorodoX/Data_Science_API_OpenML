import openml
import pandas as pd
import dash
from dash import dcc, html, dash_table
import plotly.graph_objs as go
import numpy as np
import os
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
from datetime import datetime

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Versuch, die dataset_id aus einer externen Konfigurationsdatei zu importieren
try:
    from config import dataset_id
except ImportError:
    dataset_id = None  # Fallback, wenn keine dataset_id definiert ist

test_file = 'Data_Science_API_OpenML/Downloads/heart-statlog-gaps.csv'

def download_dataset(dataset_id=None):
    dataset_info = {}
    df = pd.DataFrame()  # Initialisieren Sie df hier, um sicherzustellen, dass es immer definiert ist
    
    if dataset_id:
        try:
            dataset = openml.datasets.get_dataset(dataset_id)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute, dataset_format='dataframe')
            df = pd.DataFrame(X, columns=attribute_names)
            if y is not None:
                df['target'] = y
            
            try:
                upload_date_obj = datetime.strptime(dataset.upload_date, '%Y-%m-%d')
                formatted_date = upload_date_obj.strftime('%Y-%m-%d')
            except ValueError:
                formatted_date = dataset.upload_date
            
            dataset_info = {
                'name': dataset.name,
                'features_count': len(attribute_names),
                'instances_count': df.shape[0],
                'upload_date': formatted_date
            }
        except Exception as e:
            print(f"Fehler beim Herunterladen des Datensatzes: {e}")
    else:
        print("Keine dataset_id angegeben. Versuche, Testfile zu laden...")
        try:
            df = pd.read_csv(test_file)
            dataset_info = {
                'name': 'Lokale Testdatei',
                'features_count': len(df.columns),
                'instances_count': len(df),
                'upload_date': 'Nicht verfügbar'  # Da es sich um ein lokales Testfile handelt, haben wir kein Uploaddatum
            }
        except Exception as e:
            print(f"Fehler beim Laden der Testdatei {test_file}: {e}")

    return df, dataset_info


initial_df, dataset_info = download_dataset(dataset_id)

def create_data_completeness_graph(df):
    if df.empty:
        return go.Figure()  # Gibt eine leere Figur zurück, wenn df leer ist
    total_values = np.product(df.shape)
    missing_values = df.isnull().sum().sum()
    complete_values = total_values - missing_values
    fig = go.Figure(data=[go.Pie(labels=['Vollständige Daten', 'Fehlende Datenfelder'],
                                 values=[complete_values, missing_values], hole=.6)])
    fig.update_layout(title_text="Vollständigkeit des Datensets", title_x=0.5)
    return fig

def create_feature_summary_table(df):
    if df.empty:
        return [], []  # Gibt leere Werte zurück, wenn df leer ist
    summary = df.describe().transpose().reset_index()
    summary.rename(columns={'index': 'Feature'}, inplace=True)
    return summary.to_dict('records'), [{"name": i, "id": i} for i in summary.columns]

completeness_graph = create_data_completeness_graph(initial_df)
summary_records, columns = create_feature_summary_table(initial_df)

app.layout = html.Div([
    # Flex-Container für Datensatzinformationen und Kuchendiagramm nebeneinander
    html.Div([
        # Datensatzinformationen links mit Einrückung
        html.Div([
            html.H4("Datensatzinformationen"),
            html.P(f"Name des Datensets: {dataset_info.get('name', 'Nicht verfügbar')}"),
            html.P(f"Anzahl der Features: {dataset_info.get('features_count', 'Nicht verfügbar')}"),
            html.P(f"Anzahl der Datenpunkte: {dataset_info.get('instances_count', 'Nicht verfügbar')}"), # Datenpunkte = Instanzen
            html.P(f"Uploaddatum: {dataset_info.get('upload_date', 'Nicht verfügbar')}"),
        ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '20px', 'padding-top': '17px'}),  # Margin links und Padding oben hinzugefügt

        # Kuchendiagramm rechts von den Datensatzinformationen
        html.Div([
            dcc.Graph(figure=completeness_graph),
        ], style={'width': '75%', 'display': 'inline-block'}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),


    # DataTable und Histogramm unter dem Flex-Container
    dash_table.DataTable(
        id='feature-summary-table',
        columns=columns,
        data=summary_records,
        page_size=12,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '6px'},
        style_header={'fontWeight': 'bold'},
    ),
    html.Div(id='feature-histogram'),  # Container für das Histogramm
])


@app.callback(
    Output('feature-histogram', 'children'),
    [Input('feature-summary-table', 'active_cell')]
)
def update_histogram(active_cell):
    if not active_cell or initial_df.empty:
        return "Bitte wählen Sie für die Darstellung eines Histogrammes ein Feature aus der Tabelle."
    selected_row = active_cell['row']
    selected_feature = summary_records[selected_row]["Feature"]
    if selected_feature not in initial_df.columns:
        return f"Feature {selected_feature} nicht im DataFrame gefunden."
    fig = px.histogram(initial_df, x=selected_feature, title=f'Histogramm von {selected_feature}')
    return dcc.Graph(figure=fig)

if __name__ == '__main__':
    app.run_server(debug=True)
