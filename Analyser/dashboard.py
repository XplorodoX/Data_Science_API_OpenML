import openml
import pandas as pd
import dash
from dash import dcc, html, dash_table
import plotly.graph_objs as go
import numpy as np
import os
from dash import dcc, html, Input, Output, dash_table
from dash_table.Format import Format, Scheme
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
            print(f"Error downloading the data set: {e}")
    else:
        print("No dataset_id specified. Attempts to load test file...")
        try:
            df = pd.read_csv(test_file)
            dataset_info = {
                'name': 'local testfile',
                'features_count': len(df.columns),
                'instances_count': len(df),
                'upload_date': 'Not available'  # Da es sich um ein lokales Testfile handelt, haben wir kein Uploaddatum
            }
        except Exception as e:
            print(f"Error loading the test file {test_file}: {e}")

    return df, dataset_info


initial_df, dataset_info = download_dataset(dataset_id)

def create_data_completeness_graph(df):
    if df.empty:
        return go.Figure()  # Gibt eine leere Figur zurück, wenn df leer ist
    total_values = np.product(df.shape)
    missing_values = df.isnull().sum().sum()
    complete_values = total_values - missing_values
    fig = go.Figure(data=[go.Pie(labels=['Complete data', 'Missing data fields'],
                                 values=[complete_values, missing_values], hole=.6)])
    fig.update_layout(title_text="Completeness of the dataset", title_x=0.5)
    return fig

def format_number(value):
    """Formatiert eine Zahl mit bis zu zwei Nachkommastellen, entfernt jedoch nachfolgende Nullen."""
    if pd.isnull(value):
        return None  # Umgang mit NaN-Werten
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return f"{int(value)}"  # Konvertiert den Wert in eine Ganzzahl und dann in einen String, wenn es eine ganze Zahl ist
    else:
        formatted_value = f"{value:.4f}"  # Formatieren mit zwei Nachkommastellen
        return formatted_value.rstrip('0').rstrip('.')  # Entfernt nachfolgende Nullen und den Dezimalpunkt, falls keine Nachkommastellen vorhanden sind

def create_feature_summary_table(df):
    if df.empty:
        return [], []  # Gibt leere Werte zurück, wenn df leer ist
    
    summary = df.describe(percentiles=[.25, .5, .75, .97, .997]).transpose()
    
    # Modus berechnen und zur Zusammenfassung hinzufügen
    modes = df.mode().iloc[0]
    summary['mode'] = [modes[col] if col in modes else np.nan for col in summary.index]

    summary.reset_index(inplace=True)
    summary.rename(columns={'index': 'Feature'}, inplace=True)

    # Formatieren der numerischen Werte als Strings mit bedingter Nachkommastellen-Anzeige
    for col in summary.columns[1:]:  # Überspringe die 'Feature'-Spalte
        summary[col] = summary[col].apply(format_number)

    summary_records = summary.to_dict('records')
    columns = [{"name": i, "id": i} for i in summary.columns]

    return summary_records, columns


completeness_graph = create_data_completeness_graph(initial_df)
summary_records, columns = create_feature_summary_table(initial_df)

app.layout = html.Div([
    # Flex-Container für Datensatzinformationen und Kuchendiagramm nebeneinander
    html.Div([
        # Datensatzinformationen links mit Einrückung
        html.Div([
            html.H4("Datasetinformation"),
            html.P(f"Name of Dataset: {dataset_info.get('name', 'Not available')}"),
            html.P(f"Number of Features: {dataset_info.get('features_count', 'Not available')}"),
            html.P(f"Number of Instances: {dataset_info.get('instances_count', 'Not available')}"), # Datenpunkte = Instanzen
            html.P(f"Upload date: {dataset_info.get('upload_date', 'Not available')}"),
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
        style_table={'overflowX': 'auto', 'height': '391px'},
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
        return "Please select a feature from the table to display a histogram."
    selected_row = active_cell['row']
    selected_feature = summary_records[selected_row]["Feature"]
    if selected_feature not in initial_df.columns:
        return f"Feature {selected_feature} not found in dataframe."

    # Erstellung des Histogramms für das ausgewählte Feature
    fig = px.histogram(initial_df, x=selected_feature, title=f'Histogram of {selected_feature}')

    # Berechnung der Quantile für das ausgewählte Feature
    quantiles = initial_df[selected_feature].quantile([0.25, 0.5, 0.75, 0.97, 0.997]).to_dict()

    # Hinzufügen von vertikalen Linien für jedes Quantil
    for quantile, value in quantiles.items():
        fig.add_vline(x=value, line_dash="solid", line_color="blue",
                      annotation_text=f"{quantile * 100}th: {value:.2f}", 
                      annotation_position="top right",
                      annotation=dict(font_size=10, font_color="green", showarrow=False))

    # Verbesserung der Hover-Funktionalität
    fig.update_traces(hoverinfo='x+y', selector=dict(type='histogram'))
    
    return dcc.Graph(figure=fig)



if __name__ == '__main__':
    app.run_server(debug=True)
