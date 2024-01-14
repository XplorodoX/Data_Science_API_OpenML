import os
import time
import warnings
from contextlib import contextmanager
import dash
import pandas as pd
import openml
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import json
import logging
from datetime import datetime, timedelta
from flask_caching import Cache
import requests
import Helper as helper
from tqdm import tqdm

#TODO: Progressbar weiter implementieren  -> nur Meta Daten werden geladen von mir und die Datenset id wird weiter gegeben
#TODO: Schieberegler wieder einbauen und dynamisch anpassen lassen die features die Werte am Regler
#TODO: Vielleicht caching aber nicht in eine Datenbank speichern!
#TODO: Caching mit OpenML api??? -> Ansatz

# Dash app setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

global_max_number_of_instances = 0
global_max_number_of_features = 0
global_max_number_of_numeric_features = 100
global_max_number_of_symbolic_features = 222

def updateGlobalMaxValues(ranges):
    global global_max_number_of_instances
    global global_max_number_of_features
    global global_max_number_of_numeric_features
    global global_max_number_of_symbolic_features

    global_max_number_of_instances = ranges['NumberOfInstances'][1]
    global_max_number_of_features = ranges['NumberOfFeatures'][1]
    global_max_number_of_numeric_features = ranges['NumberOfNumericFeatures'][1]
    global_max_number_of_symbolic_features = ranges['NumberOfSymbolicFeatures'][1]

# Laden der Daten und Setzen der globalen Variablen
datasets = helper.fetchDataList()

if datasets is not None and not datasets.empty:
    numericalRange = helper.calcRangeDatasets(datasets)
    updateGlobalMaxValues(numericalRange)

# Umwandlung der Maximalwerte in Ganzzahlen
max_features = int(global_max_number_of_features)
max_numeric_features = int(global_max_number_of_numeric_features)
max_categorical_features = int(global_max_number_of_symbolic_features)
max_instances = int(global_max_number_of_instances)

# Function to create a placeholder graph
def create_placeholder_figure():
    fig = go.Figure(data=[
        go.Scatter(x=[1, 2, 3], y=[4, 1, 2], mode='markers', marker=dict(color='LightSkyBlue'), name='Placeholder Data'),
        go.Scatter(x=[1, 2, 3], y=[2, 3, 1], mode='markers', marker=dict(color='Violet'), name='Placeholder Data 2')
    ])
    fig.update_layout(title='Placeholder Figure', xaxis_title='X Axis', yaxis_title='Y Axis')
    return fig

# Function to create a statistics graph
def create_statistics_figure():
    fig = go.Figure(data=[
        go.Bar(x=['Dataset 1', 'Dataset 2', 'Dataset 3'], y=[50, 30, 70], name='Anzahl der Features')
    ])
    fig.update_layout(title='Statistik aller Datensätze', xaxis_title='Datensätze', yaxis_title='Anzahl der Features')
    return fig

@app.callback(
    Output('test_output', 'children'),  # Erstellen Sie ein entsprechendes Ausgabe-Element im Layout
    [Input('search_button', 'n_clicks')]
)
def test_button_click(n_clicks):
    if n_clicks:
        return "Button wurde geklickt!"
    return "Button wurde noch nicht geklickt."


@app.callback(
    [
        Output('list_group', 'children'),
        Output('statistics_figure', 'figure'),
        Output('statistics_figure', 'style')
    ],
    [Input('search_button', 'n_clicks')],
    [
        State('date_range', 'start_date'),
        State('date_range', 'end_date'),
        State('range_data_points', 'value'),
        State('range_features', 'value'),
        State('range_numerical_features', 'value'),
        State('range_categorical_features', 'value')
    ]
)
def on_search_button_click(n_clicks, start_date, end_date, data_points_range, features_range, numerical_features_range, categorical_features_range):
    # Initialize variables at the start of the function
    list_group_items = []
    statistics_figure = go.Figure()  # Default empty figure
    statistics_style = {'display': 'none'}  # Default style

    # Check if the button was clicked
    if n_clicks is None:
        return list_group_items, statistics_figure, statistics_style

    # start_date=None, end_date=None, num_attributes_range=None, num_features_range=None, limit=None, datasets_list
    # Datenabrufen und Verarbeitung
    filtered_data = processData(start_date, end_date, features_range, numerical_features_range, categorical_features_range, data_points_range)

    # Anzeigen der Datensätze in Liste und Graph
    for idx, dataset in enumerate(datasets, start=1):
        dataset_name = dataset[1]
        rows, columns = int(dataset[2]), int(dataset[3])
        data_dimensions = f"{rows}x{columns}"
        total_datasets = len(datasets)

        list_group_item = dbc.ListGroupItem(
            [
                html.Div(
                    [
                        html.H5(f"{idx}. {dataset_name}", className="mb-1"),  # Added numbering here
                        html.Small(f"Downloads: 1000", className="text-secondary"),
                        html.Small(f"Dimension: {data_dimensions}", className="text-secondary"),
                    ],
                    className="d-flex flex-column",
                    id={"type": "toggle", "index": idx},
                    style={
                        "cursor": "pointer",
                        "padding": "10px",
                        "margin-bottom": "5px",
                        "background-color": "#f8f9fa",
                        "border": "1px solid #ddd",
                        "border-radius": "5px",
                        "box-shadow": "0 2px 2px rgba(0,0,0,0.1)"
                    },
                )
            ]
        )
        collapse = dbc.Collapse(
            dbc.Card(dbc.CardBody([dcc.Graph(figure=create_placeholder_figure())])),
            id={"type": "collapse", "index": idx}
        )
        list_group_items.append(list_group_item)
        list_group_items.append(collapse)

    # Update statistics_figure only if datasets are available
    if datasets:
        statistics_figure = create_statistics_figure()

    statistics_style = {'display': 'block'}

    return list_group_items, statistics_figure, statistics_style


# Formatiere Datum
def parse_date(date_str):
    """
    Converts a date string to a datetime object.

    :param date_str: The date string to convert.
    :return: A datetime object or None if date_str is None.
    """
    if date_str:
        try:
            # Try parsing with fractional seconds
            return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%f')
        except ValueError:
            try:
                # Try parsing without fractional seconds
                return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
            except ValueError as e:
                print(f"Error parsing date '{date_str}': {e}")
                return None
    else:
        return None

# Function to get dataset information and file
def get_dataset_info_and_file(dataset_id, save_directory='.', dataset_List=None):
    """
        Gets dataset information and file.
    """
    try:
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True, download_qualities=True, download_features_meta_data=True)

        X, y, _, _ = dataset.get_data(dataset_format="dataframe")

        info = {
            'Name': dataset.name,
            'Version': dataset.version,
            'Format': dataset.format,
            'Upload Date': dataset.upload_date,
            'Licence': dataset.licence,
            'Download URL': dataset.url,
            'OpenML URL': f"https://www.openml.org/d/{dataset_id}",
            'Number of Features': len(dataset.features),
            'Number of Instances': dataset.qualities['NumberOfInstances']
        }
        return info, dataset_file_path
    except Exception as e:
        logging.exception(f"Error processing dataset {dataset_id}: {e}")
        return None, None

# Function to filter datasets by attribute types
def processData(start_date=None, end_date=None, features_range=None, numerical_features_range=None, categorical_features_range=None, data_points_range=None):
    dataset_ids = datasets['did'].tolist()
    filtered_datasets = []

    # Umwandlung der Datumsstrings in datetime Objekte
    start_date = parse_date(start_date) if start_date else None
    end_date = parse_date(end_date) if end_date else None

    if start_date and end_date and start_date > end_date:
        raise ValueError("Start date must be before end date.")

    for dataset_id in dataset_ids:
        if limit is not None and limit <= 0:
            break

        dataset_info, _ = get_dataset_info_and_file(dataset_id)
        if dataset_info is None:
            continue

        dataset_date = parse_date(dataset_info['Upload Date'])
        num_features = dataset_info['Number of Features']
        num_attributes = None  # Hier sollten Sie die Anzahl der Attribute bestimmen

        # Filterlogik für Datumsbereich, Anzahl der Features und Anzahl der Attribute
        if (not start_date or start_date <= dataset_date) and \
           (not end_date or end_date >= dataset_date) and \
           (num_features_range is None or num_features >= num_features_range) and \
           (num_attributes_range is None or (num_attributes and num_attributes >= num_attributes_range)):
            filtered_datasets.append({
                'id': dataset_id,
                'name': dataset_info['Name'],
                'instances': dataset_info['Number of Instances'],
                'features': num_features
            })

            if limit is not None:
                limit -= 1

    return filtered_datasets

# Callbacks for toggling intervals and collapsing items
@app.callback(
    Output('interval-component', 'disabled'),
    [Input('search_button', 'n_clicks')],
    [State('interval-component', 'disabled')]
)
def toggle_interval(n_clicks, disabled):
    if n_clicks:
        return False
    return True

@app.callback(
    Output({"type": "collapse", "index": dash.ALL}, "is_open"),
    [Input({"type": "toggle", "index": dash.ALL}, "n_clicks")],
    [State({"type": "collapse", "index": dash.ALL}, "is_open")],
)
def toggle_collapse(n_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open

    button_id = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])
    idx = button_id["index"] - 1

    new_is_open = is_open[:]
    new_is_open[idx] = not is_open[idx]
    return new_is_open

# Erstellen des Dash-Layouts
app.layout = dbc.Container([
    dbc.Card([
        dbc.CardHeader("Filter"),
        dbc.CardBody([
            # Reihe für Datum und Anzahl der Datenpunkte
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Datum"),
                        dbc.CardBody([
                            dcc.DatePickerRange(
                                id='date_range',
                                start_date=datetime.now() - timedelta(days=10000),
                                end_date=datetime.now(),
                                min_date_allowed=datetime(2000, 1, 1),
                                max_date_allowed=datetime.now(),
                                display_format='DD.MM.YYYY',
                                initial_visible_month=datetime.now()
                            ),
                        ]),
                    ]),
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Anzahl Datenpunkte"),
                        dbc.CardBody([
                            dcc.RangeSlider(
                                id='range_data_points',
                                min=0,
                                max=max_instances,  # Beispielwert
                                step=1,
                                value=[0, max_instances],
                                marks={i: str(i) for i in range(0, max_instances + 1, max(1, max_instances // 10))}
                            ),
                        ]),
                    ]),
                ], md=6),
            ]),
            # Reihe für Anzahl der Features und numerische Features
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Anzahl der Features"),
                        dbc.CardBody([
                            dcc.RangeSlider(
                                id='range_features',
                                min=0,
                                max=max_features,
                                step=1,
                                value=[0, max_features],
                                marks={i: str(i) for i in range(0, max_features + 1, max(1, max_features // 10))}
                            ),
                        ]),
                    ]),
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Anzahl der Numerischen Features"),
                        dbc.CardBody([
                            dcc.RangeSlider(
                                id='range_numerical_features',
                                min=0,
                                max=max_numeric_features,
                                step=1,
                                value=[0, max_numeric_features],
                                marks={i: str(i) for i in range(0, max_numeric_features + 1, max(1, max_numeric_features // 10))}
                            ),
                        ]),
                    ]),
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Anzahl der Kategorialen Features"),
                        dbc.CardBody([
                            dcc.RangeSlider(
                                id='range_categorical_features',
                                min=0,
                                max=max_categorical_features,
                                step=1,
                                value=[0, max_categorical_features],
                                marks={i: str(i) for i in range(0, max_categorical_features + 1, max(1, max_categorical_features // 10))}
                            ),
                        ]),
                    ]),
                ], md=4),
            ]),
            # Suchbutton
            dbc.Row([
                dbc.Col([
                    dbc.Button('Suchen', id='search_button', color="primary", className="mt-3 mb-3", style={'width': '100%'})
                ], md=12),
            ]),
            # Fortschrittsbalken
            dbc.Progress(id='progress_bar', value=0, style={"height": "20px", "margin-top": "15px"}, striped=True),
        ])
    ]),
    # Graph und Listengruppe
    dcc.Graph(id='statistics_figure', style={'display': 'none'}),
    dbc.ListGroup(id='list_group', flush=True, className="mt-4"),
    dcc.Interval(id='interval-component', interval=100, n_intervals=0, disabled=True),
], fluid=True)

# Hauptausführungsbereich
if __name__ == '__main__':
    app.run_server(debug=False)

