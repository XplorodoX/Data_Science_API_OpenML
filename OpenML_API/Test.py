from dash import html, dcc, Input, Output, State, callback, MATCH, dash
import threading
import time
import pandas as pd
import openml
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import json
import math
from datetime import datetime, timedelta, date
from dash.exceptions import PreventUpdate
import Helper as helper
from queue import Queue

datasets_info = [
    {"name": "Dataset 1", "num_instances": 100, "dimensions": "10x10"},
    {"name": "Dataset 2", "num_instances": 300, "dimensions": "20x10"},
    {"name": "Dataset 3", "num_instances": 120, "dimensions": "15x10"},
    {"name": "Dataset 4", "num_instances": 450, "dimensions": "30x15"},
    {"name": "Dataset 5", "num_instances": 200, "dimensions": "25x10"},
    {"name": "Dataset 6", "num_instances": 150, "dimensions": "12x12"},
    {"name": "Dataset 7", "num_instances": 180, "dimensions": "18x18"},
    {"name": "Dataset 8", "num_instances": 220, "dimensions": "22x11"},
    {"name": "Dataset 9", "num_instances": 250, "dimensions": "25x15"},
    {"name": "Dataset 10", "num_instances": 275, "dimensions": "27x17"},
    {"name": "Dataset 11", "num_instances": 125, "dimensions": "20x12"},
    {"name": "Dataset 12", "num_instances": 300, "dimensions": "30x20"},
    {"name": "Dataset 13", "num_instances": 320, "dimensions": "32x16"},
    {"name": "Dataset 14", "num_instances": 340, "dimensions": "34x17"},
    {"name": "Dataset 15", "num_instances": 360, "dimensions": "36x18"},
    {"name": "Dataset 16", "num_instances": 380, "dimensions": "38x19"},
    {"name": "Dataset 17", "num_instances": 400, "dimensions": "40x20"},
    {"name": "Dataset 18", "num_instances": 420, "dimensions": "42x21"},
    {"name": "Dataset 19", "num_instances": 440, "dimensions": "44x22"},
    {"name": "Dataset 20", "num_instances": 460, "dimensions": "46x23"},
    {"name": "Dataset 21", "num_instances": 480, "dimensions": "48x24"},
    {"name": "Dataset 22", "num_instances": 500, "dimensions": "50x25"},
    {"name": "Dataset 23", "num_instances": 520, "dimensions": "52x26"},
    {"name": "Dataset 24", "num_instances": 540, "dimensions": "54x27"},
    {"name": "Dataset 25", "num_instances": 560, "dimensions": "56x28"},
    {"name": "Dataset 26", "num_instances": 580, "dimensions": "58x29"},
    {"name": "Dataset 27", "num_instances": 600, "dimensions": "60x30"},
    {"name": "Dataset 28", "num_instances": 620, "dimensions": "62x31"},
    {"name": "Dataset 29", "num_instances": 640, "dimensions": "64x32"},
    {"name": "Dataset 30", "num_instances": 660, "dimensions": "66x33"},
]

# Setzen des Cache-Verzeichnisses
openml.config.set_root_cache_directory('cache')

# Erstellen eines leeren DataFrame
df = pd.DataFrame()

# Set global Variables
global_max_number_of_instances = 0
global_max_number_of_features = 0
global_max_number_of_numeric_features = 0
global_max_number_of_symbolic_features = 0

# Function to update the global max values
def updateGlobalMaxValues(ranges):
    global global_max_number_of_instances
    global global_max_number_of_features
    global global_max_number_of_numeric_features
    global global_max_number_of_symbolic_features

    global_max_number_of_instances = ranges['NumberOfInstances'][1]
    global_max_number_of_features = ranges['NumberOfFeatures'][1]
    global_max_number_of_numeric_features = ranges['NumberOfNumericFeatures'][1]
    global_max_number_of_symbolic_features = ranges['NumberOfSymbolicFeatures'][1]

# Umwandlung der Maximalwerte in Ganzzahlen
max_features = int(global_max_number_of_features)
max_numeric_features = int(global_max_number_of_numeric_features)
max_categorical_features = int(global_max_number_of_symbolic_features)
max_instances = int(global_max_number_of_instances)
maxDataset = 12

# Laden der Daten und Setzen der globalen Variablen
datasets = helper.fetchDataList()

if datasets is not None and not datasets.empty:
    numericalRange = helper.calcRangeDatasets(datasets)
    updateGlobalMaxValues(numericalRange)

progress_queue = Queue()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Anzahl der Elemente pro Seite
ITEMS_PER_PAGE = 10

# Funktion für die Collaps Datensätze
def create_placeholder_figure():
    fig = go.Figure(data=[
        go.Scatter(x=[1, 2, 3], y=[4, 1, 2], mode='markers', marker=dict(color='LightSkyBlue'), name='Placeholder Data'),
        go.Scatter(x=[1, 2, 3], y=[2, 3, 1], mode='markers', marker=dict(color='Violet'), name='Placeholder Data 2')
    ])
    fig.update_layout(title='Placeholder Figure', xaxis_title='X Axis', yaxis_title='Y Axis')
    return fig

# Funktion für die Allgemine Statistik
def create_statistics_figure():
    fig = go.Figure(data=[
        go.Bar(x=['Dataset 1', 'Dataset 2', 'Dataset 3'], y=[50, 30, 70], name='Anzahl der Features')
    ])
    fig.update_layout(title='Statistik aller Datensätze', xaxis_title='Datensätze', yaxis_title='Anzahl der Features')
    return fig

def long_task(progress_queue):
    for i in range(10):
        time.sleep(2)  # Simuliere Arbeit
        if not progress_queue.empty() and progress_queue.get() == 'cancel':
            progress_queue.put(0)  # Setze Fortschritt auf 0% bei Abbruch
            return
        progress_queue.put((i + 1) * 10)
    progress_queue.put(100)  # Signalisiert das Ende der Aufgabe

@app.callback(
    Output({"type": "collapse", "index": MATCH}, "is_open"),
    [Input({"type": "toggle", "index": MATCH}, "n_clicks")],
    [State({"type": "collapse", "index": MATCH}, "is_open")],
    prevent_initial_call=True
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# Sichtbarkeit der Navigationselemente
@app.callback(
    [Output('previous-page', 'style'),
     Output('next-page', 'style'),
     Output('current-page', 'style')],
    [Input('show-data', 'n_clicks')]
)
def toggle_navigation_elements_visibility(n_clicks_show_data):
    if n_clicks_show_data > 0:
        # Mache die Buttons und die Schrift sichtbar
        return {'display': 'inline-block'}, {'display': 'inline-block'}, {'display': 'block'}
    else:
        # Halte die Buttons und die Schrift versteckt
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

# Aktualisierung der Liste
@app.callback(
    Output('list-container', 'children'),
    [Input('show-data', 'n_clicks'), Input('previous-page', 'n_clicks'), Input('next-page', 'n_clicks')],
    [State('current-page', 'children'), State('task-status', 'data')], prevent_initial_call=True
)
def update_list(show_data_clicks, prev_clicks, next_clicks, current_page_text, data):
    if show_data_clicks == 0:
        return html.Div()

    # Extrahiere die aktuelle Seitenzahl aus dem Text "Seite x von y"
    current_page = int(current_page_text.split()[1])

    changed_id = dash.callback_context.triggered[0]['prop_id']
    if 'next-page' in changed_id:
        current_page = min(current_page + 1, math.ceil(len(datasets_info) / ITEMS_PER_PAGE))
    elif 'previous-page' in changed_id:
        current_page = max(current_page - 1, 1)

    for i in range(1, 31):
        print(i)

    start = (current_page - 1) * ITEMS_PER_PAGE
    filtered_info = datasets_info[start:start + ITEMS_PER_PAGE]

    list_group_items = []
    for idx, dataset in enumerate(filtered_info, start=start):
        global_index = idx + 1  # Stelle sicher, dass global_index korrekt verwendet wird, falls notwendig
        list_group_item = dbc.ListGroupItem(
            html.Div([
                html.H5(f"{global_index}. {dataset['name']}", className="mb-1"),
                html.Small(f"Zeilenanzahl: {dataset['num_instances']}", className="text-secondary"),
                html.Small(f"Dimension: {dataset['dimensions']}", className="text-secondary"),
            ], id={"type": "toggle", "index": global_index}),
            action=True,
            style={"cursor": "pointer", "padding": "10px", "margin-bottom": "5px",
                   "background-color": "#f8f9fa", "border": "1px solid #ddd",
                   "border-radius": "5px", "box-shadow": "0 2px 2px rgba(0,0,0,0.1)"}
        )
        collapse = dbc.Collapse(
            dbc.Card(dbc.CardBody([dcc.Graph(figure=create_placeholder_figure())])),
            id={"type": "collapse", "index": global_index}
        )

        list_group_items.extend([list_group_item, collapse])

    return dbc.ListGroup(list_group_items)

@callback(
    Output('current-page', 'children'),
    [Input('show-data', 'n_clicks'), Input('previous-page', 'n_clicks'), Input('next-page', 'n_clicks')],
    [State('current-page', 'children')]
)
def update_page_number(show_data_clicks, prev_clicks, next_clicks, current_page):
    # Logik identisch zum vorigen Callback, aktualisiert nur die Seitennummer
    if show_data_clicks == 0:
        return "Seite 1 von {}".format(math.ceil(len(datasets_info) / ITEMS_PER_PAGE))

    current_page = int(current_page.split()[1])  # Aktuelle Seite aus dem String extrahieren
    total_pages = math.ceil(len(datasets_info) / ITEMS_PER_PAGE)

    changed_id = dash.callback_context.triggered[0]['prop_id']
    if 'next-page' in changed_id:
        current_page = min(current_page + 1, total_pages)
    elif 'previous-page' in changed_id:
        current_page = max(current_page - 1, 1)

    return "Seite {} von {}".format(current_page, total_pages)


app.layout = html.Div([
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Datum"),
                    dbc.CardBody([
                        dcc.DatePickerRange(
                            id='date_range',
                            start_date=datetime.now() - timedelta(days=3600),
                            end_date=datetime.now(),
                            min_date_allowed=datetime(2000, 1, 1),
                            max_date_allowed=datetime.now(),
                            display_format='DD.MM.YYYY',
                            initial_visible_month=datetime.now()
                        ),
                    ]),
                ]),
            ], md=5),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Anzahl Datenpunkte"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(
                                dbc.InputGroup([
                                    dbc.InputGroupText("Min"),
                                    dbc.Input(id='min_data_points', type='number', value=0, min=0, max=max_instances),
                                ]),
                                width=6,
                            ),
                            dbc.Col(
                                dbc.InputGroup([
                                    dbc.InputGroupText("Max"),
                                    dbc.Input(id='max_data_points', type='number', value=max_instances, min=0, max=max_instances),
                                ]),
                                width=6,
                            ),
                        ]),
                        html.Div(
                            id='output_data_points',
                            style={
                                'margin-top': '10px',
                                'text-align': 'center',
                                'font-weight': 'bold',
                                'color': '#007bff'
                            }
                        )
                    ]),
                ]),
            ], md=5),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Maximale Datensatzanzahl"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(
                                dbc.Input(
                                    id='input_max_datasets',
                                    type='number',
                                    min=0,
                                    max=100000,
                                    step=1,
                                    value=20
                                ),
                                width=10,
                            ),
                        ]),
                        dbc.Tooltip(
                            "Geben Sie die maximale Anzahl der Datensätze ein, die berücksichtigt werden sollen.",
                            target="input_max_datasets",
                        ),
                    ]),
                ]),
            ], md=2),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Anzahl der Features"),
                    dbc.CardBody([
                        dbc.Row([
                                dbc.Col(
                                    dbc.InputGroup([
                                        dbc.InputGroupText("Min"),
                                        dbc.Input(id='min_features', type='number', value=0, min=0, max=max_features),
                                    ]),
                                    width=6,
                                ),
                                dbc.Col(
                                    dbc.InputGroup([
                                        dbc.InputGroupText("Max"),
                                        dbc.Input(id='max_features', type='number', value=max_features, min=0, max=max_features),
                                    ]),
                                    width=6,
                                ),
                            ]),
                            html.Div(
                                id='output_features',
                                style={
                                    'margin-top': '10px',
                                    'text-align': 'center',
                                    'font-weight': 'bold',
                                    'color': '#007bff'
                                }
                            )
                        ]),
                    ]),
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Anzahl der Numerischen Features"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col(
                                    dbc.InputGroup([
                                        dbc.InputGroupText("Min"),
                                        dbc.Input(id='min_numerical_features', type='number', value=0, min=0, max=max_numeric_features),
                                    ]),
                                    width=6,
                                ),
                                dbc.Col(
                                    dbc.InputGroup([
                                        dbc.InputGroupText("Max"),
                                        dbc.Input(id='max_numerical_features', type='number', value=max_numeric_features, min=0, max=max_numeric_features),
                                    ]),
                                    width=6,
                                ),
                            ]),
                            html.Div(
                                id='output_numerical_features',
                                style={
                                    'margin-top': '10px',
                                    'text-align': 'center',
                                    'font-weight': 'bold',
                                    'color': '#007bff'
                                }
                            )
                        ]),
                    ]),
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Anzahl der Kategorialen Features"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col(
                                    dbc.InputGroup([
                                        dbc.InputGroupText("Min"),
                                        dbc.Input(id='min_categorical_features', type='number', value=0, min=0, max=max_categorical_features),
                                    ]),
                                    width=6,
                                ),
                                dbc.Col(
                                    dbc.InputGroup([
                                        dbc.InputGroupText("Max"),
                                        dbc.Input(id='max_categorical_features', type='number', value=max_categorical_features, min=0, max=max_categorical_features),
                                    ]),
                                    width=6,
                                ),
                            ]),
                            html.Div(
                                id='output_categorical_features',
                                style={
                                    'margin-top': '10px',
                                    'text-align': 'center',
                                    'font-weight': 'bold',
                                    'color': '#007bff'
                                }
                            )
                        ]),
                    ]),
                ], md=4),
            ], className="mb-4"),
    dbc.Button('Daten anzeigen', id='show-data', n_clicks=0),
    dbc.Button("Abbrechen", id="cancel-button", color="danger", className="me-2", n_clicks=0),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
    dbc.Progress(id="progress-bar", value=0, striped=True, animated=True, label="0%", style={"width": "100%"}),
    html.H5(id="remaining-time", children=""),
    dcc.Store(id='task-status', data={'running': False, 'cancelled': False}),

    html.Div(id='list-container', className="list-container mt-4"),
    html.Div([
        dbc.Button('<-', id='previous-page', n_clicks=0, className="mr-2", style={'display': 'none'}),
        html.Span(id='current-page', children="Seite 1 von 5", className="mx-2", style={'display': 'none'}),
        dbc.Button('->', id='next-page', n_clicks=0, className="ml-2", style={'display': 'none'}),
    ], className="d-flex justify-content-center align-items-center mt-4"),
])

if __name__ == '__main__':
    app.run_server(debug=True)
