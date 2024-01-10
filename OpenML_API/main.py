import os
import time
import warnings
from contextlib import contextmanager
import dash
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
import threading

# Global variable for download folder
DOWNLOAD_FOLDER = '/Users/merluee/Documents/VSC/Data_Science_Projekt/OpenML_API/DownloadedFiles'

#data_queue = queue.Queue()

# Dash app setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

#cache = Cache(app.server, config={
#    'CACHE_TYPE': 'simple',
#    'CACHE_DEFAULT_TIMEOUT': 86400
#})

def getDataOpenML():
    while True:
        datasets_list = openml.datasets.list_datasets(output_format='dataframe')
        data_queue.put(datasets_list)
        time.sleep(86400)

# Starten des Background-Threads beim Start der App
data_thread = threading.Thread(target=background_data_processing, daemon=True)
data_thread.start()

# Function to check if dataset can be downloaded
def is_dataset_downloadable(download_url):
    try:
        response = requests.head(download_url, allow_redirects=True, timeout=60)
        if response.status_code != 200:
            logging.warning(f"Dataset URL {download_url} returned status code {response.status_code}.")
        return response.status_code == 200
    except requests.RequestException as e:
        logging.error(f"Failed to check dataset URL {download_url}: {e}")
        return False

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

# Callback for updating dataset list and statistics
@app.callback(
    [
        Output('list_group', 'children'),
        Output('statistics_figure', 'figure'),
        Output('statistics_figure', 'style'),
        Output('progress', 'value')
    ],
    [Input('search_button', 'n_clicks')],
    [
        State('date_range', 'start_date'),
        State('date_range', 'end_date'),
        State('numberFeatures', 'value'),
        State('numberNumericalFeatures', 'value'),
        State('numberCategoricalFeatures', 'value'),
        State('limit_input', 'value')
    ]
)
def update_dataset_list_and_statistics(n_clicks, start_date, end_date, number_features,
                                       number_numerical_features, number_categorical_features, limit_input):
    # Initialize variables at the start of the function
    list_group_items = []
    statistics_figure = go.Figure()  # Default empty figure
    statistics_style = {'display': 'none'}  # Default style

    progress_value = 0  # Default progress value
    progress_label = ""  # Initialize progress label

    if n_clicks is None:
        return dash.no_update

    datasets = filter_datasets_by_attribute_types(start_date, end_date, number_features, number_numerical_features, number_categorical_features, limit_input)

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

    return list_group_items, statistics_figure, {'display': 'block'}, progress_data['current']

# Static method to list datasets
@staticmethod
def list_datasets(output_format='dataframe'):
    return openml.datasets.list_datasets(output_format=output_format)

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
#@cache.memoize(timeout=60)
def get_dataset_info_and_file(dataset_id, output_format='dataframe', save_directory='.', current_progress=0, total_datasets=1):
    """
    Gets dataset information and file.

    Parameters:
    dataset_id (int): The ID of the dataset.
    output_format (str): The format in which to return the dataset. Options are 'csv' or 'dataframe'.
    save_directory (str): The directory to save the dataset file if output_format is 'csv'.

    Returns:
    tuple: A tuple containing dataset information and either the path to the dataset file (if output_format is 'csv')
           or the DataFrame itself (if output_format is 'dataframe').
    """

    try:
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True, download_qualities=True,
                                              download_features_meta_data=True)
        download_url = dataset.url
        if not is_dataset_downloadable(download_url):
            logging.warning(f"Dataset {dataset_id} is not downloadable.")
            return None, None
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
        return info
        #if output_format == 'csv':
            #dataset_file_extension = 'csv'
            #dataset_file_path = os.path.join(save_directory, f"{dataset.name}.{dataset_file_extension}")
            #X.to_csv(dataset_file_path, index=False, encoding='utf-8')
            #return info, dataset_file_path
        #elif output_format == 'dataframe':
            #return info, X
    except Exception as e:
        logging.exception(f"Error processing dataset {dataset_id}: {e}")
        return None, None


# Function to filter datasets by attribute types
def filter_datasets_by_attribute_types(start_date=None, end_date=None,
                                       num_features_range=None, num_numerical_features=None,
                                       num_categorical_features=None, limit=None):

    dataset_ids = datasets_list['did'].tolist()

    # Total datasets for progress calculation
    total_datasets = limit if limit is not None else max(limit, len(dataset_ids))

    # List to hold filtered datasets
    filtered_datasets = []

    # Set initial progress
    progress_data = {'current_progress': 0, 'total': total_datasets}

    for idx, dataset_id in enumerate(dataset_ids):
        if limit is not None and len(filtered_datasets) >= limit:
            break

        dataset_info, _, _ = get_dataset_info_and_file(dataset_id)
        if dataset_info is None:
            continue

        dataset_date = parse_date(dataset_info['Upload Date'])
        if ((not start_date or start_date <= dataset_date) and
            (not end_date or end_date >= dataset_date) and
            (num_features_range is None or dataset_info['Number of Features'] >= num_features_range)):
            filtered_datasets.append((dataset_id, dataset_info['Name'], dataset_info['Number of Instances'],
                                      dataset_info['Number of Features']))

        # Update progress
        progress_data['current_progress'] = idx + 1

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



@app.callback(
    [Output("progress", "value"), Output("progress", "label")],
    [Input("progress-interval", "n_intervals")],
)
def update_progress(n):
    # check progress of some background process, in this example we'll just
    # use n_intervals constrained to be in 0-100
    progress = min(n % 110, 100)
    # only add text after 5% progress to ensure text isn't squashed too much
    return progress, f"{progress} %" if progress >= 5 else ""

# App-Layout
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
                                start_date=datetime.now() - timedelta(10000),
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
                            dbc.Input(id='countDataPoints', type='number', value=5)
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
                            dbc.Input(id='numberFeatures', type='number', value=5)
                        ]),
                    ]),
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Anzahl der Numerischen Features"),
                        dbc.CardBody([
                            dbc.Input(id='numberNumericalFeatures', type='number', value=5)
                        ]),
                    ]),
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Anzahl der Kategorialen Features"),
                        dbc.CardBody([
                            dbc.Input(id='numberCategoricalFeatures', type='number', value=5)
                        ]),
                    ]),
                ], md=4),
            ]),
            # Reihe für Limit
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Anzahl Datensätze"),
                        dbc.CardBody([
                            dbc.Input(id='limit_input', type='number', value=10)
                        ]),
                    ]),
                ], md=12),  # Angepasst auf md=12, um den gesamten Platz zu nutzen
            ]),
            # Suchbutton
            dbc.Row([
                dbc.Col([
                    dbc.Button('Suchen', id='search_button', color="primary", className="mt-3 mb-3", style={'width': '100%'})
                ], md=12),
            ]),
            dcc.Interval(id="progress-interval", n_intervals=0, interval=500),
            dbc.Progress(id="progress", value=0, striped=True, animated=True),
        ])
    ]),
    # Graph und Listengruppe
    dcc.Graph(id='statistics_figure', style={'display': 'none'}),
    dbc.ListGroup(id='list_group', flush=True, className="mt-4"),
    dcc.Interval(id='interval-component', interval=100, n_intervals=0, disabled=True),
], fluid=True)

# Running the app
if __name__ == '__main__':
    app.run_server(debug=False)