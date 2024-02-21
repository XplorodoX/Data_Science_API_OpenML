import math
import openml
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objs as go
from datetime import datetime, timedelta, date
import Helper as helper
import hashlib
from dash.exceptions import PreventUpdate
import json
import webbrowser
from tempfile import NamedTemporaryFile
import dash

# Setzen des Cache-Verzeichnisses
openml.config.set_root_cache_directory('cache')

# Dash app setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Set global Variables
ITEMS_PER_PAGE = 10
stop = False
filtered_data = []
global_max_number_of_instances = 0
global_max_number_of_features = 0
global_max_number_of_numeric_features = 0
global_max_number_of_symbolic_features = 0

# Laden der Daten und Setzen der globalen Variablen
datasets = helper.fetchDataList()

# Function to hash a dataset
def hash_dataset(dataset):
    dataset_string = str(dataset)
    return hashlib.sha256(dataset_string.encode()).hexdigest()

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

if datasets is not None and not datasets.empty:
    numericalRange = helper.calcRangeDatasets(datasets)
    updateGlobalMaxValues(numericalRange)

# Umwandlung der Maximalwerte in Ganzzahlen
max_features = int(global_max_number_of_features)
max_numeric_features = int(global_max_number_of_numeric_features)
max_categorical_features = int(global_max_number_of_symbolic_features)
max_instances = int(global_max_number_of_instances)
maxDataset = 12

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

@app.callback(
    [
        Output('list_group', 'children'),
        Output('statistics_figure', 'figure'),
        Output('statistics_figure', 'style')
    ],
    [Input('search_button', 'n_clicks'), Input('previous-page', 'n_clicks'), Input('next-page', 'n_clicks')],
    [
        State('date_range', 'start_date'),
        State('date_range', 'end_date'),
        State('min_data_points', 'value'),
        State('max_data_points', 'value'),
        State('min_features', 'value'),
        State('max_features', 'value'),
        State('min_numerical_features', 'value'),
        State('max_numerical_features', 'value'),
        State('min_categorical_features', 'value'),
        State('max_categorical_features', 'value'),
        State('input_max_datasets', 'value'),
        State('current-page', 'children')
    ]
)
def on_search_button_click(n_clicks, prev_clicks, next_clicks, start_date, end_date, min_data_points, max_data_points, min_features, max_features, min_numerical_features, max_numerical_features, min_categorical_features, max_categorical_features, limit, current_page_text):
    global filtered_data

    # Initialize variables at the start of the function
    list_group_items = []
    statistics_figure = go.Figure()  # Default empty figure
    statistics_style = {'display': 'none'}  # Default style

    # Check if the button was clicked
    if n_clicks is None:
        return list_group_items, statistics_figure, statistics_style

    # Create ranges from min and max values
    features_range = (min_features, max_features)
    numerical_features_range = (min_numerical_features, max_numerical_features)
    categorical_features_range = (min_categorical_features, max_categorical_features)
    data_points_range = (min_data_points, max_data_points)

    # Datenabrufen und Verarbeitung
    filtered_data = processData(start_date, end_date, features_range, numerical_features_range, categorical_features_range, data_points_range, limit)

    # Anzeigen der Datensätze in Liste
    # Extrahiere die aktuelle Seitenzahl aus dem Text "Seite x von y"
    current_page = int(current_page_text.split()[1])

    changed_id = dash.callback_context.triggered[0]['prop_id']
    if 'next-page' in changed_id:
        current_page = min(current_page + 1, math.ceil(len(filtered_data) / ITEMS_PER_PAGE))
    elif 'previous-page' in changed_id:
        current_page = max(current_page - 1, 1)

    start = (current_page - 1) * ITEMS_PER_PAGE
    filtered_info = filtered_data[start:start + ITEMS_PER_PAGE]

    list_group_items = []
    for idx, dataset in enumerate(filtered_info, start=start):
        global_index = idx + 1
        item_id = {"type": "dataset-click", "index": dataset['id']}    # Eindeutige ID für jedes Element

        list_group_item = dbc.ListGroupItem(
            html.Div([
                html.Div([
                    html.H5(f"{global_index}. {dataset['name']}", className="mb-1"),
                    html.Div([
                        html.Small(f"Dataset-ID: {dataset['id']}", className="text-secondary d-block"),
                        html.Small(f"Dimension: {int(dataset['features'])}×{int(dataset['instances'])}",
                                   className="text-secondary d-block"),
                        html.Small(
                            f"Ordinale Features: {int(dataset.get('categorical_features', 0))}, Numerische Features: {int(dataset.get('numeric_features', 0))}",
                            className="text-secondary d-block"),
                        html.Small(f"Upload-Datum: {dataset['upload'][:10]}", className="text-secondary d-block")
                    ], className="mt-2")
                ], style={'flex': '1'}),
            ], id=item_id, n_clicks=0, style={'cursor': 'pointer', 'text-decoration': 'none', 'color': 'inherit'}),
            style={
                "padding": "20px",
                "margin-bottom": "10px",
                "background-color": "#f8f9fa",
                "border": "1px solid #ddd",
                "border-radius": "5px",
                "box-shadow": "0 2px 2px rgba(0,0,0,0.1)"
            }
        )
        list_group_items.append(list_group_item)

    return list_group_items, statistics_figure, statistics_style

@app.callback(
    Output('output-container', 'children'),
    [Input({'type': 'dataset-click', 'index': ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def on_item_click(n_clicks):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate

    # Überprüfen, ob alle n_clicks-Werte null sind
    if all(click is None or click == 0 for click in n_clicks):
        raise PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if 'dataset-click' not in triggered_id:
        raise PreventUpdate

    dataset_id = json.loads(triggered_id)['index']

    # Erstelle den entsprechenden Plot basierend auf der dataset_id
    #fig = create_plot(dataset_id)

    # HTML-Datei erstellen und Plot speichern
    #temp_file = NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8')
    #fig.write_html(temp_file.name)
    #temp_file_path = temp_file.name
    #temp_file.close()
    print(dataset_id)

    # Hier wird angenommen, dass der Plot bereits erstellt und gespeichert wurde.
    # Öffnet die Plotly-HTML-Datei in einem neuen Browser-Tab.
    #webbrowser.open_new_tab(f'file://{temp_file_path}')

    # Gibt eine Meldung zurück, dass der Datensatz geöffnet wurde
    return html.Div(f'Dataset {dataset_id} wurde geöffnet und Plot ist verfügbar.')



@app.callback(
    Output('memory-output', 'data'),  # Speichert den aktuellen Zustand in der dcc.Store Komponente
    Input('cancel_button', 'n_clicks'),  # Löst den Callback bei jedem Klick aus
    State('memory-output', 'data')  # Behält den vorherigen Zustand bei, ohne den Callback auszulösen
)
def StopProcess(clicks, current_state):
    global stop
    if clicks is None or current_state is None:
        # Wenn der Button noch nie geklickt wurde oder kein vorheriger Zustand vorhanden ist,
        # initialisieren Sie den Zustand als False (standardmäßig laufen)
        stop = False
        return False
    else:
        # Kehren Sie den aktuellen Zustand um, jedes Mal wenn der Button geklickt wird
        stop = True
        return not current_state

@app.callback(
    Output('current-page', 'children'),
    [Input('search_button', 'n_clicks'), Input('previous-page', 'n_clicks'), Input('next-page', 'n_clicks')],
    [State('current-page', 'children')]
)
def update_page_number(show_data_clicks, prev_clicks, next_clicks, current_page_text):
    if filtered_data:  # Stellen Sie sicher, dass Daten vorhanden sind
        current_page = int(current_page_text.split()[1])  # Aktuelle Seite aus dem Text extrahieren
        total_pages = math.ceil(len(filtered_data) / ITEMS_PER_PAGE)
        changed_id = dash.callback_context.triggered[0]['prop_id']
        if 'next-page' in changed_id:
            current_page = min(current_page + 1, total_pages)
        elif 'previous-page' in changed_id:
            current_page = max(current_page - 1, 1)
        return f"Seite {current_page} von {total_pages}"
    else:
        return "Seite 1 von 1"

#Datum Konvertierung
def parse_date(date_str):
    """
    Konvertiert einen Datumsstring in ein datetime-Objekt, wobei nur Jahr, Monat und Tag berücksichtigt werden.
    Wenn date_str bereits ein datetime.date oder datetime.datetime Objekt ist, wird es direkt zurückgegeben.

    :param date_str: Der zu konvertierende Datumsstring oder ein datetime.date/datetime.datetime Objekt.
    :return: Ein datetime.date Objekt oder None, wenn date_str None ist.
    """
    if date_str:
        if isinstance(date_str, datetime):
            return date_str.date()
        elif isinstance(date_str, date):
            return date_str
        else:
            try:
                # Extrahiert nur das Jahr, den Monat und den Tag
                parsed_date = datetime.strptime(date_str.split('T')[0], '%Y-%m-%d')
                return parsed_date.date()
            except ValueError as e:
                print(f"Fehler beim Parsen des Datums '{date_str}': {e}")
    return None

# Funktion zum Abrufen des Upload-Datums eines Datensatzes
def getUploadDate(dataset_id):
    """
        Zum abrufen des Upload-Datums eines Datensatzes.

        :param dataset_id: Datenatz-ID
        :return: Gibt das Uploaddatum des Datensets zurück.
        """
    try:
        dataset = openml.datasets.get_dataset(dataset_id, download_data=False, download_qualities=False, download_features_meta_data=False)
        return dataset.upload_date
    except Exception as e:
        print(f"Fehler beim Abrufen des Upload-Datums für Dataset {dataset_id}: {e}")
        return None

# Function to filter datasets by attribute types
def processData(start_date=None, end_date=None, features_range=None, numerical_features_range=None,
                categorical_features_range=None, data_points_range=None, limit=None):

    if limit is None:
        limit = float('inf')

    dataset_ids = datasets['did'].tolist()

    if start_date and end_date and start_date > end_date:
        raise ValueError("Start date must be before end date.")

    count = 0
    filtered_datasets = []
    for dataset_id in dataset_ids:

        global stop

        upload_date = getUploadDate(dataset_ids[dataset_id])

        start_date = parse_date(start_date)
        end_date = parse_date(end_date)

        if count >= limit or stop == True:
            print("Abbruch")
            break

        if upload_date:
            dataset_date = parse_date(upload_date)

            if start_date and dataset_date < start_date:
                continue
            if end_date and dataset_date > end_date:
                continue

            num_features = datasets.loc[datasets['did'] == dataset_id, 'NumberOfFeatures'].iloc[0]
            num_numeric_features = datasets.loc[datasets['did'] == dataset_id, 'NumberOfNumericFeatures'].iloc[0]
            num_categorical_features = datasets.loc[datasets['did'] == dataset_id, 'NumberOfSymbolicFeatures'].iloc[0]
            num_instances = datasets.loc[datasets['did'] == dataset_id, 'NumberOfInstances'].iloc[0]
            name = datasets.loc[datasets['did'] == dataset_id, 'name'].iloc[0]

            if ((not start_date or start_date <= dataset_date) and
                    (not end_date or end_date >= dataset_date) and
                    (not features_range or (features_range[0] <= num_features <= features_range[1])) and
                    (not numerical_features_range or (numerical_features_range[0] <= num_numeric_features <= numerical_features_range[1])) and
                    (not categorical_features_range or (categorical_features_range[0] <= num_categorical_features <= categorical_features_range[1])) and
                    (not data_points_range or (data_points_range[0] <= num_instances <= data_points_range[1]))):
                filtered_datasets.append({
                    'id': dataset_id,
                    'name': name,
                    'instances': num_instances,
                    'features': num_features,
                    'numeric_features': num_numeric_features,
                    'categorical_features': num_categorical_features,
                    'upload' : upload_date
                })
                count += 1

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

# Callback-Funktion
@app.callback(
    Output('output_numerical_features', 'children'),
    [Input('range_numerical_features', 'value')]
)
def update_output(value):
    return f"Ausgewählter Bereich: {value[0]} bis {value[1]}"

# Callback-Funktion für Anzahl der Features
@app.callback(
    Output('output_features', 'children'),
    [Input('range_features', 'value')]
)
def update_output_features(value):
    return f"Ausgewählter Bereich: {value[0]} bis {value[1]}"

# Callback-Funktion für Anzahl der Kategorialen Features
@app.callback(
    Output('output_categorical_features', 'children'),
    [Input('range_categorical_features', 'value')]
)
def update_output_categorical_features(value):
    return f"Ausgewählter Bereich: {value[0]} bis {value[1]}"

# Callback-Funktion für Anzahl Datenpunkte
@app.callback(
    Output('output_data_points', 'children'),
    [Input('range_data_points', 'value')]
)
def update_output_data_points(value):
    return f"Ausgewählter Bereich: {value[0]} bis {value[1]}"

app.layout = dbc.Container([
    dbc.Card([
        dbc.CardHeader("Filter"),
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
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        dbc.Col(dbc.Button('Suchen', id='search_button', color="primary", className="mt-3 mb-3", style={'width': '100%'}), width=6),
                        dbc.Col(dbc.Button('Abbrechen', id='cancel_button', color="danger", className="mt-3 mb-3", style={'width': '100%'}), width=6)
                    ])
                ], md=12),
            ]),
            dbc.Progress(id='progress_bar', value=0, style={"height": "20px", "margin-top": "15px"}, striped=True),
        ])
    ]),
    dcc.Graph(id='statistics_figure', style={'display': 'none'}),
    dbc.ListGroup(id='list_group', flush=True, className="mt-4"),
    dcc.Interval(id='interval-component', interval=100, n_intervals=0, disabled=True),
    html.Div(id='list-container', className="list-container mt-4"),
    html.Div([
        dbc.Button('<-', id='previous-page', n_clicks=0, className="mr-2", style={'display': 'block'}),
        html.Span(id='current-page', children="Seite 1 von 5", className="mx-2", style={'display': 'block'}),
        dbc.Button('->', id='next-page', n_clicks=0, className="ml-2", style={'display': 'block'}),
    ], className="d-flex justify-content-center align-items-center mt-4"),
    html.Div(id='output-container'),
    dcc.Store(id='memory-output'),
], fluid=True)

# Hauptausführungsbereich
if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)