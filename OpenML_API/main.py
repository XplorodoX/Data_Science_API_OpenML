import dash
import pandas as pd
import openml
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import json
import math
from datetime import datetime, timedelta, date
from dash.exceptions import PreventUpdate
import Helper as helper

#TODO
# - Fortschrittsbalken -> Implementierung
# - Mehrere Seiten -> Implementierung -> Performance verbesserungen?
# - Mehr Infos über Datensätze -> Implementierung / ausklappen Figur löschen und ersetzten durch Text, falls Platz fehlt!
# - Abbruchbutton -> Implementierung

# Setzen des Cache-Verzeichnisses
openml.config.set_root_cache_directory('cache')

ITEMS_PER_PAGE = 10

# Dash app setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

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
        Output('list-container', 'children'),
        Output('statistics_figure', 'figure'),
        # Removed Output for statistics_style as it's not defined
    ],
    [
        Input('search_button', 'n_clicks'),
        Input('previous-page', 'n_clicks'),
        Input('next-page', 'n_clicks'),
        Input('filtered_data_id', 'data')
    ],
    [
        State('date_range', 'start_date'),
        State('date_range', 'end_date'),
        State('current-page', 'children'),
        State('min_data_points', 'value'),
        State('max_data_points', 'value'),
        State('min_features', 'value'),
        State('max_features', 'value'),
        State('min_numerical_features', 'value'),
        State('max_numerical_features', 'value'),
        State('min_categorical_features', 'value'),
        State('max_categorical_features', 'value'),
        State('input_max_datasets', 'value')
    ]
)
def on_search_button_click(search_n_clicks, prev_n_clicks, next_n_clicks, filtered_data, start_date, end_date,
                           current_page, min_data_points, max_data_points, min_features, max_features,
                           min_numerical_features, max_numerical_features, min_categorical_features,
                           max_categorical_features, limit):
    # If the button hasn't been clicked, return placeholders for each output
    if search_n_clicks is None:
        return [], None, create_statistics_figure()

    # Create ranges from min and max values
    features_range = (min_features, max_features)
    numerical_features_range = (min_numerical_features, max_numerical_features)
    categorical_features_range = (min_categorical_features, max_categorical_features)
    data_points_range = (min_data_points, max_data_points)

    # Datenabrufen und Verarbeitung
    filtered_data = processData(start_date, end_date, features_range, numerical_features_range, categorical_features_range, data_points_range, limit)

    # Wenn keine Daten vorhanden sind, aktualisieren Sie nichts
    if not filtered_data:
        raise PreventUpdate

    # Umwandlung der gespeicherten Daten in DataFrame
    df = pd.DataFrame(filtered_data)  # Adjust to use filtered_data

    current_page = 1

    # Ermittlung des ausgelösten Buttons
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'next-page' in changed_id:
        current_page += 1
    elif 'previous-page' in changed_id:
        current_page -= 1

    # Stellen Sie sicher, dass die Seitenzahl innerhalb der gültigen Bereichsgrenzen bleibt
    current_page = max(min(current_page, math.ceil(len(df) / ITEMS_PER_PAGE)), 1)

    # Anwenden der Paginierung
    start = (current_page - 1) * ITEMS_PER_PAGE
    end = start + ITEMS_PER_PAGE
    #page_data = df.iloc[start:end]

    list_group_items = []
    for idx, dataset in enumerate(df, start=1):
        dataset_name = dataset['name']
        num_instances = dataset['instances']
        num_features = dataset['features']
        num_numeric_features = dataset['numeric_features']
        num_categorical_features = dataset['categorical_features']
        data_dimensions = f"{int(num_instances)} x {int(num_features)}"

        list_group_item = dbc.ListGroupItem(
            [
                html.Div(
                    [
                        html.H5(f"{idx}. {dataset_name}", className="mb-1"),
                        html.Small(f"Zeilenanzahl: {num_instances}", className="text-secondary"),
                        html.Small(f"Dimension: {data_dimensions}", className="text-secondary"),
                    ],
                    className="d-flex flex-column",
                    id={"type": "toggle", "index": idx + start},  # Adjust index based on pagination
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
            id={"type": "collapse", "index": idx + start}  # Adjust index based on pagination
        )
        list_group_items.append(list_group_item)
        list_group_items.append(collapse)

    statistics_figure = create_statistics_figure()

    return [list_group_items, html.Div(), statistics_figure]

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

        upload_date = getUploadDate(dataset_ids[dataset_id])

        start_date = parse_date(start_date)
        end_date = parse_date(end_date)

        if count >= limit:
            print("Limit erreicht")
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
                    'categorical_features': num_categorical_features
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

# Callbacks for toggling intervals and collapsing items
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
    dcc.Store(id='filtered_data_id', data=df.to_dict('records')),
    html.Div(
        [
            dbc.Button("Vorherige", id="previous-page", className="mr-2"),
            html.Span(children=["Seite ", "1"], id='current-page', className="mx-2"),
            dbc.Button("Nächste", id="next-page", className="ml-2"),
        ],
        className="d-flex justify-content-center align-items-center mt-4",
    ),
    # Hinzufügen des fehlenden Containers für list-container
    html.Div(id='list-container', className="list-container mt-4"),
], fluid=True)

# Hauptausführungsbereich
if __name__ == '__main__':
    app.run_server(debug=True)