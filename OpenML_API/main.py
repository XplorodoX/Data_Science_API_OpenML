# Imports der Bibliotheken
import math
import threading
import os
import pandas
import plotly.express as px
from dash import dash_table
import numpy as np
import openml
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL, MATCH
import plotly.graph_objs as go
from datetime import datetime, timedelta, date
import Helper as helper
import json
import dash
from dash.exceptions import PreventUpdate

# Setzen des Cache-Verzeichnisses
openml.config.set_root_cache_directory('cache')

# Dash app setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://use.fontawesome.com/releases/v5.8.1/css/all.css'], suppress_callback_exceptions=True)

# Set global Variables
ITEMS_PER_PAGE = 10 # Max Anzahl der Items pro page
stop = False # Stop Variable
filtered_data = [] # Alle Gefilterte Daten

#Globale Variablen für die max Nummern etc
global_max_number_of_instances = 0
global_max_number_of_features = 0
global_max_number_of_numeric_features = 0
global_max_number_of_symbolic_features = 0

# Laden der Daten und Setzen der globalen Variablen
datasets = helper.fetchDataList()

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

# Funktion für die Allgemine Statistik
def create_statistics_figure():
    fig = go.Figure(data=[
        go.Bar(x=['Dataset 1', 'Dataset 2', 'Dataset 3'], y=[50, 30, 70], name='Anzahl der Features')
    ])
    fig.update_layout(title='Statistik aller Datensätze', xaxis_title='Datensätze', yaxis_title='Anzahl der Features')
    return fig

#TODO fix für Statitischegesamtgrafik
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

    # Anfang des Callbacks oder der Funktion
    # Stellen Sie sicher, dass 'current_page_text' nicht leer ist und das erwartete Format hat
    if current_page_text and len(current_page_text.split()) > 2:
        try:
            current_page = int(current_page_text.split()[1])
        except ValueError:  # Fängt Fehler ab, falls die Konvertierung zu int fehlschlägt
            current_page = 1  # Setzen Sie einen Standardwert, falls ein Fehler auftritt
    else:
        current_page = 1  # Standardwert, falls 'current_page_text' nicht dem erwarteten Format entspricht

    # Ihr Code für die Behandlung von Seitenwechseln
    changed_id = dash.callback_context.triggered[0]['prop_id']
    if 'next-page' in changed_id:
        current_page = min(current_page + 1, math.ceil(len(filtered_data) / ITEMS_PER_PAGE))
    elif 'previous-page' in changed_id:
        current_page = max(current_page - 1, 1)

    start = (current_page - 1) * ITEMS_PER_PAGE
    filtered_info = filtered_data[start:start + ITEMS_PER_PAGE]

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


from dash.dependencies import Input, Output

from dash.exceptions import PreventUpdate
import plotly.express as px

@app.callback(
    Output('feature-histogram', 'figure'),
    [Input('feature-summary-table', 'active_cell'),
     State('feature-summary-table', 'data')],
    prevent_initial_call=True
)
def update_histogram(active_cell, table_data):
    if not active_cell or not table_data:
        # Wenn keine Zelle ausgewählt ist oder keine Daten vorhanden sind, zeige eine leere Figur
        raise PreventUpdate

    # Zugriff auf das ausgewählte Feature basierend auf der aktiven Zelle
    selected_feature = table_data[active_cell['row']]['Feature']

    # Da 'table_data' hier eine Liste von Wörterbüchern ist, extrahieren Sie alle Werte für das ausgewählte Feature
    # Wir erstellen ein neues DataFrame, das nur das ausgewählte Feature und seine Häufigkeiten enthält
    feature_values = [record[selected_feature] for record in table_data if selected_feature in record]

    # Umwandlung in DataFrame für das Histogramm
    feature_df = pd.DataFrame({selected_feature: feature_values})

    # Erstellung des Histogramms für das ausgewählte Feature
    fig = px.histogram(feature_df, x=selected_feature, title=f'Histogram of {selected_feature}')

    return fig


@app.callback(
    [Output('detail-section', 'style'),
     Output('filter-section', 'style'),
     Output('list_histogram', 'children')],
    [Input({'type': 'dataset-click', 'index': ALL}, 'n_clicks'),
     Input('back-button', 'n_clicks')],
    prevent_initial_call=True
)
def on_item_click(n_clicks, *args):

    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate

    # Ermitteln, welcher Input ausgelöst wurde
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Überprüfen, ob alle n_clicks-Werte null sind
    if all(click is None or click == 0 for click in n_clicks):
        raise PreventUpdate
    elif 'dataset-click' in button_id:

        # Dies ist ein Klick auf ein Dataset-Element
        dataset_id = json.loads(button_id.split('.')[0])['index']

        initial_df, dataset_info = download_dataset(dataset_id)
        completeness_graph = create_data_completeness_graph(initial_df)
        summary_records, columns = create_feature_summary_table(initial_df)

        detail_components = [
            dbc.ListGroupItem([
                html.H4("Datasetinformation"),
                html.P(f"Name of Dataset: {dataset_info.get('name', 'Not available')}"),
                html.P(f"Number of Features: {dataset_info.get('features_count', 'Not available')}"),
                html.P(f"Number of Instances: {dataset_info.get('instances_count', 'Not available')}"),
                html.P(f"Upload date: {dataset_info.get('upload_date', 'Not available')}"),
            ]),
            dbc.ListGroupItem([
                dcc.Graph(figure=completeness_graph)
            ]),
            dbc.ListGroupItem([
                dash_table.DataTable(
                    id='feature-summary-table',
                    columns=columns,
                    data=summary_records,
                    style_table={'overflowX': 'auto', 'height': '391px'},
                    style_cell={'textAlign': 'left', 'padding': '6px'},
                    style_header={'fontWeight': 'bold'},
                ),
                dcc.Graph(id='feature-histogram')
            ]),
        ]
        # Mache den Detailbereich sichtbar und verstecke den Filterbereich
        return {'display': 'block'}, {'display': 'none'}, detail_components

    # Detailansicht ausblenden, Filter anzeigen, wenn der Zurück-Button gedrückt wurde
    elif 'back-button' in button_id:
        return {'display': 'none'}, {'display': 'block'}, []

#TODO Überarbeiten nochmal!
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
    [
        Output('current-page', 'children'),
        Output('current-page', 'style'),  # Steuert die Sichtbarkeit der Seitennummer
        Output('previous-page', 'style'),  # Steuert die Sichtbarkeit des vorherigen Buttons
        Output('next-page', 'style'),  # Steuert die Sichtbarkeit des nächsten Buttons
        Output('pagination-container', 'style')  # Steuert die Sichtbarkeit des Gesamtcontainers
    ],
    [
        Input('search_button', 'n_clicks'),
        Input('previous-page', 'n_clicks'),
        Input('next-page', 'n_clicks')
    ],
    [
        State('current-page', 'children'),
        State('input_max_datasets', 'value')
    ]
)
def update_page_number(search_clicks, prev_clicks, next_clicks, current_page_text, maxData):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'] if ctx.triggered else ''

    # Initialisieren Sie die Anzahl der maximalen Daten, falls nicht angegeben
    maxData = maxData or 100  # Angenommen, 100 als Standardwert, falls nichts eingegeben wird

    # Berechnen Sie die Gesamtzahl der Seiten basierend auf maxData
    total_pages = math.ceil(maxData / ITEMS_PER_PAGE)

    # Bestimmen der aktuellen Seite basierend auf dem ausgelösten Ereignis
    if 'search_button' in triggered_id:
        current_page = 1  # Zurücksetzen auf die erste Seite, wenn die Suche ausgelöst wird
    elif 'next-page' in triggered_id and search_clicks:
        current_page = min(int(current_page_text.split()[1]) + 1,
                           total_pages) if current_page_text and ' ' in current_page_text else 2
    elif 'previous-page' in triggered_id and search_clicks:
        current_page = max(int(current_page_text.split()[1]) - 1,
                           1) if current_page_text and ' ' in current_page_text else 1
    else:
        current_page = int(current_page_text.split()[1]) if current_page_text and ' ' in current_page_text else 1

    # Stil und Sichtbarkeitseinstellungen basierend auf der Anzahl der Klicks
    container_style = {'display': 'flex'} if search_clicks else {'display': 'none'}
    page_style = {'visibility': 'visible', 'display': 'block'} if search_clicks else {'visibility': 'hidden','display': 'none'}
    prev_button_style = {'visibility': 'visible','display': 'inline-block'} if current_page > 1 and search_clicks else {'visibility': 'hidden', 'display': 'none'}
    next_button_style = {'visibility': 'visible', 'display': 'inline-block'} if current_page < total_pages and search_clicks else {'visibility': 'hidden', 'display': 'none'}
    page_number_text = f"Seite {current_page} von {total_pages}" if search_clicks else ""

    return page_number_text, page_style, prev_button_style, next_button_style, container_style

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
    # TODO Filter nochmal überprüfen und testen
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

######### Dennis Code ##############
def download_dataset(dataset_id=None):
    dataset_info = {}
    #df = pd.DataFrame()  # Initialisieren Sie df hier, um sicherzustellen, dass es immer definiert ist

    if dataset_id:
        try:
            dataset = openml.datasets.get_dataset(dataset_id, download_data=False, download_qualities=False, download_features_meta_data=False)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute,
                                                                            dataset_format='dataframe')
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

    return df, dataset_info

def create_data_completeness_graph(df):
    if df.empty:
        return go.Figure()  # Gibt eine leere Figur zurück, wenn df leer ist
    total_values = np.prod(df.shape)  # Verwende np.prod() statt np.product()
    missing_values = df.isnull().sum().sum()
    complete_values = total_values - missing_values
    fig = go.Figure(data=[go.Pie(labels=['Complete data', 'Missing data fields'],
                                 values=[complete_values, missing_values], hole=.6)])
    fig.update_layout(title_text="Completeness of the dataset", title_x=0.5)
    return fig

def format_number(value):
    """Formatiert eine Zahl mit bis zu vier Nachkommastellen, entfernt jedoch nachfolgende Nullen."""
    try:
        # Versuche, den Wert in einen Float umzuwandeln
        float_value = float(value)
        # Wenn der Wert eine ganze Zahl ist, gib ihn als ganze Zahl zurück
        if float_value.is_integer():
            return f"{int(float_value)}"
        else:
            # Andernfalls formatiere den Wert mit vier Nachkommastellen
            return f"{float_value:.4f}".rstrip('0').rstrip('.')
    except ValueError:
        # Wenn der Wert nicht in einen Float umgewandelt werden kann, gib ihn unverändert zurück
        return value

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

from dash import dcc
import pandas as pd  # Stellen Sie sicher, dass pandas importiert wird

# App Layout
app.layout = dbc.Container([
    # Verstecke diesen Teil standardmäßig durch Setzen von 'display': 'none' im style
    html.Div(id='detail-section', style={'display': 'none'}, children=[
        dbc.ListGroup(id='list_histogram', flush=True, className="mt-4"),
        html.Button("Zurück", id='back-button', className="btn btn-secondary mt-3"),
    ]),
    html.Div(id='filter-section', style={'display': 'block'}, children=[
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
            dbc.Button(
                html.Span(className="fas fa-chevron-left"),  # FontAwesome Pfeil nach links
                id='previous-page',
                n_clicks=0,
                className="mr-2 btn btn-lg",  # Entfernen Sie btn-primary für ein angepasstes Design
                style={
                    'visibility': 'hidden',  # Anfänglich unsichtbar, wird durch Dash Callbacks gesteuert
                    'backgroundColor': '#78909C',  # Dunkelgraue Farbe, passen Sie diese an Ihr Design an
                    'color': 'white',
                    'borderRadius': '20px',  # Abgerundete Ecken
                    'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)'  # Schatten für Tiefe
                }
            ),
            html.Span(
                id='current-page',
                children="",  # Der Text wird durch einen Dash Callback gesetzt
                className="px-3",  # Fügen Sie etwas Padding hinzu für besseren Abstand
                style={'fontSize': '20px'}  # Größere Schriftart für die Seitenzahl
            ),
            dbc.Button(
                html.Span(className="fas fa-chevron-right"),  # FontAwesome Pfeil nach rechts
                id='next-page',
                n_clicks=0,
                className="ml-2 btn btn-lg",  # Entfernen Sie btn-primary für ein angepasstes Design
                style={
                    'visibility': 'hidden',  # Anfänglich unsichtbar, wird durch Dash Callbacks gesteuert
                    'backgroundColor': '#78909C',  # Dunkelgraue Farbe, passen Sie diese an Ihr Design an
                    'color': 'white',
                    'borderRadius': '20px',  # Abgerundete Ecken
                    'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)'  # Schatten für Tiefe
                }
            )
        ], className="d-flex justify-content-center align-items-center mt-4", id='pagination-container'),
        html.Div(id='output-container'),
        dcc.Store(id='memory-output'),
    ]),
], fluid=True)

def initialize_openml_cache():
    try:
        # Überprüfen, ob der Cache-Ordner bereits existiert
        cache_dir = 'cache'  # Setzen Sie den Pfad zu Ihrem Cache-Verzeichnis
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)  # Erstellen Sie den Cache-Ordner, wenn er nicht existiert
            print(f"Cache-Verzeichnis {cache_dir} wurde erstellt.")

            # Erhalten Sie die Liste aller Datensätze im DataFrame-Format
            datasets_df = openml.datasets.list_datasets(output_format='dataframe')
            dataset_ids = datasets_df['did'].tolist()
            print(f"Es werden {len(dataset_ids)} Datensätze im Cache gespeichert...")

            # Laden Sie jeden Datensatz herunter, um den Cache zu initialisieren
            for dataset_id in dataset_ids:
                try:
                    openml.datasets.get_dataset(dataset_id, download_data=False, download_qualities=False, download_features_meta_data=False)
                    print(f"Dataset {dataset_id} wurde im Cache gespeichert.")
                except Exception as e:
                    print(f"Fehler beim Abrufen von Dataset {dataset_id}: {e}")
        else:
            print(f"Cache-Verzeichnis {cache_dir} existiert bereits. Keine erneute Initialisierung notwendig.")
    except Exception as e:
        print(f"Fehler beim Initialisieren des OpenML-Cache: {e}")


#cache_thread = threading.Thread(target=initialize_openml_cache, daemon=True)
#cache_thread.start()

if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)