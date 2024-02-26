# Imports der Bibliotheken
import math
import os
from dash import dash_table
import numpy as np
import pandas as pd
import plotly.express as px
import openml
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objs as go
from datetime import datetime, timedelta, date
from dash.exceptions import PreventUpdate
import Helper as helper
import json
import dash

# Setzen des Cache-Verzeichnisses
openml.config.set_root_cache_directory('cache')

# Dash app setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://use.fontawesome.com/releases/v5.8.1/css/all.css'], suppress_callback_exceptions=True)

# Set global Variables
ITEMS_PER_PAGE = 10 # Max Anzahl der Items pro page
stop = False # Stop Variable
filtered_data = [] # Alle Gefilterte Daten
initial_df = pd.DataFrame() # Initialisieren Sie df hier, um sicherzustellen, dass es immer definiert ist

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

def create_statistics_figure(filtered_info):
    if not filtered_info:
        return go.Figure()  # Gibt eine leere Figur zurück, wenn keine Daten vorhanden sind

    # Extrahiere die Namen der Datensätze
    dataset_names = [f"Dataset {idx+1}" for idx, dataset in enumerate(filtered_info)]
    # Extrahiere die Anzahl der numerischen und kategorialen Features
    numeric_feature_counts = [dataset['numeric_features'] for dataset in filtered_info]
    categorical_feature_counts = [dataset['categorical_features'] for dataset in filtered_info]

    # Erstelle die Figur und füge die Daten für numerische und kategoriale Features hinzu
    fig = go.Figure(data=[
        go.Bar(name='Numeric Features', x=dataset_names, y=numeric_feature_counts),
        go.Bar(name='Categorial Features', x=dataset_names, y=categorical_feature_counts)
    ])
    # Aktualisiere das Layout der Figur
    fig.update_layout(
        title='Statistics of the datasets on the current page',
        xaxis_title='Datasets',
        yaxis_title='Number of features',
        barmode='group'  # Gruppiere die Balken, um einen Vergleich zu ermöglichen
    )

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

    list_group_items = []
    statistics_style = {'display': 'none'}

    if n_clicks is None:
        return list_group_items, go.Figure(), statistics_style

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

    # Aktualisieren Sie die Statistikfigur basierend auf gefilterten Daten oder anderen Kriterien
    statistics_figure = create_statistics_figure(filtered_info)
    statistics_style = {'display': 'block'}

    return list_group_items, statistics_figure, statistics_style

# Definieren Sie eine Liste von bekannten ordinalen Features, falls zutreffend
ordinal_features = ['ordinal_feature1', 'ordinal_feature2']

@app.callback(
    Output('feature-histogram', 'figure'),
    [Input('feature-summary-table', 'active_cell'),
     State('feature-summary-table', 'data')],
    prevent_initial_call=True
)
def update_histogram(active_cell, table_data):
    if not active_cell or not table_data:
        raise PreventUpdate

    df = pd.DataFrame(table_data)
    selected_feature = df.iloc[active_cell['row']]['Feature']

    if selected_feature not in initial_df.columns:
        return {
            "data": [],
            "layout": {
                "title": f"Feature {selected_feature} not found in dataframe.",
                "xaxis": {"visible": False},
                "yaxis": {"visible": False}
            }
        }

    # Überprüfen, ob das Feature numerisch ist
    if pd.api.types.is_numeric_dtype(initial_df[selected_feature]):
        fig_type = 'histogram'
    # Überprüfen, ob das Feature ordinal ist
    elif selected_feature in ordinal_features:
        fig_type = 'bar'
    # Andernfalls behandeln wir das Feature als nominal
    else:
        fig_type = 'bar'

    # Entscheiden, welche Art von Diagramm basierend auf dem Feature-Typ zu erstellen
    if fig_type == 'histogram':
        fig = px.histogram(initial_df, x=selected_feature, title=f'Histogram of {selected_feature}')
    elif fig_type == 'bar':
        fig = px.bar(initial_df, x=selected_feature, title=f'Distribution of {selected_feature}')

    # Optional: Hinzufügen von Quantillinien für numerische Features
    if fig_type == 'histogram':
        quantiles = initial_df[selected_feature].quantile([0.25, 0.5, 0.75, 0.97, 0.997]).to_dict()
        for quantile, value in quantiles.items():
            fig.add_vline(x=value, line_dash="solid", line_color="blue",
                          annotation_text=f"{quantile * 100}th: {value:.2f}", annotation_position="top right",
                          annotation=dict(font_size=10, font_color="green", showarrow=False))

    fig.update_traces(hoverinfo='x+y', selector=dict(type='histogram' if fig_type == 'histogram' else 'bar'))

    return fig

def prepare_table_data_from_df(df):
    """Erstellt eine Liste von Dictionaries für die DataTable aus einem DataFrame."""
    # Erstellt eine Liste von Dictionaries, wobei jedes Dictionary eine Zeile repräsentiert
    table_data = [{"Feature": feature} for feature in df.columns]
    return table_data

@app.callback(
    [Output('detail-section', 'style'),
     Output('filter-section', 'style'),
     Output('list_histogram', 'children'),
     Output('dataset-store', 'data')],
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

        global initial_df

        initial_df, dataset_info = download_dataset(dataset_id)
        completeness_graph = create_data_completeness_graph(initial_df)
        summary_records, columns = create_feature_summary_table(initial_df)

        columns = [
    {"name": "Feature", "id": "Feature"},
    {"name": "Count", "id": "count"},
    {"name": "Mean", "id": "mean"},
    {"name": "Std", "id": "std"},
    {"name": "Min", "id": "min"},
    {"name": "25%", "id": "25%"},
    {"name": "50%", "id": "50%"},
    {"name": "75%", "id": "75%"},
    {"name": "97%", "id": "97%"},
    {"name": "99.7%", "id": "99.7%"},
    {"name": "Max", "id": "max"},
    {"name": "Mode", "id": "mode"}
]
        detail_components = [
            dbc.ListGroupItem([
                html.H4("Information of current dataset"),
                html.P(f"ID of Dataset: {dataset_id}"),
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
                    columns=columns,  # Verwenden Sie die vorher definierten Spaltenüberschriften
                    data=summary_records,  # Verwenden Sie hier die durch create_feature_summary_table generierten Daten
                    style_table={'overflowX': 'auto', 'height': '391px'},
                    style_cell={'textAlign': 'left', 'padding': '6px'},
                    style_header={'fontWeight': 'bold'},
                ),
                dcc.Graph(id='feature-histogram')
            ]),
        ]
        # Mache den Detailbereich sichtbar und verstecke den Filterbereich
        return {'display': 'block'}, {'display': 'none'}, detail_components, {'selected_dataset_id': dataset_id}

    # Detailansicht ausblenden, Filter anzeigen, wenn der Zurück-Button gedrückt wurde
    elif 'back-button' in button_id:
        return {'display': 'none'}, {'display': 'block'}, [], dash.no_update

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
    page_number_text = f"Page {current_page} of {total_pages}" if search_clicks else ""

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
                print(f"Error while parsing Data '{date_str}': {e}")
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
        print(f"Error on Dataset: {dataset_id}: {e}")
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
        float_value = float(value)
        if float_value.is_integer():
            return str(int(float_value))
        else:
            return f"{float_value:.4f}".rstrip('0').rstrip('.')
    except ValueError:
        return value

def create_feature_summary_table(df):
    if df.empty:
        return [], []  # Gibt leere Werte zurück, wenn df leer ist

    # Berechnung der deskriptiven Statistiken
    summary = df.describe(percentiles=[.25, .5, .75, .97, .997], include='all').transpose()

    # Modus für jede Spalte berechnen
    modes = df.mode().iloc[0]
    summary['mode'] = [modes[col] if col in modes else "N/A" for col in df.columns]

    # Formatieren der numerischen Werte
    for col in summary.columns:
        summary[col] = summary[col].apply(format_number)

    # Anpassen der Spaltennamen für die Darstellung in der DataTable
    summary.reset_index(inplace=True)
    summary.rename(columns={'index': 'Feature'}, inplace=True)

    summary_records = summary.to_dict('records')

    # Definieren der Spalten für die DataTable
    columns = [{"name": col, "id": col} for col in summary.columns]

    return summary_records, columns

########################################################################################

# Überprüfen Sie, ob der Cache-Ordner existiert
def check_cache_folder_exists():
    return os.path.exists('cache')  # Ändern Sie den Pfad zu Ihrem cache-Ordner nach Bedarf

# Callback-Funktion für die Aktualisierung des Fortschrittsbalkens und die Sichtbarkeit der Ladesektion
@app.callback(
    [
        Output('progress_bar', 'value'),
        Output('loading-section', 'style'),
        Output('filter-section', 'style', allow_duplicate=True),
        Output('cache-status-store', 'data')# Fügen Sie eine neue Output-Komponente hinzu
    ],
    [
        Input('progress_interval', 'n_intervals'),
        Input('cache-status-store', 'data')  # Verwenden der gespeicherten Cache-Status-Information
    ],
    [
        State('loading-section', 'style'),  # Zustand des aktuellen Stils der Ladesektion
        State('filter-section', 'style')  # Zustand des aktuellen Stils der Filtersektion
    ], prevent_initial_call=True
)
def update_progress_visibility_and_filter_visibility(n, cache_status, loading_style, filter_style):
    # Überprüfen Sie, ob der Cache-Ordner existiert
    cache_exists = check_cache_folder_exists()

    # Initialisieren Sie die Ausgabewerte
    new_value = 0
    new_loading_style = loading_style.copy()
    new_filter_style = filter_style.copy()
    new_cache_status = cache_status if cache_status else {'updated': False}

    # Bei Programmneustart, wenn Cache existiert und noch kein Update signalisiert wurde
    if cache_exists and n == 0:
        new_value = 100
        new_loading_style['display'] = 'none'
        new_filter_style['display'] = 'block'
        new_cache_status['updated'] = True
        return new_value, new_loading_style, new_filter_style, new_cache_status
    elif cache_exists and new_cache_status.get('updated'):
        # Wenn der Cache existiert und bereits aktualisiert wurde, verhindern Sie ein Update
        raise PreventUpdate

    # Wenn der Cache-Ordner existiert, aber die Aktion noch nicht durchgeführt wurde
    if cache_exists and not new_cache_status.get('updated'):
        new_value = 100
        new_loading_style['display'] = 'none'
        new_filter_style['display'] = 'block'
        new_cache_status['updated'] = True  # Markieren, dass die Aktion durchgeführt wurde
    elif not cache_exists:
        # Logik für den Fall, dass der Cache nicht existiert
        new_value = min((n * 10), 100)
        new_loading_style['display'] = 'block' if new_value < 100 else 'none'
        new_filter_style['display'] = 'none' if new_value < 100 else 'block'

    return new_value, new_loading_style, new_filter_style, new_cache_status

@app.callback(
    Output('detail-section', 'children'),
    Input('download-button', 'n_clicks'),
    State('dataset-store', 'data')
)
def download_set(n_clicks, store_data):
    if n_clicks is None or store_data is None:
        raise dash.exceptions.PreventUpdate

    dataset_id = store_data.get('selected_dataset_id') if store_data else None
    if dataset_id is None:
        raise dash.exceptions.PreventUpdate

    # Ersetzen Sie diesen Teil, um das Dataset lokal zu speichern
    folder_name = 'Downloaded_Dataset'  # Definieren Sie den Ordnernamen
    if not os.path.exists(folder_name):  # Erstellen Sie den Ordner, wenn er nicht existiert
        os.makedirs(folder_name)

    # Dataset von OpenML herunterladen
    dataset = openml.datasets.get_dataset(dataset_id, download_data=False, download_qualities=False, download_features_meta_data=False)
    df, _, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    # Speichern Sie das Dataset als CSV in dem neuen Ordner
    file_path = os.path.join(folder_name, f"dataset_{dataset_id}.csv")
    df.to_csv(file_path, index=False)

    # Hinweis für den Benutzer erstellen
    return f'Dataset {dataset_id} wurde als CSV gespeichert: {file_path}'

# App Layout
app.layout = dbc.Container([
    html.Div(
        id='loading-section',
        style={'display': 'block'},
        children=[
            dbc.Row(
                dbc.Col(
                    [
                        html.H4("Daten werden geladen, bitte warten...", className="text-center mb-3"),
                        dbc.Progress(id='progress_bar', value=0, striped=True, animated=True, style={"height": "30px"}),
                        html.P("Der Ladevorgang kann einige Momente dauern. Vielen Dank für Ihre Geduld.",
                               className="text-center mt-3"),
                        dcc.Store(id='cache-status-store', storage_type='session'),
                        dcc.Interval(id='progress_interval', interval=1000, n_intervals=0)
                        # Timer set to tick every second
                    ],
                    width={"size": 10, "offset": 1},
                    style={'max-width': '800px', 'margin': 'auto'}
                    # Stellen Sie sicher, dass dies korrekt innerhalb von dbc.Col ist
                )
            )
        ],
        className="mt-5"
    ),
    # Verstecke diesen Teil standardmäßig durch Setzen von 'display': 'none' im style
    html.Div(id='detail-section', style={'display': 'none'}, children=[
        dbc.ListGroup(id='list_histogram', flush=True, className="mt-4"),
        html.Button("Back", id='back-button', className="btn btn-secondary mt-3"),
        html.Button("Download", id='download-button', className="btn btn-secondary mt-3"),
        dcc.Store(id='dataset-store', storage_type='session'),
    ]),
    html.Div(id='filter-section', style={'display': 'block'}, children=[
        dbc.Card([
            dbc.CardHeader("Filter"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Upload-Date"),
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
                            dbc.CardHeader("Number of Data Points"),
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
                                            dbc.Input(id='max_data_points', type='number', value=12345, min=0, max=max_instances),
                                        ]),
                                        width=6,
                                    ),
                                ]),
                                dbc.Tooltip(f"Previous max range was 0 to {max_instances}.", target="max_data_points"),
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
                            dbc.CardHeader("Max Datasets"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(
                                        dbc.Input(
                                            id='input_max_datasets',
                                            type='number',
                                            min=0,
                                            max=5500,
                                            step=1,
                                            value=20
                                        ),
                                        width=10,
                                    ),
                                ]),
                                dbc.Tooltip("Previous max range was 0 to 5500.", target="input_max_datasets"),
                            ]),
                        ]),
                    ], md=2),
                ], className="mb-4"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Number of Features"),
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
                                            dbc.Input(id='max_features', type='number', value=50, min=0, max=max_features),
                                        ]),
                                        width=6,
                                    ),
                                ]),
                                dbc.Tooltip(f"Previous max range was 0 to {max_features}.", target="max_features"),
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
                            dbc.CardHeader("Number of Numerical Features"),
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
                                            dbc.Input(id='max_numerical_features', type='number', value=30, min=0, max=max_numeric_features),
                                        ]),
                                        width=6,
                                    ),
                                ]),
                                dbc.Tooltip(f"Previous max range was 0 to {max_numeric_features}.", target="max_numerical_features"),
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
                            dbc.CardHeader("Number of Categorical Features"),
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
                                            dbc.Input(id='max_categorical_features', type='number', value=20, min=0, max=max_categorical_features),
                                        ]),
                                        width=6,
                                    ),
                                ]),
                                dbc.Tooltip(f"Previous max range was 0 to {max_categorical_features}.", target="max_categorical_features"),
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
                            dbc.Col(dbc.Button('Search', id='search_button', color="primary", className="mt-3 mb-3", style={'width': '100%'}))
                        ])
                    ], md=12),
                ]),
            ])
        ]),
        dbc.Spinner(  # Fügen Sie den Spinner hier ein
            children=[  # Beginn der Kinder, die geladen werden
                dcc.Graph(id='statistics_figure', style={'display': 'none'}),
                dbc.ListGroup(id='list_group', flush=True, className="mt-4")
            ],
            size="lg",  # Größe des Spinners
            color="primary",  # Farbe des Spinners
            type="border",  # Art des Spinners, z.B. 'border' oder 'grow'
            fullscreen=False,  # Ob der Spinner den ganzen Bildschirm abdecken soll
        ),
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
    ]),
], fluid=True)

if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)