import pandas as pd
import openml
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from datetime import datetime, timedelta

# Verwenden einer Klasse zur Verwaltung des App-Zustands
class DataManager:
    def __init__(self):
        self.datasets = self.fetch_data_list()
        self.numerical_range = self.calc_range_datasets() if not self.datasets.empty else None
        self.update_global_max_values()

    def fetch_data_list(self):
        # Funktion zum Abrufen der Datenliste hier
        return pd.DataFrame()  # Beispiel-Rückgabe

    def calc_range_datasets(self):
        # Funktion zur Berechnung der Bereichsdatensätze hier
        return {}  # Beispiel-Rückgabe

    def update_global_max_values(self):
        # Funktion zum Aktualisieren globaler Maximalwerte
        pass

    def filter_datasets_by_date(datasets, start_date=None, end_date=None):
        if start_date:
            datasets = datasets[datasets['upload_date'] >= pd.to_datetime(start_date)]
        if end_date:
            datasets = datasets[datasets['upload_date'] <= pd.to_datetime(end_date)]
        return datasets

# Instanz der DataManager-Klasse erstellen
data_manager = DataManager()

# Dash-App initialisieren
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App-Layout definieren
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
                                start_date=datetime.now() - timedelta(days=200),
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
    dbc.Row([
        dbc.Col(dbc.Button("Vorherige", id="prev_page_button", className="mr-2"), width="auto"),
        dbc.Col(dbc.Button("Nächste", id="next_page_button"), width="auto"),
        dbc.Col(html.Div(id="page_info"), width="auto")
    ]),
], fluid=True)

@app.callback(
    Output('filtered-data-display', 'children'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_filtered_data(start_date, end_date):
    filtered_data = filter_datasets_by_date(data_manager.datasets, start_date, end_date)
    # Weiterverarbeitung und Anzeige der gefilterten Daten
    return html.Div([html.P(f'Datensatz-ID: {row["id"]}') for index, row in filtered_data.iterrows()])

# Callbacks definieren
@app.callback(
    Output('output-component', 'children'),
    [Input('input-component', 'value')]
)
def update_output(input_value):
    # Beispiel-Callback-Logik
    return f'Eingegebener Wert: {input_value}'

# Hauptausführungsbereich
if __name__ == '__main__':
    app.run_server(debug=True)
