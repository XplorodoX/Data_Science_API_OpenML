import dash
from OpenML_API import OpenML_API
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd

app = dash.Dash(external_stylesheets=[dbc.themes.COSMO])

@app.callback(
    Output('results_table', 'data'),
    [Input('search_button', 'n_clicks')],
    [State('date_range', 'start_date'),
     State('date_range', 'end_date'),
     State('number_of_attributes_slider1', 'value'),
     State('number_of_attributes_slider2', 'value'),
     State('limit_input', 'value')]
)

# TODO Erweitern Sie diese Methode, um die neuen Filterkriterien zu berücksichtigen + Lister der Datasets einzubinden in eine Liste
def update_table(n_clicks, start_date, end_date, num_attributes_range, num_features_range, limit):
    if n_clicks is None:
        return []

    try:
        api = OpenML_API()
        # Erweitern Sie diese Methode, um die neuen Filterkriterien zu berücksichtigen
        datasets = api.filter_datasets_by_attribute_types(start_date, end_date, num_attributes_range, num_features_range, limit)
        return pd.DataFrame(datasets).to_dict('records')
    except Exception as e:
        print(f"Fehler beim Abrufen von OpenML-Daten: {e}")
        return []

app.layout = dbc.Container([
    html.H1("OpenML Datensatzsuche"),
    dbc.Row([
        dbc.Col([
            #Filter für Datum
            dbc.Label("Datum"),
            dcc.DatePickerRange(
                id='date_range',
                start_date=datetime.now() - timedelta(1000),  # Startdatum auf 30 Tage zurückgesetzt
                end_date=datetime.now(),  # Heutiges Datum als Enddatum
                min_date_allowed=datetime(2000, 1, 1),  # Mindestdatum
                max_date_allowed=datetime.now(),  # Maximaldatum
                start_date_placeholder_text="Startdatum",
                end_date_placeholder_text="Enddatum",
                display_format='DD.MM.YYYY',
                initial_visible_month=datetime.now(),  # Initial sichtbarer Monat
                style={'padding': '10px'}  # Stil-Anpassungen
            ),

            #Filter für Anzahl Features (Slider)
            dbc.Label("Anzahl der Datenpunkte (0-100)"),
            dcc.RangeSlider(
                id='number_of_attributes_slider1',
                min=0,
                max=100,
                step=1,
                value=[0, 100],
                marks={i: str(i) for i in range(0, 101, 10)}
            ),

            #Filter für Anzahl Features (Slider)
            dbc.Label("Anzahl der Features (0-100)"),
            dcc.RangeSlider(
                id='number_of_attributes_slider2',
                min=0,
                max=100,
                step=1,
                value=[0, 100],
                marks={i: str(i) for i in range(0, 101, 10)}
            ),

            #Filter für Anzahl Features (Slider)
            dbc.Label("Anzahl der numerischen Features (0-100)"),
            dcc.RangeSlider(
                id='number_of_attributes_slider3',
                min=0,
                max=100,
                step=1,
                value=[0, 100],
                marks={i: str(i) for i in range(0, 101, 10)}
            ),

            #Filter für Anzahl Features (Slider)
            dbc.Label("Anzahl der Kategorialen Features (0-100)"),
            dcc.RangeSlider(
                id='number_of_attributes_slider4',
                min=0,
                max=100,
                step=1,
                value=[0, 100],
                marks={i: str(i) for i in range(0, 101, 10)}
            ),
            #Filter für Anzahl Features (Slider)
            dbc.Label("Anzahl der Ordinal Features (0-100)"),
            dcc.RangeSlider(
                id='number_of_attributes_slider5',
                min=0,
                max=100,
                step=1,
                value=[0, 100],
                marks={i: str(i) for i in range(0, 101, 10)}
            ),

            #Filter für Anzahl Features (Slider)
            dbc.Label("Anzahl der unique Features (0-100)"),
            dcc.RangeSlider(
                id='number_of_attributes_slider6',
                min=0,
                max=100,
                step=1,
                value=[0, 100],
                marks={i: str(i) for i in range(0, 101, 10)}
            ),

            #Limiter für Anzahl Datensätze
            dbc.Label("Max Datensätze"),
            dbc.Input(id='limit_input', type='number', value=10),
            html.Br(),
            dbc.Button('Suchen', id='search_button', color="primary", className="mt-3"),
        ], width=4),
    ]),
    dbc.Row([
        dbc.Col([
            dash_table.DataTable(
                id='results_table',
                columns=[{'name': 'Name', 'id': 'name'}, 
                         {'name': 'Anzahl Attribute', 'id': 'NumberOfAttributes'}],
                data=[],
                page_size=10,
                style_table={'height': '300px', 'overflowY': 'auto'}
            ),
        ], width=12)
    ])
], fluid=True)

if __name__ == '__main__':
    app.run_server(debug=True)