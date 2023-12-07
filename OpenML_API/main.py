import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import json
from datetime import datetime, timedelta

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

class OpenML_API:
    def filter_datasets_by_attribute_types(self, start_date, end_date, num_attributes_range, num_features_range, limit):
        # Platzhalter für die API-Logik
        return [{"name": f"Dataset {i}", "NumberOfAttributes": i} for i in range(1, limit + 1)]

# Funktion zum Abrufen und Filtern von Datensätzen
def fetch_datasets(start_date=None, end_date=None, num_attributes_range=None, num_features_range=None, limit=10):
    api = OpenML_API()
    datasets = api.filter_datasets_by_attribute_types(start_date, end_date, num_attributes_range, num_features_range, limit)
    return datasets

# Funktion zum Erstellen eines Diagramms für ein gegebenes Dataset
def create_figure_for_dataset(dataset_id):
    df = px.data.gapminder().query("country=='Canada'")
    fig = px.line(df, x="year", y="gdpPercap", title=f"GDP Per Capita over Time for {dataset_id}")
    return fig

# Callback für das Aktualisieren der Datensatzliste
@app.callback(
    Output('list_group', 'children'),
    [Input('search_button', 'n_clicks')],
    [State('date_range', 'start_date'),
     State('date_range', 'end_date'),
     State('number_of_attributes_slider1', 'value'),
     State('number_of_attributes_slider2', 'value'),
     State('limit_input', 'value')]
)
def update_dataset_list(n_clicks, start_date, end_date, num_attributes_range, num_features_range, limit):
    if n_clicks is None:
        return []

    datasets = fetch_datasets(start_date, end_date, num_attributes_range, num_features_range, limit)
    list_group_items = []

    for idx, dataset in enumerate(datasets, start=1):
        list_group_item = dbc.ListGroupItem(
            [
                html.Div(
                    [
                        html.H5(f"{dataset['name']}", className="mb-1"),
                        html.Small("Click to view details", className="text-secondary"),
                        html.P("Weitere Informationen hier...")
                    ],
                    className="d-flex w-100 justify-content-between",
                    id={"type": "toggle", "index": idx},
                    style={
                        "cursor": "pointer",
                        "padding": "10px",
                        "margin-bottom": "5px",
                        "background-color": "#f8f9fa",  # Leichter Hintergrund
                        "border": "1px solid #ddd",  # Subtiler Rand
                        "border-radius": "5px",  # Abgerundete Ecken
                        "box-shadow": "0 2px 2px rgba(0,0,0,0.1)"  # Schatten für Tiefe
                    },
                )
            ]
        )
        collapse = dbc.Collapse(
            dbc.Card(dbc.CardBody([dcc.Graph(figure=create_figure_for_dataset(dataset['name']))])),
            id={"type": "collapse", "index": idx},
        )
        list_group_items.append(list_group_item)
        list_group_items.append(collapse)

    return list_group_items

# Callback für das Umschalten der Collapse-Komponenten
@app.callback(
    Output({"type": "collapse", "index": dash.ALL}, "is_open"),
    [Input({"type": "toggle", "index": dash.ALL}, "n_clicks")],
    [State({"type": "collapse", "index": dash.ALL}, "is_open")],
)
def toggle_collapse(n_clicks, is_open):
    ctx = dash.callback_context

    if not ctx.triggered:
        return is_open

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    button_id = json.loads(button_id)

    idx = button_id["index"] - 1

    new_is_open = is_open[:]
    new_is_open[idx] = not is_open[idx]

    return new_is_open

# Hinzufügen der Filter- und Listenkomponenten zum Layout
app.layout = dbc.Container([
    html.H1("OpenML Datensatzsuche"),
    dbc.Row([
        dbc.Col([
            # Filter für Datum
            dbc.Label("Datum"),
            dcc.DatePickerRange(
                id='date_range',
                start_date=datetime.now() - timedelta(30),  # Startdatum auf 30 Tage zurückgesetzt
                end_date=datetime.now(),  # Heutiges Datum als Enddatum
                min_date_allowed=datetime(2000, 1, 1),  # Mindestdatum
                max_date_allowed=datetime.now(),  # Maximaldatum
                display_format='DD.MM.YYYY',
                initial_visible_month=datetime.now(),  # Initial sichtbarer Monat
            ),
            html.Br(),

            # Filter für Anzahl der Attribute und Features (Slider)
            dbc.Label("Anzahl der Attribute (Slider 1)"),
            dcc.RangeSlider(
                id='number_of_attributes_slider1',
                min=0, max=100, step=1, value=[0, 100],
                marks={i: str(i) for i in range(0, 101, 10)}
            ),
            html.Br(),

            dbc.Label("Anzahl der Features (Slider 2)"),
            dcc.RangeSlider(
                id='number_of_attributes_slider2',
                min=0, max=100, step=1, value=[0, 100],
                marks={i: str(i) for i in range(0, 101, 10)}
            ),
            html.Br(),

            # Limiter für Anzahl Datensätze
            dbc.Label("Max Datensätze"),
            dbc.Input(id='limit_input', type='number', value=10),
            html.Br(),
            dbc.Button('Suchen', id='search_button', color="primary", className="mt-3"),
        ], width=4),
    ]),
    dbc.Row([
        dbc.Col(id='list_group', width=12)
    ]),
], fluid=True)

if __name__ == '__main__':
    app.run_server(debug=True)
