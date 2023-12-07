import time
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
     State('limit_input', 'value'),
     State('filter_range_dropdown', 'value')]
)
def update_dataset_list(n_clicks, start_date, end_date, num_attributes_range, num_features_range, limit, filter_range):
    if n_clicks is None:
        return []

    time.sleep(2)

    datasets = fetch_datasets(start_date, end_date, num_attributes_range, num_features_range, limit)

    # Aktualisiere die Maximalwerte der Slider basierend auf dem Wert aus dem Dropdown-Menü
    num_attributes_slider1_max = filter_range
    num_attributes_slider2_max = filter_range

    list_group_items = []

    for idx, dataset in enumerate(datasets, start=1):
        list_group_item = dbc.ListGroupItem(
            [
                html.Div(
                    [
                        html.H5(f"{dataset['name']}", className="mb-1"),
                        html.Small(f"Anzahl der Features: {dataset['NumberOfAttributes']}", className="text-secondary"),
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
                ),
                html.Div(
                    [
                        dbc.Button("Datensatz herunterladen", id=f"download_button_{idx}", color="success", className="mr-2"),
                        dbc.Button("Zur Webseite", id=f"webpage_button_{idx}", color="primary")
                    ],
                    className="d-flex justify-content-end"
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


# Wenn ein "Datensatz herunterladen" Button geklickt wird, könntest du die Logik zum Herunterladen implementieren.
# Ebenso könntest du die Logik zum Weiterleiten auf eine Webseite implementieren, wenn der "Zur Webseite" Button geklickt wird.
# Hier sind Platzhalter-Funktionen für beide Aktionen:

@app.callback(
    Output('output', 'children'),  # Hier könntest du eine Bestätigungsnachricht anzeigen
    [Input(f'download_button_{idx}', 'n_clicks') for idx in range(1, 11)]
)
def download_dataset(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    dataset_index = int(button_id.split('_')[-1])

    # Hier kannst du die Logik zum Herunterladen des Datensatzes implementieren, z.B. mit Flask-Download-Link oder einer anderen Methode.
    return f"Datensatz {dataset_index} wurde heruntergeladen."


@app.callback(
    Output('webpage_redirect', 'href'),  # Hier leitest du auf die gewünschte Webseite weiter
    [Input(f'webpage_button_{idx}', 'n_clicks') for idx in range(1, 11)]
)
def redirect_to_webpage(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "/"
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    dataset_index = int(button_id.split('_')[-1])

    # Hier kannst du die Logik implementieren, um auf die gewünschte Webseite weiterzuleiten.
    # Zum Beispiel mit dash.redirect() oder einem ähnlichen Ansatz.
    return "/ziel-webseite"  # Ändere "/ziel-webseite" in die tatsächliche Ziel-URL.


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
    dbc.Card([
        dbc.CardHeader("Filter"),
        dbc.CardBody([
            dcc.Loading(  # Hier wird der Ladebalken für die Filterelemente hinzugefügt
                id="loading",
                type="default",
                children=[
                    dbc.CardGroup([
                        dbc.Card([
                            dbc.CardHeader("Datum"),
                            dbc.CardBody([
                                dcc.DatePickerRange(
                                    id='date_range',
                                    start_date=datetime.now() - timedelta(30),
                                    end_date=datetime.now(),
                                    min_date_allowed=datetime(2000, 1, 1),
                                    max_date_allowed=datetime.now(),
                                    display_format='DD.MM.YYYY',
                                    initial_visible_month=datetime.now()
                                ),
                            ]),
                        ]),
                        dbc.Card([
                            dbc.CardHeader("Maximalwert für Anzahl der Attribute"),
                            dbc.CardBody([
                                dbc.Input(id='max_attributes_input', type='number', value=100)
                            ]),
                        ]),
                        dbc.Card([
                            dbc.CardHeader("Anzahl der Attribute (Slider 1)"),
                            dbc.CardBody([
                                dcc.RangeSlider(
                                    id='number_of_attributes_slider1',
                                    min=0, max=100, step=1, value=[0, 100],
                                    marks={i: str(i) for i in range(0, 101, 10)}
                                ),
                            ]),
                        ]),
                        dbc.Card([
                            dbc.CardHeader("Anzahl der Features (Slider 2)"),
                            dbc.CardBody([
                                dcc.RangeSlider(
                                    id='number_of_attributes_slider2',
                                    min=0, max=100, step=1, value=[0, 100],
                                    marks={i: str(i) for i in range(0, 101, 10)}
                                ),
                            ]),
                        ]),
                        dbc.Card([
                            dbc.CardHeader("Max Datensätze"),
                            dbc.CardBody([
                                dbc.Input(id='limit_input', type='number', value=10)
                            ]),
                        ]),
                        dbc.Card([  # Dropdown-Menü für die Filter-Range hinzugefügt
                            dbc.CardHeader("Filter-Range auswählen"),
                            dbc.CardBody([
                                dcc.Dropdown(
                                    id='filter_range_dropdown',
                                    options=[
                                        {'label': '10', 'value': 10},
                                        {'label': '20', 'value': 20},
                                        {'label': '30', 'value': 30},
                                    ],
                                    value=10  # Standardwert auswählen
                                )
                            ]),
                        ]),
                    ]),
                    dbc.Button('Suchen', id='search_button', color="primary", className="mt-3")
                ],
            ),
        ])
    ]),
    dbc.Row([
        dcc.Loading(  # Hier wird der Ladebalken für die List Group hinzugefügt
            id="loading_list",
            type="default",
            children=[
                dbc.Col(id='list_group', width=12)
            ],
        )
    ]),
], fluid=True)


if __name__ == '__main__':
    app.run_server(debug=False)
