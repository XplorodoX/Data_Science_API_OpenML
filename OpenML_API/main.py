import time
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import json
from OpenML_API import OpenML_API
from datetime import datetime, timedelta

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Funktion zum Abrufen und Filtern von Datensätzen
# def filter_datasets_by_attribute_types(self, start_date=None, end_date=None, num_attributes_range=None, num_features_range=None, limit=None):
def fetch_datasets(start_date=None, end_date=None, num_attributes_range=None, num_features_range=None, limit=10):
    """
        Fetches datasets from OpenML based on various criteria.

        :param start_date: The start date for filtering datasets (datetime object).
        :param end_date: The end date for filtering datasets (datetime object).
        :param num_attributes_range: Tuple specifying the range of the number of attributes (min, max).
        :param num_features_range: Tuple specifying the range of the number of features (min, max).
        :param limit: Maximum number of datasets to fetch.
        :return: List of filtered datasets.
        """

    # Validate date range
    if start_date and end_date and start_date > end_date:
        raise ValueError("Start date must be before end date.")

    api_instance = OpenML_API()

    try:
        # Call the filter_datasets_by_attribute_types method on the instance
        datasets = api_instance.filter_datasets_by_attribute_types(
            start_date, end_date, num_attributes_range, num_features_range, limit
        )
        return datasets
    except Exception as e:
        # Handle potential errors during API call
        print(f"Error fetching datasets: {e}")
        return []

# Funktion zum Erstellen eines Diagramms für ein gegebenes Dataset
def create_placeholder_figure():
    # Create a simple scatter plot as a placeholder
    fig = go.Figure(data=[
        go.Scatter(x=[1, 2, 3], y=[4, 1, 2], mode='markers', marker=dict(color='LightSkyBlue'), name='Placeholder Data'),
        go.Scatter(x=[1, 2, 3], y=[2, 3, 1], mode='markers', marker=dict(color='Violet'), name='Placeholder Data 2')
    ])

    # Add layout details
    fig.update_layout(title='Placeholder Figure', xaxis_title='X Axis', yaxis_title='Y Axis')

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

    time.sleep(2)

    datasets = fetch_datasets(start_date, end_date, num_attributes_range, num_features_range, limit)

    list_group_items = []

    for idx, dataset in enumerate(datasets, start=1):
        dataset_name = dataset[1]
        num_downloads = 1000  # Assuming this is a placeholder and you will replace it with actual data

        # Retrieve the dimensions from the dataset tuple
        rows = int(dataset[2])
        columns = int(dataset[3])

        data_dimensions = f"{rows}x{columns}"

        list_group_item = dbc.ListGroupItem(
            [
                html.Div(
                    [
                        html.H5(dataset_name, className="mb-1"),
                        html.Small(f"Downloads: {num_downloads}", className="text-secondary"),
                        html.Small(f"Dimension: {data_dimensions}", className="text-secondary"),
                    ],
                    className="d-flex flex-column",
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
            dbc.Card(dbc.CardBody([dcc.Graph(figure=create_placeholder_figure())])),
            id={"type": "collapse", "index": idx}
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
    html.Img(src='logo.png', height=50),
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
                                    start_date=datetime.now() - timedelta(10000),
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
