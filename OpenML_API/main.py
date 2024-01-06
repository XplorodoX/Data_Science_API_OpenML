import os
import time
import dash
import openml
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import json
from OpenML_API import OpenML_API
from datetime import datetime, timedelta

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def fetch_datasets(start_date=None, end_date=None, num_attributes_range=None, num_features_range=None, limit=10):
    if start_date and end_date and start_date > end_date:
        raise ValueError("Start date must be before end date.")

    try:
        datasets = filter_datasets_by_attribute_types(
            start_date, end_date, num_attributes_range, num_features_range, limit
        )
        return datasets
    except Exception as e:
        print(f"Error fetching datasets: {e}")
        return []

def create_placeholder_figure():
    fig = go.Figure(data=[
        go.Scatter(x=[1, 2, 3], y=[4, 1, 2], mode='markers', marker=dict(color='LightSkyBlue'), name='Placeholder Data'),
        go.Scatter(x=[1, 2, 3], y=[2, 3, 1], mode='markers', marker=dict(color='Violet'), name='Placeholder Data 2')
    ])
    fig.update_layout(title='Placeholder Figure', xaxis_title='X Axis', yaxis_title='Y Axis')
    return fig

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
        Output('statistics_figure', 'style'),
        Output('progress_bar', 'value'),
        Output('progress_bar', 'style')
    ],
    [Input('search_button', 'n_clicks')],
    [State('date_range', 'start_date'),
     State('date_range', 'end_date'),
     State('number_of_attributes_slider1', 'value'),
     State('number_of_attributes_slider2', 'value'),
     State('limit_input', 'value'),
     State('interval-component', 'n_intervals')]
)
def update_dataset_list_and_statistics(n_clicks, start_date, end_date, num_attributes_range, num_features_range, limit, n_intervals):
    # Initialize variables at the start of the function
    list_group_items = []
    statistics_figure = go.Figure()  # Default empty figure
    statistics_style = {'display': 'none'}  # Default style
    progress_value = 0  # Default progress value
    progress_style = {'visibility': 'hidden'}  # Default progress style

    # Check if the button was clicked
    if n_clicks is None:
        return list_group_items, statistics_figure, statistics_style, progress_value, progress_style

    datasets = fetch_datasets(start_date, end_date, num_attributes_range, num_features_range, limit)

    list_group_items = []
    for idx, dataset in enumerate(datasets, start=1):
        dataset_name = dataset[1]
        rows, columns = int(dataset[2]), int(dataset[3])
        data_dimensions = f"{rows}x{columns}"

        list_group_item = dbc.ListGroupItem(
            [
                html.Div(
                    [
                        html.H5(dataset_name, className="mb-1"),
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

        statistics_style = {'display': 'block'} if datasets else {'display': 'none'}

        progress_value = int((idx / limit) * 100) if limit > 0 else 100

        progress_style = {'visibility': 'visible'} if n_intervals < 1 else {'visibility': 'hidden'}

    return list_group_items, statistics_figure, statistics_style, progress_value, progress_style

@staticmethod
def list_datasets(output_format='dataframe'):
    return openml.datasets.list_datasets(output_format=output_format)

def get_dataset(self, dataset_id, preferred_format='csv', save_directory='.'):
    try:
        openml_dataset = openml.datasets.get_dataset(dataset_id)
        name = openml_dataset.name

        X, y, _, _ = openml_dataset.get_data(dataset_format="dataframe")

        # Überprüfen, ob das bevorzugte Format verfügbar ist
        if preferred_format == 'csv':
            dataset_file_extension = 'csv'
            dataset_file_path = os.path.join(save_directory, f"{name}.{dataset_file_extension}")
            print(dataset_file_path)
            #X.to_csv(dataset_file_path, index=False, encoding='utf-8')
        else:
            # Fallback auf ein anderes Format, z.B. ARFF oder andere
            # Hier können Sie Ihre Fallback-Logik hinzufügen
            pass

        return dataset_file_path
    except Exception as e:
        self.logger.error(f"Fehler beim Abrufen des Datensatzes {dataset_id}: {e}")
        raise

def filter_datasets_by_attribute_types(self, start_date=None, end_date=None, num_attributes_range=None,
                                           num_features_range=None, limit=None):
    """
                Filters datasets based on upload dates and number of features.

                :param start_date: Minimum upload date for the datasets.
                :param end_date: Maximum upload date for the datasets.
                :param num_features_range: Tuple or list with two elements specifying the range of number of features.
                :param limit: Maximum number of datasets to return.
                :return: A list of filtered datasets.
            """

    datasets_list = self.list_datasets()
    dataset_ids = datasets_list['did'].tolist()
    filtered_datasets = []

    if start_date and end_date and start_date > end_date:
        raise ValueError("Startdatum muss vor dem Enddatum liegen.")

    for dataset_id in dataset_ids:
        if limit is not None and limit <= 0:
            break

        try:
            dataset = self.get_dataset(dataset_id)
            dataset_date = dataset.upload_date
            num_columns, num_rows = self.dimension(dataset_id)

            if ((not start_date or start_date <= dataset_date) and
                    (not end_date or end_date >= dataset_date) and
                    (not num_features_range or num_features_range[0] <= num_columns <= num_features_range[1])):

                filtered_datasets.append((dataset_id, dataset.name, num_rows, num_columns))
                OpenML_API.get_dataset()

                if limit is not None:
                    limit -= 1

        except Exception as e:
            print(f"Fehler bei der Verarbeitung des Datensatzes {dataset_id}: {e}")

    return filtered_datasets

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

app.layout = dbc.Container([
    dbc.Card([
        dbc.CardHeader("Filter"),
        dbc.CardBody([
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
            ]),
            dbc.Button('Suchen', id='search_button', color="primary", className="mt-3"),
            dbc.Progress(id='progress_bar', value=0, style={"height": "20px", "margin-top": "15px"}, striped=True),
        ])
    ]),
    dcc.Graph(id='statistics_figure', style={'display': 'none'}),
    dbc.ListGroup(id='list_group', flush=True, className="mt-4"),
    dcc.Interval(id='interval-component', interval=100, n_intervals=0, disabled=True),
], fluid=True)

if __name__ == '__main__':
    app.run_server(debug=False)
