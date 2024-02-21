import dash
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import math

# Dash-App initialisieren mit externen Stylesheets für Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

datasets_info = [
    {"name": "Dataset 1", "num_instances": 100, "dimensions": "10x10"},
    {"name": "Dataset 2", "num_instances": 300, "dimensions": "20x10"},
    {"name": "Dataset 3", "num_instances": 120, "dimensions": "15x10"},
    {"name": "Dataset 4", "num_instances": 450, "dimensions": "30x15"},
    {"name": "Dataset 5", "num_instances": 200, "dimensions": "25x10"},
    {"name": "Dataset 6", "num_instances": 150, "dimensions": "12x12"},
    {"name": "Dataset 7", "num_instances": 180, "dimensions": "18x18"},
    {"name": "Dataset 8", "num_instances": 220, "dimensions": "22x11"},
    {"name": "Dataset 9", "num_instances": 250, "dimensions": "25x15"},
    {"name": "Dataset 10", "num_instances": 275, "dimensions": "27x17"},
    {"name": "Dataset 11", "num_instances": 125, "dimensions": "20x12"},
    {"name": "Dataset 12", "num_instances": 300, "dimensions": "30x20"},
    {"name": "Dataset 13", "num_instances": 320, "dimensions": "32x16"},
    {"name": "Dataset 14", "num_instances": 340, "dimensions": "34x17"},
    {"name": "Dataset 15", "num_instances": 360, "dimensions": "36x18"},
    {"name": "Dataset 16", "num_instances": 380, "dimensions": "38x19"},
    {"name": "Dataset 17", "num_instances": 400, "dimensions": "40x20"},
    {"name": "Dataset 18", "num_instances": 420, "dimensions": "42x21"},
    {"name": "Dataset 19", "num_instances": 440, "dimensions": "44x22"},
    {"name": "Dataset 20", "num_instances": 460, "dimensions": "46x23"},
    {"name": "Dataset 21", "num_instances": 480, "dimensions": "48x24"},
    {"name": "Dataset 22", "num_instances": 500, "dimensions": "50x25"},
    {"name": "Dataset 23", "num_instances": 520, "dimensions": "52x26"},
    {"name": "Dataset 24", "num_instances": 540, "dimensions": "54x27"},
    {"name": "Dataset 25", "num_instances": 560, "dimensions": "56x28"},
    {"name": "Dataset 26", "num_instances": 580, "dimensions": "58x29"},
    {"name": "Dataset 27", "num_instances": 600, "dimensions": "60x30"},
    {"name": "Dataset 28", "num_instances": 620, "dimensions": "62x31"},
    {"name": "Dataset 29", "num_instances": 640, "dimensions": "64x32"},
    {"name": "Dataset 30", "num_instances": 660, "dimensions": "66x33"},
]

# Anzahl der Elemente pro Seite
ITEMS_PER_PAGE = 10

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
    Output({"type": "collapse", "index": MATCH}, "is_open"),
    [Input({"type": "toggle", "index": MATCH}, "n_clicks")],
    [State({"type": "collapse", "index": MATCH}, "is_open")],
    prevent_initial_call=True
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# Sichtbarkeit der Navigationselemente
@app.callback(
    [Output('previous-page', 'style'),
     Output('next-page', 'style'),
     Output('current-page', 'style')],
    [Input('show-data', 'n_clicks')]
)
def toggle_navigation_elements_visibility(n_clicks_show_data):
    if n_clicks_show_data > 0:
        # Mache die Buttons und die Schrift sichtbar
        return {'display': 'inline-block'}, {'display': 'inline-block'}, {'display': 'block'}
    else:
        # Halte die Buttons und die Schrift versteckt
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}


# Callbacks für Fortschrittsanzeige und Abbrechen-Funktion
@app.callback(
    [Output('progress-bar', 'value'), Output('progress-bar', 'label'), Output('interval-component', 'disabled')],
    [Input('start-button', 'n_clicks'), Input('cancel-button', 'n_clicks'), Input('interval-component', 'n_intervals')],
    [State('progress-bar', 'value')]
)
def update_progress_and_control_interval(start_clicks, cancel_clicks, n_intervals, progress_value):
    # Logik zur Aktualisierung der Fortschrittsanzeige und Steuerung des Intervals
    return progress_value, f"{progress_value}%", True

# Weitere Callbacks für Listendarstellung, Paginierung usw.
@app.callback(
    # Outputs, Inputs und States für weitere Funktionalitäten
)
def additional_logic():
    # Implementierung zusätzlicher Logiken
    return

@callback(
    Output('current-page', 'children'),
    [Input('show-data', 'n_clicks'), Input('previous-page', 'n_clicks'), Input('next-page', 'n_clicks')],
    [State('current-page', 'children')]
)
def update_page_number(show_data_clicks, prev_clicks, next_clicks, current_page):
    # Logik identisch zum vorigen Callback, aktualisiert nur die Seitennummer
    if show_data_clicks == 0:
        return "Seite 1 von {}".format(math.ceil(len(datasets_info) / ITEMS_PER_PAGE))

    current_page = int(current_page.split()[1])  # Aktuelle Seite aus dem String extrahieren
    total_pages = math.ceil(len(datasets_info) / ITEMS_PER_PAGE)

    changed_id = dash.callback_context.triggered[0]['prop_id']
    if 'next-page' in changed_id:
        current_page = min(current_page + 1, total_pages)
    elif 'previous-page' in changed_id:
        current_page = max(current_page - 1, 1)

    return "Seite {} von {}".format(current_page, total_pages)

# App-Layout
app.layout = html.Div([
    dbc.Button("Starte lange Aufgabe", id="start-button", className="me-2", n_clicks=0),
    dbc.Button("Abbrechen", id="cancel-button", color="danger", className="me-2", n_clicks=0),
    dcc.Interval(id='interval-component', interval=1*1000, n_intervals=0, disabled=True),
    dbc.Progress(id="progress-bar", value=0, striped=True, animated=True, label="0%", style={"width": "100%"}),
    html.Div(id='list-container', className="list-container mt-4"),
    html.Div([
        dbc.Button('<-', id='previous-page', n_clicks=0, className="mr-2", style={'display': 'none'}),
        html.Span(id='current-page', children="Seite 1 von 5", className="mx-2", style={'display': 'none'}),
        dbc.Button('->', id='next-page', n_clicks=0, className="ml-2", style={'display': 'none'}),
    ], className="d-flex justify-content-center align-items-center mt-4"),
])


if __name__ == '__main__':
    app.run_server(debug=True)
