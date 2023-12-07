import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import json

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Funktion zum Abrufen von Datensätzen
def fetch_datasets():
    # Hier können Sie Ihre Logik zum Abrufen von Datensätzen implementieren.
    # Zum Beispiel könnten Sie eine Liste von Datensatz-IDs von OpenML abrufen.
    # Für dieses Beispiel verwenden wir eine einfache statische Liste.
    return ["Dataset 1", "Dataset 2", "Dataset 3"]

datasets = fetch_datasets()

# Funktion zum Erstellen eines Diagramms für ein gegebenes Dataset
def create_figure_for_dataset(dataset_id):
    # Erstellen Sie hier ein Plotly-Diagramm basierend auf dem übergebenen Datensatz.
    # Für dieses Beispiel verwenden wir ein einfaches statisches Diagramm.
    df = px.data.gapminder().query("country=='Canada'")
    fig = px.line(df, x="year", y="gdpPercap", title=f"GDP Per Capita over Time for {dataset_id}")
    return fig

list_group_items = []
for idx, dataset in enumerate(datasets, start=1):
    list_group_item = dbc.ListGroupItem(
        [
            html.Div(
                [
                    html.H5(f"{dataset}", className="mb-1"),
                    html.Small("Click to view details", className="text-secondary"),
                ],
                className="d-flex w-100 justify-content-between",
                id={"type": "toggle", "index": idx},
                style={"cursor": "pointer"},
            )
        ]
    )
    collapse = dbc.Collapse(
        dbc.Card(dbc.CardBody([dcc.Graph(figure=create_figure_for_dataset(dataset))])),
        id={"type": "collapse", "index": idx},
    )
    list_group_items.append(list_group_item)
    list_group_items.append(collapse)

app.layout = html.Div([
    dbc.ListGroup(list_group_items)
])

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

if __name__ == "__main__":
    app.run_server(debug=True)
