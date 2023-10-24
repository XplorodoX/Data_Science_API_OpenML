import openml as oml
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# OpenML-Daten abrufen
def get_datasets(min_data_points=None, max_data_points=None):
    datasets_list = oml.datasets.list_datasets()
    df = pd.DataFrame(datasets_list).transpose()
    
    if min_data_points:
        df = df[df['NumberOfInstances'] >= min_data_points]
    if max_data_points:
        df = df[df['NumberOfInstances'] <= max_data_points]

    return df

# Dash App erstellen
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Label("Minimale Datenpunkte:"),
    dcc.Input(id="min-data-points", type="number"),
    html.Label("Maximale Datenpunkte:"),
    dcc.Input(id="max-data-points", type="number"),
    html.Button('Suchen', id='search-button'),
    dcc.Graph(id='results-graph')
])

@app.callback(
    Output('results-graph', 'figure'),
    [Input('search-button', 'n_clicks')],
    [dash.dependencies.State('min-data-points', 'value'),
     dash.dependencies.State('max-data-points', 'value')]
)
def update_graph(n_clicks, min_data_points, max_data_points):
    df = get_datasets(min_data_points, max_data_points)
    
    # Hier können Sie die Daten weiter analysieren und visualisieren
    # Zum Beispiel: df['NumberOfInstances'].mean()

    return {
        'data': [{
            'x': df['name'],
            'y': df['NumberOfInstances'],
            'type': 'bar'
        }],
        'layout': {
            'title': 'Anzahl der Datenpunkte pro Datensatz'
        }
    }

if __name__ == '__main__':
    app.run_server(debug=True)
