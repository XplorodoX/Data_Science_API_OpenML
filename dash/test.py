# Importiere benötigte Bibliotheken
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Lies die Daten
df = pd.DataFrame({
    "X": [1, 2, 3, 4],
    "Y": [10, 11, 12, 13]
})

# Erstelle eine Dash-Instanz
app = dash.Dash(__name__)

# Definiere das Layout der Anwendung
app.layout = html.Div([
    html.H1("Beispiel Dash Anwendung"),
    dcc.Graph(
        id='beispiel-graph',
        figure={
            'data': [
                {'x': df['X'], 'y': df['Y'], 'type': 'bar', 'name': 'SF'},
            ],
            'layout': {
                'title': 'Beispiel für einen Plotly-Graphen'
            }
        }
    )
])

# Starte den Server
if __name__ == '__main__':
    app.run_server(debug=True)
