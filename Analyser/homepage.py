import dash
from dash import html
import subprocess
import openml

# Erstellen Sie eine neue Dash-App für die Homepage
homepage_app = dash.Dash(__name__)

# Definieren Sie das Layout der Homepage
homepage_app.layout = html.Div([
    html.H1("Willkommen zur Datenanalyse"),
    html.Button("Starte das Dashboard", id="start-dashboard", n_clicks=0)
])

# Fügen Sie einen Callback hinzu, der das Dashboard startet
@homepage_app.callback(
    dash.dependencies.Output('start-dashboard', 'children'),
    [dash.dependencies.Input('start-dashboard', 'n_clicks')]
)
def start_dashboard(n_clicks):
    if n_clicks > 0:
        # Starte das Dashboard als separaten Prozess
        subprocess.Popen(["python", "./Data_Science_API_OpenML/Analyser/dashboard_test.py"])  # Stellen Sie sicher, dass der Skriptname korrekt ist
        return "Dashboard wird gestartet..."
    return "Starte das Dashboard"

if __name__ == "__main__":
    homepage_app.run_server(debug=False, port=8051)
