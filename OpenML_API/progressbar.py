import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import threading
import time
from queue import Queue

# Der Rest Ihres Codes bleibt unverändert

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

progress_queue = Queue()

app.layout = html.Div([
    dbc.Button("Starte lange Aufgabe", id="start-button", className="me-2", n_clicks=0),
    dbc.Button("Abbrechen", id="cancel-button", color="danger", className="me-2", n_clicks=0),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
    dbc.Progress(id="progress-bar", value=0, striped=True, animated=True, label="0%", style={"width": "100%"}),
    html.H5(id="remaining-time", children=""),
    dcc.Store(id='task-status', data={'running': False, 'cancelled': False}),
])

def long_task(progress_queue):
    for i in range(10):
        time.sleep(2)  # Simuliere Arbeit
        if not progress_queue.empty() and progress_queue.get() == 'cancel':
            progress_queue.put(0)  # Setze Fortschritt auf 0% bei Abbruch
            return
        progress_queue.put((i + 1) * 10)
    progress_queue.put(100)  # Signalisiert das Ende der Aufgabe

@app.callback(
    Output('interval-component', 'disabled'),
    Input('start-button', 'n_clicks'),
    State('task-status', 'data'),
    prevent_initial_call=True,
)
def start_long_task(n_clicks, data):
    if n_clicks > 0:
        if not data['running']:
            progress_queue.queue.clear()
            threading.Thread(target=long_task, args=(progress_queue,), daemon=True).start()
            return False  # Aktiviere das Interval
        else:
            return True  # Deaktiviere das Interval, falls die Aufgabe bereits läuft
    return True

@app.callback(
    Output('progress-bar', 'value'),
    Output('progress-bar', 'label'),
    Output('task-status', 'data'),
    Input('interval-component', 'n_intervals'),
    Input('cancel-button', 'n_clicks'),
    State('task-status', 'data'),
)
def update_progress(n_intervals, cancel_clicks, data):
    ctx = dash.callback_context

    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'cancel-button.n_clicks':
        progress_queue.queue.clear()
        progress_queue.put('cancel')  # Sendet Abbruchsignal
        return 0, "0%", {'running': False, 'cancelled': True}

    if not progress_queue.empty():
        progress = progress_queue.get()
        if progress == 100 or data['cancelled']:
            return progress, f"{progress}%", {'running': False, 'cancelled': False}
        return progress, f"{progress}%", {'running': True, 'cancelled': False}

    return dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)
