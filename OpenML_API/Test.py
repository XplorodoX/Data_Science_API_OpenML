import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import math

# Beispiel-Daten
df = pd.DataFrame({'Nummer': range(1, 101)})  # 100 Beispieldatensätze

app = dash.Dash(__name__)

# Anzahl der Elemente pro Seite
ITEMS_PER_PAGE = 10

app.layout = html.Div([
    html.Div(id='table-container'),  # Container für die anzuzeigenden Daten, zunächst leer
    html.Button('Daten anzeigen', id='show-data', n_clicks=0),
    html.Button('Zurück', id='previous-page', n_clicks=0),
    html.Span(id='current-page', children="1"),  # Anzeige der aktuellen Seite
    html.Button('Vor', id='next-page', n_clicks=0),
])


@app.callback(
    Output('table-container', 'children'),
    [Input('show-data', 'n_clicks'), Input('previous-page', 'n_clicks'), Input('next-page', 'n_clicks')],
    [State('current-page', 'children')]
)
def update_table(show_data_clicks, prev_clicks, next_clicks, current_page):
    # Überprüfung, ob der Button zum Anzeigen der Daten geklickt wurde
    if show_data_clicks == 0:
        # Wenn nicht, leere Antwort zurückgeben
        return ""

    # Umwandlung der aktuellen Seite in eine Zahl
    current_page = int(current_page)

    # Ermittlung des ausgelösten Buttons
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'next-page' in changed_id and show_data_clicks > 0:
        current_page = min(current_page + 1, math.ceil(len(df) / ITEMS_PER_PAGE))
    elif 'previous-page' in changed_id and show_data_clicks > 0:
        current_page = max(current_page - 1, 1)

    # Aktualisierung der Tabelle basierend auf der aktuellen Seite
    start = (current_page - 1) * ITEMS_PER_PAGE
    end = start + ITEMS_PER_PAGE
    filtered_df = df.iloc[start:end]
    return [html.Table(
        # Header
        [html.Tr([html.Th(col) for col in filtered_df.columns])] +

        # Body
        [html.Tr([
            html.Td(filtered_df.iloc[i][col]) for col in filtered_df.columns
        ]) for i in range(min(len(filtered_df), ITEMS_PER_PAGE))]
    ), html.Div(f"Seite {current_page} von {math.ceil(len(df) / ITEMS_PER_PAGE)}")]


@app.callback(
    Output('current-page', 'children'),
    [Input('show-data', 'n_clicks'), Input('previous-page', 'n_clicks'), Input('next-page', 'n_clicks')],
    [State('current-page', 'children')]
)
def update_page_number(show_data_clicks, prev_clicks, next_clicks, current_page):
    if show_data_clicks == 0:
        # Wenn der Button zum Anzeigen der Daten noch nicht geklickt wurde, keine Aktion
        return current_page

    # Logik zur Aktualisierung der Seitennummer
    current_page = int(current_page)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'next-page' in changed_id:
        current_page = min(current_page + 1, math.ceil(len(df) / ITEMS_PER_PAGE))
    elif 'previous-page' in changed_id:
        current_page = max(current_page - 1, 1)

    return str(current_page)


if __name__ == '__main__':
    app.run_server(debug=True)
