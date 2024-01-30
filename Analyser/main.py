import openml
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import numpy as np
import os

app = dash.Dash(__name__)
# https://www.openml.org/search?type=data&status=active&id=53
# heart-statlog
dataset_id = 53 

def download_dataset(dataset_id):
    try:
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
        df = pd.DataFrame(X, columns=attribute_names)
        if y is not None:
            df['target'] = y

        # Name des Datensatzes für die Dateinamenbildung verwenden
        dataset_name = dataset.name
        filename = f"{dataset_name.replace(' ', '_')}.csv"

        # Speichern im spezifizierten Verzeichnis
        
        current_path = os.getcwd()
        # print("TEST PRINT CURRENT PATH ", current_path)
        file_path = os.path.join('Data_Science_API_OpenML\Downloads', filename)
        # print("TEST PRINT FILE PATH ", file_path)
        df.to_csv(file_path, index=False)

        return df
    except Exception as e:
        print(f"Fehler beim Herunterladen des Datensatzes: {e}")
        return None


def analyze_dataset(df):
    if df is None:
        return "Keine Daten zum Analysieren vorhanden.", []

    stats_output = ""
    graphs = []
    
    # Statistische Zusammenfassung
    # summary = df.describe()


    # Mittelwert, Minimum, Maximum berechnen
    for column in df.columns:
        if df[column].dtype in [np.int64, np.float64]:
            mean_value = df[column].mean()
            min_value = df[column].min()
            max_value = df[column].max()
            stats_output += f"Statistiken für {column}: Mittelwert: {mean_value}, Minimum: {min_value}, Maximum: {max_value}\n"

            # Plotly-Grafik erstellen
            fig = px.histogram(df, x=column, title=f"Verteilung von {column}")
            graphs.append(fig)

    return stats_output, graphs

initial_df = download_dataset(dataset_id)
initial_stats, initial_graphs = analyze_dataset(initial_df)

app.layout = html.Div([
    dcc.Input(id='dataset-id-input', type='text', value=str(dataset_id), placeholder='Dataset ID eingeben'),
    html.Button('Analyse starten', id='analyze-button'),
    dcc.Graph(id='data-visualization', figure=initial_graphs[0] if initial_graphs else {}),
    html.Div(id='stats-output', children=initial_stats)
])

@app.callback(
    [dash.dependencies.Output('data-visualization', 'figure'),
     dash.dependencies.Output('stats-output', 'children')],
    [dash.dependencies.Input('analyze-button', 'n_clicks')],
    [dash.dependencies.State('dataset-id-input', 'value')]
)
def update_output(n_clicks, dataset_id):
    if n_clicks is None or not dataset_id.isdigit():
        return {}, 'Bitte geben Sie eine gültige Dataset-ID ein.'

    df = download_dataset(int(dataset_id))
    if df is None:
        return {}, 'Fehler beim Herunterladen des Datensatzes.'

    stats, graphs = analyze_dataset(df)
    return graphs[0] if graphs else {}, stats

if __name__ == '__main__':
    app.run_server(debug=True)
