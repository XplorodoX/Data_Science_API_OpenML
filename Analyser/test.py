import openml
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
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
        print("TEST PRINT CURRENT PATH ", current_path)
        file_path = os.path.join('Data_Science_API_OpenML\Downloads', filename)
        print("TEST PRINT FILE PATH ", file_path)
        df.to_csv(file_path, index=False)

        return df
    except Exception as e:
        print(f"Fehler beim Herunterladen des Datensatzes: {e}")
        return None

def create_summary_graphs(df):
    graphs = []
    for column in df.select_dtypes(include=[np.number]).columns:
        data = [
            go.Bar(
                x=['Mittelwert', 'Minimum', 'Maximum'],
                y=[df[column].mean(), df[column].min(), df[column].max()],
                name=column
            )
        ]
        layout = go.Layout(title=f'Statistiken für {column}')
        fig = go.Figure(data=data, layout=layout)
        graphs.append(fig)
    return graphs

def create_boxplots(df):
    boxplots = []
    for column in df.select_dtypes(include=[np.number]).columns:
        fig = px.box(df, y=column, title=f'Boxplot für {column}')
        boxplots.append(fig)
    return boxplots


def analyze_dataset(df):
    if df is None:
        return []

    bar_charts = create_summary_graphs(df)
    box_plots = create_boxplots(df)
    
    # Kombiniere alle erstellten Graphen
    all_graphs = bar_charts + box_plots

    return all_graphs

initial_df = download_dataset(dataset_id)
# initial_stats, initial_graphs = analyze_dataset(initial_df)
initial_graphs = analyze_dataset(initial_df)

app.layout = html.Div([
    dcc.Input(id='dataset-id-input', type='text', value=str(dataset_id), placeholder='Dataset ID eingeben'),
    html.Button('Analyse starten', id='analyze-button'),
    html.Div([dcc.Graph(figure=fig) for fig in initial_graphs])
])

@app.callback(
    [dash.dependencies.Output('data-visualization-container', 'children')],
    [dash.dependencies.Input('analyze-button', 'n_clicks')],
    [dash.dependencies.State('dataset-id-input', 'value')]
)
def update_output(n_clicks, dataset_id):
    if n_clicks is None or not dataset_id.isdigit():
        return [html.P('Bitte geben Sie eine gültige Dataset-ID ein.')]

    df = download_dataset(int(dataset_id))
    if df is None:
        return [html.P('Fehler beim Herunterladen des Datensatzes.')]

    graphs = analyze_dataset(df)
    return [dcc.Graph(figure=fig) for fig in graphs]

if __name__ == '__main__':
    app.run_server(debug=True)
