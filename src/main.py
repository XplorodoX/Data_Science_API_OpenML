# Libraries Imports
import math
import os
from pathlib import Path
from dash import dash_table, callback_context
import numpy as np
import pandas as pd
import plotly.express as px
import openml
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objs as go
from datetime import datetime, timedelta, date
from dash.exceptions import PreventUpdate
import Helper as helper
import json
import dash

# Set the Cache Directory
openml.config.set_root_cache_directory('cache') # <- Set the cache directory

# Dash app setup
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://use.fontawesome.com/releases/v5.8.1/css/all.css'],
                suppress_callback_exceptions=True)

# Set global Variables
ITEMS_PER_PAGE = 10  # <- Max number of items per page
filtered_data = []  # All filtered data
initial_df = pd.DataFrame()  # Initialize df here to ensure it's always defined
datasets_length = 0  # Length of the datasets

# (FM)
class DatasetMetrics:
    def __init__(self):
        # Initialize maximum values
        self.max_number_of_instances = 0
        self.max_number_of_features = 0
        self.max_number_of_numeric_features = 0
        self.max_number_of_symbolic_features = 0

    def update_max_values(self, ranges):
        """Update the maximum values based on the provided ranges."""
        # Update maximum values for each metric
        self.max_number_of_instances = max(self.max_number_of_instances, ranges['NumberOfInstances'][1])
        self.max_number_of_features = max(self.max_number_of_features, ranges['NumberOfFeatures'][1])
        self.max_number_of_numeric_features = max(self.max_number_of_numeric_features,
                                                  ranges['NumberOfNumericFeatures'][1])
        self.max_number_of_symbolic_features = max(self.max_number_of_symbolic_features,
                                                   ranges['NumberOfSymbolicFeatures'][1])

metrics = DatasetMetrics()
datasets = helper.fetchDataList()  # Fetch data lists

# Update metrics if datasets are not empty
if not datasets.empty:
    numericalRange = helper.calcRangeDatasets(datasets)
    metrics.update_max_values(numericalRange)

# Convert maximum values to integers and use them directly from the metrics object
maxi_features = int(metrics.max_number_of_features)
max_numeric_features = int(metrics.max_number_of_numeric_features)
max_categorical_features = int(metrics.max_number_of_symbolic_features)
max_instances = int(metrics.max_number_of_instances)
maxDataset = len(datasets)

def create_statistics_figure(filtered_info):
    """
    Create a statistics figure based on the filtered dataset information. (DS)

    Args:
        filtered_info (list): A list of dictionaries containing information about filtered datasets.
            Each dictionary should contain keys like 'name', 'numeric_features', and 'categorical_features'.

    Returns:
        plotly.graph_objs._figure.Figure: A Plotly figure displaying statistics of numeric and categorical features
            for each dataset on the current page.
    """
    # Check if filtered_info is empty
    if not filtered_info:
        return go.Figure()  # Returns an empty figure if no data is available

    # Extract actual dataset names
    dataset_names = [dataset['name'] for dataset in filtered_info]
    # Extract counts of numeric and categorical features
    numeric_feature_counts = [dataset['numeric_features'] for dataset in filtered_info]
    categorical_feature_counts = [dataset['categorical_features'] for dataset in filtered_info]

    # Create the figure and add data for numeric and categorical features
    fig = go.Figure(data=[
        go.Bar(name='Numeric Features', x=dataset_names, y=numeric_feature_counts),
        go.Bar(name='Categorical Features', x=dataset_names, y=categorical_feature_counts)
    ])
    # Update the layout of the figure
    fig.update_layout(
        title='Statistics of the datasets on the current page',
        xaxis_title='Datasets',
        yaxis_title='Number of features',
        barmode='group'  # Group bars to enable comparison
    )
    return fig

@app.callback(
    [
        Output('list_group', 'children'),
        Output('statistics_figure', 'figure'),
        Output('statistics_figure', 'style'),
        Output('error-modal', 'is_open'),  # Output for opening/closing the modal
        Output('error-invalid-range', 'children'),
        Output('dataset_length_store', 'data')
    ],
    [Input('search_button', 'n_clicks'), Input('previous-page', 'n_clicks'), Input('next-page', 'n_clicks')],
    [
        State('date_range', 'start_date'),
        State('date_range', 'end_date'),
        State('min_data_points', 'value'),
        State('max_data_points', 'value'),
        State('min_features', 'value'),
        State('max_features', 'value'),
        State('min_numerical_features', 'value'),
        State('max_numerical_features', 'value'),
        State('min_categorical_features', 'value'),
        State('max_categorical_features', 'value'),
        State('input_max_datasets', 'value'),
        State('current-page', 'children'),
        State('dataset_length_store', 'data')
    ]
)
def on_search_button_click(n_clicks, prev_clicks, next_clicks, start_date, end_date, min_data_points, max_data_points,
                           min_features, max_features, min_numerical_features, max_numerical_features,
                           min_categorical_features, max_categorical_features, limit, current_page_text, datasets_length):
    # (FM)
    global filtered_data

    list_group_items = []
    statistics_style = {'display': 'none'}

    if n_clicks is None:
        return list_group_items, go.Figure(), statistics_style, False, "", datasets_length

    # Create ranges from min and max values for validation
    features_range = (min_features, max_features)
    numerical_features_range = (min_numerical_features, max_numerical_features)
    categorical_features_range = (min_categorical_features, max_categorical_features)
    data_points_range = (min_data_points, max_data_points)

    # Example call to the modified check_input_ranges function
    valid, messages = check_input_ranges(
        ((min_features, max_features), 'Features range', maxi_features),
        ((min_numerical_features, max_numerical_features), 'Numerical Features range', max_numeric_features),
        ((min_categorical_features, max_categorical_features), 'Categorical Features range', max_categorical_features),
        ((min_data_points, max_data_points), 'Data Points range', max_instances),
        ((1, limit), 'Max Datasets', maxDataset)
    )

    # If the validation fails, return error message and open modal
    if not valid:
        error_message = '\n'.join(messages)  # Combine all error messages into a single string
        return [], go.Figure(), {'display': 'none'}, True, error_message, None  # Use the actual error message

    # If the inputs are valid, continue with the data processing
    filtered_data = processData(start_date, end_date, features_range, numerical_features_range,
                                categorical_features_range, data_points_range, limit)

    datasets_length = len(filtered_data) # Update the length of the datasets for pagination

    # Start of the callback or function
    # Make sure 'current_page_text' is not empty and has the expected format
    if current_page_text and len(current_page_text.split()) > 2:
        try:
            current_page = int(current_page_text.split()[1])
        except ValueError:  # Catch errors if conversion to int fails
            current_page = 1  # Set a default value if an error occurs
    else:
        current_page = 1  # Default value if 'current_page_text' does not match the expected format

    # Your code for handling page navigation
    changed_id = dash.callback_context.triggered[0]['prop_id']
    if 'next-page' in changed_id:
        current_page = min(current_page + 1, math.ceil(len(filtered_data) / ITEMS_PER_PAGE))
    elif 'previous-page' in changed_id:
        current_page = max(current_page - 1, 1)

    start = (current_page - 1) * ITEMS_PER_PAGE
    filtered_info = filtered_data[start:start + ITEMS_PER_PAGE]

    for idx, dataset in enumerate(filtered_info, start=start):
        global_index = idx + 1
        item_id = {"type": "dataset-click", "index": dataset['id']}  # Unique ID for each item

        list_group_item = dbc.ListGroupItem(
            html.Div([
                html.Div([
                    html.H5(f"{global_index}. {dataset['name']}", className="mb-1"),
                    html.Div([
                        html.Small(f"Dataset-ID: {dataset['id']}", className="text-secondary d-block"),
                        html.Small(f"Dimension: {int(dataset['features'])}×{int(dataset['instances'])}",
                                   className="text-secondary d-block"),
                        html.Small(
                            f"Categorical Features: {int(dataset.get('categorical_features', 0))}, Numeric Features: {int(dataset.get('numeric_features', 0))}",
                            className="text-secondary d-block"),
                        html.Small(f"Upload Date: {dataset['upload'].strftime('%Y-%m-%d')}", className="text-secondary d-block")
                    ], className="mt-2")
                ], style={'flex': '1'}),
            ], id=item_id, n_clicks=0, style={'cursor': 'pointer', 'text-decoration': 'none', 'color': 'inherit'}),
            style={
                "padding": "20px",
                "margin-bottom": "10px",
                "background-color": "#f8f9fa",
                "border": "1px solid #ddd",
                "border-radius": "5px",
                "box-shadow": "0 2px 2px rgba(0,0,0,0.1)"
            }
        )
        list_group_items.append(list_group_item)

    # Update statistics figure based on filtered data or other criteria
    statistics_figure = create_statistics_figure(filtered_info)
    statistics_style = {'display': 'block'}

    # Return the normal state if there are no errors
    return list_group_items, statistics_figure, statistics_style, False, "", datasets_length  # Keep the modal closed, no error message

@app.callback(
    Output('error-modal', 'is_open', allow_duplicate=True),
    [Input('close-error-modal', 'n_clicks')],
    [State('error-modal', 'is_open')], prevent_initial_call=True
)
def toggle_error_modal(n_clicks, is_open):
    if n_clicks:
        return False  # Close the modal
    return is_open  # Leave the state unchanged if the button hasn't been clicked


# Define a list of known categorical features if applicable
categorical_features = ['categorical_feature1', 'categorical_feature2']

@app.callback(
    Output('feature-histogram', 'figure'),
    [Input('feature-summary-table', 'active_cell'),
     State('feature-summary-table', 'data')],
    prevent_initial_call=True
)
def update_histogram(active_cell, table_data):
    """
    Callback function to update the feature histogram based on user selection. (DS)

    Args:
        active_cell (dict): The active cell selected in the feature summary table.
        table_data (list): Data of the feature summary table.

    Returns:
        dict: A dictionary containing the figure data and layout for the feature histogram.
    """
    # Check if active cell and table data are available
    if not active_cell or not table_data:
        raise PreventUpdate

    # Convert table data to a DataFrame
    df = pd.DataFrame(table_data)
    # Get the selected feature from the active cell
    selected_feature = df.iloc[active_cell['row']]['Feature']

    # Check if the selected feature exists in the initial DataFrame
    if selected_feature not in initial_df.columns:
        return {
            "data": [],
            "layout": {
                "title": f"Feature {selected_feature} not found in dataframe.",
                "xaxis": {"visible": False},
                "yaxis": {"visible": False}
            }
        }

    # Determine the type of figure to create based on the feature type
    if pd.api.types.is_numeric_dtype(initial_df[selected_feature]):
        fig_type = 'histogram'  # Numeric feature
    elif selected_feature in categorical_features:
        fig_type = 'bar'  # Categorical feature
    else:
        fig_type = 'bar'  # Treat as nominal feature if not numeric or categorical

    # Create the appropriate type of chart based on the feature type
    if fig_type == 'histogram':
        fig = px.histogram(initial_df, x=selected_feature, title=f'Histogram of {selected_feature}')
    elif fig_type == 'bar':
        fig = px.bar(initial_df, x=selected_feature, title=f'Distribution of {selected_feature}')

    # Optional: Add quantile lines for numeric features
    if fig_type == 'histogram':
        quantiles = initial_df[selected_feature].quantile([0.25, 0.5, 0.75, 0.97, 0.997]).to_dict()
        for quantile, value in quantiles.items():
            fig.add_vline(x=value, line_dash="solid", line_color="blue",
                          annotation_text=f"{quantile * 100}th: {value:.2f}", annotation_position="top right",
                          annotation=dict(font_size=10, font_color="green", showarrow=False))

    # Update hover information for traces
    fig.update_traces(hoverinfo='x+y', selector=dict(type='histogram' if fig_type == 'histogram' else 'bar'))
    return fig

def prepare_table_data_from_df(df):
    """
    Creates a list of dictionaries for the DataTable from a DataFrame.  (DS)

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        list: A list of dictionaries, where each dictionary represents a row in the DataTable.
    """
    # Create a list of dictionaries, where each dictionary represents a row
    table_data = [{"Feature": feature} for feature in df.columns]
    return table_data

@app.callback(
    [Output('detail-section', 'style'),  # Output for the detail section style
     Output('filter-section', 'style'),  # Output for the filter section style
     Output('list_histogram', 'children'),  # Output for the list histogram children
     Output('dataset-store', 'data')],  # Output for the dataset store data
    [Input({'type': 'dataset-click', 'index': ALL}, 'n_clicks'),  # Input trigger for dataset clicks
     Input('back-button', 'n_clicks')],  # Input trigger for back button click
    prevent_initial_call=True  # Prevent initial callback upon page load
)
def on_item_click(n_clicks, *args):
    """
    Callback function to handle clicks on dataset items and back button. (DS)

    Args:
        n_clicks (list): List of click counts for dataset items.
        *args: Variable number of additional arguments.

    Returns:
        tuple: A tuple containing outputs for detail section style, filter section style,
               list histogram children, and dataset store data.
    """
    # Get the callback context
    ctx = dash.callback_context

    # If no input triggered the callback, prevent update
    if not ctx.triggered:
        raise PreventUpdate

    # Determine which input triggered the callback
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Check if all n_clicks values are None or 0
    if all(click is None or click == 0 for click in n_clicks):
        raise PreventUpdate
    elif 'dataset-click' in button_id:
        # This is a click on a dataset item
        dataset_id = json.loads(button_id.split('.')[0])['index']  # Extract dataset ID

        # Download dataset and retrieve information
        global initial_df
        initial_df, dataset_info = download_dataset(dataset_id)
        completeness_graph = create_data_completeness_graph(initial_df)
        summary_records, columns = create_feature_summary_table(initial_df)

        # Define columns for the feature summary table
        columns = [
            {"name": "Feature", "id": "Feature"},
            {"name": "Count", "id": "count"},
            {"name": "Mean", "id": "mean"},
            {"name": "Std", "id": "std"},
            {"name": "Min", "id": "min"},
            {"name": "25%", "id": "25%"},
            {"name": "50%", "id": "50%"},
            {"name": "75%", "id": "75%"},
            {"name": "97%", "id": "97%"},
            {"name": "99.7%", "id": "99.7%"},
            {"name": "Max", "id": "max"},
            {"name": "Mode", "id": "mode"}
        ]

        # Define components for the detail view
        detail_components = [
            dbc.ListGroupItem([
                html.H4("Information of current dataset"),
                html.P(f"ID of Dataset: {dataset_id}"),
                html.P(f"Name of Dataset: {dataset_info.get('name', 'Not available')}"),
                html.P(f"Number of Features: {dataset_info.get('features_count', 'Not available')}"),
                html.P(f"Number of Instances: {dataset_info.get('instances_count', 'Not available')}"),
                html.P(f"Upload date: {dataset_info.get('upload_date', 'Not available')}"),
            ]),
            dbc.ListGroupItem([
                dcc.Graph(figure=completeness_graph)
            ]),
            dbc.ListGroupItem([
                dash_table.DataTable(
                    id='feature-summary-table',
                    columns=columns,  # Use the previously defined column headers
                    data=summary_records,  # Use the data generated by create_feature_summary_table
                    style_table={'overflowX': 'auto', 'height': '391px'},
                    style_cell={'textAlign': 'left', 'padding': '6px'},
                    style_header={'fontWeight': 'bold'},
                ),
                dcc.Graph(id='feature-histogram')
            ]),
        ]

        # Make the detail section visible and hide the filter section
        return {'display': 'block'}, {'display': 'none'}, detail_components, {'selected_dataset_id': dataset_id}

    # Hide detail view, show filter if back button clicked
    elif 'back-button' in button_id:
        return {'display': 'none'}, {'display': 'block'}, [], dash.no_update

@app.callback(
    [
        Output('current-page', 'children'),
        Output('current-page', 'style'),
        Output('previous-page', 'style'),
        Output('next-page', 'style'),
        Output('pagination-container', 'style')
    ],
    [
        Input('search_button', 'n_clicks'),
        Input('previous-page', 'n_clicks'),
        Input('next-page', 'n_clicks'),
        Input('interval-component_page', 'n_intervals')  # Neuer Input von Interval
    ],
    [
        State('current-page', 'children'),
        State('input_max_datasets', 'value'),
        State('dataset_length_store', 'data')
    ]
)
def update_page_number(search_clicks, prev_clicks, next_clicks, n_intervals, current_page_text, maxData, datasets_length):
    """
    Callback function to update the page number and pagination controls. (FM)

    Parameters:
        - search_clicks: Number of clicks on the search button.
        - prev_clicks: Number of clicks on the previous page button.
        - next_clicks: Number of clicks on the next page button.
        - current_page_text: Text indicating the current page number.
        - maxData: Maximum number of datasets.

    Returns:
        - current_page_text: Text indicating the current page number.
        - page_style: Style for displaying the current page number.
        - prev_button_style: Style for displaying the previous page button.
        - next_button_style: Style for displaying the next page button.
        - container_style: Style for displaying the pagination container.
    """

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'] if ctx.triggered else ''

    # Initialisieren Sie die Anzahl der Datensätze, falls nicht angegeben
    datasets_length = datasets_length or 0  # Angenommen, 0 als Standardwert, falls nichts eingegeben wird
    # Berechnen Sie die Gesamtzahl der Seiten basierend auf der Anzahl der Datensätze
    total_pages = math.ceil(datasets_length / ITEMS_PER_PAGE)

    # Determine the current page based on the triggered event
    if 'search_button' in triggered_id:
        current_page = 1  # Reset to the first page if search is triggered
    elif 'next-page' in triggered_id and search_clicks:
        current_page = min(int(current_page_text.split()[1]) + 1,
                           total_pages) if current_page_text and ' ' in current_page_text else 2
    elif 'previous-page' in triggered_id and search_clicks:
        current_page = max(int(current_page_text.split()[1]) - 1,
                           1) if current_page_text and ' ' in current_page_text else 1
    else:
        current_page = int(current_page_text.split()[1]) if current_page_text and ' ' in current_page_text else 1

    # Style and visibility settings based on the number of clicks
    container_style = {'display': 'flex'} if search_clicks else {'display': 'none'}
    page_style = {'visibility': 'visible', 'display': 'block'} if search_clicks else {'visibility': 'hidden',
                                                                                      'display': 'none'}
    prev_button_style = {'visibility': 'visible',
                         'display': 'inline-block'} if current_page > 1 and search_clicks else {'visibility': 'hidden',
                                                                                                'display': 'none'}
    next_button_style = {'visibility': 'visible',
                         'display': 'inline-block'} if current_page < total_pages and search_clicks else {
        'visibility': 'hidden', 'display': 'none'}
    page_number_text = f"Page {current_page} of {total_pages}" if search_clicks else ""

    return page_number_text, page_style, prev_button_style, next_button_style, container_style

# Convert string dates to datetime objects for comparison
def parse_date(date_string):
    if isinstance(date_string, datetime):
        # If date_string is already a datetime object, return it directly
        return date_string

    formats = ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]   # Keep your existing formats
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    raise ValueError(f"time data {date_string!r} does not match any of the expected formats: {formats}")


def check_input_ranges(*range_labels_max):
    """
    Checks if the input ranges are valid, do not exceed the specified maximum values,
    and are not None due to exceeding the predefined bounds.

    Parameters:
        - range_labels_max (tuple): Variable number of tuples, each representing a range in the form ((min, max), label, max_allowed).

    Returns:
        - valid (bool): True if all ranges are valid and within limits, False otherwise.
        - messages (list): List of error messages for invalid or exceeded ranges.
    """
    valid = True
    messages = []

    # Iterate through each range
    for index, (range_tuple, label, max_allowed) in enumerate(range_labels_max):
        # Check if the range is non-None and unpack it
        if range_tuple is not None:
            min_val, max_val = range_tuple

            # Check if min_val or max_val are None, which indicates they're out of the allowed range
            if min_val is None:
                valid = False
                messages.append(f"Error in {label}: Minimum value is missing or below the allowed range.")
            if max_val is None:
                valid = False
                messages.append(f"Error in {label}: Maximum value is missing or above the allowed maximum of {max_allowed}.")

            # If both values are not None, check if they are within the allowed range
            if min_val is not None and max_val is not None:
                if min_val > max_val:
                    valid = False
                    messages.append(f"Error in {label}: Minimum value ({min_val}) is greater than maximum value ({max_val}).")
                if max_val > max_allowed:
                    valid = False
                    messages.append(f"Error in {label}: Maximum value ({max_val}) exceeds the allowed maximum of {max_allowed}.")

    return valid, messages


def processData(start_date=None, end_date=None, features_range=None, numerical_features_range=None,
                categorical_features_range=None, data_points_range=None, limit=None):
    """
    Processes datasets based on specified filters. (FM)

    Parameters:
        - start_date (str): Start date for filtering datasets.
        - end_date (str): End date for filtering datasets.
        - features_range (tuple): Range of features count (min, max) for filtering datasets.
        - numerical_features_range (tuple): Range of numerical features count (min, max) for filtering datasets.
        - categorical_features_range (tuple): Range of categorical features count (min, max) for filtering datasets.
        - data_points_range (tuple): Range of data points count (min, max) for filtering datasets.
        - limit (int): Maximum number of datasets to be processed.

    Returns:
        - filtered_datasets (list): List of dictionaries containing filtered dataset information.
    """

    # Set a default limit if not provided
    if limit is None:
        limit = float('inf')

    # Get the list of dataset IDs
    dataset_ids = datasets['did'].tolist()

    # Initialize a counter for processed datasets
    count = 0

    # Initialize an empty list to store filtered datasets
    filtered_datasets = []

    # Iterate through each dataset ID
    for dataset_id in dataset_ids:

        # Check if the processing needs to be stopped
        if count >= limit:
            break

        # Retrieve the dataset information without downloading data, qualities, or features metadata
        dataset = openml.datasets.get_dataset(dataset_id, download_data=False, download_qualities=False,
                                                  download_features_meta_data=False)
        upload_date = dataset.upload_date

        # Assuming 'start_date' and 'end_date' are strings initially:
        start_date = parse_date(start_date) if start_date else None
        end_date = parse_date(end_date) if end_date else None
        upload_date = parse_date(upload_date) if upload_date else None

        # Inside your dataset loop
        if upload_date:  # This check is now for a datetime object
            # Ensure that upload_date is compared as datetime object
            if ((not start_date or start_date <= upload_date) and
                    (not end_date or upload_date <= end_date)):

                # Retrieve dataset information from the dataset list
                num_features = datasets.loc[datasets['did'] == dataset_id, 'NumberOfFeatures'].iloc[0]
                num_numeric_features = datasets.loc[datasets['did'] == dataset_id, 'NumberOfNumericFeatures'].iloc[0]
                num_categorical_features = datasets.loc[datasets['did'] == dataset_id, 'NumberOfSymbolicFeatures'].iloc[0]
                num_instances = datasets.loc[datasets['did'] == dataset_id, 'NumberOfInstances'].iloc[0]
                name = datasets.loc[datasets['did'] == dataset_id, 'name'].iloc[0]

                # Check if the dataset satisfies all filtering criteria
                if ((not features_range or (features_range[0] <= num_features <= features_range[1])) and
                        (not numerical_features_range or (
                                numerical_features_range[0] <= num_numeric_features <= numerical_features_range[1])) and
                        (not categorical_features_range or (
                                categorical_features_range[0] <= num_categorical_features <= categorical_features_range[
                            1])) and
                        (not data_points_range or (data_points_range[0] <= num_instances <= data_points_range[1]))):
                    # Append the dataset information to the filtered datasets list
                    filtered_datasets.append({
                        'id': dataset_id,
                        'name': name,
                        'instances': num_instances,
                        'features': num_features,
                        'numeric_features': num_numeric_features,
                        'categorical_features': num_categorical_features,
                        'upload': upload_date
                    })
                    count += 1
            else:
                continue

    return filtered_datasets

# Callback for toggling intervals based on search button clicks
@app.callback(
    Output('interval-component', 'disabled'),  # Output: Disabling the interval component
    [Input('search_button', 'n_clicks')],  # Input: Clicks on the search button
    [State('interval-component', 'disabled')]  # State: Current disabled state of the interval component
)
def toggle_interval(n_clicks, disabled):
    """
    Callback function to toggle the interval component based on search button clicks. (FM)

    Args:
        n_clicks (int): The number of clicks on the search button.
        disabled (bool): The current disabled state of the interval component.

    Returns:
        bool: True if the interval component should be disabled, False otherwise.
    """
    if n_clicks:
        return False  # Enable interval component if search button is clicked
    return True  # Disable interval component if search button is not clicked


# Callback function to update output for numerical features range
@app.callback(
    Output('output_numerical_features', 'children'),  # Output: Display for selected numerical features range
    [Input('range_numerical_features', 'value')]  # Input: Selected range of numerical features
)
def update_output(value):
    """
    Callback function to update the displayed range of selected numerical features. (FM)

    Args:
        value (tuple): The selected range of numerical features.

    Returns:
        str: A string indicating the selected range of numerical features.
    """
    return f"Selected range: {value[0]} to {value[1]}"


# Callback function to update output for total features range
@app.callback(
    Output('output_features', 'children'),  # Output: Display for selected total features range
    [Input('range_features', 'value')]  # Input: Selected range of total features
)
def update_output_features(value):
    """
    Callback function to update the displayed range of selected total features. (FM)

    Args:
        value (tuple): The selected range of total features.

    Returns:
        str: A string indicating the selected range of total features.
    """
    return f"Selected range: {value[0]} to {value[1]}"


# Callback function to update output for categorical features range
@app.callback(
    Output('output_categorical_features', 'children'),  # Output: Display for selected categorical features range
    [Input('range_categorical_features', 'value')]  # Input: Selected range of categorical features
)
def update_output_categorical_features(value):
    """
    Callback function to update the displayed range of selected categorical features. (FM)

    Args:
        value (tuple): The selected range of categorical features.

    Returns:
        str: A string indicating the selected range of categorical features.
    """
    return f"Selected range: {value[0]} to {value[1]}"


# Callback function to update output for data points range
@app.callback(
    Output('output_data_points', 'children'),  # Output: Display for selected data points range
    [Input('range_data_points', 'value')]  # Input: Selected range of data points
)
def update_output_data_points(value):
    """
    Callback function to update the displayed range of selected data points. (FM)

    Args:
        value (tuple): The selected range of data points.

    Returns:
        str: A string indicating the selected range of data points.
    """
    return f"Selected range: {value[0]} to {value[1]}"

# Check if the cache folder exists
def check_cache_folder_exists():
    """
    Checks if the cache folder exists. (FM)

    Returns:
        bool: True if the cache folder exists, False otherwise.
    """
    return os.path.exists('../cache')  # Modify the path to your cache folder as needed

@app.callback(
    [
        Output('progress_bar', 'value'),
        Output('progress_bar', 'label'),  # Label output added to show progress percentage
        Output('loading-section', 'style'),
        Output('filter-section', 'style', allow_duplicate=True),
        Output('cache-status-store', 'data')  # New Output component added for cache status
    ],
    [
        Input('progress_interval', 'n_intervals'),
    ],
    [
        State('loading-section', 'style'),  # State of the current loading section style
        State('filter-section', 'style'),  # State of the current filter section style
        State('cache-status-store', 'data')  # Current cache status
    ],
    prevent_initial_call=True
)
def update_progress(n, loading_section_style, filter_section_style, cache_status):
    """
    Updates the progress bar and cache status based on caching progress.

    Args:
        n (int): The number of interval updates.
        loading_section_style (dict): The current style of the loading section.
        filter_section_style (dict): The current style of the filter section.
        cache_status (dict): The current cache status.

    Returns:
        tuple: A tuple containing updated values for the progress bar, loading section style, filter section style,
        and cache status.
    """
    # Prepare and setup
    current_working_directory = Path(os.getcwd())
    cache_directory_path = current_working_directory / 'cache/org/openml/www/datasets'
    new_cache_status = cache_status.copy()
    dataset_ids = datasets['did'].tolist()  # Assumption: datasets is defined somewhere
    total_datasets = len(dataset_ids)
    processed_datasets = new_cache_status.get('processed', 0)
    cache_directory_count = sum(1 for _ in cache_directory_path.iterdir()) if cache_directory_path.is_dir() else 0
    tolerance = 60  # Define a tolerance range

    # Check cache conditions
    if cache_directory_count == 0 and processed_datasets == 0:  # Check if cache was originally empty
        new_cache_status['originally_empty'] = True
    if abs(cache_directory_count - total_datasets) > tolerance:
        new_cache_status['processing'] = True
    else:
        new_cache_status['processing'] = False

    # If cache was originally empty and is not being processed anymore
    if new_cache_status.get('originally_empty', False) and not new_cache_status.get('processing', False):
        new_cache_status['originally_empty'] = False  # Reset to avoid repeated triggers
        return 100, "100%", {'display': 'none'}, {'display': 'block'}, new_cache_status
    elif new_cache_status.get('processing', True):
        # Process datasets if necessary
        for dataset_id in dataset_ids[processed_datasets:processed_datasets + 5]:  # Process up to 5 datasets per call
            try:
                # Try to cache the dataset
                dataset = openml.datasets.get_dataset(dataset_id, download_data=False, download_qualities=False,
                                                      download_features_meta_data=False)
                processed_datasets += 1  # Successful processing
            except Exception as e:
                print(f"Error caching dataset {dataset_id}: {e}")
        new_cache_status['processed'] = processed_datasets
        progress = int((processed_datasets / total_datasets) * 100) if total_datasets > 0 else 100
        return progress, f"{progress}%", {'display': 'block'}, {'display': 'none'}, new_cache_status
    else:
        raise PreventUpdate  # No change in state

# Update your callback function
@app.callback(
    [
        Output('download-modal', 'is_open'),  # To open or close the modal
        Output('download-modal-body', 'children')  # To update the message in the modal
    ],
    [
        Input('download-button', 'n_clicks')
    ],
    [
        State('dataset-store', 'data')
    ],
    prevent_initial_call=True  # Prevents the modal from being displayed on the first load of the page
)

def download_set(n_clicks, store_data):
    """
    Callback function to handle dataset download. (FM)

    Args:
        n_clicks (int): Number of clicks on the download button.
        store_data (dict): Data stored in the dataset store.

    Returns:
        tuple: Tuple containing the open/close status of the modal and the message to display.
    """
    if n_clicks is None or store_data is None:
        raise dash.exceptions.PreventUpdate

    dataset_id = store_data.get('selected_dataset_id') if store_data else None
    if dataset_id is None:
        raise dash.exceptions.PreventUpdate

    folder_name = 'Downloaded_Dataset' # <- Define the folder name for the downloaded dataset
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    dataset = openml.datasets.get_dataset(dataset_id, download_data=False, download_qualities=False,
                                          download_features_meta_data=False)
    df, _, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    file_path = os.path.join(folder_name, f"dataset_{dataset_id}.csv")
    df.to_csv(file_path, index=False)

    # Return the new message for the modal and the command to open the modal
    return True, f'Dataset {dataset_id} has been saved as CSV: {file_path}'


# Callback to close the modal
@app.callback(
    Output('download-modal', 'is_open', allow_duplicate=True),
    [
        Input('close-modal', 'n_clicks')
    ],
    [
        State('download-modal', 'is_open')
    ],
    prevent_initial_call=True
)
def close_modal(n_clicks, is_open):
    """
    Callback function to close the modal. (FM)

    Args:
        n_clicks (int): Number of clicks on the close button.
        is_open (bool): Current open status of the modal.

    Returns:
        bool: New open status of the modal.
    """
    if n_clicks:
        return False
    return is_open

# (FM)
modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Download Complete")),
        dbc.ModalBody("Your dataset has been successfully downloaded.", id='download-modal-body'),  # Add the ID here
        dbc.ModalFooter(
            dbc.Button("Close", id="close-modal", className="ms-auto", n_clicks=0)
        ),
    ],
    id="download-modal",
    is_open=False,  # This ensures the modal is not shown initially
)

# (FM)
error_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Error: Invalid Input")),
        dbc.ModalBody("", id='error-invalid-range'),  # Leave the body empty; the content will be updated dynamically.
        dbc.ModalFooter(
            dbc.Button("Close", id="close-error-modal", className="ms-auto", n_clicks=0)
        ),
    ],
    id="error-modal",
    is_open=False,
    centered=True,
    keyboard=True,
    backdrop="static",
)

def download_dataset(dataset_id=None):
    """
    Downloads a dataset based on the provided dataset_id. (DS)

    Args:
        dataset_id (int): The ID of the dataset to download.

    Returns:
        df (DataFrame): DataFrame containing the dataset.
        dataset_info (dict): Information about the downloaded dataset.
    """
    dataset_info = {}
    if dataset_id:
        try:
            dataset = openml.datasets.get_dataset(dataset_id, download_data=False, download_qualities=False,
                                                  download_features_meta_data=False)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute,
                                                                            dataset_format='dataframe')
            df = pd.DataFrame(X, columns=attribute_names)
            if y is not None:
                df['target'] = y

            try:
                upload_date_obj = datetime.strptime(dataset.upload_date, '%Y-%m-%d')
                formatted_date = upload_date_obj.strftime('%Y-%m-%d')
            except ValueError:
                formatted_date = dataset.upload_date

            dataset_info = {
                'name': dataset.name,
                'features_count': len(attribute_names),
                'instances_count': df.shape[0],
                'upload_date': formatted_date
            }
        except Exception as e:
            print(f"Error downloading the data set: {e}")
    else:
        print("No dataset_id specified. Attempts to load test file...")

    return df, dataset_info

def create_data_completeness_graph(df):
    """
    Creates a donut chart to visualize the completeness of the provided DataFrame, 
    including a hover feature to show missing features and total missing data fields. (DS)

    Args:
        df (DataFrame): DataFrame containing the dataset.

    Returns:
        fig (plotly.graph_objs.Figure): Plotly figure representing the data completeness.
    """
    if df.empty:
        return go.Figure()  # Returns an empty figure if df is empty

    total_values = np.prod(df.shape)
    missing_values = df.isnull().sum().sum()
    complete_values = total_values - missing_values

    # Identifying missing features and their missing value count
    missing_info = df.isnull().sum()
    missing_features = missing_info[missing_info > 0]
    missing_features_info = "<br>".join([f"{feature}: {missing} missing" 
                                         for feature, missing in missing_features.items()])

    # Adding total missing data fields information to the hover text
    missing_features_summary = f"Total missing data fields: {missing_values}<br>{missing_features_info}"

    # Configuring hover text for the 'Missing data fields' segment
    hover_text = [None, missing_features_summary]  # No hover info for 'Complete data'

    fig = go.Figure(data=[go.Pie(labels=['Complete data', 'Missing data fields'],
                                 values=[complete_values, missing_values], hole=.6,
                                 hoverinfo='label+percent+text', hovertext=hover_text)])
    fig.update_layout(title_text="Completeness of the dataset", title_x=0.5)

    return fig



def format_number(value):
    """
    Formats a number with up to four decimal places, but removes trailing zeros. (DS)

    Args:
        value: The number to format.

    Returns:
        str: The formatted number.
    """
    try:
        float_value = float(value)
        if float_value.is_integer():
            return str(int(float_value))
        else:
            return f"{float_value:.4f}".rstrip('0').rstrip('.')
    except ValueError:
        return value


def create_feature_summary_table(df):
    """
    Creates a summary table containing descriptive statistics for the provided DataFrame. (DS)

    Args:
        df (DataFrame): DataFrame containing the dataset.

    Returns:
        summary_records (list of dict), columns (list of dict): Summary records and columns for DataTable.
    """
    if df.empty:
        return [], []  # Returns empty values if df is empty
    
    # Convert sparse columns to dense
    for col in df.columns:
        if isinstance(df[col].dtype, pd.SparseDtype):
            df[col] = df[col].sparse.to_dense()
            
    # Calculate descriptive statistics
    summary = df.describe(percentiles=[.25, .5, .75, .97, .997], include='all').transpose()

    # Calculate mode for each column
    modes = df.mode().iloc[0]
    summary['mode'] = [modes[col] if col in modes else "N/A" for col in df.columns]

    # Format numerical values
    for col in summary.columns:
        summary[col] = summary[col].apply(format_number)

    # Adjust column names for display in DataTable
    summary.reset_index(inplace=True)
    summary.rename(columns={'index': 'Feature'}, inplace=True)

    summary_records = summary.to_dict('records')

    # Define columns for the DataTable
    columns = [{"name": col, "id": col} for col in summary.columns]

    return summary_records, columns

# App Layout (FM)
app.layout = dbc.Container([
    # Loading section displayed at app start
    html.Div(
        id='loading-section',
        style={'display': 'None'},  # Initially visible
        children=[
            dbc.Row(
                dbc.Col(
                    [
                        html.H4("Data is loading, please wait...", className="text-center mb-3"),
                        dbc.Progress(id='progress_bar', value=0, striped=True, animated=True, label="0%",
                                     style={"height": "30px"}),
                        html.P("The data must be pre-cached, which can take up to an 1 houre. Thank you for your patience.",
                               className="text-center mt-3"),
                        dcc.Store(id='cache-status-store', data={'processed': 0, 'processing': False}),
                        dcc.Interval(id='progress_interval', interval=500, n_intervals=0)
                        # Timer set to tick every second
                    ],
                    width={"size": 10, "offset": 1},
                    style={'max-width': '800px', 'margin': 'auto'}
                )
            )
        ],
        className="mt-5"
    ),
    html.Div(id='detail-section', style={'display': 'none'}, children=[
        dbc.ListGroup(id='list_histogram', flush=True, className="mt-4"),

        html.Button("Back", id='back-button', className="btn btn-primary mt-3 me-2", style={
            'boxShadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)',
            'transition': '0.3s',
            'border': 'none'
        }),

        html.Button("Download", id='download-button', className="btn btn-success mt-3", style={
            'boxShadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)',
            'transition': '0.3s',
            'border': 'none'
        }),

        modal,
        dcc.Store(id='dataset-store', storage_type='session'),
    ]),
    # Filter section for filtering data
    html.Div(id='filter-section', style={'display': 'Block'}, children=[
        dbc.Card([
            dbc.CardHeader("Filter"),
            dbc.CardBody([
                dbc.Row([
                    # Filter for upload date
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Upload-Date"),
                            dbc.CardBody([
                                dcc.DatePickerRange(
                                    id='date_range',
                                    start_date=datetime.now() - timedelta(days=3600),
                                    end_date=datetime.now(),
                                    min_date_allowed=datetime(2000, 1, 1),
                                    max_date_allowed=datetime.now(),
                                    display_format='DD.MM.YYYY',
                                    initial_visible_month=datetime.now()
                                ),
                            ]),
                        ]),
                    ], md=5),
                    # Filter for number of data points
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Number of Data Points"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(
                                        dbc.InputGroup([
                                            dbc.InputGroupText("Min"),
                                            dbc.Input(id='min_data_points', type='number', value=0, min=0,
                                                      max=max_instances),
                                        ]),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dbc.InputGroup([
                                            dbc.InputGroupText("Max"),
                                            dbc.Input(id='max_data_points', type='number', value=12345, min=0,
                                                      max=max_instances),
                                        ]),
                                        width=6,
                                    ),
                                ]),
                                dbc.Tooltip(f"Previous max range was 0 to {max_instances}.", target="max_data_points"),
                                html.Div(
                                    id='output_data_points',
                                    style={
                                        'margin-top': '10px',
                                        'text-align': 'center',
                                        'font-weight': 'bold',
                                        'color': '#007bff'
                                    }
                                )
                            ]),
                        ]),
                    ], md=5),
                    # Filter for maximum datasets
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Max Datasets"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(
                                        dbc.Input(
                                            id='input_max_datasets',
                                            type='number',
                                            min=0,
                                            max=maxDataset,
                                            step=1,
                                            value=20
                                        ),
                                        width=10,
                                    ),
                                ]),
                                dbc.Tooltip(f"Previous max range was 0 to {maxDataset}.", target="input_max_datasets"),
                            ]),
                        ]),
                    ], md=2),
                ], className="mb-4"),
                # Filter for number of features
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Number of Features"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(
                                        dbc.InputGroup([
                                            dbc.InputGroupText("Min"),
                                            dbc.Input(id='min_features', type='number', value=0, min=0,
                                                      max=maxi_features),
                                        ]),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dbc.InputGroup([
                                            dbc.InputGroupText("Max"),
                                            dbc.Input(id='max_features', type='number', value=50, min=0,
                                                      max=maxi_features),
                                        ]),
                                        width=6,
                                    ),
                                ]),
                                dbc.Tooltip(f"Previous max range was 0 to {maxi_features}.", target="max_features"),
                                html.Div(
                                    id='output_features',
                                    style={
                                        'margin-top': '10px',
                                        'text-align': 'center',
                                        'font-weight': 'bold',
                                        'color': '#007bff'
                                    }
                                )
                            ]),
                        ]),
                    ], md=4),
                    # Filter for number of numerical features
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Number of Numerical Features"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(
                                        dbc.InputGroup([
                                            dbc.InputGroupText("Min"),
                                            dbc.Input(id='min_numerical_features', type='number', value=0, min=0,
                                                      max=max_numeric_features),
                                        ]),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dbc.InputGroup([
                                            dbc.InputGroupText("Max"),
                                            dbc.Input(id='max_numerical_features', type='number', value=30, min=0,
                                                      max=max_numeric_features),
                                        ]),
                                        width=6,
                                    ),
                                ]),
                                dbc.Tooltip(f"Previous max range was 0 to {max_numeric_features}.",
                                            target="max_numerical_features"),
                                html.Div(
                                    id='output_numerical_features',
                                    style={
                                        'margin-top': '10px',
                                        'text-align': 'center',
                                        'font-weight': 'bold',
                                        'color': '#007bff'
                                    }
                                )
                            ]),
                        ]),
                    ], md=4),
                    # Filter for number of categorical features
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Number of Categorical Features"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(
                                        dbc.InputGroup([
                                            dbc.InputGroupText("Min"),
                                            dbc.Input(id='min_categorical_features', type='number', value=0, min=0,
                                                      max=max_categorical_features),
                                        ]),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dbc.InputGroup([
                                            dbc.InputGroupText("Max"),
                                            dbc.Input(id='max_categorical_features', type='number', value=20, min=0,
                                                      max=max_categorical_features),
                                        ]),
                                        width=6,
                                    ),
                                ]),
                                dbc.Tooltip(f"Previous max range was 0 to {max_categorical_features}.",
                                            target="max_categorical_features"),
                                html.Div(
                                    id='output_categorical_features',
                                    style={
                                        'margin-top': '10px',
                                        'text-align': 'center',
                                        'font-weight': 'bold',
                                        'color': '#007bff'
                                    }
                                )
                            ]),
                        ]),
                    ], md=4),
                ], className="mb-4"),
                # Search button
                dbc.Row([
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(dbc.Button('Search', id='search_button', color="primary", className="mt-3 mb-3",
                                               style={'width': '100%'}))
                        ])
                    ], md=12),
                ]),
            ])
        ]),
        # Spinner component to indicate data processing
        dbc.Spinner(
            children=[
                dcc.Graph(id='statistics_figure', style={'display': 'none'}),
                dbc.ListGroup(id='list_group', flush=True, className="mt-4")
            ],
            size="lg",
            color="primary",
            type="border",
            fullscreen=False,
        ),
        error_modal,
        dcc.Interval(id='interval-component', interval=100, n_intervals=0, disabled=True),
        html.Div(id='list-container', className="list-container mt-4"),
        dcc.Store(
            id='dataset_length_store',
            data=0
        ),

        # Pagination buttons
        html.Div([
            dcc.Interval(
                id='interval-component_page',  # Die ID, die im Callback verwendet wird
                interval=5*100,          # in Millisekunden (5*1000ms = 5s)
                n_intervals=0
            ),
            dbc.Button(
                html.Span(className="fas fa-chevron-left"),
                id='previous-page',
                n_clicks=0,
                className="mr-2 btn btn-lg",
                style={
                    'visibility': 'hidden',
                    'backgroundColor': '#78909C',
                    'color': 'white',
                    'borderRadius': '20px',
                    'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)'
                }
            ),
            html.Span(
                id='current-page',
                children="",
                className="px-3",
                style={'fontSize': '20px'}
            ),
            dbc.Button(
                html.Span(className="fas fa-chevron-right"),
                id='next-page',
                n_clicks=0,
                className="ml-2 btn btn-lg",
                style={
                    'visibility': 'hidden',
                    'backgroundColor': '#78909C',
                    'color': 'white',
                    'borderRadius': '20px',
                    'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)'
                }
            )
        ], className="d-flex justify-content-center align-items-center mt-4", id='pagination-container'),
    ]),
], fluid=True)

# Run Server
if __name__ == '__main__':
    app.run_server(threaded=True)