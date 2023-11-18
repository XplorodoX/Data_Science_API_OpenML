from dash import Dash, html, Input, Output, callback

app = Dash(__name__)

app.layout = html.Div(
    [
        html.Div(
            "Div with n_clicks event listener",
            id="click-div",
            style={"color": "red", "font-weight": "bold"},
        ),
        html.P(id="click-output"),
    ]
)


@callback(
    Output("click-output", "children"),
    Input("click-div", "n_clicks")
    )
def click_counter(n_clicks):
    return f"The html.Div above has been clicked this many times: {n_clicks}"


app.run(debug=True)