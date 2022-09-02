import dash
import dash_daq as daq
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import sqlite3
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from ast import literal_eval

app = dash.Dash(__name__,
                title='Symbolic DNN-Tuner',
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
                )

# Load extra layouts
cyto.load_extra_layouts()

database = "../database/database.db"

app.layout = html.Div(
    [
        ################
        ###  Header  ###
        ################

        html.Div(
            [
                dbc.Row([
                    dbc.Col(
                        html.Img(src=app.get_asset_url('DNN-Tuner.png'), style={'height': '58px'}),
                        width=3,
                        style={"marginLeft": "12px"},
                        className="py-2"),
                ])
            ], className='mx-5 mt-2 header-style'
        ),

        # refresh button #
        dbc.Button(html.I(id='tooltip-target', n_clicks=0, style={'height': '20px'},
                          className='fas fa-sync-alt'),
                   color="dark",
                   id='refresh-val',
                   n_clicks=0,
                   className='refresh-btn btn btn-outline-success'),

        html.Div([
            dbc.Tabs(
                [
                    ###########################
                    #   Network Architecture  #
                    ###########################

                    dbc.Tab(html.Div(id='iframe-netron'),
                            label="Network architecture", tab_id="network_arch", className="mt-2"),

                    ###############
                    ##  Metrics  ##
                    ###############

                    dbc.Tab(html.Div([
                        dcc.Interval(
                            id='graph-component',
                            interval=1 * 60000,
                            n_intervals=0,
                        ),

                        dbc.Row([
                            # graph of Training Vs Accuracy
                            dbc.Col(dcc.Graph(id='graph-TA'), width=6, className="col-12 col-lg-6"),
                            # graph of Training Vs Loss
                            dbc.Col(dcc.Graph(id='graph-TL'), width=6, className="col-12 col-lg-6")
                        ], className='graphs'),

                        dbc.Row([
                            # graph of Validation Vs Accuracy
                            dbc.Col(dcc.Graph(id='graph-VA'), width=6, className="col-12 col-lg-6"),
                            # graph of Validation Vs Loss
                            dbc.Col(dcc.Graph(id='graph-VL'), width=6, className="col-12 col-lg-6")
                        ], className='graphs')

                    ], className="justify-content-center mt-2"),
                        label="Metrics",
                        tab_id="metrics"),

                    ############################
                    #  Probabilistics weights  #
                    ############################

                    dbc.Tab(html.Div([
                        # update all weights every 'interval' milliseconds
                        dcc.Interval(
                            id='weights-component',
                            interval=1 * 60000,
                            n_intervals=0,
                        ),

                        dbc.Row([
                            dbc.Col([

                                html.Div([
                                    html.H5("Overfitting"),
                                ], className='mx-3 mt-2 mb-3 header-prob-weights'),

                                dbc.Row([
                                    dbc.Col(daq.Gauge(id='inc_dropout',
                                                      color="#00bc8c",
                                                      showCurrentValue=True,
                                                      units="%",
                                                      label='Increment dropout',
                                                      labelPosition="bottom",
                                                      max=100,
                                                      min=0,
                                                      size=260,
                                                      ), width=3, className="col-12 col-sm weights-height"),

                                    dbc.Col(daq.Gauge(id='data_augmentation',
                                                      color="#00bc8c",
                                                      showCurrentValue=True,
                                                      units="%",
                                                      label='Data augmentation',
                                                      labelPosition="bottom",
                                                      max=100,
                                                      min=0,
                                                      size=260
                                                      ), width=3, className="col-12 col-sm weights-height"),

                                ], className="weights mt-2"),
                            ], width=6, className="col-12 col-sm-12 col-lg-12 col-xl-6"),

                            dbc.Col([

                                html.Div([
                                    html.H5("Floating Loss"),
                                ], className='mx-3 mt-2 mb-3 header-prob-weights'),

                                dbc.Row([

                                    dbc.Col(daq.Gauge(id='inc_batch_size',
                                                      color="#00bc8c",
                                                      showCurrentValue=True,
                                                      units="%",
                                                      label='Increment Batch size',
                                                      labelPosition="bottom",
                                                      max=100,
                                                      min=0,
                                                      size=260
                                                      ), width=3, className="col-12 col-sm weights-height"),

                                    dbc.Col(daq.Gauge(id='decr_lr_fl',
                                                      color="#00bc8c",
                                                      showCurrentValue=True,
                                                      units="%",
                                                      label='Decrement Learning',
                                                      labelPosition="bottom",
                                                      max=100,
                                                      min=0,
                                                      size=260
                                                      ), width=3, className="col-12 col-sm weights-height"),

                                ], className="weights mt-2"),
                            ], width=6, className="col-12 col-sm-12 col-lg-12 col-xl-6"),

                        ], className="weights"),

                        html.Div([
                            html.H5("Underfitting"),
                        ], className='mx-3 mt-2 mb-3 header-prob-weights'),

                        dbc.Row([

                            dbc.Col(daq.Gauge(id='decr_lr',
                                              color="#00bc8c",
                                              showCurrentValue=True,
                                              units="%",
                                              label='Decrement Learning rate',
                                              labelPosition="bottom",
                                              max=100,
                                              min=0,
                                              size=260
                                              ), width=3, className="col-12 col-sm weights-height"),
                            dbc.Col(daq.Gauge(id='inc_neurons',
                                              color="#00bc8c",
                                              showCurrentValue=True,
                                              units="%",
                                              label='Increment Neurons',
                                              labelPosition="bottom",
                                              max=100,
                                              min=0,
                                              size=260
                                              ), width=3, className="col-12 col-sm weights-height"),
                            dbc.Col(daq.Gauge(id='new_fc_layer',
                                              color="#00bc8c",
                                              showCurrentValue=True,
                                              units="%",
                                              label='New Fully connected layers',
                                              labelPosition="bottom",
                                              max=100,
                                              min=0,
                                              size=260
                                              ), width=3, className="col-12 col-sm weights-height"),
                            dbc.Col(daq.Gauge(id='new_conv_layer',
                                              color="#00bc8c",
                                              showCurrentValue=True,
                                              units="%",
                                              label='New Convolutional layers',
                                              labelPosition="bottom",
                                              max=100,
                                              min=0,
                                              size=260
                                              ), width=3, className="col-12 col-sm weights-height")

                        ], className="weights"),

                    ], className="mt-2 pt-2", ),
                        label="Probabilistic weights", tab_id="prob_weights"),

                    #####################
                    #  Hyperparameters  #
                    #####################

                    dbc.Tab(html.Div([

                        dbc.Row([
                            # update cytoscape every 'interval' milliseconds
                            dcc.Interval(
                                id='hyp-cyto',
                                interval=1 * 60000,
                                n_intervals=0,
                            ),
                            cyto.Cytoscape(
                                id='cytoscape',
                                style={'width': '100%', 'height': '100%'},
                                layout={'name': 'cola', 'edgeLength': 200},
                            )

                        ], style={"height": "780px"}, className="py-2", justify="center"),
                    ], className="mt-2", ),
                        label="Hyper-parameters",
                        tab_id="hyper_param"
                    ),

                ], className="mt-3", style={"backgroundColor": "231F1F"},
                id="tabs",
                active_tab="network_arch",
            )], className="mx-5"),
        html.Div(id="tab-content", className="p-4"),
    ]
)


############################
# Functions
############################

def get_id(db_conn):
    """ Get the last id iteration from the database """
    cur = db_conn.cursor()
    cur.execute("SELECT COUNT(id) FROM iteration")
    n = cur.fetchone()
    n = int(n[0])

    return n


def l2dict(row):
    """
    List to dictionary transform a <class 'list'> with a single element to a <class 'dict'>
    """
    to_tuple = row[0]
    to_string = to_tuple[0]
    to_dictionary = literal_eval(to_string)

    return to_dictionary


def format_string(target):
    """
    This function convert int or float to string but if the target is a <class 'float'>
    it removes all digits after the fifth digit after zero or if the target is < 10^-4 it removes all digits
    after the eight digit after zero

    Return a <class 'str'>
    """
    target = str(target)  # to compare the various cases it is necessary to convert to string
    try:  # case integer class
        target = int(target)
        return str(target)
    except ValueError:
        pass
    try:  # case float class
        new_val = float(target)
        if new_val < 10 ** -4:
            new_val = round(new_val, 8)
            return str(new_val)
        else:
            new_val = round(new_val, 5)
        return str(new_val)
    except ValueError:  # case string class
        return str(target)


############################
# Callbacks
############################

############## NETWORK ARCHITECTURE ################

@app.callback(Output('iframe-netron', 'children'),
              [Input('graph-component', 'n_intervals'),
               Input('refresh-val', 'n_clicks')])
def update_netron(value, n_clicks):
    if value is None or n_clicks is None:
        raise PreventUpdate
    # Attenzione, se si modifica la porta contenuta nell'iframe, e' necessario aggiornare il dato anche in launcher.py
    source = 'http://localhost:8080'
    width = '100%'
    height = '780px'
    id_val = 'value' + str(n_clicks)
    return html.Iframe(id=id_val, src=source, width=width, height=height)


############### HYPERPARAMETERS ###############

@app.callback(Output('cytoscape', 'elements'),
              [Input('hyp-cyto', 'n_intervals'),
               Input('refresh-val', 'n_clicks')])
def update_cytoscape(data, button):
    if data is None or button is None:
        raise PreventUpdate

    conn = sqlite3.connect(database)
    cur = conn.cursor()
    cur.execute("SELECT hps FROM hyperparameters NATURAL JOIN iteration")

    data = cur.fetchall()

    id_ = 1
    iteration = 1
    data_nodes = {}
    data_edges = {}

    for x in data:  # List to tuple
        for z in x:  # Tuple to string
            data_nodes[iteration] = "IT_{}".format(iteration)  # result: '#' : 'IT_#'
            temp = literal_eval(z)  # String to dictionary
            for key, value in temp.items():
                data_nodes["id_{}".format(id_)] = "{}: {}".format(key, format_string(value))
                data_edges["id_{}".format(id_)] = iteration  # Add source and target for edges
                id_ += 1
            iteration += 1

    for x in range(1, iteration - 1):  # Add source and target for nodes
        data_edges[x] = (x + 1)

    nodes = [
        {
            'data': {'id': iteration, 'label': label},
        }
        for iteration, label in data_nodes.items()
    ]

    edges = [
        {'data': {'source': source, 'target': target}}
        for source, target in data_edges.items()
    ]

    elements = nodes + edges

    return elements


@app.callback(Output('cytoscape', 'stylesheet'),
              [Input('hyp-cyto', 'n_intervals'),
               Input('refresh-val', 'n_clicks')])
def update_cytoscape(data, button):
    if data is None or button is None:
        raise PreventUpdate

    conn = sqlite3.connect(database)
    last_id = get_id(conn)

    default_stylesheet = [
        {
            'selector': '[source !*= "id"]',
            'style': {
                'mid-target-arrow-color': 'red',
                'mid-target-arrow-shape': 'triangle',
                'line-color': '#00bc8e',
                'arrow-scale': '3',
            },
        },
        {
            'selector': '[label *= "IT_"]',
            'style': {
                'width': 60,
                'height': 60,
                'label': 'data(label)',
                'color': '#00bc8e',
                'font-size': '25px',
                'font-weight': 'bold',
            },
        },
        {
            'selector': '[label = "IT_{}'.format(last_id) + '"]',
            'style': {
                'background-color': '#fc7703',
            },
        },
        {
            'selector': '[label !*= "IT"]',
            'style': {
                'label': 'data(label)',
                'color': '#00bc8e',
                'font-size': '16px',
                'font-weight': 'bold',
            }
        },
    ]

    return default_stylesheet


############### METRICS ################

# Training - Accuracy
@app.callback(Output('graph-TA', 'figure'),
              [Input("graph-component", "n_intervals"),
               Input('refresh-val', 'n_clicks')]
              )
def update_fig(data, button):
    if data is None or button is None:
        raise PreventUpdate

    conn = sqlite3.connect(database)
    id_iter = get_id(conn)
    cur = conn.cursor()

    cur.execute("SELECT accuracy FROM training NATURAL JOIN iteration WHERE id=?", (id_iter,))
    accuracy = cur.fetchall()

    for row in accuracy:
        tavalue = row[0].split(',')

    layout = go.Layout(
        xaxis={
            'title': 'Accuracy',
            'titlefont': {
                'family': 'Arial, Helvetica, sans-serif',
                'size': 18,
                'color': '#a1a0a0',
            },
            'color': '#a1a0a0',
        },
        yaxis={
            'title': 'Training',
            'titlefont': {
                'family': 'Arial, Helvetica, sans-serif',
                'size': 18,
                'color': '#a1a0a0',
            },
            'color': '#a1a0a0',
        },
        paper_bgcolor='#231f1f',
        plot_bgcolor='#231f1f',
    )

    return {'data': [go.Line(y=tavalue)],
            'layout': layout}


# Training - Loss
@app.callback(Output('graph-TL', 'figure'),
              [Input("graph-component", "n_intervals"),
               Input('refresh-val', 'n_clicks')]
              )
def update_fig(data, button):
    if data is None or button is None:
        raise PreventUpdate

    conn = sqlite3.connect(database)
    id_iter = get_id(conn)
    cur = conn.cursor()

    cur.execute("SELECT loss FROM training NATURAL JOIN iteration WHERE id=?", (id_iter,))
    loss = cur.fetchall()

    for row in loss:
        tlvalue = row[0].split(',')

    layout = go.Layout(
        xaxis={
            'title': 'Loss',
            'titlefont': {
                'family': 'Arial, Helvetica, sans-serif',
                'size': 18,
                'color': '#a1a0a0',
            },
            'color': '#a1a0a0',
        },
        yaxis={'title': 'Training',
               'titlefont': {
                   'family': 'Arial, Helvetica, sans-serif',
                   'size': 18,
                   'color': '#a1a0a0',
               },
               'color': '#a1a0a0'},
        paper_bgcolor='#231f1f',
        plot_bgcolor='#231f1f',
    )

    return {'data': [go.Line(y=tlvalue)],
            'layout': layout, }


# Validation - Accuracy
@app.callback(Output('graph-VA', 'figure'),
              [Input("graph-component", "n_intervals"),
               Input('refresh-val', 'n_clicks')]
              )
def update_fig(data, button):
    if data is None or button is None:
        raise PreventUpdate

    conn = sqlite3.connect(database)
    id_iter = get_id(conn)
    cur = conn.cursor()

    cur.execute("SELECT accuracy FROM validation NATURAL JOIN iteration WHERE id=?", (id_iter,))
    accuracy = cur.fetchall()

    for row in accuracy:
        vavalue = row[0].split(',')

    layout = go.Layout(
        xaxis={
            'title': 'Accuracy',
            'titlefont': {
                'family': 'Arial, Helvetica, sans-serif',
                'size': 18,
                'color': '#a1a0a0',
            },
            'color': '#a1a0a0',
        },
        yaxis={
            'title': 'Validation',
            'titlefont': {
                'family': 'Arial, Helvetica, sans-serif',
                'size': 18,
                'color': '#a1a0a0',
            },
            'color': '#a1a0a0',
        },
        paper_bgcolor='#231f1f',
        plot_bgcolor='#231f1f',
    )

    return {'data': [go.Line(y=vavalue)],
            'layout': layout, }


# Validation - Loss
@app.callback(Output('graph-VL', 'figure'),
              [Input("graph-component", "n_intervals"),
               Input('refresh-val', 'n_clicks')]
              )
def update_fig(data, button):
    if data is None or button is None:
        raise PreventUpdate

    conn = sqlite3.connect(database)
    id_iter = get_id(conn)
    cur = conn.cursor()

    cur.execute("SELECT loss FROM validation NATURAL JOIN iteration WHERE id=?", (id_iter,))
    loss = cur.fetchall()

    for row in loss:
        vlvalue = row[0].split(',')

    layout = go.Layout(
        xaxis={
            'title': 'Loss',
            'titlefont': {
                'family': 'Arial, Helvetica, sans-serif',
                'size': 18,
                'color': '#a1a0a0',
            },
            'color': '#a1a0a0',
        },
        yaxis={
            'title': 'Validation',
            'titlefont': {
                'family': 'Arial, Helvetica, sans-serif',
                'size': 18,
                'color': '#a1a0a0',
            },
            'color': '#a1a0a0',
        },
        paper_bgcolor='#231f1f',
        plot_bgcolor='#231f1f',
    )

    return {'data': [go.Line(y=vlvalue)],
            'layout': layout, }


############## PROBABILITY WEIGHTS ################

# Increment Dropout
@app.callback(Output('inc_dropout', 'value'),
              [Input("weights-component", "n_intervals"),
               Input('refresh-val', 'n_clicks')]
              )
def update_fig(data, button):
    if data is None or button is None:
        raise PreventUpdate

    conn = sqlite3.connect(database)
    id_iter = get_id(conn)
    cur = conn.cursor()
    cur.execute("SELECT weights FROM weights NATURAL JOIN iteration WHERE id=?", (id_iter,))

    row = cur.fetchall()
    weights = l2dict(row)

    return float(weights['inc_dropout']) * 100


# Data augmentation
@app.callback(Output('data_augmentation', 'value'),
              [Input("weights-component", "n_intervals"),
               Input('refresh-val', 'n_clicks')]
              )
def update_fig(data, button):
    if data is None or button is None:
        raise PreventUpdate

    conn = sqlite3.connect(database)
    id_iter = get_id(conn)
    cur = conn.cursor()
    cur.execute("SELECT weights FROM weights NATURAL JOIN iteration WHERE id=?", (id_iter,))

    row = cur.fetchall()
    weights = l2dict(row)

    return float(weights['data_augmentation']) * 100


# Decrement Learining rate
@app.callback(Output('decr_lr', 'value'),
              [Input("weights-component", "n_intervals"),
               Input('refresh-val', 'n_clicks')]
              )
def update_fig(data, button):
    if data is None or button is None:
        raise PreventUpdate

    conn = sqlite3.connect(database)
    id_iter = get_id(conn)
    cur = conn.cursor()
    cur.execute("SELECT weights FROM weights NATURAL JOIN iteration WHERE id=?", (id_iter,))

    row = cur.fetchall()
    weights = l2dict(row)

    return float(weights['decr_lr']) * 100


# Increment Neurons
@app.callback(Output('inc_neurons', 'value'),
              [Input("weights-component", "n_intervals"),
               Input('refresh-val', 'n_clicks')]
              )
def update_fig(data, button):
    if data is None or button is None:
        raise PreventUpdate

    conn = sqlite3.connect(database)
    id_iter = get_id(conn)
    cur = conn.cursor()
    cur.execute("SELECT weights FROM weights NATURAL JOIN iteration WHERE id=?", (id_iter,))

    row = cur.fetchall()
    weights = l2dict(row)

    return float(weights['inc_neurons']) * 100


# New Fully connected layers
@app.callback(Output('new_fc_layer', 'value'),
              [Input("weights-component", "n_intervals"),
               Input('refresh-val', 'n_clicks')]
              )
def update_fig(data, button):
    if data is None or button is None:
        raise PreventUpdate

    conn = sqlite3.connect(database)
    id_iter = get_id(conn)
    cur = conn.cursor()
    cur.execute("SELECT weights FROM weights NATURAL JOIN iteration WHERE id=?", (id_iter,))

    row = cur.fetchall()
    weights = l2dict(row)

    return float(weights['new_fc_layer']) * 100


# New Convolutional layers
@app.callback(Output('new_conv_layer', 'value'),
              [Input("weights-component", "n_intervals"),
               Input('refresh-val', 'n_clicks')]
              )
def update_fig(data, button):
    if data is None or button is None:
        raise PreventUpdate

    conn = sqlite3.connect(database)
    id_iter = get_id(conn)
    cur = conn.cursor()
    cur.execute("SELECT weights FROM weights NATURAL JOIN iteration WHERE id=?", (id_iter,))

    row = cur.fetchall()
    weights = l2dict(row)

    return float(weights['new_conv_layer']) * 100


# Increment Batch size
@app.callback(Output('inc_batch_size', 'value'),
              [Input("weights-component", "n_intervals"),
               Input('refresh-val', 'n_clicks')]
              )
def update_fig(data, button):
    if data is None or button is None:
        raise PreventUpdate

    conn = sqlite3.connect(database)
    id_iter = get_id(conn)
    cur = conn.cursor()
    cur.execute("SELECT weights FROM weights NATURAL JOIN iteration WHERE id=?", (id_iter,))

    row = cur.fetchall()
    weights = l2dict(row)

    return float(weights['inc_batch_size']) * 100


# Decrement Learning
@app.callback(Output('decr_lr_fl', 'value'),
              [Input("weights-component", "n_intervals"),
               Input('refresh-val', 'n_clicks')]
              )
def update_fig(data, button):
    if data is None or button is None:
        raise PreventUpdate

    conn = sqlite3.connect(database)
    id_iter = get_id(conn)
    cur = conn.cursor()
    cur.execute("SELECT weights FROM weights NATURAL JOIN iteration WHERE id=?", (id_iter,))

    row = cur.fetchall()
    weights = l2dict(row)

    return float(weights['decr_lr_fl']) * 100


############################
# Run app
############################

if __name__ == "__main__":
    app.run_server(debug=False, port=8082)
