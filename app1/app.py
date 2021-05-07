import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import itertools
from copy import deepcopy
from algo import THSimulationAdv

n_bandits = 10

thsim = THSimulationAdv(nb_bandits=n_bandits)

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

white_button_style = {'background-color': 'white', 'color': 'black'}
green_button_style = {'background-color': 'green', 'color': 'white'}
red_button_style = {'background-color': 'red', 'color': 'white'}

def create_row(nb=1, wd=1, pb='0.5'):
    return dbc.Row(children=[
        dbc.Col(dbc.Input(id='bandit{}_prob'.format(str(nb)), type="number", min=0, max=1, 
                          step=0.01, value=pb), width=wd),
        dbc.Col(dbc.Card(html.Div(id='bandit{}_hits'.format(str(nb))), color="success"),width=wd),
        dbc.Col(dbc.Card(html.Div(id='bandit{}_miss'.format(str(nb))), color="danger"),width=wd),
        dbc.Col(dbc.Card(html.Div(id='bandit{}_total'.format(str(nb))), color="light"),width=wd),
    ], align="center", justify="start")

def create_table():
    row_list = [create_row(nb=i) for i in range(1,n_bandits+1)]
    return html.Div(row_list)

app.layout = html.Div(children=[
    dbc.Button("Start Simulation", color="primary"),
    create_table(),
    dcc.Interval(
            id='interval-component',
            interval=1000, # in milliseconds
            n_intervals=0
        ),
    html.Div(id='p_bandits'),
])

p_bandits = [np.random.rand() for i in range(n_bandits)]
last_update = thsim.step(p_bandits)

input_list = [eval(f"Input('bandit{i}_prob', 'value')") for i in range(1,n_bandits+1)]

@app.callback(
    Output('p_bandits', 'children'),
    input_list)
def update_probs(*args):
    global p_bandits
    p_bandits = [float(prob) for prob in args] 
    return ""

output_list_hits = [eval(f"Output('bandit{i}_hits', 'children')") for i in range(1,n_bandits+1)]
output_list_miss = [eval(f"Output('bandit{i}_miss', 'children')") for i in range(1,n_bandits+1)]
output_list_total = [eval(f"Output('bandit{i}_total', 'children')") for i in range(1,n_bandits+1)]
output_list = list(itertools.chain(output_list_hits,
                                   output_list_miss,
                                   output_list_total)
                  )

@app.callback(
    output_list,
    Input('interval-component', 'n_intervals'))
def update_metrics(n):
    x = thsim.step(p_bandits)
    totals = x[0]
    hits = x[1]
    global last_update
    hitlist=[]; misslist=[]; totallist=[]
    for i in range(n_bandits):
        hit_style = green_button_style if hits[i]!=last_update[1][i] else white_button_style
        miss_style = red_button_style if (totals[i]-hits[i])!=(last_update[0][i]-last_update[1][i]) else white_button_style
        hitlist.append(html.Div(hits[i], style=hit_style))
        misslist.append(html.Div(totals[i]-hits[i], style=miss_style))
        totallist.append(totals[i])
    last_update = deepcopy(x)
    return list(itertools.chain(hitlist,misslist,totallist))

if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8051)