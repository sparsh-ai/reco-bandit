import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
from copy import deepcopy
from vowpalwabbit import pyvw
from algo import VWCSimulation
import numpy as np
import pandas as pd
import itertools
import pathlib
import dash_daq as daq
import plotly.graph_objs as go

from itertools import product

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

def generate_input_cards(preference='Random'):
    card_content = [
    dbc.CardImg(src="assets/img.jpg", top=True),
    dbc.CardBody([html.P(preference, className="card-title")])
    ]
    card = dbc.Card(card_content, color="primary", outline=True)
    return dbc.Col([card], width={"size": 2})

pref_grid = []

mapping_users = {
    'Alex':'usera',
    'Ben':'userb',
    'Cindy': 'userc'
}
    
mapping_context1 = {
    'Morning':'ctx11',
    'Evening':'ctx12',
}

mapping_context2 = {
    'Weekday':'ctx21',
    'Weekend':'ctx22'
}

mapping_items = {
    'Politics':'item1',
    'Economics':'item2',
    'Technology':'item3',
    'Weather':'item4',
    'Business':'item5',
    'History':'item6'
}

mapping_users_reverse = {v:k for k,v in mapping_users.items()}
mapping_context1_reverse = {v:k for k,v in mapping_context1.items()}
mapping_context2_reverse = {v:k for k,v in mapping_context2.items()}
mapping_items_reverse = {v:k for k,v in mapping_items.items()}

users = list(mapping_users.values())
items = list(mapping_items.values())
context1 = list(mapping_context1.values())
context2 = list(mapping_context2.values())

context = pd.DataFrame(list(product(users, context1, context2, items)),
                       columns=['users', 'context1', 'context2', 'items'])
context['reward'] = np.random.choice([0,1],len(context),p=[0.8,0.2])

vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")
vws = VWCSimulation(vw, context)
last_update = vws.step()

contextdf = context.copy()
countDF = contextdf.copy()
countDF['prob'] = 0

def generate_input_boxes():
    dropdown_users = dcc.Dropdown(
        id='ddown_users',
        options=[{"label":k, "value":v} for k,v in mapping_users.items()],
        clearable=False,
        value="usera",
        className="m-1",
    )
    dropdown_context1 = dcc.Dropdown(
        id='ddown_ctx1',
        options=[{"label":k, "value":v} for k,v in mapping_context1.items()],
        clearable=False,
        value="ctx11",
        className="m-1",
    )
    dropdown_context2 = dcc.Dropdown(
        id='ddown_ctx2',
        options=[{"label":k, "value":v} for k,v in mapping_context2.items()],
        clearable=False,
        value="ctx21",
        className="m-1",
    )
    dropdown_items = dcc.Dropdown(
        id='ddown_items',
        options=[{"label":k, "value":v} for k,v in mapping_items.items()],
        clearable=False,
        value="item1",
        className="m-1",
    )
    return html.Div(
        [
            dropdown_users,
            dropdown_context1,
            dropdown_context2,
            dropdown_items,
        ],
        style={"display": "flex", "flex-direction": "column"},
    )

def generate_context_boxes():
    dropdown_outcontext1 = dcc.Dropdown(
        id='ddown_outctx1',
        options=[{"label":k, "value":v} for k,v in mapping_context1.items()],
        clearable=False,
        value="ctx11",
        className="m-1",
    )
    dropdown_outcontext2 = dcc.Dropdown(
        id='ddown_outctx2',
        options=[{"label":k, "value":v} for k,v in mapping_context2.items()],
        clearable=False,
        value="ctx21",
        className="m-1",
    )
    return html.Div(
        [
            dropdown_outcontext1,
            dropdown_outcontext2
        ],
        style={"display": "flex", "flex-direction": "column"},
    )

app.layout = html.Div([
        generate_input_boxes(),
        dbc.Button("Register your Preference", color="primary", className="m-1", 
                   id='pref-button', block=True),
        html.Div(id='pref-grid'),
        dbc.Button("Clear the context", color="secondary", 
                   className="m-1", id='clr-button', block=True),
        dbc.Button("Start rewarding Agent for these Preferences", color="success", 
                   className="m-1", id='updt-button', block=True),
        generate_context_boxes(),
        dcc.Interval(
            id='interval-component',
            interval=100, # in milliseconds
            n_intervals=0),
        html.Div(id='placeholder'),
        html.Div(id='placeholder2'),

])

@app.callback(
    Output("pref-grid", "children"),
    Input("pref-button", "n_clicks"),   
    Input("clr-button", "n_clicks"),
    State('ddown_users', 'value'),
    State('ddown_items', 'value'),
    State('ddown_ctx1', 'value'), 
    State('ddown_ctx2', 'value'),
)
def update_pref_grid(nclick_pref, nclick_clr, pref_user, pref_item, pref_ctx1, pref_ctx2):
    global pref_grid
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if "pref-button" in changed_id:
        global contextdf
        card_text = '{} prefers {} related news in {} {}s'.format(mapping_users_reverse[pref_user],
                                                  mapping_items_reverse[pref_item],
                                                  mapping_context2_reverse[pref_ctx2],
                                                  mapping_context1_reverse[pref_ctx1])
        
        contextdf.loc[(contextdf.users==pref_user) & \
            (contextdf.context1==pref_ctx1) & \
            (contextdf.context2==pref_ctx2), \
            'reward'] = 0
        contextdf.loc[(contextdf.users==pref_user) & \
            (contextdf.context1==pref_ctx1) & \
            (contextdf.context2==pref_ctx2) & \
            (contextdf['items']==pref_item), \
            'reward'] = 1
        pref_grid.append(generate_input_cards(card_text))
        return dbc.Row(children=pref_grid,
                      style={'max-width': '100%',
                             'display': 'flex',
                             'align-items': 'center',
                             'padding': '2rem 5rem',
                             'overflow': 'auto',
                             'height': 'fit-content',
                             'flex-direction': 'row',
                            })
    elif "clr-button" in changed_id:
        pref_grid = []
        return dbc.Row(children=pref_grid)

@app.callback(
    Output("placeholder2", "children"),
    Input("updt-button", "n_clicks")
)
def update_context(nclick):
    if nclick:
        global vws
        global contextdf
        vws.update_context(contextdf)
    return ''


@app.callback(
    Output("placeholder", "children"),
    Input('interval-component', 'n_intervals'),
    Input('ddown_outctx1', 'value'), 
    Input('ddown_outctx2', 'value'),
)
def update_metrics(n, octx1, octx2):
    global countDF
    countDF = countDF.append(pd.Series(vws.step(),countDF.columns),ignore_index=True)
    _x = countDF.copy()
    _x = _x[(_x.context1==octx1) & (_x.context2==octx2)]
    _x['reward']*=-1
    pv = pd.pivot_table(_x, index=['users'], columns=["items"], values=['reward'], aggfunc=sum, fill_value=0)
    pv.index = [mapping_users_reverse[x] for x in pv.index]
    pv.columns = pv.columns.droplevel(0)
    pv = pv.rename_axis('User').reset_index().rename_axis(None, axis=1).set_index('User').T.reset_index()
    pv['index'] = pv['index'].map(mapping_items_reverse)
    pv = pv.rename(columns={"index": "Preferences"})
    out = html.Div([
        dbc.Table.from_dataframe(pv, striped=True, bordered=True, hover=True, responsive=True)
    ])
    return out

if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8051)