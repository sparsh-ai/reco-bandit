import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import pickle
import pathlib
import os
import random
import base64
from algo import topk_similar

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

APP_PATH = str(pathlib.Path(__file__).parent.resolve())
styles = pd.read_csv(os.path.join(APP_PATH,'assets/data/styles.csv'))

artifact_path = 'assets/data/artifacts'
item_vectors = pickle.load(open(os.path.join(APP_PATH,artifact_path,"item_vectors.p"), "rb"))

epsilon = 0.8
def topk_dice(x):
  if (np.random.random()>epsilon) or (x<10):
    return 10
  else:
    return 2

def item_info(ipath):
    _id = int(os.path.basename(ipath).split('.jpg')[0])
    _imgmeta = styles[styles.id==_id].productDisplayName.values[0]
    return _id, _imgmeta

def create_card(ipath):
    _, imgmeta = item_info(ipath)
    card_content = [
    dbc.CardImg(src=ipath, top=True),
    dbc.CardBody(
            [
                # html.P(imgmeta, className="card-text"),
             dbc.Button("Like", color="success", 
                   className="m-1", id='like-button', block=True),
             dbc.Button("Hate", color="failure", 
                   className="m-1", id='hate-button', block=True),
             ]
            )
    ]
    card = dbc.Card(card_content, color="primary", outline=True)
    return dbc.Col([card], width={"size": 2})

def random_pid():
    return np.random.choice(list(item_vectors.keys()))

feedbackdf = pd.DataFrame(columns=['imgid','feedback'])
feedbackdf = feedbackdf.append(pd.Series((random_pid(),'1'), index=['imgid','feedback']),ignore_index=True)
feedbackdf = feedbackdf.append(pd.Series((random_pid(),'1'), index=['imgid','feedback']),ignore_index=True)

def recommend():
    impressions = feedbackdf.imgid.unique()
    likes = feedbackdf[feedbackdf.feedback=='1']
    likedlist = likes.imgid.unique()
    # hatedlist = [x for x in impressions if x not in likedlist]
    likevecs = np.array([item_vectors[x] for x in likedlist])
    # hatevecs = np.array([item_vectors[x] for x in hatedlist])
    _x = np.average(likevecs[2:,:], axis=0)
    # hatevec = np.average(hatevecs, axis=0)
    # _x = likevec - hatevec
    # weights=np.square(np.arange(_x.shape[0])+1)
    # lastN = 10
    # _x = np.average(_x[-lastN:,:], axis=0, weights=weights)
    # _x = np.average(_x, axis=0)
    _x = _x.astype(np.float32)
    # topk = topk_dice(len(impressions))
    topk = 50
    _x = topk_similar(_x,topk)
    _x = [x for x in _x if x not in impressions]
    # _x = random.choice(_x) if _x else random_pid()
    _x = _x[0] if _x else random_pid()
    _y = '{}/{}.jpg'.format('assets/data/images', _x)
    return _x, _y

app.layout = html.Div([
        html.Div(id='img-card', children=[create_card('assets/data/images/2704.jpg')]),
        html.Div(id='stats')
])

@app.callback(
    Output("img-card", "children"), 
    Input("like-button", "n_clicks"),
    Input('hate-button', 'n_clicks'),
)
def update_pref_grid(nlike, nhate):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    feedback = '0'
    if "like-button" in changed_id:
        feedback = '1'
    _x,_y = recommend()
    global feedbackdf
    feedbackdf = feedbackdf.append(pd.Series((_x,feedback), index=['imgid','feedback']),ignore_index=True)
    print(feedbackdf.tail())
    return create_card(_y)

if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8051)