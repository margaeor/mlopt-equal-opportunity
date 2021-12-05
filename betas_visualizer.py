
import time
import os
os.environ['PATH'] += ';'+'C:\\Program Files\\Poppler\\bin'

import pandas as pd
import numpy as np
from constants import *
import re
from collections import defaultdict
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
from xgboost import XGBRegressor
import faiss
from ast import literal_eval
import selenium
import seaborn as sns
import pathlib
from gurobipy import Model, GRB
import random
from tqdm import tqdm
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import plotly.offline as pyo
from sklearn import linear_model
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS

def split_in_half_space(s):

    num_spaces = s.count(' ')

    parts = s.split(' ')

    ss = ''
    for i,p in enumerate(parts):
        ss += ' ' + p

        if len(parts) > 1 and i == num_spaces//2:
            ss += '<br>'


    return ss

if __name__ == '__main__':

    betas_file = os.path.join('data','weights', 'betas_iai1.csv')
    df = pd.read_csv(betas_file)

    df_meta = pd.read_csv(METADATA_FILE)
    df_cols = pd.read_csv(FIELD_MAPPING_FILE)

    N_COLS = 3
    N_ROWS = 2
    fig = make_subplots(
        rows=2, cols=N_COLS,
        specs=[[{'rowspan': 2}, {},{'rowspan': 2}], [None, {}, None]],
        horizontal_spacing=0.01
        #subplot_titles=titles
    )

    assignments = [
        (1,2),
        (2,2),
        (1,1)
    ]

    assignments = [
        (1,3),
        (1,2),
        (2,2),
        (1,1)
    ]

    # assignments = [
    #     (1,1),
    #     (1,2),
    #     (2,2)
    # ]

    idx = 0
    for i in range(1,5):
        beta_col, name_col = f'grp{i}betas', f'grp{i}cols'
        final_mapper = {}
        #if i==3: continue

        if i < 3:
            og_col = 'SOCP' if i == 2 else 'FOD1P'
            intervention_col_name = 'occupation_code' if i==2 else 'field_of_degree'
            intervention_cols = list(filter(lambda x: re.match(rf"{intervention_col_name}_([0-9.]+|nan)$", x),
                                            df.loc[~df[name_col].isna(),name_col].unique().tolist()))
            intervention_vals = [float((re.match(rf"{intervention_col_name}_([0-9.]+|nan)$", col).group(1))) for col in
                                 intervention_cols]

            intervention_cols = [c for c in intervention_cols if c != f'{intervention_col_name}_nan']
            intervention_vals = [v for v in intervention_vals if not pd.isna(v)]
            #new_names = [mapper[val] for i, val in enumerate(intervention_vals)]

            cola, colb = f'{og_col}_map', f'{og_col}_desc_map2'
            mapper = {a: b for _, (a, b) in df_cols.loc[:, [cola, colb]].drop_duplicates().iterrows()}
            final_mapper = {intervention_cols[i]:mapper[val] for i, val in enumerate(intervention_vals)}
            print(final_mapper)
        else:
            df_tmp = pd.read_csv(os.path.join('data','weights','hol_map.csv'))
            final_mapper = {a:b for _,(a, b) in df_tmp.loc[:, ['key', 'value']].drop_duplicates().iterrows() if not pd.isna(b) and 'nan' not in b}

        df_new = df[[beta_col, name_col]].dropna()
        df_new['abs_beta'] = df_new[beta_col].abs()
        df_new = df_new.loc[df_new[name_col].isin(final_mapper)]\
            .sort_values(by='abs_beta', ascending=False)\
            .head(10)\
            .sort_values(by=beta_col,ascending=True)

        df_new["Color"] = np.where(df_new[beta_col] < 0, 'rgba(200,0,3,0.8)', 'rgba(9,129,74,0.8)')

        row = idx//N_COLS+1
        col = idx%N_COLS+1
        row = assignments[idx][0]
        col = assignments[idx][1]
        #if i ==4: col += 1
        #fig = go.Figure()
        y_vals = df_new[name_col].apply(lambda x: split_in_half_space(final_mapper[x]))
        fig.add_trace(
            go.Bar(name='Net',
                   y=y_vals,#.replace(' ','<br>') if final_mapper[x].count(' ')<2 else final_mapper[x]),
                   x=df_new[beta_col],
                   text=y_vals,
                   #textposition="inside",
                   #insidetextanchor="start",

                   #showlegend=False,
                   marker_color=df_new['Color'], orientation='h'),row ,col)#(i-1)//2+1, (i-1)%2+1),


        idx += 1
    fig.update_layout(barmode='stack',showlegend=False)#,template="plotly_dark"
    fig.update_yaxes(showticklabels=False)
    fig.write_image('exports/sparse.svg',scale=2, width=1200, height=600)
    pyo.plot(fig)


            # new_df_dict = defaultdict(list)
            #
            # for _, row in df.iterrows():
            #     row[]



    print('hi')