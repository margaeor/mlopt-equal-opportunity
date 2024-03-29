import time
import os
os.environ['PATH'] += ';'+'C:\\Program Files\\Poppler\\bin'

import pandas as pd
import numpy as np
from constants import *
import re
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

#coefficient_of_dermination = r2_score(y, p(x))

np.random.seed(100)
random.seed(42)
EXPORT = True

sns.set_theme(style="darkgrid")

try:
    from julia import Julia
    Julia(compiled_modules=False, runtime='C:/Users/marga/AppData/Local/julias/bin/julia-1.6.cmd')
except:
    print("Cannot load julia")

from interpretableai import iai

def perform_kmeans(X, k, sk=True):

    if sk:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        distances = kmeans.transform(X).min(axis=1).reshape((-1,1))
        clusters = kmeans.predict(X).reshape((-1,1))

        distortion = kmeans.inertia_
        #distances, clusters = kmeans.transform(X)
    else:
        kmeans = faiss.Kmeans(d=X.shape[1], k=k, niter=100, nredo=10)
        kmeans.train(X.astype(np.float32))

        distances, clusters = kmeans.index.search(X.astype(np.float32), 1)

        d_indexed_clust = np.concatenate([clusters, distances], axis=1)
        d_indexed_clust = d_indexed_clust[d_indexed_clust[:, 0].argsort()]
        cluster_errors = np.split(d_indexed_clust[:, 1], np.unique(d_indexed_clust[:, 0], return_index=True)[1][1:])
        distortion = np.mean([np.mean(tmp ** 2) for tmp in cluster_errors if tmp.shape[0] > 0])

    indexed_clust = np.concatenate([clusters, np.arange(clusters.shape[0]).reshape((-1, 1))], axis=1)
    indexed_clust = indexed_clust[indexed_clust[:, 0].argsort()]

    cluster_to_idx_list = np.split(indexed_clust[:, 1], np.unique(indexed_clust[:, 0], return_index=True)[1][1:])

    #print(distortion)

    return distortion, clusters, kmeans

def plot_spider(df, categorical_cols, cluster_col, titles, labels=None, export=None):


    rows = (len(categorical_cols)+1)//2
    cols = 2

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{"type": "polar"} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=titles
    )

    for j, categorical_col in enumerate(categorical_cols):
        df_tmp = df.groupby(cluster_col)[categorical_col].value_counts(normalize=True).mul(100).rename(
            'percentage').reset_index()

        if labels is None:
            labels = df_tmp[[categorical_col, 'percentage']].groupby(categorical_col).std().dropna().sort_values('percentage').tail(
                10).index.to_list()

        labels = [*labels, labels[0]]

        spider_vals = []

        for c in df[cluster_col].sort_values().unique():
            tmp = []
            for l in labels:
                df_row = df_tmp.loc[(df_tmp[cluster_col] == c) & (df_tmp[categorical_col] == l), 'percentage']
                tmp.append(0 if df_row.shape[0] == 0 else df_row.iloc[0])
            spider_vals.append(tmp)

        # spider_vals = [[df_tmp.loc[(df_tmp.cluster == c) & (df_tmp.z_id_desc == l), 'percentage'].iloc[0] for l in all_labels] for c in range(self.k)]
        spider_vals = [[*val, val[0]] for val in spider_vals]

        show_legend = j==0
        for i, cluster in enumerate(spider_vals):
            if i==0:
                fig.add_trace(
                    go.Scatterpolar(r=cluster, theta=labels, name=f'Cluster {i}', fill='toself',
                                    legendgroup=f'group{i}', showlegend=show_legend), j // 2 + 1, j % 2 + 1
                )
            else:
                fig.append_trace(
                    go.Scatterpolar(r=cluster, theta=labels, name=f'Cluster {i}', fill='toself',legendgroup=f'group{i}',
                                    showlegend=show_legend), j//2+1, j%2+1
                )
        # fig.add_trace(go.Figure(
        #     data=[
        #         go.Scatterpolar(r=cluster, theta=labels, name=f'Cluster {i}', fill='toself') for i, cluster in
        #         enumerate(spider_vals)
        #     ],
        #     layout=go.Layout(
        #         title=go.layout.Title(text='Restaurant comparison'),
        #         polar={'radialaxis': {'visible': True}},
        #         showlegend=True,
        #         template="plotly_dark"
        #         # paper_bgcolor='rgb(0,0,0)',
        #     )
        # ))
    fig.update_layout(
        #title=go.layout.Title(text='Restaurant comparison'),
        polar={'radialaxis': {'visible': True}},
        showlegend=True,
        template="plotly_dark"
    )
    if export:
        fig.write_image(export)

    pyo.plot(fig)

def find_col_names_from_dummy_cols(df, clust_cols):

    col_name_map = {}

    for col in df.columns:

        match = re.match(r"^(.*)_([0-9.]+|nan)$", col)
        mapping = match.group(1) if match else col

        if mapping in clust_cols:
            col_name_map[col] = mapping


    return col_name_map

def elbow_plot(X_train, mn=2, mx=10):

    distortions = []
    K = range(mn, mx)
    for k in tqdm(K):
        # kmeanModel = KMeans(n_clusters=k)
        # kmeanModel.fit(df)
        # distortions.append(kmeanModel.inertia_)
        err, clusters_assignment, m = perform_kmeans(X_train, k)

        distortions.append(err)

    plt.figure(figsize=(16, 8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()



def train_knn(df_X, outcome_col, k=5):

    X_train = df_X.loc[:, [col for col in df_X.columns if col != outcome_col]].to_numpy()
    X_train = X_train.astype(np.float32)
    index = faiss.IndexFlatL2(X_train.shape[1])

    index.add(X_train)

    return index


class KNNPredictor:
    def __init__(self, outcome_col, k=5 ):
        self.index = None
        self.y = None
        self.k = k
        self.scaler = preprocessing.StandardScaler()
        self.outcome_col = outcome_col
        #self.excluded_cols = excluded_cols
        self.used_cols = None
        self.xgb = XGBRegressor()
        #self.xgb = linear_model.Lasso(alpha=0.1)
        self.train_metrics = None
        self.test_metrics = None

    def get_X_from_df(self, df):

        if self.used_cols is None:
            X = df.loc[:, [col for col in df.columns if col != self.outcome_col]]
            self.used_cols = X.columns.to_list()
        else:
            X = df.loc[:, self.used_cols]
        X = X.to_numpy()
        return X

    def get_y_from_df(self, df):
        y = df.loc[:, self.outcome_col].to_numpy()
        return y

    def fit(self, df):

        X, y = self.get_X_from_df(df), self.get_y_from_df(df)
        self.X = X
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        self.xgb.fit(X, y)
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(np.ascontiguousarray(X).astype(np.float32))
        self.y = y

        self.evaluate(df, is_train=True)

    def predict(self, df, is_train=False):

        X = self.get_X_from_df(df)

        X = self.scaler.transform(X)
        distances, indices = self.index.search(np.ascontiguousarray(X).astype(np.float32), k=self.k)
        y_neighs = self.y[indices]
        predictions = np.mean(y_neighs, axis=1)

        predictions = self.xgb.predict(X)

        return indices, predictions

    def evaluate(self, df, is_train=False):

        _, y_pred = self.predict(df)

        s = 'train' if is_train else 'test'
        print(f"R squared {s} {df.shape[0]}: {r2_score(df[self.outcome_col].to_numpy(), y_pred)}")



class CustomClustering:

    def __init__(self, k, y_col_name, possible_interventions, treatment_desc, intervention_cols, use_global_predictor=True):

        self.X = None
        self.y = None
        self.y_clust_avg = None
        self.z_clust_avg = None
        self.N_clust_avg = None
        self.outcome_col = y_col_name
        self.k = k
        self.scaler = None
        self.clusters = None
        self.kmeans = None
        self.final_cluster_cols = None
        self.possible_interventions = [i for i in possible_interventions if not pd.isna(i) and i != 0]
        self.intervention_cols = intervention_cols
        self.knn_predictors = []
        self.global_predictor = None
        self.treatment_desc = {k:v for k,v in treatment_desc.items() if k in self.possible_interventions}
        self.z_avg_all = None
        self.intervention_col = 'occupation_code'
        self.use_global_predictor = use_global_predictor
        self.df_train = None

    def fit(self, df, clust_cols, knn_cols):

        self.df_train = df
        df = df.copy()

        KNN_K = 10
        knn_excluded_cols = ['z', 'z_id', self.outcome_col]

        clust_col_name_map = find_col_names_from_dummy_cols(df, clust_cols)
        knn_col_name_map = find_col_names_from_dummy_cols(df, knn_cols)

        self.final_cluster_cols = set(clust_col_name_map.keys()) & set(df.columns)
        self.final_knn_cols = set(knn_col_name_map.keys()) & set(df.columns)

        self.knn_predictors = []


        if self.use_global_predictor:
            self.global_predictor = KNNPredictor(outcome_col=self.outcome_col, k=KNN_K)
            self.global_predictor.fit(df.loc[:, self.final_knn_cols | {'income_total'}])
        else:
            for intervention in self.possible_interventions:

                # For every intervention train a KNN predictor to predict the
                # outcome
                pred = KNNPredictor(outcome_col=self.outcome_col, k=KNN_K)
                pred.fit(df.loc[df.z_id == intervention, self.final_knn_cols | {'income_total'}])
                self.knn_predictors.append(pred)



        X_train = df.loc[:, self.final_cluster_cols].to_numpy()

        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(X_train)
        X_train = np.ascontiguousarray(self.scaler.transform(X_train), dtype=np.float32)

        def calculate_agg_for_each_cluser_treatment_comb(df, col, func):


            # aggregate = []
            # for intervention in self.possible_interventions:
            #     mask = (df.z_id == intervention)
            #     df_subset = df.loc[mask, col]
            #     indexed_clust = np.concatenate(
            #         [self.clusters[np.argwhere(mask.to_numpy()).reshape((-1)),:], df_subset.to_numpy().reshape((-1, 1))], axis=1
            #     )
            #     indexed_clust = indexed_clust[indexed_clust[:, 0].argsort()]
            #
            #     #cluster_to_idx_list = np.split(indexed_clust[:, 1], np.unique(indexed_clust[:, 0], return_index=True)[1][1:])
            #     cluster_groups = np.split(indexed_clust[:, 1], np.unique(indexed_clust[:, 0], return_index=True)[1][1:])
            #
            #     cluster_z_aggregate = [func(clust) if clust.shape[0]>0 else 0 for clust in cluster_groups]
            #     aggregate.append(cluster_z_aggregate)

            results = []
            for cluster in range(self.k):

                cluster_interventions = []
                for intervention in self.possible_interventions:

                    df_sub = df[(df.z_id == intervention) & (df.cluster == cluster)]
                    if df_sub.shape[0] == 0:
                        cluster_interventions.append(0)
                    else:
                        cluster_interventions.append(func(df_sub[col]))
                results.append(cluster_interventions)

            results = np.array(results)
            return results


        _, self.clusters, self.kmeans = perform_kmeans(X_train, self.k)
        df['cluster'] = self.clusters

        # Dummy col just used to count individuals whith each
        # intervention on each cluster
        df['dummy_col'] = 1

        self.y_clust_avg = calculate_agg_for_each_cluser_treatment_comb(df, self.outcome_col, func=np.median)
        #self.z_clust_avg = calculate_agg_for_each_cluser_treatment_comb(df, 'z', func=np.mean)
        self.z_avg_all = df['z'].mean()
        self.N_clust_avg = calculate_agg_for_each_cluser_treatment_comb(df, 'dummy_col', func=np.sum)
        #df['cluster'] = self.clusters

    def predict(self, df):

        X_test = df.loc[:, self.final_cluster_cols].to_numpy()
        X_test = np.ascontiguousarray(self.scaler.transform(X_test), dtype=np.float32)

        return self.kmeans.predict(X_test)

    def visualize_cluster(self, df):

        from selenium import webdriver
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.common.by import By

        grid = iai.GridSearch(
            iai.OptimalTreeClassifier(
                random_seed=1,
            ),
            max_depth=range(3, 5),
            # cp=0.001
            localsearch=False
        )

        y = self.predict(df).reshape(-1)
        grid.fit(df.loc[:, self.final_cluster_cols], y)

        grid.get_learner().write_html('exports/test.html')

        return
        #grid.show_in_browser()

        cur_dir = str(pathlib.Path(__file__).parent.resolve()).replace('\\','/')

        options = webdriver.ChromeOptions()
        dir = cur_dir.replace('/','\\')+'\exports'+'\\'
        prefs = {"download.default_directory": dir}
        #options.add_argument('--headless')
        options.add_experimental_option("prefs", prefs)
        driver = webdriver.Chrome(executable_path=r'./chromedriver.exe', chrome_options=options)
        driver.get(f"file:///{cur_dir}/exports/test.html")

        print('Exporting IAI')
        time.sleep(3)
        driver.find_elements(By.CSS_SELECTOR, '.button-row > button')[2].click()
        time.sleep(3)
        print('Exported IAI')

    def prescribe_group(self, df, y_knn_all, rhos):

        N = len(self.possible_interventions)
        M = df.shape[0]

        df['z_pred_3'] = [[]] * df.shape[0]

        for rho in tqdm(rhos):
            m = Model()
            m.setParam('OutputFlag', 0)
            m.modelSense = GRB.MAXIMIZE

            # Initialize variables
            z = m.addVars(M, N, vtype=GRB.BINARY, name='z')
            t = m.addVars(N, vtype=GRB.CONTINUOUS, name='t')
            m.addConstr(sum(t[i] for i in range(N)) <= rho/N)
            m.addConstrs((t[j] >=  sum(z[i,j] for i in range(M)) / M - self.z_avg_all[j]) for j in range(N))
            m.addConstrs((t[j] >= -sum(z[i,j] for i in range(M)) / M + self.z_avg_all[j]) for j in range(N))
            m.addConstrs((sum(z[i, j] for j in range(N)) <= 1) for i in range(M))
            m.setObjective(sum(sum(z[i, j]*y_knn_all[i, j] for i in range(N)) for j in range(N)))
            m.optimize()

            z_vals = z.values()
            z_matrix = np.array([[z_vals[i * N + j].X for i in range(M)] for j in range(N)]).T

            df['z_pred_3'] = df.apply(lambda x: x['z_pred_3']+[self.possible_interventions[np.argmax(z_matrix[x.name,:])]] ,axis=1)
        #df['z_pred_3'] = [self.possible_interventions[np.argmax(l)] for l in z_matrix.T]


    def calculate_prediction_tradedoffs(self, df, rho_vals, y_knn_all):
        pred_col = 'z_pred_3'


        for i, rho in enumerate(rho_vals):
            ###################HEREE
            df['estimated_income'] = df.apply(lambda x: y_knn_all[x.name, x['z_pred_3_idx'][i]], axis=1)
            #df['estimated_income'] = df.apply(lambda x: y_knn_all[x.name, x['z_idx']], axis=1)
            df[f'profit_{i}'] = df['estimated_income']-df[self.outcome_col]
            print(f"[ RHO = {rho}")
            print(f"min: {df[f'profit_{i}'].min()}")
            print(f"mean: {df[f'profit_{i}'].mean()}")
            print(f"std: {df[f'profit_{i}'].std()}")
            print(f"max: {df[f'profit_{i}'].max()}")

    def prescribe_separately(self, df, plot=True):

        df_og = df

        df = df.copy()
        df['cluster'] = clusterer.predict(df)

        # Model (1): z = argmin_z_i y_clust_avg(x,z_i)*N_clust(x,z_i)
        df['z_pred_1'] = df.cluster.apply(
            lambda i: self.possible_interventions[np.argmax(self.N_clust_avg[i, :] * self.y_clust_avg[i, :])])

        print(df['z_pred_1'].value_counts())

        # Model (2): z = argmin_z_i y_knn_avg(x,z_i)*N_clust(x,z_i)
        # where y_knn is calculated as the average of the ys of the k
        # nearest neighbors of y which have treatment z_i

        y_knn_all = []

        if self.use_global_predictor:
            df_tmp = df_og.copy()

            for i, z in enumerate(self.possible_interventions):

                df_tmp.loc[:, self.intervention_cols] = 0
                df_tmp.loc[:, self.intervention_cols[i]] = 1
                _, y_pred = self.global_predictor.predict(df_tmp)
                #self.holistic_predictor.evaluate(df_tmp, is_train=False)
                y_knn_all.append(y_pred)

            self.global_predictor.evaluate(df_og, is_train=False)
        else:

            for z, predictor in zip(self.possible_interventions, self.knn_predictors):
                _, y_pred = predictor.predict(df_og)
                predictor.evaluate(df_og[df_og.z_id == z], is_train=False)
                y_knn_all.append(y_pred)

        y_knn_all = np.array(y_knn_all).T

        df = df.reset_index()
        df['z_pred_2'] = df.apply(
            lambda x: self.possible_interventions[np.argmax(self.N_clust_avg[x.cluster, :] * y_knn_all[x.name, :])],
            axis=1)

        rho_list = [0.5, 1, 3, 5, 7, 10]

        self.prescribe_group(df, y_knn_all, rho_list)

        map_to_name = lambda x: self.treatment_desc[x]
        map_to_idx = lambda x: {v:i for i,v in enumerate(self.possible_interventions)}[x]
        for i in range(1,4):

            if i<3:
                df[f'z_pred_{i}_idx'] = df[f'z_pred_{i}'].apply(map_to_idx)
                df[f'z_pred_{i}_desc'] = df[f'z_pred_{i}'].apply(map_to_name)
                df['estimated_income'] = df.apply(lambda x: y_knn_all[x.name, x[f'z_pred_{i}_idx']], axis=1)
                df[f'profit_{i}p'] = df['estimated_income'] - df[self.outcome_col]
            else:
                df[f'z_pred_{i}_idx'] = df[f'z_pred_{i}'].apply(lambda x: [map_to_idx(j) for j in x])
                df[f'z_pred_{i}_desc'] = df[f'z_pred_{i}'].apply(lambda x: [map_to_name(j) for j in x])
            #df['z_pred_2_desc'] = df['z_pred_2'].apply(map_to_name)
            #df['z_pred_3_desc'] = df['z_pred_3'].apply(map_to_name)

        self.df_train['z_id_desc'] = self.df_train['z_id'].apply(map_to_name)
        df['z_id_desc'] = df['z_id'].apply(map_to_name)
        df['z_idx'] = df['z_id'].apply(map_to_idx)
        label_order = pd.value_counts(df.z_id_desc).iloc[:10].index

        ax = sns.countplot(x="z_id_desc", data=self.df_train, order=label_order)
        plt.xticks(rotation=45)
        plt.title('Original Occupation Distribution')
        if EXPORT: plt.savefig(f'exports/distr_orig.eps')
        plt.show()

        for i in range(len(rho_list)):
            rho = rho_list[i]
            df['tmp'] = df[f'z_pred_3_desc'].apply(lambda x: x[i])
            ax = sns.countplot(x="tmp", data=df, order=label_order)
            plt.xticks(rotation=45)
            plt.title(rf'Prescripted distribution in test: $\rho={rho}$')
            if EXPORT: plt.savefig(f'exports/distr_{rho}.eps')
            plt.show()


        self.calculate_prediction_tradedoffs(df, rho_list, y_knn_all)

        mean_profits = []
        for i in range(len(rho_list)):
            mean_profits.append(df[f'profit_{i}'].mean())

        #rho_list = [rho_list[0]]
        profits = [x for i in range(len(rho_list)) for x in df[f'profit_{i}'].to_list()]
        rhos_tmp = [x for rho in rho_list for x in [rho]*df.shape[0]]
        df_stat = pd.DataFrame({'rhos': rhos_tmp, 'profits': profits})
        sns.lineplot(data=df_stat, x="rhos", y="profits")
        #plt.xscale('log')
        #plt.yscale('log')
        plt.xlabel(r'$\rho$')
        plt.ylabel('Mean profit (\$/year/person)')
        plt.title('Expected income increase after prescription')
        if EXPORT: plt.savefig('exports/profits.eps')
        plt.show()
        print('hi')

        df_tmp = df.groupby('cluster')['z_id_desc'].value_counts(normalize=True).mul(100).rename(
            'percentage').reset_index()



        ######################################
        # Spider plot

        labels = df_tmp[['z_id_desc', 'percentage']].groupby('z_id_desc').sum().dropna().sort_values(
            'percentage').tail(
            10).index.to_list()+['ENG']

        chosen_rho_id = 1
        df['z_pred_3_desc_chosen'] = df[f'z_pred_3_desc'].apply(lambda x: x[chosen_rho_id])

        spider_col_names = ['z_id_desc',  'z_pred_3_desc_chosen']
        titles = ['Original distribution', f'Distribution after prescription $\\rho={rho_list[chosen_rho_id]}$']
        plot_spider(df, spider_col_names, 'cluster', labels=labels, export='exports/grouped.svg', titles=titles)
        time.sleep(1)


        #plot_spider(df, 'z_pred_3_desc_chosen', 'cluster', labels=labels, export='exports/prescribed.svg')

        #####################################

        #df_tmp = df.groupby('cluster')['z_id_desc'].value_counts(normalize=True).mul(100).rename('percentage').reset_index().pipe((sns.catplot, 'data'), kind="bar", x='z_id_desc',col='cluster')
        df_tmp.pipe((sns.catplot, 'data'), x='z_id_desc', y='percentage', kind='bar', order=label_order, col='cluster',col_wrap=2)
        [ax.set_xticklabels(label_order.to_list(), rotation=45) for ax in plt.gcf().axes]
        [ax.set_xlabel('Occupation Category') for ax in plt.gcf().axes]
        plt.show()

        print("Method 1 ", df['profit_1p'].mean())
        ax = sns.countplot(x="z_pred_1_desc", data=df, order=label_order)
        plt.xticks(rotation=45)
        plt.title('Occupation distribution in test: Framework 1.1')
        if EXPORT: plt.savefig('exports/distr_p1.eps')
        plt.show()

        print("Method 2 ", df['profit_2p'].mean())
        ax = sns.countplot(x="z_pred_2_desc", data=df, order=label_order)
        plt.xticks(rotation=45)
        plt.title('Occupation distribution in test: Framework 1.2')
        if EXPORT: plt.savefig('exports/distr_p2.eps')
        plt.show()

        ax = sns.countplot(x="z_pred_3_desc_chosen", data=df, order=label_order)
        plt.xticks(rotation=45)
        plt.title('Occupation distribution in test: Framework 2')
        plt.show()

        # ax = sns.countplot(x="z_id_desc", data=df)
        # sns.catplot(x="z_id_desc", col="cluster", data=df, kind="count", height=4, aspect=1,
        #             order=label_order, col_wrap=3)
        # #plt.show()
        # [ax.set_xticklabels(label_order.to_list(), rotation=45) for ax in plt.gcf().axes]
        # #[ax.set_xticklabels(ax.get_xticklabels(), rotation=45) for ax in plt.gcf().axes]
        # #plt.title('Actual Occupation distribution in test')
        # plt.show()
        # #plt.xticks()
        # print(df['z_pred_2'].value_counts())


preprocessed_path = os.path.join(OUTPUT_PATH, 'preprocessed.csv')

if __name__ == '__main__':
    df = pd.read_csv(preprocessed_path).head(100000)

    intervention_col_name = 'occupation_code'
    outcome_col_name = 'income_total'

    intervention_cols = list(filter(lambda x: re.match(rf"{intervention_col_name}_([0-9.]+|nan)$", x), df.columns))
    intervention_vals = [float((re.match(rf"{intervention_col_name}_([0-9.]+|nan)$", col).group(1))) for col in intervention_cols] + [0]

    df['z'] = df.apply(lambda x: np.array([x[col] for col in intervention_cols]), axis=1)
    df['z_id'] = df['z'].apply(lambda x: 0 if np.sum(x)==0 else intervention_vals[np.argmax(x)])

    map_to_idx = lambda x: {v: i for i, v in enumerate(intervention_vals)}[x]

    df = df.loc[~df['z_id'].isna(), :]
    df['z_id'] = df['z_id'].astype(int)
    df['z_idx'] = df['z_id'].apply(map_to_idx)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=4
                                         , shuffle=True)

    df_meta = pd.read_csv(METADATA_FILE)
    df_map = pd.read_csv(FIELD_MAPPING_FILE)

    intervention_var_name = df_meta.loc[df_meta.VariableRename == intervention_col_name, 'Variable'].iloc[0]
    clust_cols = set(df_meta.loc[df_meta.UseInClustering2 == 1, 'VariableRename'].to_list())
    knn_cols = set(df_meta.loc[df_meta.UseInKNN == 1, 'VariableRename'].to_list())

    df_interv_descs = df_map[[f'{intervention_var_name}_desc_map',f'{intervention_var_name}_map']]
    interv_desc = {row.iloc[1]: row.iloc[0] for _, row in df_interv_descs.drop_duplicates().iterrows()}

    clusterer = CustomClustering(k=4, y_col_name=outcome_col_name,
                                 possible_interventions=intervention_vals,
                                 treatment_desc=interv_desc,
                                 intervention_cols=intervention_cols
    )

    clusterer.fit(df_train, clust_cols, knn_cols)

    NUM_PRESCRIPTIONS = 1500
    df_prescr = df_test.head(NUM_PRESCRIPTIONS)

    clusterer.prescribe_separately(df_prescr)


    # exit(0)
    # p = KNNPredictor(outcome_col_name)
    #
    # knn_cols_m = list(find_col_names_from_dummy_cols(df_train, knn_cols).keys())
    # #print(knn_cols_m)
    # p.fit(df_train.loc[:, knn_cols_m + [outcome_col_name]])
    # p.evaluate(df_test.loc[:, knn_cols_m + [outcome_col_name]])
    #df_test['cluster'] = clusterer.predict(df_test).tolist()

    #print('hi')
    clusterer.visualize_cluster(df_test)

    # y_train = clusters.reshape((-1))


    #grid.show_in_browser()
    #grid.get_learner()

    #lnr = grid.get_learner()


    # income_col = 'income_total'
    # cols_for_prediction = final_cols
    #
    # pred = KNNPredictor(k=5)
    # pred.fit(df, income_col)
    # y_neighs, _ = pred.predict(df_test.head(100))







