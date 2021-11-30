import pandas as pd
import numpy as np
from constants import *
import re
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import faiss
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
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k
        self.scaler = preprocessing.StandardScaler()

    def get_X_from_df(self, df):
        X = df.loc[:, [col for col in df.columns if col != self.outcome_col]].to_numpy()
        return X

    def get_y_from_df(self, df):
        y = df.loc[:, self.outcome_col].to_numpy()
        return y

    def fit(self, df, outcome_col):

        self.outcome_col = outcome_col

        X, y = self.get_X_from_df(df), self.get_y_from_df(df)

        self.scaler.fit(X)
        X = self.scaler.transform(X)

        self.outcome_col = outcome_col
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(np.ascontiguousarray(X).astype(np.float32))
        self.y = y

    def predict(self, df):

        X = self.get_X_from_df(df)
        distances, indices = self.index.search(np.ascontiguousarray(X).astype(np.float32), k=self.k)
        y_neighs = self.y[indices]
        #predictions = np.mean(y_neighs, axis=1)

        return y_neighs, indices

# def CustomClustering:
#
#     def __init__(self):
#
#         self.X = None
#         self.y = None
#
#     def fit(self, df):



preprocessed_path = os.path.join(OUTPUT_PATH, 'test_preprocessed.csv')

if __name__ == '__main__':
    df = pd.read_csv(preprocessed_path)

    target_col = 'field_of_degree'

    intervention_cols = list(filter(lambda x: re.match(rf"{target_col}_([0-9.]+|nan)$", x), df.columns))
    intervention_vals = [int(re.match(rf"{target_col}_([0-9.]+|nan)$", col).group(1)) for col in intervention_cols]+[0]

    df['z'] = df.apply(lambda x: np.array([x[col] for col in intervention_cols]), axis=1)
    df['z'] = df['z'].apply(lambda x: 0 if np.sum(x)==0 else intervention_vals[np.argmax(x)])

    df_map = pd.read_csv(METADATA_FILE)

    clust_cols = set(df_map.loc[df_map.UseInClustering == 1, 'VariableRename'].to_list())

    col_name_map = {}

    for col in df.columns:

        match = re.match(r"^(.*)_([0-9.]+|nan)$", col)
        mapping = match.group(1) if match else col

        if mapping in clust_cols:
            col_name_map[col] = mapping

    final_cols = set(col_name_map.keys()) & set(df.columns)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    X_train, X_test = df_train.loc[:, final_cols].to_numpy(), df_test.loc[:, final_cols].to_numpy()

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = np.ascontiguousarray(scaler.transform(X_train), dtype=np.float32)
    X_test = np.ascontiguousarray(scaler.transform(X_test), dtype=np.float32)

    k = 5

    _, clusters, kmeans = perform_kmeans(X_train, k)

    y_train = clusters.reshape((-1))

    grid = iai.GridSearch(
        iai.OptimalTreeClassifier(
            random_seed=1,
        ),
        max_depth=range(3, 5),
        #cp=0.001
        localsearch=False
    )
    grid.fit(df_train.loc[:, final_cols], y_train)
    #grid.show_in_browser()
    #grid.get_learner()

    #lnr = grid.get_learner()


    income_col = 'income_total'
    cols_for_prediction = final_cols

    pred = KNNPredictor(k=5)
    pred.fit(df, income_col)
    y_neighs, _ = pred.predict(df_test.head(100))







