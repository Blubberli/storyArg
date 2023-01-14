import argparse
import pickle
import pandas as pd
## Import libraries
from nltk.cluster import KMeansClusterer
import nltk
import numpy as np


def load_embeddings(path):
    with open(path, "rb") as fIN:
        return pickle.load(fIN)


def clustering(dataframe, embeddings, NUM_CLUSTERS=15):
    X = np.array(embeddings)

    kclusterer = KMeansClusterer(
        NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance,
        repeats=25, avoid_empty_clusters=True)

    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)

    dataframe['cluster'] = pd.Series(assigned_clusters, index=dataframe.index)
    dataframe['centroid'] = dataframe['cluster'].apply(lambda x: kclusterer.means()[x])

    return dataframe


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='path to data to do the clustering on')
    parser.add_argument('--embedding_path', type=str, help='path from which to load the embeddings')
    parser.add_argument('--output_path', type=str, help='path to save the predictions')
    parser.add_argument('--num_clusters', type=int, help='number of clusters')
    parser.add_argument('--filter', type=str, help='column to filter on')

    args = parser.parse_args()

    embeddings = load_embeddings(args.embedding_path)
    embeddings = embeddings['whitening']
    df = pd.read_csv(args.dataset, sep="\t")
    # get the indices of the dataframe that have filter == 1
    indices = df[df[args.filter] == 1].index.tolist()
    # get the embeddings of the two-dim array 'embeddings' that correspond to the indices (index = row number)
    embeddings = np.take(embeddings, indices, axis=0)
    # filter the dataset to contain comments with label == 1
    df = df[df[args.filter] == 1]
    df = clustering(dataframe=df, embeddings=embeddings, NUM_CLUSTERS=args.num_clusters)
    df.to_csv(args.output_path, sep="\t", index=False)
