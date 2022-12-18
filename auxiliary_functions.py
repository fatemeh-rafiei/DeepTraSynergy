from google.colab import drive
import os
from collections import defaultdict
import math
import networkx as nx
import random
from tqdm import tqdm
from zipfile import ZipFile
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from pubchempy import *
import networkx as nx
from sklearn.cluster import KMeans
import tensorflow
import keras
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, Softmax
from keras.layers import Conv2D, GRU
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, Flatten, \
    Concatenate, Lambda
from keras.models import Model
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers, layers
from tensorflow.keras import regularizers
import tensorflow as tf


path_dataset = 'Project/DDSynergy/GraphSynergy/'
dataset = 'OncologyScreen'



def get_list_of_proteins(path_dataset, dataset):
    # path_dataset = '/content/drive/My Drive/Project/DDSynergy/GraphSynergy/GraphSynergy/data/'
    data = pd.read_excel(path_dataset + dataset + '/protein-protein_network.xlsx')

    yy = data.iloc[:, 1]
    list_of_prots = yy.unique()

    yy = data.iloc[:, 0]
    tmp = yy.unique()
    list_of_prots = np.concatenate((list_of_prots, tmp))
    list_of_prots = np.unique(list_of_prots)

    data = pd.read_csv(path_dataset + dataset + '/drug_protein.csv')

    yy = data.iloc[:, 0]
    list_of_drugs = yy.unique()

    yy = data.iloc[:, 1]
    tmp = yy.unique()
    list_of_prots = np.concatenate((list_of_prots, tmp))
    list_of_prots = np.unique(list_of_prots)

    data = pd.read_csv(path_dataset + dataset + '/cell_protein.csv')

    yy = data.iloc[:, 2]
    tmp = yy.unique()
    list_of_prots = np.concatenate((list_of_prots, tmp))
    list_of_prots = np.unique(list_of_prots)
    return list_of_prots




def get_list_of_drugs(path_dataset, dataset):
    data = pd.read_csv(path_dataset + dataset + '/drug_protein.csv')

    yy = data.iloc[:, 0]
    list_of_drugs = yy.unique()

    data = pd.read_csv(path_dataset + dataset + '/drug_combinations.csv')

    yy = data.iloc[:, 3]
    tmp = yy.unique()
    list_of_drugs = np.concatenate((list_of_drugs, tmp))
    list_of_drugs = np.unique(list_of_drugs)

    yy = data.iloc[:, 4]
    tmp = yy.unique()
    list_of_drugs = np.concatenate((list_of_drugs, tmp))

    print(list_of_drugs)
    list_of_drugs = np.unique(list_of_drugs)


    print(list_of_drugs.shape)
    return list_of_drugs



def get_list_of_cells(path_dataset, dataset):

    data = pd.read_csv(path_dataset + dataset + '/cell_protein.csv')

    yy = data.iloc[:, 3]
    list_of_cells = yy.unique()

    data = pd.read_csv(path_dataset + dataset + '/drug_combinations.csv')

    yy = data.iloc[:, 2]
    tmp = yy.unique()
    list_of_cells = np.concatenate((list_of_cells, tmp))
    list_of_cells = np.unique(list_of_cells)
    return list_of_cells



## Read drug smiles
CHARCANSMISET = {"#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
                 ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
                 "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
                 "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
                 "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
                 "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
                 "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
                 "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
                 "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
                 "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
                 "t": 61, "y": 62, "@": 63, "/": 64, "\\": 0}


def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):  # x, smi_ch_ind, y
        X[i] = smi_ch_ind[ch]

    return X  # .tolist()


def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros(MAX_SEQ_LEN)

    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]

    return X  # .tolist()


smiles_dict_len = 64
smiles_max_len = 100


def read_protein_protein_graph(path_dataset, dataset, list_of_prots):
    data = pd.read_excel(path_dataset + dataset + '/protein-protein_network.xlsx')

    protein_protein_graph = nx.Graph()

    for i in range(0, 217160):
        tmp = np.argwhere(list_of_prots == data.iloc[i, 0])[0]
        tmp1 = np.argwhere(list_of_prots == data.iloc[i, 1])[0]
        # data.iloc[i,0:2]
        # tmp=np.array(tmp)
        protein_protein_graph.add_edge(tmp[0], tmp1[0], weight=1)


    return protein_protein_graph


def read_protein_protein_matrix(path_dataset, dataset, list_of_prots, list_of_drugs):
    data = pd.read_csv(path_dataset + dataset + '/drug_protein.csv')
    indx_of_drugs_interaction = []
    indx_of_prots_interaction = []
    print(list_of_prots.shape)
    protein_drug_matrix = np.zeros((list_of_prots.shape[0], list_of_drugs.shape[0]))
    for i in range(0, data.shape[0]):
        tmp = np.argwhere(list_of_drugs == data.iloc[i, 0])[0]
        # indx_of_drugs_interaction.append(tmp[0])

        tmp2 = np.argwhere(list_of_prots == data.iloc[i, 1])[0]
        # indx_of_prots_interaction.append(tmp[0])

        protein_drug_matrix[tmp2[0], tmp[0]] = 1
        # protein_interaction.append(data.iloc[i,1])


    return protein_drug_matrix

	

def read_protein_cell_matrix(path_dataset, dataset, list_of_prots, list_of_cells):
    data = pd.read_csv(path_dataset + dataset + '/cell_protein.csv')

    protein_cell_matrix = np.zeros((list_of_prots.shape[0], list_of_cells.shape[0]))
    for i in range(0, data.shape[0]):
        tmp = np.argwhere(list_of_prots == data.iloc[i, 2])[0]
        tmp2 = np.argwhere(list_of_cells == data.iloc[i, 3])[0]
        protein_cell_matrix[tmp[0], tmp2[0]] = 1

    return protein_cell_matrix


def read_drug_combinations_data(path_dataset, dataset, list_of_prots, list_of_cells):
    data = pd.read_csv(path_dataset + dataset + '/drug_combinations.csv')

    indx_of_drugs1_combinations = []
    indx_of_drugs2_combinations = []
    indx_of_cells_combinations = []
    drugs_combinations = []
    num_training_samples = data.shape[0]
    sort_lbl = np.sort(np.array(data.iloc[:, 5]))
    print(sort_lbl.shape[0])
    threshold1 = sort_lbl[17359]
    threshold2 = sort_lbl[52077]
    for i in range(0, data.shape[0]):
        tmp = np.argwhere(list_of_drugs == data.iloc[i, 3])[0]
        indx_of_drugs1_combinations.append(tmp[0])
        tmp = np.argwhere(list_of_drugs == data.iloc[i, 4])[0]
        indx_of_drugs2_combinations.append(tmp[0])
        tmp = np.argwhere(list_of_cells == data.iloc[i, 2])[0]
        indx_of_cells_combinations.append(tmp[0])
        """if data.iloc[i,5]<threshold1:
           drugs_combinations.append([1,0,0])
        if data.iloc[i,5]>threshold2:
          drugs_combinations.append([0,0,1])
        else:
          drugs_combinations.append([0,1,0])"""
        if data.iloc[i, 5] < 0:  # (threshold1+threshold2)/2:
            drugs_combinations.append([1])
        else:
            drugs_combinations.append([0])


    return drugs_combinations, indx_of_drugs1_combinations, indx_of_drugs2_combinations, indx_of_cells_combinations


def apply_clustering_on_proteins(protein_embeddings):
    
    kmeans = KMeans(n_clusters=200, random_state=0).fit(protein_embeddings[0:15970, :])
    protein_embeddings = kmeans.cluster_centers_
    return protein_embeddings, kmeans


def update_protein_drug_matrix_by_clustering(protein_drug_matrix, kmeans):
    protein_drug_matrix_kmeans = np.zeros((200, protein_drug_matrix.shape[1]))
    for i in range(200):
        protein_drug_matrix_kmeans[i, :] = np.max(protein_drug_matrix[kmeans.labels_ == i, :], axis=0)
    protein_drug_matrix = protein_drug_matrix_kmeans
    return protein_drug_matrix



def update_protein_cell_matrix_by_clustering(protein_cell_matrix, kmeans):
    protein_cell_matrix_kmeans = np.zeros((200, protein_cell_matrix.shape[1]))
    for i in range(200):
        protein_cell_matrix_kmeans[i, :] = np.max(protein_cell_matrix[kmeans.labels_ == i, :], axis=0)
    protein_cell_matrix = protein_cell_matrix_kmeans
    return protein_cell_matrix

