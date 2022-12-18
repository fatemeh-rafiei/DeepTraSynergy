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
from PPI_Node2vec import node2vec_on_PPI
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
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, merge, Flatten, \
    Concatenate, Lambda
from keras.models import Model
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers, layers
from tensorflow.keras import regularizers
import tensorflow as tf
from auxiliary_functions import *
from ppi_node2vec import *

path_dataset = 'Project/DDSynergy/GraphSynergy/'
dataset = 'OncologyScreen'



################################################################################
#        This part extract the protein protein graph and protein drug graph    #
################################################################################

list_of_prots = get_list_of_proteins(path_dataset, dataset)

list_of_drugs = get_list_of_drugs(path_dataset, dataset)

list_of_cells = get_list_of_cells(path_dataset, dataset)

protein_protein_graph = read_protein_protein_graph(path_dataset, dataset, list_of_prots)

protein_drug_matrix = read_protein_protein_matrix(path_dataset, dataset, list_of_prots, list_of_drugs)

protein_cell_matrix = read_protein_cell_matrix(path_dataset, dataset, list_of_prots, list_of_cells)

###################################################################################################
#     This part of code reads compound ID and get its SMILES sequence and save it as numpy array  #
###################################################################################################
# smiles_max_len = 200
# smiles_unique1 = []
# for i in range(0, list_of_drugs.shape[0]):
#     print(i)
#     cs = get_compounds(list_of_drugs[i], 'name')
#     c = Compound.from_cid(cs[0].cid)
#     smiles_unique1.append(label_smiles(c.canonical_smiles, smiles_max_len, CHARCANSMISET))
#
# print(list_of_drugs.shape[0])
# np.save('/content/drive/My Drive/Project/DDSynergy/GraphSynergy/GraphSynergy/data/' + dataset + '/smiles3.npy',
#         np.array(smiles_unique1))

########################################################################################################

#   Read the stored SMILES sequencs
smiles_unique = np.load(path_dataset + dataset + '/smiles_200.npy')


# Read drug combinations data
drugs_combinations, indx_of_drugs1_combinations, indx_of_drugs2_combinations, indx_of_cells_combinations = read_drug_combinations_data(path_dataset, dataset, list_of_prots, list_of_cells)

# Apply node2vec on PPI
protein_embeddings = node2vec_on_PPI(protein_protein_graph)


# Apply k-means on the learned protein representation
protein_embeddings, kmeans = apply_clustering_on_proteins(protein_embeddings)

# Update the protein drug and protein cell matrixes using the center of clusters

protein_drug_matrix = update_protein_drug_matrix_by_clustering(protein_drug_matrix, kmeans)

protein_cell_matrix = update_protein_cell_matrix_by_clustering(protein_cell_matrix, kmeans)




####################################################################
#          Define Auxiliary functions which are used in model       #
####################################################################

def dot_batch(inp):
    
    return tf.keras.backend.batch_dot(inp[0], inp[1], axes=(1, 2))  # tf.keras.activations.sigmoid()


def dot_batch_axs_pro(inp):
    
    return tf.keras.backend.batch_dot(inp[0], inp[1], axes=(2, 2))


def dot_batch_axs_drug(inp):
    
    return tf.keras.activations.sigmoid(tf.keras.backend.batch_dot(inp[0], inp[1], axes=(1, 1)))


def dot_batch_axs_cell(inp):
    
    return tf.keras.activations.sigmoid(
        tf.keras.backend.batch_dot(inp[0], inp[1], axes=(1, 1)))  # tf.math.multiply(inp[0],inp[1]))


def dot_batch_axs_toxic(inp):
    
    return tf.keras.backend.batch_dot(inp[0], inp[1], axes=(1, 1))

def compactness_loss(actual, features):
    features = Flatten()(features)
    k = 2000
    batch_size = 10
    dim = (batch_size, k)

    def zero(i):
        z = tf.zeros((1, dim[1]), dtype=tf.dtypes.float32)
        o = tf.ones((1, dim[1]), dtype=tf.dtypes.float32)
        arr = []
        for k in range(dim[0]):
            arr.append(o if k != i else z)
        res = tf.concat(arr, axis=0)
        return res

    masks = [zero(i) for i in range(batch_size)]
    m = (1 / (batch_size - 1)) * tf.map_fn(
        # row-wise summation
        lambda mask: tf.math.reduce_sum(features * mask, axis=0),
        masks,
        dtype=tf.float32,
    )
    dists = features - m
    sqrd_dists = tf.pow(dists, 2)
    red_dists = tf.math.reduce_sum(sqrd_dists, axis=1)
    compact_loss = (1 / (batch_size * k)) * tf.math.reduce_sum(red_dists)
    return compact_loss
	


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=tf.expand_dims(images, axis=-1),
            sizes=[1, self.patch_size / 4, 1, 1],
            strides=[1, self.patch_size / 4, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    # augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


##############################################################
#           Feature extractor for Drugs                      #
##############################################################

embedding_size = 20
num_filters = 64
protein_filter_lengths = 8
smiles_filter_lengths = 4
smiles_max_len = 200
protein_embeding_dim = 200


num_classes = 2
input_shape = (200, 1)

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
drug_size = 200  # We'll resize input sequences to this size
patch_size = 20  # Size of the patches to be extract from the input images
num_patches = (drug_size // patch_size) * 4
projection_dim = 50
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [1024, 1024, 512]  # Size of the dense layers of the final classifier


METRICS = [
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR')  # precision-recall curve
]


num_prot_cluster = 200

# Define Shared Layers
Drug1_input = Input(shape=(smiles_max_len, 1), name='drug1_input')  # dtype='int32',
Drug2_input = Input(shape=(smiles_max_len, 1), name='drug2_input')  # dtype='int32',

Protein_Protein_input = Input(shape=(num_prot_cluster, protein_embeding_dim), name='protein_protein_input')
Protein_Cell_input = Input(shape=(num_prot_cluster), name='protein_cell_input')

# network for Drug 1
# out1 = Embedding(input_dim=smiles_dict_len+1, output_dim = embedding_size, input_length=smiles_max_len,name='smiles_embedding') (Drug1_input)
patches = Patches(patch_size)(Drug1_input)
# Encode patches.
encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

# Create multiple layers of the Transformer block.
for _ in range(transformer_layers):
    # Layer normalization 1.
    x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # Create a multi-head attention layer.
    num_heads = 1
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim, dropout=0.1
    )(x1, x1)
    # Skip connection 1.
    x2 = layers.Add()([attention_output, encoded_patches])
    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    # MLP.
    x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
    # Skip connection 2.
    encoded_patches = layers.Add()([x3, x2])

# Create a [batch_size, projection_dim] tensor.
representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
representation = layers.Flatten()(representation)
representation = layers.Dropout(0.5)(representation)

# out_drug1 = GlobalMaxPooling1D()(out_drug1)

num_classes = 2
input_shape = (200, 1)

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
drug_size = 200  # We'll resize input sequences to this size
patch_size = 20  # Size of the patches to be extract from the input images
num_patches = (drug_size // patch_size) * 4
projection_dim = 50
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [1024, 1024, 512]

# out2 = Embedding(input_dim=smiles_dict_len+1, output_dim = embedding_size, input_length=smiles_max_len,name='smiles_embedding2') (Drug2_input)
patches_2 = Patches(patch_size)(Drug2_input)
# Encode patches.
encoded_patches_2 = PatchEncoder(num_patches, projection_dim)(patches_2)

# Create multiple layers of the Transformer block.
for _ in range(transformer_layers):
    # Layer normalization 1.
    x1_2 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches_2)
    # Create a multi-head attention layer.
    num_heads = 1
    attention_output_2 = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim, dropout=0.1
    )(x1_2, x1_2)
    # Skip connection 1.
    x2_2 = layers.Add()([attention_output_2, encoded_patches_2])
    # Layer normalization 2.
    x3_2 = layers.LayerNormalization(epsilon=1e-6)(x2_2)
    # MLP.
    x3_2 = mlp(x3_2, hidden_units=transformer_units, dropout_rate=0.1)
    # Skip connection 2.
    encoded_patches_2 = layers.Add()([x3_2, x2_2])

# Create a [batch_size, projection_dim] tensor.
representation_2 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches_2)
representation_2 = layers.Flatten()(representation_2)
representation_2 = layers.Dropout(0.5)(representation_2)


##############################################################
#           Drug-Target prediction network                   #
##############################################################

out_drug1 = layers.Dense(200)(representation)
out_drug2 = layers.Dense(200)(representation_2)
# out_drug2 = GlobalMaxPooling1D()(out_drug2)
# Protein_Protein_input_ = Conv1D(500,1)(Protein_Protein_input)
# Protein_Protein_input_ = Conv1D(200,1)(Protein_Protein_input_)

attention_prot = layers.MultiHeadAttention(
    num_heads=num_heads, key_dim=200, dropout=0.1
)(Protein_Protein_input, Protein_Protein_input)
predictions1 = layers.Dense(200, name='pred1', activation='sigmoid')(Lambda(dot_batch)([out_drug1, attention_prot]))
# predictions1 = tf.keras.layers.Activation('sigmoid',name='pred1')(predictions1)
predictions2 = layers.Dense(200, name='pred2', activation='sigmoid')(Lambda(dot_batch)([out_drug2, attention_prot]))
# predictions2 = tf.keras.layers.Activation('sigmoid',name='pred2')(predictions2)

# pro_matrix_mul = Lambda(dot_batch_axs_pro)([Protein_Protein_input,Protein_Protein_input])
# drug1_matrix_mul = Lambda(dot_batch_axs_drug)([predictions1,pro_matrix_mul])
# drug2_matrix_mul = Lambda(dot_batch_axs_drug)([predictions2,pro_matrix_mul])
# cell_drug1_mul = Lambda(dot_batch_axs_cell)([drug1_matrix_mul,Protein_Cell_input])
# cell_drug2_mul = Lambda(dot_batch_axs_cell)([drug2_matrix_mul,Protein_Cell_input])


##########################################################
#             Synergy prediction network                 #
##########################################################


drug1_matrix_mul = Lambda(dot_batch_axs_drug)([predictions1, attention_prot])
# drug1_matrix_mul = Softmax()(drug1_matrix_mul)
drug2_matrix_mul = Lambda(dot_batch_axs_drug)([predictions2, attention_prot])
# drug2_matrix_mul = Softmax()(drug2_matrix_mul)
cell_drug1_mul = Lambda(dot_batch_axs_drug)([predictions1, Protein_Cell_input])
# cell_drug1_mul = Softmax()(cell_drug1_mul)
cell_drug2_mul = Lambda(dot_batch_axs_drug)([predictions2, Protein_Cell_input])
# cell_drug2_mul = Softmax()(cell_drug2_mul)

drug1_matrix_mul = Flatten()(drug1_matrix_mul)
drug2_matrix_mul = Flatten()(drug2_matrix_mul)
cell_drug1_mul = Flatten()(cell_drug1_mul)
cell_drug2_mul = Flatten()(cell_drug2_mul)

concate_out1 = Concatenate()([representation, representation_2])
concate_out2 = Concatenate()([drug1_matrix_mul, drug2_matrix_mul])
concate_out3 = Concatenate()([cell_drug1_mul, cell_drug2_mul])

features1 = mlp(concate_out1, hidden_units=mlp_head_units, dropout_rate=0.1)
features2 = mlp(concate_out2, hidden_units=mlp_head_units, dropout_rate=0.1)
features3 = mlp(concate_out3, hidden_units=mlp_head_units, dropout_rate=0.1)
features = layers.Add()([features1,features2])
features = layers.Add()([features,features3])
# Classify outputs.
Synergy = layers.Dense(1, name='synergy', activation='sigmoid')(features)
# Create the Keras model.
# model = keras.Model(inputs=inputs, outputs=logits)

#################################################
#           Toxic prediction network            #
#################################################

predictions11 = tf.keras.layers.LayerNormalization(axis=-1)(predictions1)
predictions21 = tf.keras.layers.LayerNormalization(axis=-1)(predictions2)
Toxic = Lambda(dot_batch_axs_toxic)([predictions11, predictions21])
# Toxic = Softmax()(Toxic)
# ,activity_regularizer=regularizers.l2(1e-5)

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR')  # precision-recall curve
]
adam = tf.optimizers.Adam(learning_rate=0.01)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=300,
    decay_rate=0.9)
adam = tf.optimizers.Adam(learning_rate=lr_schedule)

# adam=tf.optimizers.Adam(learning_rate=0.01)
model1 = Model(inputs=[Drug1_input, Drug2_input,Protein_Protein_input,Protein_Cell_input], outputs=[predictions1,predictions2,representation, representation_2])
model1.compile(optimizer=adam, loss=['mse','mse', compactness_loss , compactness_loss])


model = Model(inputs=[Drug1_input, Drug2_input, Protein_Protein_input, Protein_Cell_input], outputs=[Toxic, Synergy])
model.compile(optimizer=adam, loss=['mse', 'binary_crossentropy'], metrics={'synergy': [METRICS]})



indx_of_drugs1_combinations = np.array(indx_of_drugs1_combinations)
indx_of_drugs2_combinations = np.array(indx_of_drugs2_combinations)
drugs_combinations = np.array(drugs_combinations)

smiles_unique = np.array(smiles_unique)
indx_of_cells_combinations = np.array(indx_of_cells_combinations)

choice = np.random.choice(range(indx_of_drugs1_combinations.shape[0]),
                          size=(int(indx_of_drugs1_combinations.shape[0] * 0.8),), replace=False)
ind = np.zeros(indx_of_drugs1_combinations.shape[0], dtype=bool)
ind[choice] = True
rest = np.argwhere(~ind)

print(choice.shape)
print(rest.shape)

############################################
#  Splitting data into test and train      #
############################################

indx_of_drugs1_combinations_train = indx_of_drugs1_combinations[choice]
indx_of_drugs2_combinations_train = indx_of_drugs2_combinations[choice]
drugs_combinations_train = drugs_combinations[choice, :]
indx_of_cells_combinations_train = indx_of_cells_combinations[choice]

indx_of_drugs1_combinations_test = indx_of_drugs1_combinations[rest]
indx_of_drugs1_combinations_test = indx_of_drugs1_combinations_test.reshape(
    (indx_of_drugs1_combinations_test.shape[0],))
indx_of_drugs2_combinations_test = indx_of_drugs2_combinations[rest]
indx_of_drugs2_combinations_test = indx_of_drugs2_combinations_test.reshape(
    (indx_of_drugs2_combinations_test.shape[0],))
drugs_combinations_test = drugs_combinations[rest, :]
drugs_combinations_test = np.squeeze(drugs_combinations_test)
indx_of_cells_combinations_test = indx_of_cells_combinations[rest]
indx_of_cells_combinations_test = indx_of_cells_combinations_test.reshape((indx_of_cells_combinations_test.shape[0],))

print(indx_of_cells_combinations_test.shape)
print(drugs_combinations_train.shape)




def generate_data(batch_size, protein_embeddings, indx_of_drugs1_combinations, indx_of_drugs2_combinations,
                  indx_of_cells_combinations, drugs_combinations, num_training_samples, flag):
    i_c = 0
    drugs1 = []
    drugs2 = []
    combinations = []
    cells = []
    interactions1 = []
    interactions2 = []
    while True:
        if i_c >= np.floor(num_training_samples / batch_size):
            i_c = 0
        drugs1 = smiles_unique[indx_of_drugs1_combinations[i_c * batch_size:(i_c + 1) * batch_size]]
        drugs2 = smiles_unique[indx_of_drugs2_combinations[i_c * batch_size:(i_c + 1) * batch_size]]
        combinations = drugs_combinations[i_c * batch_size:(i_c + 1) * batch_size]
        idx = indx_of_cells_combinations[i_c * batch_size:(i_c + 1) * batch_size]
        cells = protein_cell_matrix[:, idx]
        idx = indx_of_drugs1_combinations[i_c * batch_size:(i_c + 1) * batch_size]
        interactions1 = protein_drug_matrix[:, idx]
        idx = indx_of_drugs2_combinations[i_c * batch_size:(i_c + 1) * batch_size]
        interactions2 = protein_drug_matrix[:, idx]
        Toxic = np.zeros((batch_size, 1))
        protein_embeddings1 = np.expand_dims(protein_embeddings, axis=0)
        protein_embeddings2 = protein_embeddings1  # [0:15970,:]
        zero_out = np.zeros((batch_size, 1))
		
        i_c = i_c + 1
        
        if flag == 1:
            # yield np.concatenate((np.expand_dims(drugs1,axis=-1), np.expand_dims(drugs2,axis=-1)),axis=-1), combinations
            yield [np.expand_dims(drugs1, axis=-1), np.expand_dims(drugs2, axis=-1), protein_embeddings2, np.transpose(
                cells)], combinations  # [np.transpose(interactions1),np.transpose(interactions2),Toxic,combinations]#np.transpose(interactions1),np.transpose(interactions2),
        else:
            yield [np.expand_dims(drugs1, axis=-1), np.expand_dims(drugs2, axis=-1), protein_embeddings2, np.transpose(
                cells)], [np.transpose(interactions1), np.transpose(interactions2), zero_out, zero_out]


def generate_data_val(batch_size, protein_embeddings, indx_of_drugs1_combinations, indx_of_drugs2_combinations,
                      indx_of_cells_combinations, drugs_combinations, num_training_samples, flag):
    i_c = 0
    drugs1 = []
    drugs2 = []
    combinations = []
    cells = []
    interactions1 = []
    interactions2 = []
    while True:
        if i_c >= np.floor(num_training_samples / batch_size):
            i_c = 0
        drugs1 = smiles_unique[indx_of_drugs1_combinations[i_c * batch_size:(i_c + 1) * batch_size]]
        drugs2 = smiles_unique[indx_of_drugs2_combinations[i_c * batch_size:(i_c + 1) * batch_size]]
        combinations = drugs_combinations[i_c * batch_size:(i_c + 1) * batch_size]
        idx = indx_of_cells_combinations[i_c * batch_size:(i_c + 1) * batch_size]
        cells = protein_cell_matrix[:, idx]
        idx = indx_of_drugs1_combinations[i_c * batch_size:(i_c + 1) * batch_size]
        interactions1 = protein_drug_matrix[:, idx]
        idx = indx_of_drugs2_combinations[i_c * batch_size:(i_c + 1) * batch_size]
        interactions2 = protein_drug_matrix[:, idx]
        Toxic = np.zeros((batch_size, 1))
        protein_embeddings1 = np.expand_dims(protein_embeddings, axis=0)
        protein_embeddings2 = protein_embeddings1[0:15970, :]
        zero_out = np.zeros((batch_size, 1))
        
        i_c = i_c + 1
        if flag == 1:
            # yield np.concatenate((np.expand_dims(drugs1,axis=-1), np.expand_dims(drugs2,axis=-1)),axis=-1), combinations
            yield [np.expand_dims(drugs1, axis=-1), np.expand_dims(drugs2, axis=-1), protein_embeddings2, np.transpose(
                cells)], combinations  # [np.transpose(interactions1),np.transpose(interactions2),Toxic,combinations]#np.transpose(interactions1),np.transpose(interactions2)
        else:
            yield [np.expand_dims(drugs1, axis=-1), np.expand_dims(drugs2, axis=-1), protein_embeddings2, np.transpose(
                cells)], [np.transpose(interactions1), np.transpose(interactions2), zero_out, zero_out]


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=300,
    decay_rate=0.9)
adam = tf.optimizers.Adam(learning_rate=lr_schedule)
es = EarlyStopping(monitor=adam, mode='min', verbose=1, patience=15)


model1.fit_generator(
	generate_data(10, protein_embeddings, indx_of_drugs1_combinations_train, indx_of_drugs2_combinations_train,
				  indx_of_cells_combinations_train, drugs_combinations_train,
				  indx_of_drugs1_combinations_train.shape[0], 2),
	steps_per_epoch=54, epochs=100,
	validation_data=generate_data_val(10, protein_embeddings, indx_of_drugs1_combinations_test,
									  indx_of_drugs2_combinations_test, indx_of_cells_combinations_test,
									  drugs_combinations_test, indx_of_drugs1_combinations_test.shape[0], 2),
	validation_steps=13)

model.fit_generator(
	generate_data(1000, protein_embeddings, indx_of_drugs1_combinations_train, indx_of_drugs2_combinations_train,
				  indx_of_cells_combinations_train, drugs_combinations_train,
				  indx_of_drugs1_combinations_train.shape[0], 1),
	steps_per_epoch=54, epochs=300,
	validation_data=generate_data_val(1000, protein_embeddings, indx_of_drugs1_combinations_test,
									  indx_of_drugs2_combinations_test, indx_of_cells_combinations_test,
									  drugs_combinations_test, indx_of_drugs1_combinations_test.shape[0], 1),
	validation_steps=13)

