
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


drive.mount('/content/drive')

# %cd '/content/drive/My Drive/Project/DDSynergy/GraphSynergy/'
!dir
!git clone https://github.com/JasonJYang/GraphSynergy.git

dataset='OncologyScreen'

!pip install openpyxl==3.0.9
data = pd.read_excel('/content/drive/My Drive/Project/DDSynergy/GraphSynergy/GraphSynergy/data/'+dataset+'/protein-protein_network.xlsx')

yy=data.iloc[:,1]
list_of_prots=yy.unique()

yy=data.iloc[:,0]
tmp=yy.unique()
list_of_prots=np.concatenate((list_of_prots,tmp))
list_of_prots=np.unique(list_of_prots)

data = pd.read_csv('/content/drive/My Drive/Project/DDSynergy/GraphSynergy/GraphSynergy/data/'+dataset+'/drug_protein.csv')

yy=data.iloc[:,0]
list_of_drugs=yy.unique()

yy=data.iloc[:,1]
tmp=yy.unique()
list_of_prots=np.concatenate((list_of_prots,tmp))
list_of_prots=np.unique(list_of_prots)

data = pd.read_csv('/content/drive/My Drive/Project/DDSynergy/GraphSynergy/GraphSynergy/data/'+dataset+'/cell_protein.csv')

yy=data.iloc[:,3]
list_of_cells=yy.unique()

yy=data.iloc[:,2]
tmp=yy.unique()
list_of_prots=np.concatenate((list_of_prots,tmp))
list_of_prots=np.unique(list_of_prots)

data = pd.read_csv('/content/drive/My Drive/Project/DDSynergy/GraphSynergy/GraphSynergy/data/'+dataset+'/drug_combinations.csv')

yy=data.iloc[:,3]
tmp=yy.unique()
list_of_drugs=np.concatenate((list_of_drugs,tmp))
list_of_drugs=np.unique(list_of_drugs)

yy=data.iloc[:,4]
tmp=yy.unique()
list_of_drugs=np.concatenate((list_of_drugs,tmp))
list_of_drugs=np.unique(list_of_drugs)

yy=data.iloc[:,2]
tmp=yy.unique()
list_of_cells=np.concatenate((list_of_cells,tmp))
list_of_cells=np.unique(list_of_cells)

print(list_of_drugs.shape)

!pip install pubchempy
from pubchempy import *
import networkx as nx



## Read drug smiles
CHARCANSMISET = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, 
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
	for i, ch in enumerate(line[:MAX_SMI_LEN]): #	x, smi_ch_ind, y
		X[i] = smi_ch_ind[ch]

	return X #.tolist()

def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
	X = np.zeros(MAX_SEQ_LEN)

	for i, ch in enumerate(line[:MAX_SEQ_LEN]):
		X[i] = smi_ch_ind[ch]

	return X #.tolist()

smiles_dict_len = 64
smiles_max_len=100


data = pd.read_excel('/content/drive/My Drive/Project/DDSynergy/GraphSynergy/GraphSynergy/data/'+dataset+'/protein-protein_network.xlsx')


protein_protein_graph = nx.Graph()

for i in range(0,217160):
  tmp=np.argwhere(list_of_prots==data.iloc[i,0])[0]
  tmp1=np.argwhere(list_of_prots==data.iloc[i,1])[0]
  #data.iloc[i,0:2]
  #tmp=np.array(tmp)
  protein_protein_graph.add_edge(tmp[0], tmp1[0], weight=1)


data = pd.read_csv('/content/drive/My Drive/Project/DDSynergy/GraphSynergy/GraphSynergy/data/'+dataset+'/drug_protein.csv')


indx_of_drugs_interaction=[]
indx_of_prots_interaction=[]
print(list_of_prots.shape)
protein_drug_matrix=np.zeros((list_of_prots.shape[0],list_of_drugs.shape[0]))
for i in range(0,data.shape[0]):
     tmp=np.argwhere(list_of_drugs==data.iloc[i,0])[0]
     #indx_of_drugs_interaction.append(tmp[0])

     tmp2=np.argwhere(list_of_prots==data.iloc[i,1])[0]
     #indx_of_prots_interaction.append(tmp[0])

     protein_drug_matrix[tmp2[0],tmp[0]]=1
     #protein_interaction.append(data.iloc[i,1])

smiles_unique=np.load('/content/drive/My Drive/Project/DDSynergy/GraphSynergy/GraphSynergy/data/'+dataset+'/smiles.npy')
smiles_unique1=np.load('/content/drive/My Drive/Project/DDSynergy/GraphSynergy/GraphSynergy/data/'+dataset+'/smiles2.npy')
smiles_unique=np.concatenate((smiles_unique,smiles_unique1),axis=0)
smiles_unique1=np.load('/content/drive/My Drive/Project/DDSynergy/GraphSynergy/GraphSynergy/data/'+dataset+'/smiles3.npy')
smiles_unique=np.concatenate((smiles_unique,smiles_unique1),axis=0)
print(smiles_unique.shape)

smiles_max_len=200
smiles_unique1=[]
for i in range(0,list_of_drugs.shape[0]):
     print(i)
     cs = get_compounds(list_of_drugs[i], 'name')
     c = Compound.from_cid(cs[0].cid)
     smiles_unique1.append(label_smiles(c.canonical_smiles,smiles_max_len,CHARCANSMISET))

print(list_of_drugs.shape[0])
np.save('/content/drive/My Drive/Project/DDSynergy/GraphSynergy/GraphSynergy/data/'+dataset+'/smiles3.npy',np.array(smiles_unique1))


#smiles_unique=np.concatenate((smiles_unique,smiles_unique[507-list_of_drugs.shape[0]+508:507,:]),axis=0)


data = pd.read_csv('/content/drive/My Drive/Project/DDSynergy/GraphSynergy/GraphSynergy/data/'+dataset+'/cell_protein.csv')


protein_cell_matrix=np.zeros((list_of_prots.shape[0],list_of_cells.shape[0]))
for i in range(0,data.shape[0]):
     tmp=np.argwhere(list_of_prots==data.iloc[i,2])[0]
     tmp2=np.argwhere(list_of_cells==data.iloc[i,3])[0]
     protein_cell_matrix[tmp[0],tmp2[0]]=1

data = pd.read_csv('/content/drive/My Drive/Project/DDSynergy/GraphSynergy/GraphSynergy/data/'+dataset+'/drug_combinations.csv')

indx_of_drugs1_combinations=[]
indx_of_drugs2_combinations=[]
indx_of_cells_combinations=[]
drugs_combinations=[]
num_training_samples=data.shape[0]
sort_lbl=np.sort(np.array(data.iloc[:,5]))
print(sort_lbl.shape[0])
threshold1=sort_lbl[17359]
threshold2=sort_lbl[52077]
for i in range(0,data.shape[0]):
     tmp=np.argwhere(list_of_drugs==data.iloc[i,3])[0]
     indx_of_drugs1_combinations.append(tmp[0])
     tmp=np.argwhere(list_of_drugs==data.iloc[i,4])[0]
     indx_of_drugs2_combinations.append(tmp[0])
     tmp=np.argwhere(list_of_cells==data.iloc[i,2])[0]
     indx_of_cells_combinations.append(tmp[0])
     """if data.iloc[i,5]<threshold1:
        drugs_combinations.append([1,0,0])
     if data.iloc[i,5]>threshold2:
       drugs_combinations.append([0,0,1])
     else:
       drugs_combinations.append([0,1,0])"""
     if data.iloc[i,5]<0:#(threshold1+threshold2)/2:
        drugs_combinations.append([1])
     else:
       drugs_combinations.append([0])

smiles_unique1=np.load('/content/drive/My Drive/Project/DDSynergy/GraphSynergy/GraphSynergy/data/'+dataset+'/smiles_200.npy')
print(protein_cell_matrix.shape)
print(np.sum(drugs_combinations,axis=0))


vocabulary_protein = ["NA"] + list(protein_protein_graph.nodes)
vocabulary_lookup_protein = {token: idx for idx, token in enumerate(vocabulary_protein)}


def next_step(graph, previous, current, p, q):
    neighbors = list(graph.neighbors(current))

    weights = []
    # Adjust the weights of the edges to the neighbors with respect to p and q.
    for neighbor in neighbors:
        if neighbor == previous:
            # Control the probability to return to the previous node.
            weights.append(graph[current][neighbor]["weight"] / p)
        elif graph.has_edge(neighbor, previous):
            # The probability of visiting a local node.
            weights.append(graph[current][neighbor]["weight"])
        else:
            # Control the probability to move forward.
            weights.append(graph[current][neighbor]["weight"] / q)

    # Compute the probabilities of visiting each neighbor.
    weight_sum = sum(weights)
    probabilities = [weight / weight_sum for weight in weights]
    # Probabilistically select a neighbor to visit.
    next = np.random.choice(neighbors, size=1, p=probabilities)[0]
    return next


def random_walk(graph, num_walks,vocabulary_lookup, num_steps, p, q):
    walks = []
    nodes = list(graph.nodes())
    # Perform multiple iterations of the random walk.
    for walk_iteration in range(num_walks):
        random.shuffle(nodes)

        for node in tqdm(
            nodes,
            position=0,
            leave=True,
            desc=f"Random walks iteration {walk_iteration + 1} of {num_walks}",
        ):
            # Start the walk with a random node from the graph.
            walk = [node]
            # Randomly walk for num_steps.
            while len(walk) < num_steps:
                current = walk[-1]
                previous = walk[-2] if len(walk) > 1 else None
                # Compute the next node to visit.
                next = next_step(graph, previous, current, p, q)
                walk.append(next)
            # Replace node ids (movie ids) in the walk with token ids.
            walk = [vocabulary_lookup[token] for token in walk]
            # Add the walk to the generated sequence.
            walks.append(walk)

    return walks

##############################################

# Random walk return parameter.
p = 1
# Random walk in-out parameter.
q = 1
# Number of iterations of random walks.
num_walks = 3
# Number of steps of each random walk.
num_steps = 10

walks_protein = random_walk(protein_protein_graph, num_walks,vocabulary_lookup_protein, num_steps, p, q)


print("Number of protein walks generated:", len(walks_protein))

#################################################

def generate_examples(sequences, window_size, num_negative_samples, vocabulary_size):
    example_weights = defaultdict(int)
    # Iterate over all sequences (walks).
    for sequence in tqdm(
        sequences,
        position=0,
        leave=True,
        desc=f"Generating postive and negative examples",
    ):
        # Generate positive and negative skip-gram pairs for a sequence (walk).
        pairs, labels = keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocabulary_size,
            window_size=window_size,
            negative_samples=num_negative_samples,
        )
        for idx in range(len(pairs)):
            pair = pairs[idx]
            label = labels[idx]
            target, context = min(pair[0], pair[1]), max(pair[0], pair[1])
            if target == context:
                continue
            entry = (target, context, label)
            example_weights[entry] += 1

    targets, contexts, labels, weights = [], [], [], []
    for entry in example_weights:
        weight = example_weights[entry]
        target, context, label = entry
        targets.append(target)
        contexts.append(context)
        labels.append(label)
        weights.append(weight)

    return np.array(targets), np.array(contexts), np.array(labels), np.array(weights)


num_negative_samples = 10#4


# generate examples for Protein-Protein Network

targets_p, contexts_p, labels_p, weights_p = generate_examples(
    sequences=walks_protein,
    window_size=num_steps,
    num_negative_samples=num_negative_samples,
    vocabulary_size=len(vocabulary_protein),
)



batch_size = 1024


def create_dataset(targets, contexts, labels, weights, batch_size):
    inputs = {
        "target": targets,
        "context": contexts,
    }
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels, weights))
    dataset = dataset.shuffle(buffer_size=batch_size * 2)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


dataset_protein = create_dataset(
    targets=targets_p,
    contexts=contexts_p,
    labels=labels_p,
    weights=weights_p,
    batch_size=batch_size,
)


learning_rate = 0.001



def create_model(vocabulary_size, embedding_dim):

    inputs = {
        "target": layers.Input(name="target", shape=(), dtype="int32"),
        "context": layers.Input(name="context", shape=(), dtype="int32"),
    }
    # Initialize item embeddings.
    embed_item = layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dim,
        embeddings_initializer="he_normal",
        embeddings_regularizer=keras.regularizers.l2(1e-6),
        name="item_embeddings",
    )
    # Lookup embeddings for target.
    target_embeddings = embed_item(inputs["target"])
    # Lookup embeddings for context.
    context_embeddings = embed_item(inputs["context"])
    # Compute dot similarity between target and context embeddings.
    logits = layers.Dot(axes=1, normalize=False, name="dot_similarity")(
        [target_embeddings, context_embeddings]
    )
    # Create the model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


protein_embeding_dim=200


model_protein = create_model(len(vocabulary_protein), protein_embeding_dim)
model_protein.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
)

history_protein = model_protein.fit(dataset_protein, epochs=100)

protein_embeddings = model_protein.get_layer("item_embeddings").get_weights()[0]
print("Protein Embeddings shape:", protein_embeddings.shape)

np.save('/content/drive/My Drive/Project/DDSynergy/GraphSynergy/GraphSynergy/data/'+dataset+'/protein_embeddings_1.npy',protein_embeddings)
protein_embeddings=np.load('/content/drive/My Drive/Project/DDSynergy/GraphSynergy/GraphSynergy/data/'+dataset+'/protein_embeddings_1.npy')

hkj

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=200, random_state=0).fit(protein_embeddings[0:15970,:])
protein_embeddings = kmeans.cluster_centers_
protein_drug_matrix_kmeans=np.zeros((200,protein_drug_matrix.shape[1]))
for i in range(200):
    protein_drug_matrix_kmeans[i,:]=np.max(protein_drug_matrix[kmeans.labels_==i,:],axis=0)
protein_drug_matrix = protein_drug_matrix_kmeans

protein_cell_matrix
protein_cell_matrix_kmeans=np.zeros((200,protein_cell_matrix.shape[1]))
for i in range(200):
    protein_cell_matrix_kmeans[i,:]=np.max(protein_cell_matrix[kmeans.labels_==i,:],axis=0)
protein_cell_matrix = protein_cell_matrix_kmeans

import tensorflow
import keras
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, Softmax
from keras.layers import Conv2D, GRU
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, merge, Flatten, Concatenate, Lambda
from keras.models import Model
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers, layers
from tensorflow.keras import regularizers

#protein_embeddings1=protein_embeddings[0:15970,:]
#protein_embeddings=protein_embeddings1

def dot_batch(inp):
    import tensorflow as tf
    return tf.keras.backend.batch_dot(inp[0],inp[1],axes=(1, 2))#tf.keras.activations.sigmoid()
    
pro_matrix_mul_axs=(2,2)
drug_matrix_mul_axs=(1,2)
cell_drug_mul_axs=(1,1)

def dot_batch_axs_pro(inp):
    import tensorflow as tf
    return tf.keras.backend.batch_dot(inp[0],inp[1],axes=(2,2))

def dot_batch_axs_drug(inp):
    import tensorflow as tf
    return tf.keras.activations.sigmoid(tf.keras.backend.batch_dot(inp[0],inp[1],axes=(1,1)))

def dot_batch_axs_cell(inp):
    import tensorflow as tf
    return tf.keras.activations.sigmoid(tf.keras.backend.batch_dot(inp[0],inp[1],axes=(1,1)))#tf.math.multiply(inp[0],inp[1]))

def dot_batch_axs_toxic(inp):
    import tensorflow as tf
    return tf.keras.backend.batch_dot(inp[0],inp[1],axes=(1,1))

embedding_size=20
num_filters=64
protein_filter_lengths=8
smiles_filter_lengths=4
smiles_max_len=200
protein_embeding_dim=200
# Define Shared Layers
Drug1_input = Input(shape=(smiles_max_len,), name='drug1_input') #dtype='int32',
Drug2_input = Input(shape=(smiles_max_len,), name='drug2_input')#dtype='int32',

Protein_Protein_input = Input(shape=(list_of_prots.shape[0],protein_embeding_dim),name='protein_protein_input')
Protein_Cell_input = Input(shape=(list_of_prots.shape[0]),name='protein_cell_input')


METRICS = [
      #keras.metrics.TruePositives(name='tp'),
      #keras.metrics.FalsePositives(name='fp'),
      #keras.metrics.TrueNegatives(name='tn'),
      #keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR') # precision-recall curve
]


!pip install -U tensorflow-addons


print(tmp)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

num_classes = 2
input_shape = (200,1)

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
drug_size = 200  # We'll resize input sequences to this size
patch_size = 20  # Size of the patches to be extract from the input images
num_patches = (drug_size // patch_size) *4
projection_dim = 50
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [1024, 1024, 512]  # Size of the dense layers of the final classifier

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
            images=tf.expand_dims(images,axis=-1),
            sizes=[1, self.patch_size/4, 1, 1],
            strides=[1, self.patch_size/4, 1, 1],
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
    #augmented = data_augmentation(inputs)
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
	
def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            #keras.metrics.AUC(),
            #keras.metrics.AUC(curve='PR'),
            keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit_generator(generate_data(1000,protein_embeddings,indx_of_drugs1_combinations_train,indx_of_drugs2_combinations_train,indx_of_cells_combinations_train,drugs_combinations_train,indx_of_drugs1_combinations_train.shape[0],1),
                    steps_per_epoch = 54,epochs=100,validation_data=generate_data_val(1000,protein_embeddings,indx_of_drugs1_combinations_test,indx_of_drugs2_combinations_test,indx_of_cells_combinations_test,drugs_combinations_test,indx_of_drugs1_combinations_test.shape[0],1),
                    validation_steps=13)



    return history

num_prot_cluster=200

# Define Shared Layers
Drug1_input = Input(shape=(smiles_max_len,1), name='drug1_input') #dtype='int32',
Drug2_input = Input(shape=(smiles_max_len,1), name='drug2_input')#dtype='int32',

Protein_Protein_input = Input(shape=(num_prot_cluster,protein_embeding_dim),name='protein_protein_input')
Protein_Cell_input = Input(shape=(num_prot_cluster),name='protein_cell_input')


# network for Drug 1
#out1 = Embedding(input_dim=smiles_dict_len+1, output_dim = embedding_size, input_length=smiles_max_len,name='smiles_embedding') (Drug1_input)
patches = Patches(patch_size)(Drug1_input)
# Encode patches.
encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

# Create multiple layers of the Transformer block.
for _ in range(transformer_layers):
    # Layer normalization 1.
    x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # Create a multi-head attention layer.
    num_heads=1
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
out_drug1 = layers.Dense(200)(representation)
#out_drug1 = GlobalMaxPooling1D()(out_drug1)

num_classes = 2
input_shape = (200,1)

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
drug_size = 200  # We'll resize input sequences to this size
patch_size = 20  # Size of the patches to be extract from the input images
num_patches = (drug_size // patch_size) *4
projection_dim = 50
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [1024, 1024, 512]

#out2 = Embedding(input_dim=smiles_dict_len+1, output_dim = embedding_size, input_length=smiles_max_len,name='smiles_embedding2') (Drug2_input)
patches_2 = Patches(patch_size)(Drug2_input)
# Encode patches.
encoded_patches_2 = PatchEncoder(num_patches, projection_dim)(patches_2)

# Create multiple layers of the Transformer block.
for _ in range(transformer_layers):
    # Layer normalization 1.
    x1_2 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches_2)
    # Create a multi-head attention layer.
    num_heads=1
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
out_drug2 = layers.Dense(200)(representation_2)
#out_drug2 = GlobalMaxPooling1D()(out_drug2)
#Protein_Protein_input_ = Conv1D(500,1)(Protein_Protein_input)
#Protein_Protein_input_ = Conv1D(200,1)(Protein_Protein_input_)

attention_prot = layers.MultiHeadAttention(
       num_heads=num_heads, key_dim=200, dropout=0.1
    )(Protein_Protein_input,Protein_Protein_input)
predictions1 = layers.Dense(200,name='pred1',activation='sigmoid')(Lambda(dot_batch)([out_drug1, attention_prot]))
#predictions1 = tf.keras.layers.Activation('sigmoid',name='pred1')(predictions1)
predictions2 = layers.Dense(200,name='pred2',activation='sigmoid')(Lambda(dot_batch)([out_drug2, attention_prot]))
#predictions2 = tf.keras.layers.Activation('sigmoid',name='pred2')(predictions2)

#pro_matrix_mul = Lambda(dot_batch_axs_pro)([Protein_Protein_input,Protein_Protein_input])
#drug1_matrix_mul = Lambda(dot_batch_axs_drug)([predictions1,pro_matrix_mul])
#drug2_matrix_mul = Lambda(dot_batch_axs_drug)([predictions2,pro_matrix_mul])
#cell_drug1_mul = Lambda(dot_batch_axs_cell)([drug1_matrix_mul,Protein_Cell_input])
#cell_drug2_mul = Lambda(dot_batch_axs_cell)([drug2_matrix_mul,Protein_Cell_input])

drug1_matrix_mul = Lambda(dot_batch_axs_drug)([predictions1,attention_prot])
#drug1_matrix_mul = Softmax()(drug1_matrix_mul)
drug2_matrix_mul = Lambda(dot_batch_axs_drug)([predictions2,attention_prot])
#drug2_matrix_mul = Softmax()(drug2_matrix_mul)
cell_drug1_mul = Lambda(dot_batch_axs_drug)([predictions1,Protein_Cell_input])
#cell_drug1_mul = Softmax()(cell_drug1_mul)
cell_drug2_mul = Lambda(dot_batch_axs_drug)([predictions2,Protein_Cell_input])
#cell_drug2_mul = Softmax()(cell_drug2_mul)

drug1_matrix_mul = Flatten()(drug1_matrix_mul)
drug2_matrix_mul = Flatten()(drug2_matrix_mul)
cell_drug1_mul = Flatten()(cell_drug1_mul)
cell_drug2_mul = Flatten()(cell_drug2_mul)

concate_out1 = Concatenate()([representation,representation_2])
concate_out2 = Concatenate()([drug1_matrix_mul,drug2_matrix_mul])
concate_out3 = Concatenate()([cell_drug1_mul,cell_drug2_mul])

features = mlp(concate_out1, hidden_units=mlp_head_units, dropout_rate=0.1)
#features2 = mlp(concate_out2, hidden_units=mlp_head_units, dropout_rate=0.1)
#features3 = mlp(concate_out3, hidden_units=mlp_head_units, dropout_rate=0.1)
#features = layers.Add()([features1,features2])
#features = layers.Add()([features,features3])
# Classify outputs.
Synergy = layers.Dense(1, name='synergy', activation='sigmoid')(features)
# Create the Keras model.
#model = keras.Model(inputs=inputs, outputs=logits)

#concate_out2 = Concatenate()([out1,out2])
#from tensorflow.keras import regularizers
#FC_Synergy1 = Dense(100, activation='relu', name='dense_synergy1')(concate_out)
#DO_Synergy1 = Dropout(0.5)(FC_Synergy1)
#FC_Synergy2 = Dense(100, activation='relu', name='dense_synergy2')(FC_Synergy1)
""",kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
     bias_regularizer=regularizers.l2(1e-4),
     activity_regularizer=regularizers.l2(1e-5)"""
#Synergy = Dense(2, activation='softmax', name='synergy')(FC_Synergy2)
#,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#    bias_regularizer=regularizers.l2(1e-4),
#    activity_regularizer=regularizers.l2(1e-5)

predictions11 = tf.keras.layers.LayerNormalization(axis=-1)(predictions1)
predictions21 = tf.keras.layers.LayerNormalization(axis=-1)(predictions2)
Toxic = Lambda(dot_batch_axs_toxic)([predictions11,predictions21])
#Toxic = Softmax()(Toxic)
#,activity_regularizer=regularizers.l2(1e-5)

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR') # precision-recall curve
]
adam=tf.optimizers.Adam(learning_rate=0.01)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=300,
    decay_rate=0.9)
adam=tf.optimizers.Adam(learning_rate=lr_schedule)

#adam=tf.optimizers.Adam(learning_rate=0.01)
#model = Model(inputs=[Drug1_input, Drug2_input,Protein_Protein_input,Protein_Cell_input], outputs=[predictions1,predictions2,Toxic,Synergy])
#model.compile(optimizer=adam, loss=['mse','mse','mse','binary_crossentropy'], metrics={'synergy':[METRICS]})
model = Model(inputs=[Drug1_input, Drug2_input,Protein_Protein_input,Protein_Cell_input], outputs=[Synergy])
model.compile(optimizer=adam, loss=['binary_crossentropy'], metrics={'synergy':[METRICS]})
indx_of_drugs1_combinations=np.array(indx_of_drugs1_combinations)
indx_of_drugs2_combinations=np.array(indx_of_drugs2_combinations)
drugs_combinations=np.array(drugs_combinations)
smiles_unique=smiles_unique1
smiles_unique=np.array(smiles_unique)
indx_of_cells_combinations=np.array(indx_of_cells_combinations)

choice = np.random.choice(range(indx_of_drugs1_combinations.shape[0]), size=(int(indx_of_drugs1_combinations.shape[0]*0.8),), replace=False)    
ind = np.zeros(indx_of_drugs1_combinations.shape[0], dtype=bool)
ind[choice] = True
rest = np.argwhere(~ind)

print(choice.shape)
print(rest.shape)

indx_of_drugs1_combinations_train=indx_of_drugs1_combinations[choice]
indx_of_drugs2_combinations_train=indx_of_drugs2_combinations[choice]
drugs_combinations_train=drugs_combinations[choice,:]
indx_of_cells_combinations_train=indx_of_cells_combinations[choice]

indx_of_drugs1_combinations_test=indx_of_drugs1_combinations[rest]
indx_of_drugs1_combinations_test=indx_of_drugs1_combinations_test.reshape((indx_of_drugs1_combinations_test.shape[0],))
indx_of_drugs2_combinations_test=indx_of_drugs2_combinations[rest]
indx_of_drugs2_combinations_test=indx_of_drugs2_combinations_test.reshape((indx_of_drugs2_combinations_test.shape[0],))
drugs_combinations_test=drugs_combinations[rest,:]
drugs_combinations_test=np.squeeze(drugs_combinations_test)
indx_of_cells_combinations_test=indx_of_cells_combinations[rest]
indx_of_cells_combinations_test=indx_of_cells_combinations_test.reshape((indx_of_cells_combinations_test.shape[0],))

print(indx_of_cells_combinations_test.shape)
print(drugs_combinations_train.shape)

import tensorflow as tf

def generate_data(batch_size,protein_embeddings,indx_of_drugs1_combinations,indx_of_drugs2_combinations,indx_of_cells_combinations,drugs_combinations,num_training_samples,flag):
  i_c = 0
  drugs1=[]
  drugs2=[]
  combinations=[]
  cells=[]
  interactions1=[]
  interactions2=[]
  while True:
    if i_c>=np.floor(num_training_samples/batch_size):
      i_c=0
    drugs1=smiles_unique[indx_of_drugs1_combinations[i_c*batch_size:(i_c+1)*batch_size]]
    drugs2=smiles_unique[indx_of_drugs2_combinations[i_c*batch_size:(i_c+1)*batch_size]]
    combinations=drugs_combinations[i_c*batch_size:(i_c+1)*batch_size]
    idx=indx_of_cells_combinations[i_c*batch_size:(i_c+1)*batch_size]
    cells=protein_cell_matrix[:,idx]
    idx=indx_of_drugs1_combinations[i_c*batch_size:(i_c+1)*batch_size]
    interactions1=protein_drug_matrix[:,idx]
    idx=indx_of_drugs2_combinations[i_c*batch_size:(i_c+1)*batch_size]
    interactions2=protein_drug_matrix[:,idx]
    Toxic=np.zeros((batch_size,1))
    protein_embeddings1=np.expand_dims(protein_embeddings,axis=0)
    protein_embeddings2=protein_embeddings1#[0:15970,:]
    """print(drugs1.shape)
    print(drugs2.shape)
    print(cells.shape)
    print(interactions1.shape)
    print(interactions2.shape)
    print(combinations.shape)
    print(Toxic.shape)
    print(i_c)"""
    i_c=i_c+1
    """print(drugs1)
    print(drugs2)
    print(drugs1.shape)
    print(drugs2.shape)
    print(np.sum(combinations,axis=0))"""
    if flag==1:
       #yield np.concatenate((np.expand_dims(drugs1,axis=-1), np.expand_dims(drugs2,axis=-1)),axis=-1), combinations
       yield [np.expand_dims(drugs1,axis=-1), np.expand_dims(drugs2,axis=-1),protein_embeddings2,np.transpose(cells)], combinations#[np.transpose(interactions1),np.transpose(interactions2),Toxic,combinations]#np.transpose(interactions1),np.transpose(interactions2),
    else:
       yield [drugs1, drugs2,protein_embeddings2], [np.transpose(interactions1),np.transpose(interactions2)]
    
def generate_data_val(batch_size,protein_embeddings,indx_of_drugs1_combinations,indx_of_drugs2_combinations,indx_of_cells_combinations,drugs_combinations,num_training_samples,flag):
  i_c = 0
  drugs1=[]
  drugs2=[]
  combinations=[]
  cells=[]
  interactions1=[]
  interactions2=[]
  while True:
    if i_c>=np.floor(num_training_samples/batch_size):
      i_c=0
    drugs1=smiles_unique[indx_of_drugs1_combinations[i_c*batch_size:(i_c+1)*batch_size]]
    drugs2=smiles_unique[indx_of_drugs2_combinations[i_c*batch_size:(i_c+1)*batch_size]]
    combinations=drugs_combinations[i_c*batch_size:(i_c+1)*batch_size]
    idx=indx_of_cells_combinations[i_c*batch_size:(i_c+1)*batch_size]
    cells=protein_cell_matrix[:,idx]
    idx=indx_of_drugs1_combinations[i_c*batch_size:(i_c+1)*batch_size]
    interactions1=protein_drug_matrix[:,idx]
    idx=indx_of_drugs2_combinations[i_c*batch_size:(i_c+1)*batch_size]
    interactions2=protein_drug_matrix[:,idx]
    Toxic=np.zeros((batch_size,1))
    protein_embeddings1=np.expand_dims(protein_embeddings,axis=0)
    protein_embeddings2=protein_embeddings1[0:15970,:]
    """print(drugs1.shape)
    print(drugs2.shape)
    print(cells.shape)
    print(interactions1.shape)
    print(interactions2.shape)
    print(combinations.shape)
    print(protein_embeddings2.shape)
    print(i_c)"""
    i_c=i_c+1
    if flag==1:
       #yield np.concatenate((np.expand_dims(drugs1,axis=-1), np.expand_dims(drugs2,axis=-1)),axis=-1), combinations
       yield [np.expand_dims(drugs1,axis=-1), np.expand_dims(drugs2,axis=-1),protein_embeddings2,np.transpose(cells)], combinations#[np.transpose(interactions1),np.transpose(interactions2),Toxic,combinations]#np.transpose(interactions1),np.transpose(interactions2)
    else:
       yield [drugs1, drugs2,protein_embeddings2], [np.transpose(interactions1),np.transpose(interactions2)]

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=300,
    decay_rate=0.9)
adam=tf.optimizers.Adam(learning_rate=lr_schedule)
es = EarlyStopping(monitor=adam, mode='min', verbose=1, patience=15)
#model.compile(optimizer=adam, loss=['mse','mse','mse','categorical_crossentropy'], loss_weights=[1,1,1,2], metrics={'synergy':[METRICS]}) 

#model2.fit_generator(generate_data(1000,protein_embeddings,indx_of_drugs1_combinations_train,indx_of_drugs2_combinations_train,indx_of_cells_combinations_train,drugs_combinations_train,indx_of_drugs1_combinations_train.shape[0],2),
#                    steps_per_epoch = 54,epochs=5,validation_data=generate_data_val(1000,protein_embeddings,indx_of_drugs1_combinations_test,indx_of_drugs2_combinations_test,indx_of_cells_combinations_test,drugs_combinations_test,indx_of_drugs1_combinations_test.shape[0],2),
#
#                    validation_steps=13)
#model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics={'synergy':[METRICS]}) #, loss_weights=[1,2]

#model3.fit_generator(generate_data(1000,protein_embeddings,indx_of_drugs1_combinations_train,indx_of_drugs2_combinations_train,indx_of_cells_combinations_train,drugs_combinations_train,indx_of_drugs1_combinations_train.shape[0],1),
#                    steps_per_epoch = 54,epochs=100,validation_data=generate_data_val(1000,protein_embeddings,indx_of_drugs1_combinations_test,indx_of_drugs2_combinations_test,indx_of_cells_combinations_test,drugs_combinations_test,indx_of_drugs1_combinations_test.shape[0],1),
#                    validation_steps=13)

model.fit_generator(generate_data(1000,protein_embeddings,indx_of_drugs1_combinations_train,indx_of_drugs2_combinations_train,indx_of_cells_combinations_train,drugs_combinations_train,indx_of_drugs1_combinations_train.shape[0],1),
                    steps_per_epoch = 54,epochs=300,validation_data=generate_data_val(1000,protein_embeddings,indx_of_drugs1_combinations_test,indx_of_drugs2_combinations_test,indx_of_cells_combinations_test,drugs_combinations_test,indx_of_drugs1_combinations_test.shape[0],1),
                    validation_steps=13)


