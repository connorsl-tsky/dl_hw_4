import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import scipy.sparse as sp
import pandas as pd
import sklearn.preprocessing as skpre
import ast



class GCNLayer(layers.Layer):
    """
    copied from gemini, mostly
    """

    def __init__(self, output_dim, activation=None, drouput=0.5, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        self.dropout = dropout

    def build(self, input_shape):
        # input_shape expects feature dimensions from inputs[0]
        feature_dim = input_shape[0][-1]
        self.weight = self.add_weight(
            name="weight",
            shape=(feature_dim, self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs, training=False):
        print("TRAINING: ", training)
        features, adj_normalized = inputs

        # dropout
        features = tf.nn.dropout(features, self.drop) if training else features


        # Linear transformation: XW
        support = tf.matmul(features, self.weight)

        # Graph convolution: A_hat * (XW)
        # If adj_normalized is a SparseTensor, use sparse_dense_matmul
        if isinstance(adj_normalized, tf.SparseTensor):
            output = tf.sparse.sparse_dense_matmul(adj_normalized, support)
        else:
            output = tf.matmul(adj_normalized, support)

        if self.activation is not None:
            output = self.activation(output)

        return output

def build_1layer_gcn(input_dim, hidden_dim, num_classes):
    """copied from gemini"""
    # Inputs: Node features (X) and normalized adjacency matrix (A_hat)
    x_in = layers.Input(shape=(input_dim,), name="node_features")
    adj_in = tf.keras.Input(
        shape=(None,), sparse=True, name="normalized_adjacency"
    )

    # 1-Layer GCN with Softmax for classification
    h = GCNLayer(hidden_dim, activation="relu")([x_in, adj_in])

    # average across all nodes in the graph
    graph_embed = layers.GlobalAveragePooling1D()(h)

    outputs = layers.Dense(num_classes, activation="softmax")(graph_embed)

    model = tf.keras.Model(inputs=[x_in, adj_in], outputs=outputs, name="1Layer_GCN")
    return model

def build_2layer_gcn(input_dim, hidden_dim, num_classes):
    x_in = layers.Input(shape=(input_dim,), name="node_features")
    adj_in = tf.keras.Input(
        shape=(None,), sparse=True, name="normalized_adjacency"
    )

    # Layer 1: Hidden layer with ReLU
    h = GCNLayer(hidden_dim, activation="relu")([x_in, adj_in])
    h = GCNLayer(hidden_dim, activation="relu")([h, adj_in])


    # average across all nodes in the graph
    graph_embed = layers.GlobalAveragePooling1D()(h)

    outputs = layers.Dense(num_classes, activation="softmax")(graph_embed)

    model = tf.keras.Model(inputs=[x_in, adj_in], outputs=outputs, name="2Layer_GCN")
    return model

def preprocess_adjacency(adj):
    """copied from gemini"""
    # Add self-loops: A_tilde = A + I
    adj_tilde = adj + sp.eye(adj.shape[0])

    # Calculate degree matrix D_tilde
    row_sum = np.array(adj_tilde.sum(1)).flatten()
    deg_inv_sqrt = np.power(row_sum, -0.5, where=row_sum > 0)
    deg_inv_sqrt[row_sum == 0] = 0.0
    d_mat_inv_sqrt = sp.diags(deg_inv_sqrt)

    # Symmetric normalization: D^(-1/2) * A_tilde * D^(-1/2)
    return d_mat_inv_sqrt.dot(adj_tilde).dot(d_mat_inv_sqrt).tocoo()

def standardize(row):
    return (row - np.mean(row)) / np.std(row) 

def convert_to_arr(row):
    new = []
    row = row.split("],[")
    for s in row:
        s = s.replace("[", "")
        s = s.replace("]", "")
        new.append(list(map(float, s.split(","))))
    return new





if __name__ == "__main__":
    # PREPROCESSING

    # load
    train = pd.read_csv("train.csv")


    # build model
    # remember to add l2

    # train model or load model

    # save weights

    # evaluate model

    # load test and preprocess

    # test model

    # build submission.csv (name on flag, or based on model name)

    # output AUC curve (on flag)
    

