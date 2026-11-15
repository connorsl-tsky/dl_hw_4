import tensorflow as tf
from tensorflow.keras import layers
import keras
import numpy as np
import scipy.sparse as sp
import pandas as pd
import sklearn.preprocessing as skpre
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class GCNLayer(layers.Layer):
    """
    copied from gemini, mostly
    """

    def __init__(self, output_dim, activation=None, dropout=0.5, **kwargs):
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

    # Average across all nodes in the graph (axis=0) to collapse to a graph-level vector
    # keepdims=True preserves a 2D shape of (1, hidden_dim) for the dense classifier
    graph_embed = layers.Lambda(lambda t: tf.reduce_mean(t, axis=0, keepdims=True))(h)

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

def convert_to_arr(row):
    new = []
    row = row.split("],[")
    for s in row:
        s = s.replace("[", "")
        s = s.replace("]", "")
        new.append(list(map(float, s.split(","))))
    return new


TRAIN = True
CHECKPOINT = False # Ignored if not training
SAVE = True # ignored when not training
SAVE_NAME = "gcn1.keras" # ends in .keras

if __name__ == "__main__":
    # PREPROCESSING
    # sorry i used AI (Gemini) to do a lot of this, I was not willing to spend hours figuring all this out myself
    # I knew what to do, I just didn't know how to do it (standardize features per row, one hot encode labels and append to features, sparse adj matrix)

    # load
    train = pd.read_csv("train.csv")

    # 1. Load DataFrame columns
    feature_strings = train['node_features'].to_numpy()
    label_strings = train['node_labels'].to_numpy()
    edge_strings = train['edge_index'].to_numpy()
    num_nodes_arr = train['num_nodes'].to_numpy()
    graph_labels = train['label'].to_numpy()  # Your target prediction

    # 2. Parse raw strings into lists of numpy arrays
    parsed_features = []
    parsed_adjs = []

    for f_str, l_str, e_str, n_nodes in zip(
        feature_strings, label_strings, edge_strings, num_nodes_arr
    ):
        # Parse continuous features (shape: num_nodes x 18)
        node_feats = np.array(ast.literal_eval(f_str), dtype=np.float32)

        # Parse and one-hot encode node labels (shape: num_nodes x 3)
        node_labs = np.array(ast.literal_eval(l_str)) - 1  # 0-indexed
        one_hot = np.eye(3)[node_labs] # 3 for classes 0,1,2 (1,2,3)

        # Temporarily combine them (we will scale features next)
        combined = np.concatenate([node_feats, one_hot], axis=1)
        parsed_features.append(combined) # contains features + one-hot

        # Parse edge_index and build SciPy COO adjacency matrix
        edge_index = ast.literal_eval(e_str)
        sources, targets = edge_index[0], edge_index[1]
        edge_weights = np.ones(len(sources), dtype=np.float32)
        adj_coo = sp.coo_matrix(
            (edge_weights, (sources, targets)), shape=(n_nodes, n_nodes)
        )
        parsed_adjs.append(adj_coo)

    # 3. Create train/validation/test splits of your graphs
    (
        X_train,
        X_val,
        adj_train,
        adj_val,
        y_train,
        y_val,
    ) = train_test_split(parsed_features, parsed_adjs, graph_labels, test_size=0.2)


    # 1. Collect all node features from the training set only
    # Note: indices [:, :18] slices out the 18 continuous features, leaving the last 3 one-hot columns alone!
    train_continuous = np.concatenate([g[:, :18] for g in X_train], axis=0)

    # 2. Fit the scaler on training data only
    scaler = StandardScaler()
    scaler.fit(train_continuous)

    # 3. Define a helper function to scale continuous features while leaving one-hot columns untouched
    def scale_graph_features(graph_list, scaler):
        scaled_graphs = []
        for g in graph_list:
            continuous_part = g[:, :18]
            one_hot_part = g[:, 18:]
            
            # Transform continuous features using the training-fitted scaler
            scaled_continuous = scaler.transform(continuous_part)
            
            # Re-concatenate with the unscaled one-hot labels
            scaled_combined = np.concatenate([scaled_continuous, one_hot_part], axis=1)
            scaled_graphs.append(scaled_combined)
        return scaled_graphs

    # 4. Apply transformation to train and validation sets
    X_train = scale_graph_features(X_train, scaler)
    X_val = scale_graph_features(X_val, scaler)

    # build model
    # remember to add l2
    if TRAIN:
        if CHECKPOINT:
            loaded = "check_"+SAVE_NAME
            print(f"LOADING {loaded}")
            model = keras.models.load_model("check_"+SAVE_NAME)
        else:
            num_feats = 18
            hidden_layers = 16
            num_classes = 6
            model = build_1layer_gcn(num_feats, hidden_layers, num_classes)

            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.01),
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=[keras.metrics.Accuracy, keras.metrics.F1Score(average="macro", threshold=0.5)]
            )

        model.fit(
            [X_train, adj_train], 
            y_train, 
            batch_size=32, 
            epochs=200, 
            validation_data=([X_val, adj_val], y_val),
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True,
                ),
                keras.callbacks.ModelCheckpoint("check_"+SAVE_NAME,save_best_only=True, monitor="val_loss")
            ]
        )

        if SAVE:
            model.save(SAVE_NAME)            
    else: 
        model = keras.models.load_model(SAVE_NAME)



    # evaluate model

    # load test and preprocess

    # test model

    # build submission.csv (name on flag, or based on model name)

    # output AUC curve (on flag)
    

