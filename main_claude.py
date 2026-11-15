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
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import skew

@keras.saving.register_keras_serializable(package="gcn")
class GCNLayer(layers.Layer):
    def __init__(self, output_dim, activation=None, dropout=0.5, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.activation_name = activation
        self.activation = keras.activations.get(activation)
        self.dropout = dropout

    def build(self, input_shape):
        self.w = self.add_weight(name="w", shape=(input_shape[0][-1], self.output_dim),
                                 initializer="glorot_uniform", trainable=True)

    def call(self, inputs, training=False):
        x, adj = inputs
        if training and self.dropout > 0:
            x = tf.nn.dropout(x, self.dropout)
        out = tf.sparse.sparse_dense_matmul(adj, tf.matmul(x, self.w))
        return self.activation(out)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(output_dim=self.output_dim, activation=self.activation_name,
                   dropout=self.dropout)
        return cfg


@keras.saving.register_keras_serializable(package="gcn")
class MeanPool(layers.Layer):
    def call(self, x):                       # (nodes, hidden) -> (1, hidden)
        return tf.reduce_mean(x, axis=0, keepdims=True)


def build_gcn(input_dim, hidden_dim, num_classes, n_layers=1, dropout=0.5):
    x_in = layers.Input(shape=(input_dim,), name="node_features")
    adj_in = layers.Input(shape=(None,), sparse=True, name="normalized_adjacency")
    h = x_in
    for _ in range(n_layers):
        h = GCNLayer(hidden_dim, activation="relu", dropout=dropout)([h, adj_in])
    logits = layers.Dense(num_classes)(MeanPool()(h))     # logits, no softmax
    return keras.Model([x_in, adj_in], logits)

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

def preprocess(train, test=False, scale=None):
    # 1. Load DataFrame columns
    feature_strings = train['node_features'].to_numpy()
    label_strings = train['node_labels'].to_numpy()
    edge_strings = train['edge_index'].to_numpy()
    num_nodes_arr = train['num_nodes'].to_numpy()
    if not test:
        graph_labels = train['label'].to_numpy()-1  # Your target prediction

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
    if not test:
        (
            X_train,
            X_val,
            adj_train,
            adj_val,
            y_train,
            y_val,
        ) = train_test_split(parsed_features, parsed_adjs, graph_labels, test_size=0.2, random_state=1, stratify=graph_labels)
    else:
        # test data
        X_train = parsed_features
        adj_train = parsed_adjs


    if not test and scale is None:
        # 1. Collect all node features from the training set only
        # Note: indices [:, :18] slices out the 18 continuous features, leaving the last 3 one-hot columns alone!
        train_continuous = np.concatenate([g[:, :18] for g in X_train], axis=0)
        # 2. Fit the scaler on training data only
        scaler = StandardScaler()
        scaler.fit(train_continuous)
    elif test and scale is not None:
        scaler = scale
    else:
        raise ValueError(f"test is {test}, scale is {scale}")

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
    if not test:
        X_val = scale_graph_features(X_val, scaler)

    def to_sparse(adj, undirected=True):
        if undirected:                                    
            adj = ((adj + adj.T) > 0).astype(np.float32)
        a = preprocess_adjacency(adj.tocoo())
        idx = np.column_stack([a.row, a.col]).astype(np.int64)
        return tf.sparse.reorder(tf.SparseTensor(idx, a.data.astype(np.float32), list(a.shape)))

    X_train = [tf.constant(g, dtype=tf.float32) for g in X_train]
    adj_train = [to_sparse(a) for a in adj_train]
    if test:
        return X_train, adj_train
    else:
        X_val   = [tf.constant(g, dtype=tf.float32) for g in X_val]
        adj_val   = [to_sparse(a) for a in adj_val]
        y_train = np.asarray(y_train, dtype=np.int32)
        y_val   = np.asarray(y_val, dtype=np.int32)
        return X_train, X_val, adj_train, adj_val, y_train, y_val, scaler

def predict(model, X, A):
    return tf.concat([model([x, a], training=False) for x, a in zip(X, A)], axis=0)


TRAIN = False
# CHECKPOINT = False # Ignored if not training
SAVE = True # ignored when not training
SAVE_NAME = "gcn1.keras" # ends in .keras
SUBMISSION = False
AUC = True

if __name__ == "__main__":
    # PREPROCESSING
    # sorry i used AI (Gemini) to do a lot of this, I was not willing to spend hours figuring all this out myself
    # I knew what to do, I just didn't know how to do it (standardize features per row, one hot encode labels and append to features, sparse adj matrix)

    # load
    train = pd.read_csv("train.csv")
    X_train, X_val, adj_train, adj_val, y_train, y_val, scaler = preprocess(train, test=False, scale=None)
    
    # build model
    # remember to add l2
    if TRAIN:
        # if CHECKPOINT:
        #     loaded = "check_"+SAVE_NAME
        #     print(f"LOADING {loaded}")
        #     model = keras.models.load_model("check_"+SAVE_NAME)
        # else:
        model = build_gcn(21, 16, num_classes=6, n_layers=1)
        opt = keras.optimizers.Adam(0.01)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        L2 = 5e-4
        first_w = [l for l in model.layers if isinstance(l, GCNLayer)][0].w   # L2 on first layer, per the paper
        BATCH, PATIENCE = 32, 25


        rng = np.random.default_rng(0)
        best_val, best_weights, wait = np.inf, None, 0

        for epoch in range(200):
            order = rng.permutation(len(X_train))
            for start in range(0, len(order), BATCH):
                idx = order[start:start + BATCH]
                with tf.GradientTape() as tape:
                    # one forward pass per graph, then a single loss over the 32 graphs
                    logits = tf.concat([model([X_train[i], adj_train[i]], training=True) for i in idx], axis=0)
                    loss = loss_fn(y_train[idx], logits) + L2 * tf.nn.l2_loss(first_w)
                grads = tape.gradient(loss, model.trainable_variables)
                opt.apply_gradients(zip(grads, model.trainable_variables))

            val_logits = predict(model, X_val, adj_val)
            val_loss = float(loss_fn(y_val, val_logits))
            val_pred = tf.argmax(val_logits, axis=1).numpy()
            print(f"epoch {epoch:3d}  train_loss={float(loss):.3f}  val_loss={val_loss:.3f}  "
                f"val_acc={accuracy_score(y_val, val_pred):.3f}  "
                f"val_f1={f1_score(y_val, val_pred, average='macro'):.3f}")
            train_pred = tf.argmax(predict(model, X_train, adj_train), axis=1).numpy()
            print(f"train_acc {accuracy_score(y_train, train_pred):.3f} train_f1={f1_score(y_train, train_pred, average='macro'):.3f}")

            if val_loss < best_val:
                best_val, best_weights, wait = val_loss, model.get_weights(), 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        model.set_weights(best_weights)          # same as restore_best_weights=True
        if SAVE:
            model.save(SAVE_NAME)         
    else: 
        print(f"LOADING {SAVE_NAME}")
        model = keras.models.load_model(SAVE_NAME)

    if SUBMISSION:
        # load test and preprocess
        test = pd.read_csv("test.csv")
        X_test, adj_test = preprocess(test, test=True, scale=scaler)

        # test model
        test_logits = predict(model, X_test, adj_test)
        test_pred = tf.argmax(test_logits, axis=1).numpy()
        print(test_pred)

        # build submission.csv (name on flag, or based on model name)
        sub = pd.DataFrame({"label": test_pred})
        print(sub)
        sub.insert(0, "graph_id", test["graph_id"])
        print(sub, "after insert")
        sub.to_csv(SAVE_NAME+"_submission.csv", index=False)

        # output AUC curve (on flag)
        if AUC:
            pass
    

"""
runs

after first run to diagnose low acc/f1
epoch 127  train_loss=1.405  val_loss=1.525  val_acc=0.396  val_f1=0.380
train_acc 0.555 train_f1=0.380
nope f1 is bad
train_acc 0.516 train_f1=0.513, it should be close
BIN COUNT [64 64 64 64 64 64] - np.bincount(y_train)
RFC acc - 0.7291666666666666
RFC f1 - 0.726920413815575

"""

