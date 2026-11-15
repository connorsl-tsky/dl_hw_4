import tensorflow as tf
from tensorflow.keras import layers
import keras
import numpy as np
import scipy.sparse as sp
import pandas as pd
import ast
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, roc_auc_score
import sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
import joblib


@keras.saving.register_keras_serializable(package="gcn")
class GCNLayer(layers.Layer):
    """
    custom keras GCN layer (tf/keras doesn't have a native one)
    written by claude
    """
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
class MeanMaxPool(layers.Layer):
    """
    custom layer to aggregate over the vector of node outputs from GCNlayer
    to be fed to a dense layer for estimation
    written by claude
    """
    def call(self, x):   # x: (nodes, hidden)
        # Average over nodes, and also take the biggest value over nodes,
        # then place the two side by side -> shape (1, 2*hidden).
        return tf.concat([tf.reduce_mean(x, axis=0, keepdims=True),
                          tf.reduce_max(x, axis=0, keepdims=True)], axis=1)


def build_gcn(input_dim=21, hidden_dim=32, num_classes=6, n_layers=1, dropout=0.2):
    """
    builds a gcn model
    written by claude, but it's an improvement over the one I had before
    """
    x_in = layers.Input(shape=(input_dim,), name="node_features")
    adj_in = layers.Input(shape=(None,), sparse=True, name="normalized_adjacency")
    h = x_in
    for _ in range(n_layers):
        h = GCNLayer(hidden_dim, activation="relu", dropout=dropout)([h, adj_in])
    logits = layers.Dense(num_classes)(MeanMaxPool()(h))     # logits, no softmax
    return keras.Model([x_in, adj_in], logits)

def preprocess_adjacency(adj):
    """
    copied from gemini
    takes the adjacency matrix and preprocesses it to match the
    symmetric normalized matrix found in the GCN paper
    """
    # Add self-loops: A_tilde = A + I
    adj_tilde = adj + sp.eye(adj.shape[0])

    # Calculate degree matrix D_tilde
    row_sum = np.array(adj_tilde.sum(1)).flatten()
    deg_inv_sqrt = np.power(row_sum, -0.5, where=row_sum > 0)
    deg_inv_sqrt[row_sum == 0] = 0.0
    d_mat_inv_sqrt = sp.diags(deg_inv_sqrt)

    # Symmetric normalization: D^(-1/2) * A_tilde * D^(-1/2)
    return d_mat_inv_sqrt.dot(adj_tilde).dot(d_mat_inv_sqrt).tocoo()

def preprocess(train, test=False, scale=None):
    """
    preprocesses input dataframe
    sorry for bad naming, i'm lazy. train should be df or data
    turns node_labels to one_hot, and appends to features
    standardizes features
    computes the adjacency matrices, and preprocesses them per preprocess_adjacency
    and turns the adj matrices to SparseTensors for more efficient memory usage
    a lot of it was written by gemini, some of it was modified by claude
    """
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
        raise ValueError(f"test is {test}, scale is {scale}. If scale is None, test should be False, if scale is not None, test should be True")

    # 3. Define a helper function to scale continuous features while leaving one-hot columns untouched
    def scale_graph_features(graph_list, scaler):
        scaled_graphs = []
        for g in graph_list:
            continuous_part = g[:, :18]
            one_hot_part = g[:, 18:]
            
            # Transform continuous features using the training-fitted scaler
            scaled_continuous = np.clip(scaler.transform(continuous_part), -5, 5)            
            
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
    """
    quick function to create a prediction array
    outputs logits
    you do have to `val_pred = tf.argmax(val_logits, axis=1).numpy()`
    in order to convert it to predictions
    """
    return tf.concat([model([x, a], training=False) for x, a in zip(X, A)], axis=0)

def plot_auc(y_true, y_proba, save_path="roc_curve.png", title="ROC curves (one-vs-rest)"):
    """
    y_true:  array of true labels, 0-5, shape (n_graphs,)
    y_proba: array of predicted probabilities, shape (n_graphs, 6).
            Each row is one graph; each column is the model's confidence in one class.
    written by claude
    plots an auc/roc graph, saves it as a png
    """
    n_classes = y_proba.shape[1]

    # Turn labels like 3 into [0,0,0,1,0,0], so each class gets its own yes/no column.
    y_onehot = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=(7, 6))
    for c in range(n_classes):
        # roc_curve sweeps a threshold and returns the false-positive and true-positive rates
        fpr, tpr, _ = roc_curve(y_onehot[:, c], y_proba[:, c])
        plt.plot(fpr, tpr, label=f"class {c + 1} (AUC = {auc(fpr, tpr):.2f})")

    # Average of the 6 per-class AUCs
    macro_auc = roc_auc_score(y_onehot, y_proba, average="macro")

    plt.plot([0, 1], [0, 1], "k--", label="random guessing")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"{title}\nmacro AUC = {macro_auc:.3f}")
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return macro_auc


TRAIN = False
SAVE = True # ignored when not training
SAVE_NAME = "gcn3.keras" # ends in .keras
SUBMISSION = True
AUC = True
LAYERS = 3

if __name__ == "__main__":
    # load and preprocess
    train = pd.read_csv("train.csv")
    X_train, X_val, adj_train, adj_val, y_train, y_val, scaler = preprocess(train, test=False, scale=None)
    
    # build model
    if TRAIN:
        model = build_gcn(21, 32, num_classes=6, n_layers=LAYERS)
        opt = keras.optimizers.Adam(0.01)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        L2 = 5e-4
        first_w = [l for l in model.layers if isinstance(l, GCNLayer)][0].w   # L2 on first layer, per the paper
        BATCH, PATIENCE = 32, 25

        rng = np.random.default_rng(0)
        best_val, best_weights, wait = np.inf, None, 0
        best_epoch = 0

        for epoch in range(200):
            order = rng.permutation(len(X_train))

            # batch training
            for start in range(0, len(order), BATCH):
                idx = order[start:start + BATCH]
                with tf.GradientTape() as tape:
                    # one forward pass per graph, then a single loss over the 32 graphs
                    logits = tf.concat([model([X_train[i], adj_train[i]], training=True) for i in idx], axis=0)
                    loss = loss_fn(y_train[idx], logits) + L2 * tf.nn.l2_loss(first_w)
                grads = tape.gradient(loss, model.trainable_variables)
                opt.apply_gradients(zip(grads, model.trainable_variables))
            
            # validation output
            val_logits = predict(model, X_val, adj_val)
            val_loss = float(loss_fn(y_val, val_logits))
            val_pred = tf.argmax(val_logits, axis=1).numpy()
            print(f"epoch {epoch:3d}  train_loss={float(loss):.3f}  val_loss={val_loss:.3f}  "
                f"val_acc={accuracy_score(y_val, val_pred):.3f}  "
                f"val_f1={f1_score(y_val, val_pred, average='macro'):.3f}")
            train_pred = tf.argmax(predict(model, X_train, adj_train), axis=1).numpy()
            print(f"train_acc {accuracy_score(y_train, train_pred):.3f} train_f1={f1_score(y_train, train_pred, average='macro'):.3f}")
            print(f"best epoch: {best_epoch}")

            # early stop
            if val_loss < best_val:
                best_val, best_weights, wait = val_loss, model.get_weights(), 0
                best_epoch = epoch
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        
        # saves best weights to SAVE_NAME
        model.set_weights(best_weights)          # same as restore_best_weights=True
        if SAVE:
            model.save(SAVE_NAME)         
    else: 
        print(f"LOADING {SAVE_NAME}")
        model = keras.models.load_model(SAVE_NAME)

    if AUC:
        # outputs AUC for GCN model
        val_logits = predict(model, X_val, adj_val)
        val_proba = tf.nn.softmax(val_logits).numpy()
        plot_auc(y_val, val_proba, save_path=SAVE_NAME.split(".")[0]+"_roc.png", title=f"{SAVE_NAME.split(".")[0]} ROC")


    if SUBMISSION:
        # train ETC on all data, test on test set and create submission.csv
        def summarize(x, adj):                      # x: (n, 21) numpy, adj: tf.SparseTensor
            n = len(x)
            deg = np.bincount(adj.indices[:, 0].numpy(), minlength=n) - 1   # minus self-loop
            cont = x[:, :18]
            return np.concatenate([
                cont.mean(0), cont.std(0), cont.min(0), cont.max(0),
                np.percentile(cont, [25, 50, 75], axis=0).ravel(),
                x[:, 18:].mean(0),                  # fraction of nodes with label 1/2/3
                [n, np.log(n), deg.mean(), deg.std(), deg.max(), (deg == 0).mean(),
                deg.sum() / max(n * (n - 1), 1)],  # size, degree stats, density
            ])

        X_all = np.stack([summarize(x.numpy(), a) for x, a in
                        zip(X_train + X_val, adj_train + adj_val)])
        y_all = np.concatenate([y_train, y_val])

        # Train on all 480 labeled graphs (train + validation together).
        etc = ExtraTreesClassifier(n_estimators=500, random_state=0)
        etc.fit(X_all, y_all)

        # Load the test file and apply the SAME preprocessing and scaler as before.
        test = pd.read_csv("test.csv")
        X_test, adj_test = preprocess(test, test=True, scale=scaler)

        # Turn every test graph into one row of summary numbers, like X_all.
        X_test_tab = np.stack([summarize(x.numpy(), a) for x, a in zip(X_test, adj_test)])

        # The model predicts 0-5, but your original labels were shifted down by 1, so shift back up.
        pred = etc.predict(X_test_tab) + 1

        pd.DataFrame({"graph_id": test["graph_id"], "label": pred}).to_csv(
            "etc_submission.csv", index=False)

        # save model/scaler
        joblib.dump(etc, "etc_model.joblib")
        joblib.dump(scaler, "etc_scaler.joblib")
        # print(sklearn.__version__) # 1.9.1

        # validation scores for report
        etc_test = ExtraTreesClassifier(n_estimators=500, random_state=0)
        X_train_tab = np.stack([summarize(x.numpy(), a) for x, a in zip(X_train, adj_train)])
        etc_test.fit(X_train_tab, y_train)

        X_val_tab = np.stack([summarize(x.numpy(), a) for x, a in zip(X_val, adj_val)])
        val_pred = etc_test.predict(X_val_tab) 
        print(f"ETC: acc: {accuracy_score(y_val, val_pred)}, f1: {f1_score(y_val, val_pred, average="macro")}")

        if AUC:
            # plot AUC of ETC 
            cv = StratifiedKFold(5, shuffle=True, random_state=0)
            proba = cross_val_predict(etc,
                                    X_all, y_all, cv=cv, method="predict_proba")

            plot_auc(y_all, proba, save_path="etc_roc.png", title="Extra Trees ROC")


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

SKEW: [-13.65   1.82   1.55   0.22   1.54   1.72   1.63   1.68   1.42   1.77
   1.55   1.33   1.3    1.65   1.67   1.64   1.64   1.42]
0.6645833333333333 0.03448530282759761 - rfc
0.6895833333333333 0.03186887195995491 - extra tress classifier
0.6583333333333333 0.015309310892394868 - hist gradient boosting classifier

feature 0: min=-213, median=6, max=91
feature 1: min=0, median=12.4, max=149


GCN1 real
epoch  89  train_loss=0.710  val_loss=1.438  val_acc=0.469  val_f1=0.467
train_acc 0.781 train_f1=0.781
epoch  74  train_loss=1.022  val_loss=1.466  val_acc=0.531  val_f1=0.511
train_acc 0.766 train_f1=0.756
oh i should probably grab the ones that 25 before the last one huh
epoch  76  train_loss=0.966  val_loss=1.409  val_acc=0.542  val_f1=0.532
train_acc 0.742 train_f1=0.737


GCN2 real
epoch  42  train_loss=1.065  val_loss=1.214  val_acc=0.604  val_f1=0.583
train_acc 0.760 train_f1=0.757
best epoch: 19
epoch  19  train_loss=1.125  val_loss=1.191  val_acc=0.552  val_f1=0.537
train_acc 0.643 train_f1=0.637
mmm there's a bug
no there isn't
oh well

GCN3 why not
epoch  50  train_loss=1.096  val_loss=1.162  val_acc=0.562  val_f1=0.549
train_acc 0.805 train_f1=0.802
best epoch: 49
epoch  56  train_loss=0.819  val_loss=1.199  val_acc=0.604  val_f1=0.599
train_acc 0.810 train_f1=0.808
best epoch: 50
eh cool

ETC: acc: 0.7604166666666666, f1: 0.7577361452361452
"""