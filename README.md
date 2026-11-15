# dl_hw_4
deep learning hw 4 graph classification

https://www.kaggle.com/competitions/cse-5173-6073-hw-4-enzymes-graph-classification

main.py is the main python file
run `./.venv/bin/python main.py`

There are a few control variables above the `if __name__ == "__main__":` line:
 - `TRAIN` - set to `True` to train a GCN model, and `False` to load a GCN model specified by `SAVE_NAME` 
 - `SAVE` - set to `True` to save a GCN model's weights to the file given by `SAVE_NAME`. Ignored when `TRAIN` is `False` 
 - `SAVE_NAME` - the name of the file to save model weights to, or to load model weights from. has to end in `.keras`
 - `SUBMISSION` - set to `True` to create a submission.csv. It uses my best model - ExtraTreesClassifier for training and on testing. It does not create a submission.csv for the GCN model. 
 - `AUC` - set to `True` to save an AUC ROC curve graph as a .png to the current working directory. This creates two pngs - one for ExtraTreesClassifier if `SUBMISSION` is set to `True`, and it will create one for the loaded/trained GCN model. 
 - `LAYERS` - integer to indicate how many layers to build the GCN model with. Ignored when `TRAIN` is `False`

