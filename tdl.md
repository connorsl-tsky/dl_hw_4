 - [x] read hw and generate tdls - 11/14/26 814am
 - [x] read kaggle page and download data - 11/14/26 821am
 - [x] create venv and download dependencies - 11/14/26 833am
 - [x] init the hw report - 11/14/26 843am
 - [x] pseudo - 11/14/26 859am
 - [x] initial research / prompting - 11/14/26 1200pm (distracted for 45 min + chores)
 - [x] design and develop - 11/15/26 1007am
 - [x] hw report
 - [x] submit best model to kaggle - 11/15/26 1043am
 - [x] clean up code
 - [ ] create zip file of connor_slutsky_hw4.zip with 1) hw report, 2) prompt report, 3) all code, and 4) best model weights






train - 480 graphs with labels, test - 120 graphs
enzymes dataset
row is a graph, graph id, number of nodes, node featuers, node types, edge connections, last three are json arrays.
6 classes
graph id and label - submission.csv
macro f1
enzymes graph indicators - ith line is graph id of node with node id i
enzeme graph labels - ith line is class label of graph with id i
enzymes a - sparse adjancecy matrix for all graphs, each line correpsond to (row, col) resp (node id, node id)
enzymes node attributes - matrix of node attributes, ith line is node i

gcn 1 layer and 2 layer

pip install "tensorflow[and-cuda]"