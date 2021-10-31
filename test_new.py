from bin.pruning import prune_tree, prune_with_cross_validation
from bin.decision_tree import create_decision_tree
from bin.plot_tree import createPlot

#fro sklearn import tree
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

loaded_data = np.loadtxt("./wifi_db/clean_dataset.txt")

dataset = loaded_data[:, [0, 1, 2, 3, 4, 5, 6]]
label = loaded_data[:, 7]

training_folds = loaded_data[:int(len(dataset) * 0.8)]
validation_fold = loaded_data[int(len(dataset) * 0.9):]

training_set = dataset[:int(len(dataset) * 0.8)]
training_label = label[:int(len(label) * 0.8)]

test_set = dataset[int(len(dataset) * 0.9):]
test_label = label[int(len(label) * 0.9):]

depth = 0
decision_tree, depth = create_decision_tree(training_set,training_label,depth)
pruned_decision_tree_model = prune_tree(training_folds, validation_fold, decision_tree)     
createPlot(decision_tree,depth,"original_tree.jpg")
createPlot(pruned_decision_tree_model,13,"pruned_tree.jpg")