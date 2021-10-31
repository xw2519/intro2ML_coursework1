from bin.pruning import prune_tree, prune_with_cross_validation
from bin.evaluation import cross_validation
from bin.decision_tree import create_decision_tree
from bin.plot_tree import createPlot
from numpy.random import default_rng
#fro sklearn import tree
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

filepath = "./wifi_db/noisy_dataset.txt"
loaded_data = np.loadtxt(filepath)

x = loaded_data[:, [0, 1, 2, 3, 4, 5, 6]]
y = loaded_data[:, 7]

rg = np.random.default_rng(seed)
shuffled_order = rg.permutation(len(x))
#print("x: ", shuffled_order.shape)

n_test = round(len(x) * 0.1)
n_train = round(len(x) * 0.8)
n_valid = len(x)-n_train-n_test

x_train = x[shuffled_order[:n_train]]
y_train = y[shuffled_order[:n_train]]
x_test = x[shuffled_order[n_train:-n_valid]]
y_test = y[shuffled_order[n_train:-n_valid]]
x_valid = x[shuffled_order[-n_valid::]]
y_valid = y[shuffled_order[-n_valid::]]

#print(x_train)
#print("x_train size is", x_train.shape)

training_folds = loaded_data[shuffled_order[:n_train]]
#print(training_folds)
#print("training_fold size is", training_folds.shape)
validation_fold = loaded_data[shuffled_order[-n_valid::]]

'''
training_folds = loaded_data[:int(len(x) * 0.8)]
validation_fold = loaded_data[int(len(x) * 0.9):]

training_set = x[:int(len(x) * 0.8)]
training_y = y[:int(len(y) * 0.8)]

test_set = x[int(len(x) * 0.9):]
test_y = y[int(len(y) * 0.9):]
'''
'''
depth = 0
decision_tree, depth = create_decision_tree(x_train,y_train,depth)
createPlot(decision_tree,depth,"original_tree.jpg")
pruned_decision_tree_model = prune_tree(training_folds, validation_fold, decision_tree)     
createPlot(pruned_decision_tree_model,depth,"pruned_tree.jpg")
'''
seed = 2
unpruned_accuracies, unpruned_precision, unpruned_recall, unpruend_f1_score, unpruned_confusion_matrix = cross_validation(filepath,seed)
average_accuracy, average_precision, average_recall, average_f1_score, average_confusion_matrix = prune_with_cross_validation(filepath,seed)
print(unpruned_accuracies, unpruned_precision, unpruned_recall, unpruend_f1_score, unpruned_confusion_matrix)
print(average_accuracy, average_precision, average_recall, average_f1_score, average_confusion_matrix)