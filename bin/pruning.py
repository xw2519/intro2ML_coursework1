from bin.decision_tree import create_decision_tree, predict_dataset
from bin.evaluation import calculate_confusion_matrix, calculate_evaluation_metrics, cross_validation

import numpy as np
import collections

'''
the tree is pruned if a full iteration of the while loop completes without removing nodes

while tree is not pruned:
    for every node in tree:
        if node is connected to two leaves:
            prune and compare accuracy

'''
def prune_tree(training_set, validation_set, trained_tree):

    pruned         = False             # pruning is complete if a whole pass of the tree is made without removing nodes

    stack          = []                # stores current path in tree, used for traversing tree without recursion and finding subsets of training set

    current_node   = trained_tree      # current node of interest

    explored_nodes = []                # contains all explored nodes


    while not (pruned and (stack == [])):

        if stack == []:                # reset when all nodes of tree have been visited
            pruned = True
            explored_nodes = []

        if ((current_node['left']['leaf']) and (current_node['right']['leaf'])):  # if the current node has 2 leaves, try to prune the tree


            result = predict_dataset(validation_set[:,:7], trained_tree)                                    # evaluate before pruning
            confusion_matrix = calculate_confusion_matrix(result, validation_set[:,7])
            unpruned_accuracy, precision, recall, f_score = calculate_evaluation_metrics(confusion_matrix)

            search_set = np.copy(training_set)                                                             # find the label of the leaf to replace the current node
            for i in range(len(stack)):
                if i < len(stack)-1:
                    if stack[i]['left'] == stack[i+1]:
                        search_set = search_set[ search_set[:,stack[i]['attribute']] <= stack[i]['value'] ]
                    else:
                        search_set = search_set[ search_set[:,stack[i]['attribute']] >  stack[i]['value'] ]
                else:
                    if stack[i]['left'] == current_node:
                        search_set = search_set[ search_set[:,stack[i]['attribute']] <= stack[i]['value'] ]
                    else:
                        search_set = search_set[ search_set[:,stack[i]['attribute']] >  stack[i]['value'] ]
            value = collections.Counter(search_set[:,7]).most_common(1)[0][0]                              # get the most common label from this subset of the training set

            new_leaf = {"class" : value, "leaf" : True}                                                    # create leaf to replace current node

            if stack[-1]['left'] == current_node:                                                          # replace current node with leaf
                stack[-1]['left']  = new_leaf
            else:
                stack[-1]['right'] = new_leaf

            result = predict_dataset(validation_set[:,:7], trained_tree)                                    # evaluate before pruning
            confusion_matrix = calculate_confusion_matrix(result, validation_set[:,7])
            pruned_accuracy, precision, recall, f_score = calculate_evaluation_metrics(confusion_matrix)

            #print('Compare: ', unpruned_accuracy, pruned_accuracy)

            if pruned_accuracy <= unpruned_accuracy:                                           # if pruning reduces performance, undo the pruning and continue traversing tree
                if stack[-1]['left'] == new_leaf:
                    stack[-1]['left']  = current_node
                else:
                    stack[-1]['right'] = current_node
                explored_nodes.append(current_node['left'])
                explored_nodes.append(current_node['right'])
            else:                                                                              # if pruning improved performance, set flag and continue
                pruned = False
                #print('Removed: ', current_node)

            current_node = stack.pop()


        else:                                                                     # if current node does NOT have 2 leaves, keep traversing tree

            if (current_node['left']['leaf']) and (current_node['left'] not in explored_nodes):
                explored_nodes.append(current_node['left'])
            if (current_node['right']['leaf']) and (current_node['right'] not in explored_nodes):
                explored_nodes.append(current_node['right'])

            if (current_node['left'] in explored_nodes) and (current_node['right'] in explored_nodes):     # if both connected nodes have been explored, go to parent node
                current_node = stack.pop()
            elif (not current_node['left']['leaf']) and (current_node['left'] not in explored_nodes):      # go to left node if it is not a leaf and has not been explored
               stack.append(current_node)
               current_node = current_node['left']
            else:                                                                                          # if left node is a leaf or has been explored, go to right node
                stack.append(current_node)
                current_node = current_node['right']

        explored_nodes.append(current_node)

    return trained_tree

def prune_with_cross_validation(filepath):
    '''
    Loads and divides the data set into 10 folds. Performs cross validation with each fold as a test set and 9 folds as training/validation set.

    Calculates metrics for each fold to get the average metric.

    Parameters:
    - filepath: File path to the dataset

    Return:
    - average_accuracy: Number of correctly classified examples divided by the total number of examples
    - average_precision: Number of correctly classified positive divided by the total number of predicted positive
    - average_recall: Number of correctly classified positive divided by the total number of positive
    - average_f1_score: Performance measure of the classifier
    '''
    # Load and shuffle the data
    loaded_data = np.loadtxt(filepath)
    np.random.shuffle(loaded_data)

    # Create 10 folds
    loaded_data = loaded_data.reshape((10, -1, 8))

    # Perform cross validation
    confusion_matrix_list = []
    for index, test_fold in enumerate(loaded_data):
        # Remove test fold from training fold
        training_folds = np.vstack(np.delete(loaded_data, index, axis = 0))
        training_dataset = training_folds[:, [0, 1, 2, 3, 4, 5, 6]]
        training_labels = training_folds[:, 7]

        # Train decision tree model
        decision_tree_model, max_depth = create_decision_tree(training_dataset=training_dataset, label=training_labels, tree_depth=0)

        # Predict using test fold
        predictions = predict_dataset(test_fold, decision_tree_model)

        # Calculate confusion matrix
        confusion_matrix_list.append(calculate_confusion_matrix(predictions, test_fold[:, -1]))

    # Calculate and return cross validation results
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    confusion_matrix_sum = np.zeros((4, 4))

    for i in range(len(confusion_matrix_list)):
        accuracy, precision, recall, f1_score = calculate_evaluation_metrics(confusion_matrix_list[i])

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)
        confusion_matrix_sum += confusion_matrix_list[i]

    average_accuracy = np.mean(np.array(accuracy_list), axis=0)
    average_precision = np.mean(np.array(precision_list), axis=0)
    average_recall = np.mean(np.array(recall_list), axis=0)
    average_f1_score = np.mean(np.array(f1_score_list), axis=0)
    average_confusion_matrix = confusion_matrix_sum / 10

    return average_accuracy, average_precision, average_recall, average_f1_score, average_confusion_matrix
