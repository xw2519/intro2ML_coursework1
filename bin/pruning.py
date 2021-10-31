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
    # Store current path in tree.
    # Traverses tree without recursion and finding subsets of training set. Stores recently explored nodes.
    stack = []                
    current_node = trained_tree # current node of interest      
    explored_nodes = []               

    while not ((stack == []) and (current_node['right'] in explored_nodes)):
        # If current node has 2 leaves, try prune the tree
        if ((current_node['left']['leaf']) and (current_node['right']['leaf'])):  
            # Evaluate unpruned model metrics
            result = predict_dataset(validation_set[:, :7], trained_tree)                                    
            confusion_matrix = calculate_confusion_matrix(result, validation_set[:,7])
            unpruned_accuracy, unpruned_precision, unpruned_recall, unpruned_f_score = calculate_evaluation_metrics(confusion_matrix)

            # Find label of the leaf to replace the current node
            search_set = np.copy(training_set)                                                             
            for i in range(len(stack)):
                if i < len(stack)-1:
                    if (stack[i]['left'] == stack[i+1]):
                        search_set = search_set[ search_set[:,stack[i]['attribute']] <= stack[i]['value'] ]
                    else:
                        search_set = search_set[ search_set[:,stack[i]['attribute']] >  stack[i]['value'] ]
                else:
                    if (stack[i]['left'] == current_node):
                        search_set = search_set[ search_set[:,stack[i]['attribute']] <= stack[i]['value'] ]
                    else:
                        search_set = search_set[ search_set[:,stack[i]['attribute']] >  stack[i]['value'] ]
                        
            # Get most common label from this subset of the training set
            value = collections.Counter(search_set[:, 7]).most_common(1)[0][0]                              

            # Create leaf to replace current node
            new_leaf = {"class" : value, "leaf" : True}                                                  

            # Replace current node with leaf
            if (stack[-1]['left'] == current_node):                                                          
                stack[-1]['left']  = new_leaf
                side = 'left'
            else:
                stack[-1]['right'] = new_leaf
                side = 'right'

            # Evaluate pruned model metrics
            result = predict_dataset(validation_set[:, :7], trained_tree)                                    
            confusion_matrix = calculate_confusion_matrix(result, validation_set[:,7])
            pruned_accuracy, pruned_precision, pruned_recall, pruned_f_score = calculate_evaluation_metrics(confusion_matrix)

            if pruned_accuracy <= unpruned_accuracy: 
                # Pruning reduced performance, undo the pruning and continue traversing tree
                stack[-1][side] = current_node
            else:
                # Pruning improved performance                                 
                pass

            if current_node['left']  in explored_nodes: explored_nodes.remove(current_node['left'])
            if current_node['right'] in explored_nodes: explored_nodes.remove(current_node['right'])
            current_node = stack.pop()
        
        # If current node does NOT have 2 leaves, keep traversing tree
        else:                                                                    
            # Update explored nodes
            if (current_node['left']['leaf']) and (current_node['left'] not in explored_nodes): explored_nodes.append(current_node['left'])
            if (current_node['right']['leaf']) and (current_node['right'] not in explored_nodes): explored_nodes.append(current_node['right'])

            # If both connected nodes have been explored, go to parent node
            if (current_node['left'] in explored_nodes) and (current_node['right'] in explored_nodes):     
                explored_nodes.remove(current_node['left'])
                explored_nodes.remove(current_node['right'])
                current_node = stack.pop()
            # Go to left node if it is not a leaf and has not been explored
            elif (not current_node['left']['leaf']) and (current_node['left'] not in explored_nodes):      
               stack.append(current_node)
               current_node = current_node['left']
            # Go to right node
            else:                                                                                          
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

    # Perform 10-fold nested cross validation
    confusion_matrix_list = []

    for i, test_fold in enumerate(loaded_data):
        # Remove test fold from training/validation folds
        train_valid_folds = np.delete(loaded_data, i, axis = 0)                                  

        best_valid_tree = ({},0)

        for j, validation_fold in enumerate(train_valid_folds):
            training_folds   = np.vstack(np.delete(train_valid_folds, j, axis = 0))
            training_dataset = training_folds[:, :7]
            training_labels  = training_folds[:, 7]
            
            validation_dataset = validation_fold[:, :7]
            validation_labels =  validation_fold[:, -1]

            decision_tree_model, max_depth = create_decision_tree(training_dataset=training_dataset, label=training_labels, tree_depth=0)    
            pruned_decision_tree_model = prune_tree(training_folds, validation_fold, decision_tree_model)                                 

            # Predict using validation fold
            pruned_predictions = predict_dataset(validation_dataset, pruned_decision_tree_model)               
            pruned_confusion_matrix = calculate_confusion_matrix(pruned_predictions, validation_labels)          
            pruned_accuracy, pruned_precision, pruned_recall, pruned_f1_score = calculate_evaluation_metrics(pruned_confusion_matrix)

            # Select the best model
            if(pruned_accuracy > best_valid_tree[1]): best_valid_tree = (pruned_decision_tree_model, pruned_accuracy)

        # Evaluate performance of the best model selected with validation fold
        predictions = predict_dataset(test_fold, best_valid_tree[0])                                    

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
