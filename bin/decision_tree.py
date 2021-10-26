from bin.util import find_split
import numpy as np    

'''
tree.py
    
Creates and manipulates the decision tree
'''

# Global variable declaration 
tree_depth = 0
width = [0,0,0,0,0]

# Decision tree functions
def create_decision_tree(training_dataset, label, tree_depth):
    '''
    Given a training dataset, labels and initial tree depth, create the decision tree
    
    Parameters:
    - training_dataset: Array of values to train the decision tree 
    - label: Array of labels associated with provided training dataset
    - tree_depth: Value indicating the current depth of the tree
    
    Return:
    - node: A decision tree model in dictionary form
    - max_depth: The maximum depth of the decision tree model
    '''
    [classes, unique_label] = np.unique(label, return_inverse=True) 
    
    if(len(classes) == 1):
        leaf_node = {"class" : classes[0], "leaf" : True}

        if tree_depth >= len(width): width.append(1)
        else: width[tree_depth] += 1
        
        return leaf_node, tree_depth
    else:
        attribute, value, cut_point = find_split(training_dataset, label) 

        order = np.argsort(training_dataset[:, attribute])
        data = training_dataset[order]
        label = label[order]
        
        l_dataset=data[:cut_point] 
        l_label=label[:cut_point]
        
        r_dataset=data[cut_point:]
        r_label=label[cut_point:]

        node = {"attribute" : attribute, "value" : value, "left" : {} , "right" : {}, "leaf" : False}
        
        if tree_depth >= len(width): width.append(1)
        else: width[tree_depth] += 1
        
        l_branch, l_depth = create_decision_tree(l_dataset, l_label, tree_depth + 1)
        r_branch, r_depth = create_decision_tree(r_dataset, r_label, tree_depth + 1)

        node["left"] = l_branch
        node["right"] = r_branch

        return node, max(l_depth, r_depth) 


def predict_dataset(test_dataset, decision_tree_model):
    '''
    Given a test dataset and trained decision tree model, predict the appropriate labels of the test dataset
    
    Parameters:
    - test_dataset: Array of test values to predict 
    - decision_tree_model: Trained decision tree using training dataset
    
    Return:
    - predicted_labels: Array of predicted labels of 'test_dataset'
    '''
    return [ predict_single_instance(test_dataset[i, :], decision_tree_model) for i in range(len(test_dataset)) ]


def predict_single_instance(test_instance, decision_tree_model):
    '''
    Given a single test instance and trained decision tree model, predict the appropriate label of that single test instance
    
    Parameters:
    test_instance: Test value to predict 
    decision_tree_model: Trained decision tree using training dataset
    
    Return:
    predicted_labels: Predicted label of 'test_instance'
    '''
    if (decision_tree_model["leaf"] == True): 
        # If "leaf" node is reached, prediction is complete and return the predicted class
        return int(decision_tree_model["class"])
    else: 
        # If "leaf" node is not reached, traverse the decision tree
        if test_instance[ decision_tree_model["attribute"] ] <= decision_tree_model["value"]: return predict_single_instance(test_instance, decision_tree_model["left"])
        else: return predict_single_instance(test_instance, decision_tree_model["right"])



    