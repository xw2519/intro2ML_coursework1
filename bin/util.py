import numpy as np
import matplotlib.pyplot as matplot

'''
util.py

Contains helper functions
'''

def calculate_set_entropy(dataset_list):
    '''
    Calculates the entropy for a given list of datasets
    
    :param1 dataset_list: List of dataset
    :return entropy_result
    '''
    entropy_values = [] 
    total_count = 0
    
    for dataset in dataset_list:
        # Calculate dataset entropy
        dataset_length = len(dataset)
        total_count += dataset_length
        unique_values = np.unique(dataset, return_counts=True)[1]
        entropy = np.sum([-(value/dataset_length) * np.log2(value/dataset_length) for value in unique_values])
        entropy_values.append((entropy, dataset_length))
        
    # Calculate and return the average entropy of the entire dataset 
    return np.sum([(subset_length/(total_count)) * entropy_value for entropy_value, subset_length in iter(entropy_values)])
    
    
def find_split(dataset: np.ndarray):
    '''
    Given a dataset instance, choose the attribute and the value that results in the highest information gain
    
    :param1 dataset: Training dataset
    :return node_attribute, cut_point, left_subset, right_subset
    '''
    previous_entropy = calculate_set_entropy([dataset[:, 7]])
    
    # Sort values of attribute (excluding the last column) and consider only split points between the two values 
    split_point_values = [] 
    
    return_attribute = None
    cut_point = 0
    return_left_subset = None 
    return_right_subset = None
    
    for column in range(dataset.shape[1] - 1): 
        unique_values = np.unique(dataset[:, column])
        split_values = []
        
        for row in range(len(unique_values) - 1): split_values.append((unique_values[row] + unique_values[row+1])/2)
        split_point_values.append(split_values)
        
    # Split data at split points and calculate the entropy
    for attribute in range(len(split_point_values)):
        for value in range(len(split_point_values[attribute])):
            left_dataset = dataset[ dataset[:, attribute] < split_point_values[attribute][value] ]
            right_dataset = dataset[ dataset[:, attribute] >= split_point_values[attribute][value] ]
            
            # Calculate the entropy of left and right sub-dataset        
            set_entropy = calculate_set_entropy([left_dataset[:, 7], right_dataset[:, 7]])
            
            if set_entropy < previous_entropy:
                previous_entropy = set_entropy
                return_attribute = attribute + 1
                cut_point = split_point_values[attribute][value]
                return_left_subset = left_dataset
                return_right_subset = right_dataset

    return return_attribute, cut_point, return_left_subset, return_right_subset
        

def generate_decison_tree_graph():
    '''
    Visualises the decision tree trained on a particular dataset
    '''
    ...