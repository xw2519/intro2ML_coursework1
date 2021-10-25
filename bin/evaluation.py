import numpy as np    

'''
evaluation.py
    
Contains functions that evaluates the performance of a decision tree model  
'''

def calculate_confusion_matrix(predicted_labels, actual_labels):
    confusion_matrix = np.zeros((4, 4))
    
    for actual_value, predicted_value in zip(actual_labels, predicted_labels): 
        actual_value, predicted_value = int(actual_value), int(predicted_value)
        confusion_matrix[actual_value - 1][predicted_value - 1] += 1

    return confusion_matrix 