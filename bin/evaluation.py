import numpy as np    

'''
evaluation.py
    
Contains functions that evaluates the performance of a decision tree model  
'''

def calculate_confusion_matrix(predicted_labels, actual_labels) -> np.ndarray:
    '''
    Given arrays of 'predicted_labels' and 'actual_labels, calculate and return the confusion matrix as an 4x4 array
    
    Parameters:
    - predicted_labels: Array of predicted values from the 'predict_dataset' function 
    - actual_labels: Array of actual labels of the test dataset
    
    Return:
    - confusion_matrix: The 4x4 confusion matrix array
    '''
    confusion_matrix = np.zeros((4, 4))
    
    for actual_value, predicted_value in zip(actual_labels, predicted_labels): 
        actual_value, predicted_value = int(actual_value), int(predicted_value)
        confusion_matrix[actual_value - 1][predicted_value - 1] += 1

    return confusion_matrix 


def calculate_evaluation_metrics(confusion_matrix):
    '''
    Given the 'confusion_matrix' array, calculate and return evaluation metrics derived from the confusion matrix
    
    Parameters:
    - confusion_matrix: Array of confusion matrix 
    
    Return:
    - accuracy: Number of correctly classified examples divided by the total number of examples
    - precision: Number of correctly classified positive divided by the total number of predicted positive 
    - recall: Number of correctly classified positive divided by the total number of positive
    - f_score: Performance measure of the classifier
    '''
    accuracy = np.sum(np.diagonal(confusion_matrix)) / np.sum(confusion_matrix)
    precision = np.diagonal(confusion_matrix) / np.sum(confusion_matrix, axis = 0)
    recall = np.diagonal(confusion_matrix) / np.sum(confusion_matrix, axis = 1)
    f_score = 2 * (precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, f_score
    
    