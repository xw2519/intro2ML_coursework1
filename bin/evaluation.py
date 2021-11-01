from .decision_tree import create_decision_tree, predict_dataset
import numpy as np   

np.seterr(invalid='ignore')


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
    
def cross_validation(filepath,seed):
    '''
    Loads and divides the data set into 10 folds. Performs cross validation with each fold as a test set and 9 folds as training set.
    
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
    #np.random.shuffle(loaded_data)
    rg = np.random.default_rng(seed)
    shuffled_order = rg.permutation(len(loaded_data))
    loaded_data = loaded_data[shuffled_order]
    
    # Create 10 folds
    loaded_data = loaded_data.reshape((10, -1, 8))
    
    # Perform cross validation 
    confusion_matrix_list = [] 
    depth_list=[]
    for index, test_fold in enumerate(loaded_data):
        # Remove test fold from training fold
        training_folds = np.vstack(np.delete(loaded_data, index, axis = 0))
        training_dataset = training_folds[:, [0, 1, 2, 3, 4, 5, 6]]
        training_labels = training_folds[:, 7]
        
        # Train decision tree model 
        decision_tree_model, max_depth = create_decision_tree(training_dataset=training_dataset, label=training_labels, tree_depth=0)
        depth_list.append(max_depth+1)
        # Predict using test fold
        predictions = predict_dataset(test_fold, decision_tree_model)
        
        # Calculate confusion matrix
        confusion_matrix_list.append(calculate_confusion_matrix(predictions, test_fold[:, -1]))
        
    # Calculate and return cross validation results
    average_depth=np.mean(np.array(depth_list))
    print("average_depth : ", average_depth)
    print("depth list : ", depth_list)
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
    