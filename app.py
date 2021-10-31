from bin.decision_tree import create_decision_tree, width
from bin.evaluation import cross_validation
from bin.util import read_and_shuffle_dataset, plot_decision_tree, plot_confusion_matrix, print_evaluation_metrics, print_cross_validation_metrics

import numpy as np

'''
app.py

Main python program accessible to the user and links all subprograms together
'''

if __name__ == "__main__":
    while True:
        # User prompt
        while True:
            filepath = input('\nInput filepath to dataset:')
            try:
                loaded_dataset = np.loadtxt(filepath)
                break
            except:
                print('Failed to load dataset at specified filepath. Please try again.')
        print()
        print('Program functions: [Please enter the corresponding number]')
        selected_function = input(' 1: Plot Decision Tree \n 2: 10-fold cross validation \n 3: Nested 10-fold cross validation with post-pruning \n \nChoice: ')
        print()
        
        # Function execution
        if (selected_function == '1'):
            print("-------------------- Program execution log --------------------")
            print("1: Plot Decision Tree")
            print()
            print("Loading dataset from: ", filepath)
            training_set, training_label = read_and_shuffle_dataset(filepath)
            print("Plotting decision tree")
            decision_tree_model, max_tree_depth = create_decision_tree(training_dataset = training_set, label = training_label, tree_depth = 0)  
            plot_decision_tree(decision_tree_model, width)
            print("Plotting completed. Plot saved to 'output/decision_tree_model.jpg'")
            print() 
            input("To select other functions, press 'Enter'")
        
        elif (selected_function == '2'):
            print("-------------------- Program execution log --------------------")
            print("2: 10-fold cross validation")
            print()
            print("Loading dataset from: ", filepath)
            print()
            average_accuracy, average_precision, average_recall, average_f1_score, average_confusion_matrix = cross_validation(filepath)
            print_cross_validation_metrics(average_accuracy, average_precision, average_recall, average_f1_score, average_confusion_matrix)
            print("Plotting confusion matrix")
            plot_confusion_matrix(average_confusion_matrix)
            print("Plotting completed. Plot saved to 'output/confusion_matrix.jpg'")
            print() 
            input("To select other functions, press 'Enter'")
                    
        elif (selected_function == '3'):
            print("-------------------- Program execution log --------------------")
            print("3: Nested 10-fold cross validation with post-pruning")
            
            print() 
            input("To select other functions, press 'Enter'")
            ...
            
    
    