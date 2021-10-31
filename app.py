from bin.decision_tree import create_decision_tree, predict_dataset, width
from bin.evaluation import calculate_confusion_matrix, calculate_evaluation_metrics, cross_validation
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
            filepath = input('\nInput path to dataset: ')
            try:
                loaded_dataset = np.loadtxt(filepath)
                break
            except:
                print('Path not valid! Try again')
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
            
    # Load and shuffle data
    dataset, label = read_and_shuffle_dataset("./wifi_db/clean_dataset.txt")
    
    training_set = dataset[:int(len(dataset) * 0.7)]
    training_label = label[:int(len(label) * 0.7)]
    
    test_set = dataset[int(len(dataset) * 0.7):]
    test_label = label[int(len(label) * 0.7):]
    
    # Declare and create decision tree model
    decision_tree_model, max_tree_depth = create_decision_tree(training_dataset = training_set, label = training_label, tree_depth = 0)  
    result = predict_dataset(test_set, decision_tree_model)
    
    confusion_matrix = calculate_confusion_matrix(result, test_label)
    accuracy, precision, recall, f_score = calculate_evaluation_metrics(confusion_matrix)
    
    print_evaluation_metrics(accuracy, precision, recall, f_score) 
    
    # Evaluation matrix 
    average_accuracy, average_precision, average_recall, average_f1_score, average_confusion_matrix = cross_validation("./wifi_db/clean_dataset.txt")
    print_cross_validation_metrics(average_accuracy, average_precision, average_recall, average_f1_score, average_confusion_matrix)
    
    ''' 
    print(max_tree_depth)
    print_evaluation_metrics(accuracy, precision, recall, f_score) 
    plot_confusion_matrix(confusion_matrix=confusion_matrix)
    
    sklearn_tree = tree.DecisionTreeClassifier()
    sklearn_tree = sklearn_tree.fit(training_set, training_label)
    
    fig = matplotlib.pyplot.figure(figsize=(50, 50))
    _ = tree.plot_tree(sklearn_tree, filled=True)

    
    fig.savefig("decistion_tree.png", bbox_inches = "tight")

    accuracy, precision, recall, f_score = calculate_evaluation_metrics(confusion_matrix)
    
    
    print("Width: ", width)
    print("Max depth: ", max_tree_depth)
    
    print("SK depth: ", sklearn_tree.tree_.max_depth)
    
    dataset=np.array([[1,2,3,4,5,6,7],[2,3,4,5,6,7,8]])
    decision_tree={"attribute":0,"value":0,"left":{"attribute":3,"value":4,"left":{"class":15,"leaf":True},"right":{"class":14,"leaf":True},"leaf":False},"right":{"attribute":5,"value":8,"left":{"class":11,"leaf":True},"right":{"class":12,"leaf":True},"leaf":False},"leaf":False}
    y = predict_dataset(dataset, decision_tree)
    print(y)
    
    plot_decision_tree(decision_tree_model, width)
    '''
    
    