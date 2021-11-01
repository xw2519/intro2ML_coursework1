
import matplotlib

from bin.pruning import prune_tree, prune_with_cross_validation

#fro sklearn import tree
import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)

from bin.decision_tree import create_decision_tree, width
from bin.evaluation import cross_validation
from bin.util import read_and_shuffle_dataset, plot_decision_tree, plot_confusion_matrix, print_evaluation_metrics, print_cross_validation_metrics

import numpy as np

'''
app.py

Main python program accessible to the user and links all subprograms together
'''

if __name__ == "__main__":
    '''
    seed = int(input("Please choose the seed : "))

    unpruned_accuracies, average_precision, average_recall, average_f1_score, average_confusion_matrix = cross_validation("./wifi_db/clean_dataset.txt",seed)
    print("Unpruned tree ----------------------------------------------------------- ")
    print("accuracy : ", unpruned_accuracies)
    print("precision per class : ", average_precision)
    print("recall per class : ", average_recall)
    print("f1_score : ", average_f1_score)
    print("confusion_matrix : ")
    print(average_confusion_matrix)
    #print(unpruned_accuracies, average_precision, average_recall, average_f1_score, average_confusion_matrix)
    pruned_accuracies, pruned_average_precision, pruned_average_recall, pruned_average_f1_score, pruned_average_confusion_matrix = prune_with_cross_validation("./wifi_db/clean_dataset.txt",seed)
    print("Pruned tree ----------------------------------------------------------- ")
    print("accuracy : ", pruned_accuracies)
    print("precision per class : ", pruned_average_precision)
    print("recall per class : ", pruned_average_recall)
    print("f1_score : ", pruned_average_f1_score)
    print("confusion_matrix : ")
    print(pruned_average_confusion_matrix)
    #print(unpruned_accuracies, pruned_accuracies)


    
    # Load and shuffle data
    dataset, label = read_and_shuffle_dataset("./wifi_db/clean_dataset.txt")

    training_set = dataset[:int(len(dataset) * 0.7)]
    training_label = label[:int(len(label) * 0.7)]

    test_set = dataset[int(len(dataset) * 0.7):]
    test_label = label[int(len(label) * 0.7):]


    # Declare and create decision tree model
    decision_tree_model, max_tree_depth = create_decision_tree(training_dataset = training_set, label = training_label, tree_depth = 0)
    result = predict_dataset(test_set[len(test_set)//2:], decision_tree_model)

    confusion_matrix = calculate_confusion_matrix(result, test_label[len(test_label)//2:])
    accuracy, precision, recall, f_score = calculate_evaluation_metrics(confusion_matrix)

    print_evaluation_metrics(accuracy, precision, recall, f_score)

    plot_decision_tree(decision_tree_model, width)

    # Prune decision tree model and evaluate after pruning
    pruned_decision_tree_model = prune_tree(numpy.hstack((training_set,training_label.reshape([len(training_label),1]))), numpy.hstack((test_set[:len(test_set)//2],test_label[:len(test_label)//2].reshape([len(test_label[:len(test_label)//2]),1]))), decision_tree_model)
    pruned_result = predict_dataset(test_set[len(test_set)//2:], pruned_decision_tree_model)

    pruned_confusion_matrix = calculate_confusion_matrix(pruned_result, test_label[len(test_label)//2:])
    pruned_accuracy, pruned_precision, pruned_recall, pruned_f_score = calculate_evaluation_metrics(pruned_confusion_matrix)

    print_evaluation_metrics(pruned_accuracy, pruned_precision, pruned_recall, pruned_f_score)

    # Evaluation matrix
    average_accuracy, average_precision, average_recall, average_f1_score, average_confusion_matrix = cross_validation("./wifi_db/clean_dataset.txt")
    print_cross_validation_metrics(average_accuracy, average_precision, average_recall, average_f1_score, average_confusion_matrix)



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
            
