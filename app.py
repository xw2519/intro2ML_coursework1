from bin.decision_tree import create_decision_tree, predict_dataset
from bin.evaluation import calculate_confusion_matrix
from bin.util import read_and_shuffle_dataset, plot_decision_tree, plot_confusion_matrix
from bin.decision_tree import width

'''
app.py

Main python program accessible to the user and links all subprograms together
'''

if __name__ == "__main__":
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
    print(plot_confusion_matrix(confusion_matrix))
    
    
    '''
    dataset=np.array([[1,2,3,4,5,6,7],[2,3,4,5,6,7,8]])
    decision_tree={"attribute":0,"value":0,"left":{"attribute":3,"value":4,"left":{"class":15,"leaf":True},"right":{"class":14,"leaf":True},"leaf":False},"right":{"attribute":5,"value":8,"left":{"class":11,"leaf":True},"right":{"class":12,"leaf":True},"leaf":False},"leaf":False}
    y = predict_dataset(dataset, decision_tree)
    print(y)
    
    
    plot_decision_tree(decision_tree_model, width)
    '''
    
    