
from bin.decision_tree import decision_tree, predict_dataset
from bin.util import read_dataset

import numpy as np

'''
app.py

Main python program accessible to the user and links all subprograms together
'''

if __name__ == "__main__":
    # Load data
    dataset, label = read_dataset("./wifi_db/clean_dataset.txt")
     
    # Shuffle and divide the data set appropriately
    np.random.shuffle(dataset)
    
    training_set = dataset[:int(len(dataset) * 0.7)]
    test_set = dataset[int(len(dataset) * 0.7):]
    
    # Declare and create decision tree model
    decision_tree_model, max_tree_depth = decision_tree(dataset = training_set, label = label, tree_depth = 0)    
    
    '''
    result = predict_dataset(test_set, decision_tree_model)
    
    print(result)
    '''
    
    dataset=np.array([[1,2,3,4,5,6,7],[2,3,4,5,6,7,8]])
    decision_tree={"attribute":0,"value":0,"left":{"attribute":3,"value":4,"left":{"class":15,"leaf":True},"right":{"class":14,"leaf":True},"leaf":False},"right":{"attribute":5,"value":8,"left":{"class":11,"leaf":True},"right":{"class":12,"leaf":True},"leaf":False},"leaf":False}
    y = predict_dataset(dataset, decision_tree)
    print(y)