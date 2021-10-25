
from bin.decision_tree import decision_tree 
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
    decision_tree_model = decision_tree(dataset = training_set, label = label, tree_depth = 0)
    
    
        