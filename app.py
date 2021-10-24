
from bin.decision_tree import Decision_tree 

import numpy as np

'''
app.py

Main python program accessible to the user and links all subprograms together
'''

if __name__ == "__main__":
    # Load data
    clean_data = np.loadtxt("./wifi_db/clean_dataset.txt")
     
    # Shuffle and divide the data set appropriately
    np.random.shuffle(clean_data)
    
    training_set = clean_data[:int(len(clean_data) * 0.7)]
    test_set = clean_data[int(len(clean_data) * 0.7):]
    
    # Declare and create decision tree model
    decision_tree_model = Decision_tree(clean_data, -1)
    
    
        