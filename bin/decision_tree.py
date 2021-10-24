'''
tree.py
    
Creates and manipulates the decision tree
'''

from bin.util import find_split
import numpy as np    
    
class Decision_tree: 
    def __init__(self, dataset, tree_depth) -> None:
         
        if len(np.unique(dataset[:, -1])) == 1: 
            self.label = dataset[0, -1]
        else: 
            self.tree_depth = tree_depth + 1
            self.label, self.value, left_sub_dataset, right_sub_dataset = find_split(dataset)
            
            print(self.value)
                        
            self.left_child = Decision_tree(left_sub_dataset, self.tree_depth)
            self.right_child = Decision_tree(right_sub_dataset, self.tree_depth)
                   
        
    def predict(data_value):
        '''
        Given an unseen value, predict the label based off the created decision tree 
        '''
        ... 
        
        
        