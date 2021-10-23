'''
tree.py
    
Creates and manipulates the decision tree
'''

class Node: 
    def __init__(self) -> None:
        self.value = None 
        self.label = None
        self.tree_depth = None 
        
        self.right_child = None 
        self.left_child = None

    def create_node(data_subset, current_tree_depth):
        '''
        Determines the required values of the specified decision tree node
        '''
        ...
        
    def propagate_node(data_value):
        '''
        Propagate through the decision tree nodes to appropriate node based on Binary Search Tree principals
        '''
        ...
    
    
class Decision_tree: 
    def __init__(self) -> None:
        self.root_node = None
        
    def create_decision_tree(dataset): 
        '''
        Given a dataset, create the decision tree 
        
        :param dataset: Training dataset
        '''
        ...
        
    def predict(data_value):
        '''
        Given an unseen value, predict the label based off the created decision tree 
        '''
        ... 
        
        
        