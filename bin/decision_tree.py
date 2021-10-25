'''
tree.py
    
Creates and manipulates the decision tree
'''

from bin.util import find_split
import numpy as np    

# Global variable declaration 
tree_depth = 0
width = [0,0,0,0,0]

def decision_tree(dataset, label, tree_depth):
    [classes, unique_label] = np.unique(label, return_inverse=True) 
    
    if(len(classes) == 1):
        leaf_node = {"class" : classes[0], "leaf" : True}

        if tree_depth >= len(width): width.append(1)
        else: width[tree_depth] += 1
        
        return leaf_node, tree_depth
    else:
        attribute, value, cut_point = find_split(dataset, label) 

        order = np.argsort(dataset[:, attribute])
        data = dataset[order]
        label = label[order]
        
        l_dataset=data[:cut_point] 
        l_label=label[:cut_point]
        
        r_dataset=data[cut_point:]
        r_label=label[cut_point:]

        node = {"attribute" : attribute, "value" : value, "left" : {} , "right" : {}, "leaf" : False}
        
        if tree_depth >= len(width): width.append(1)
        else: width[tree_depth] += 1
        
        l_branch, l_depth = decision_tree(l_dataset, l_label,tree_depth+1)
        r_branch, r_depth = decision_tree(r_dataset, r_label,tree_depth+1)

        node["left"] = l_branch
        node["right"] = r_branch

        return node, max(l_depth,r_depth) 