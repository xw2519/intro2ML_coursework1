import numpy as np 
import matplotlib as plt
import sys
import find_split as fs
np.set_printoptions(threshold=sys.maxsize)

'''
dict = {
    attribute :  
    value : 
    left :
    right :  
    leaf : True
}
'''

class Node:
    def __init__(self, attribute = None, value = None, leaf = False):
        self.attribute = attribute #attribute 1-7
        self.value = value #split point
        self.left = None 
        self.right = None
        self.leaf = False #whether it is a leaf node or not


def read_dataset(filepath): 
    x = np.loadtxt(filepath,usecols=[0,1,2,3,4,5,6]) 
    y = np.loadtxt(filepath,usecols=[7])
    return x, y



'''
1: procedure decision tree learning(training dataset, depth)
2:      if all samples have the same label then
3:          return (a leaf node with this value, depth)
4:      else
5:          split <- find split(training dataset)
6:          node <- a new decision tree with root as split value
7:          l branch, l depth <- DECISION TREE LEARNING(l dataset, depth+1)
8:          r branch, r depth <- DECISION TREE LEARNING(r dataset, depth+1)
9:          return (node, max(l depth, r depth))
10:     end if
11: end procedure
'''
#nested_dict = {}
def DecisionTree(data,label,depth) :
    #print("hiya")
    depth=0
    #node_num=0
    [classes, label_unique] = np.unique(label, return_inverse=True) 
    #print(len(classes))
    #print("classes is", classes)
    #print("shape of data is", data.shape)
    if len(classes) == 1 :
        # 
        #print("I am here 1")
        return Node(leaf = True), depth
        #return depth,node
    else:
        #print("I am here 2")
        attribute,value,cut_point = fs.find_split(data,label) 
        #nested_dict[node_num]={"attribute":attribute,"value":value,"left":-1,"right":-1,"leaf":False}
        #node_num+=1
        #split current data into left and right
        #tbc
        order=np.argsort(data[:,attribute])
        data=data[order]
        label=label[order]
        
        #can combine into a function later
        l_dataset=data[:cut_point] 
        l_label=label[:cut_point]
        r_dataset=data[cut_point:]
        r_label=label[cut_point:]
        #print(len(l_dataset))
        #print(len(r_dataset))
        #print(len(l_label))
        #print(len(r_label))
        #node <- a new decision tree with root as split value
        node = Node(attribute = attribute, value = value)
        l_branch, l_depth = DecisionTree(l_dataset,l_label,depth+1)
        r_branch, r_depth = DecisionTree(r_dataset,r_label,depth+1)
        node.left = l_branch
        node.right = r_branch
        return node, max(l_depth,r_depth)

def main() :
    data, label = read_dataset("./wifi_db/clean_dataset.txt")
    DecisionTree(data,label,depth = 0)

main()