import numpy as np
import matplotlib.pyplot as matplot

'''
util.py

Contains helper functions
'''

def read_dataset(filepath):
    '''
    Given a filepath, read the file 
    
    Parameters:
    - filepath: Path to the file 
    
    Return:
    - dataset: Values of readings taken (columns 0 - 6)
    - labels: Labels associated with each reading (column 7)
    '''
    dataset = np.loadtxt(filepath, usecols=[0,1,2,3,4,5,6]) 
    labels = np.loadtxt(filepath, usecols=[7])
    
    return dataset, labels  
 
    
def find_split(dataset: np.ndarray, label: np.ndarray):
    '''
    Given a dataset instance and labels, choose the attribute and the value that results in the highest information gain
    
    Parameters:
    - dataset: Training dataset 
    - label: Label array of dataset
    
    Return:
    - node_attribute: 
    - value: 
    - cut_point: 
    '''
    total_information_gain = np.zeros((7, 3))
    
    if (len(dataset) == 2):
        node_attribute = 0
        value = np.around(np.mean(dataset), 2)
        cut_point = 1 
        
        return node_attribute, value, int(cut_point)
    
    for n in range(7):
        A = dataset[:, n]

        order_A = np.argsort(A)
        A = A[order_A]
        label = label[order_A]

        prev_data = A[0]
        cut_prev = A[0]
        current = np.zeros(4)
        highest_information_gain = 0
        cut = prev_data
        cut_point = 0
        total = [(label == 1).sum(), (label == 2).sum(), (label == 3).sum(), (label == 4).sum()]

        for (i, value) in enumerate(A):
            if (i == (len(A) - 1)): break
            elif (i == 0): continue

            if (label[i-1] == 1): current[0] += 1
            elif (label[i-1] == 2): current[1] += 1
            elif (label[i-1] == 3): current[2] += 1
            elif (label[i-1] == 4): current[3] += 1
            else: 
                print("ERROR")
                break

            if (value == prev_data):
                prev_data=value
                continue
            else:         
                left_total = i
                right_total = len(A) - i

                prob_left = current/left_total
                prob_right = (total - current)/right_total

                entropy_left = 0
                for prob in prob_left:
                    if (prob == 0): continue
                    else: entropy_left += -prob*np.log2(prob)

                entropy_right=0
                for prob in prob_right:
                    if (prob == 0): continue
                    else: entropy_right+=-prob*np.log2(prob)

                rate1 = (i)/float(len(A))
                rate2 = (len(A) - i)/float(len(A))

                total_prob = np.array(total) / len(label)
                previous_entropy = 0

                for prob in total_prob:
                    if (prob == 0): continue
                    else: previous_entropy += -prob*np.log2(prob)
                    
                information_gain = previous_entropy - rate1*entropy_left-rate2*entropy_right

                if information_gain > highest_information_gain:
                    highest_information_gain = information_gain
                    cut = (value + cut_prev)/2
                    cut_point = i
                    
                prev_data = value 
                cut_prev = value
                
        total_information_gain[n] = [highest_information_gain, cut, cut_point]
    
    node_attribute = np.argmax(total_information_gain[:, 0], axis=0)
    value = total_information_gain[node_attribute, 1]
    cut_point = total_information_gain[node_attribute, 2]

    return node_attribute, value, int(cut_point)


# Define style and colour of node and arrow
node = dict(boxstyle = "round4", fc = "0.8")
arrow_args = dict(arrowstyle="<-")


def create_decision_tree_graph(decision_tree, max_depth, width):
    ...

