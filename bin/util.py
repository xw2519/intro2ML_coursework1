import numpy as np
import matplotlib.pyplot as matplot

'''
util.py

Contains helper functions
'''
def read_dataset(filepath):
    dataset = np.loadtxt(filepath, usecols=[0,1,2,3,4,5,6]) 
    y = np.loadtxt(filepath, usecols=[7])
    
    return dataset, y


def calculate_set_entropy(dataset_list):
    '''
    Calculates the entropy for a given list of datasets
    
    :param1 dataset_list: List of dataset
    :return entropy_result
    '''
    entropy_values = [] 
    total_count = 0
    
    for dataset in dataset_list:
        # Calculate dataset entropy
        dataset_length = len(dataset)
        total_count += dataset_length
        unique_values = np.unique(dataset, return_counts=True)[1]
        entropy = np.sum([-(value/dataset_length) * np.log2(value/dataset_length) for value in unique_values])
        entropy_values.append((entropy, dataset_length))
        
    # Calculate and return the average entropy of the entire dataset 
    return np.sum([(subset_length/(total_count)) * entropy_value for entropy_value, subset_length in iter(entropy_values)])
    
    
def find_split(dataset: np.ndarray, label: np.ndarray):
    '''
    Given a dataset instance and labels, choose the attribute and the value that results in the highest information gain
    
    :param1 dataset: Training dataset
    :param2 label: Label array of dataset
    :return node_attribute, value, cut_point
    '''
    total_information_gain = np.zeros((7,3))
    
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
                    if prob == 0: continue
                    else: entropy_right+=-prob*np.log2(prob)

                rate1 = (i)/float(len(A))
                rate2 = (len(A) - i)/float(len(A))

                total_prob = np.array(total) / len(label)
                previous_entropy = 0

                for prob in total_prob:
                    if prob == 0: continue
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
        
        