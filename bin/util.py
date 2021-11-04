from matplotlib import colors

import numpy as np
import matplotlib.pyplot as matplot

'''
util.py

Contains helper functions
'''

def read_and_shuffle_dataset(filepath):
    '''
    Given a filepath, read the file 
    
    Parameters:
    - filepath: Path to the file 
    
    Return:
    - dataset: Values of readings taken (columns 0 - 6)
    - labels: Labels associated with each reading (column 7)
    '''
    loaded_data = np.loadtxt(filepath)
    
    np.random.shuffle(loaded_data)
     
    dataset = loaded_data[:, [0, 1, 2, 3, 4, 5, 6]]
    labels = loaded_data[:, 7]
    
    return dataset, labels  
 
    
def find_split(dataset, label):
    '''
    Given a dataset instance and labels, choose the attribute and the value that results in the highest information gain
    
    Parameters:
    - dataset: Training dataset 
    - label: Label array of dataset
    
    Return:
    - node_attribute: 
    - split_value:                               
    - cutting_points: 
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
        ordered_label = label[order_A]

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

            if (ordered_label[i-1] == 1): current[0] += 1
            elif (ordered_label[i-1] == 2): current[1] += 1
            elif (ordered_label[i-1] == 3): current[2] += 1
            elif (ordered_label[i-1] == 4): current[3] += 1
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


def plot_confusion_matrix(confusion_matrix):
    '''
    Given a 'confusion_matrix', plot and show the confusion matrix using matplotlib
    
    Parameters:
    - confusion_matrix: Confusion matrix 
    
    Return: None
    '''
    figure, axis = matplot.subplots(1, 1)
    
    # Define colours used 
    cmap = colors.ListedColormap(['lightcoral', 'lightgreen'])
    
    matplot.imshow(confusion_matrix, cmap = cmap)
    matplot.xlabel("Predicted Labels")
    matplot.ylabel("Actual Labels")

    label_list = ['1', '2', '3', '4']

    axis.set_xticks([0, 1, 2, 3])
    axis.set_yticks([0, 1, 2, 3])

    axis.set_xticklabels(label_list)
    axis.set_yticklabels(label_list)

    for i in range(4):
        for j in range(4): axis.text(j, i, np.round(confusion_matrix[i][j], 4), ha = "center", va = "center", color = 'black')
    
    matplot.savefig('output/confusion_matrix.jpg', bbox_inches = 'tight')


def print_evaluation_metrics(accuracy, precision, recall, f_score):
    '''
    Prints the evaluation metrics of a testset
    
    Parameters:
    - accuracy: Number of correctly classified examples divided by the total number of examples
    - precision: Number of correctly classified positive divided by the total number of predicted positive 
    - recall: Number of correctly classified positive divided by the total number of positive
    - f_score: Performance measure of the classifier
    
    Return: None
    '''
    print('--------- Evaluation Metrics ---------')
    print('Accuracy:                     ', accuracy)
    print('Precision:                   ', precision)
    print('Recall:                      ', recall)
    print('F1 Score:                    ', f_score)
    print()


def print_cross_validation_metrics(average_accuracy, accuracy_standard_deviation, average_precision, average_recall, average_f_score, average_confusion_matrix, prune_metrics = False, keep_previous_metrics = False):
    '''
    Prints the evaluation metrics of a testset
    
    Parameters:
    - accuracy: Number of correctly classified examples divided by the total number of examples
    - accuracy_standard_deviation: Standard deviation of accuracy
    - precision: Number of correctly classified positive divided by the total number of predicted positive 
    - recall: Number of correctly classified positive divided by the total number of positive
    - f_score: Performance measure of the classifier
    - prune_metrics: Flag checking if the metrics are for pruning and required two entries into output file
    - keep_previous_metrics: Flag differentiating between unpruned and pruned cross validation metric
    
    Return: None
    '''
    print("---------------------------------------- Cross Validation Metrics ----------------------------------------------")  
    print('Average Confusion Matrix:')
    print(np.round(average_confusion_matrix.astype(np.float), 3))
    print()
    print('Average Accuracy:             ', np.round(average_accuracy, 3))
    print('Accuracy Standard Deviation:  ', np.round(accuracy_standard_deviation, 5))
    print('Average Precision:           ', np.round(average_precision, 3))
    print('Average Recall:              ', np.round(average_recall, 3))
    print('Average F1 Score:            ', np.round(average_f_score, 3))
    print()
    
    # Save results to output file
    print("Saving cross validation result to 'output/cross_validation_results.txt'")
    
    # Print standard cross validation metrics
    if not prune_metrics:
        output_file = open('output/cross_validation_results.txt', 'w')
        print('---------------------------------------- Cross Validation Metrics ----------------------------------------------', file = output_file)    
        print('Average Confusion Matrix:', file = output_file)
        print(np.round(average_confusion_matrix.astype(np.float), 3), file = output_file)
        print('', file = output_file)
        print('Average Accuracy:             ', average_accuracy, file = output_file)
        print('Accuracy Standard Deviation:  ', accuracy_standard_deviation, file = output_file)
        print('Average Precision:           ', average_precision, file = output_file)
        print('Average Recall:              ', average_recall, file = output_file)
        print('Average F1 Score:            ', average_f_score, file = output_file)
        print()
        output_file.close()
        
    # Print pruning metrics
    else: 
        if not keep_previous_metrics: 
            output_file = open('output/cross_validation_results.txt', 'w')
            print('----------------------------------- Unpruned Cross Validation Metrics ------------------------------------------', file = output_file)   
            print('Average Confusion Matrix:', file = output_file)
            print(np.round(average_confusion_matrix.astype(np.float), 3), file = output_file)
            print('', file = output_file)
            print('Average Accuracy:             ', average_accuracy, file = output_file)
            print('Accuracy Standard Deviation:  ', accuracy_standard_deviation, file = output_file)
            print('Average Precision:           ', average_precision, file = output_file)
            print('Average Recall:              ', average_recall, file = output_file)
            print('Average F1 Score:            ', average_f_score, file = output_file)
            print(file = output_file)
            output_file.close()
            
        else: 
            output_file = open('output/cross_validation_results.txt', 'a')
            print('------------------------------------ Pruned Cross Validation Metrics -------------------------------------------', file = output_file)    
            print('Average Confusion Matrix:', file = output_file)
            print(np.round(average_confusion_matrix.astype(np.float), 3), file = output_file)
            print('', file = output_file)
            print('Average Accuracy:             ', average_accuracy, file = output_file)
            print('Accuracy Standard Deviation:  ', accuracy_standard_deviation, file = output_file)
            print('Average Precision:           ', average_precision, file = output_file)
            print('Average Recall:              ', average_recall, file = output_file)
            print('Average F1 Score:            ', average_f_score, file = output_file)
            print(file = output_file)
            output_file.close()
    
    return


#define style and colour of the node and arrow
node = dict(boxstyle="round4",fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt,parentPt,centerPt,nodeType):
    # plot a single node or leaf with array
    plot_decision_tree.ax1.annotate(nodeTxt, xy=parentPt, \
    xycoords='axes fraction',
    xytext=centerPt, textcoords='axes fraction',\
    va="center",ha="center", bbox=nodeType, arrowprops=arrow_args)


def plot_decision_tree(decision_tree,max_depth,filename):
    '''
    Given a decision tree and its maximum depth,
    Plot the tree depends on its maximum depth using matplotlib
    Finally save the picture into a jpg with the given name as filename
        
    Parameters:
    - decision_tree: a dictionary containing all the tree nodes and leaves
    - max_depth: an int, presenting the maximum depth of the decision tree
    - filename: a string, the name of the file which the final image will save to, e.g. 'tree1.jpg'
    
    Return: None
    '''
    def plotNode(nodeTxt,parentPt,centerPt,nodeType):
        # plot a single node or leaf with array
        plot_decision_tree.ax1.annotate(nodeTxt, xy=parentPt, \
        xycoords='axes fraction',
        xytext=centerPt, textcoords='axes fraction',\
        va="center",ha="center", bbox=nodeType, arrowprops=arrow_args)

    def plotTree(decision_tree,max_depth,cur_depth,parentPt,left):
        # left = -1.0 means left branch, left = 1.0 means right branch 
        width=0.01
        if max_depth>=6:
            param=max_depth-cur_depth-5
        else:   
            param=max_depth-cur_depth
        change_height=0.5
        curPt=parentPt[0]+left*width*pow(2,param)/4,parentPt[1]-change_height
        if decision_tree["leaf"]==True:
            message='leaf:'+str(decision_tree["class"])
            plotNode(message,(parentPt[0],parentPt[1]-0.03),curPt,node)
        else:
            message='X'+str(decision_tree["attribute"])+'<'+str(decision_tree["value"])
            plotNode(message,(parentPt[0],parentPt[1]-0.03),curPt,node)
            plotTree(decision_tree["left"],max_depth,cur_depth+1,curPt,-1.0)
            plotTree(decision_tree["right"],max_depth,cur_depth+1,curPt,1.0)

    # Draw a plot depending on depth
    print("Plotting decision tree using depth ...")
    fig = matplot.figure(1,facecolor='white')
    fig.clf()
    plot_decision_tree.ax1 = matplot.subplot(111,frameon=False,xticks=[], yticks=[])
    message='X'+str(decision_tree["attribute"])+'<'+str(decision_tree["value"])
    plot_decision_tree.ax1.annotate(message, xy=(0,1), xytext=(0,1), va="center", ha="center", bbox=node)
    parentPt = (0,1)
    plotTree(decision_tree["left"], max_depth, 1, parentPt, -1.0)
    plotTree(decision_tree["right"], max_depth, 1, parentPt, 1.0)

    matplot.savefig(filename, bbox_inches = 'tight')
    print("Plotting completed. Plot saved to ", filename)


def plot_decision_tree_v2(decision_tree,width,filename):
    '''
    Given a decision tree and its width at each depth,
    Plot the tree depends on its width at each depth using matplotlib
    Finally save the picture into a jpg with the given name as filename
        
    Parameters:
    - decision_tree: a dictionary containing all the tree nodes and leaves
    - width: a list, contain the number of nodes and leaves at each depth of the tree
    - filename: a string, the name of the file which the final image will save to, e.g. 'tree1.jpg'
    
    Return: None
    '''
    def plotNode(nodeTxt,parentPt,centerPt,nodeType):
        # Plot a single node or leaf with array
        plot_decision_tree_v2.ax1.annotate(nodeTxt, xy=parentPt, \
        xycoords='axes fraction',
        xytext=centerPt, textcoords='axes fraction',\
        va="center",ha="center", bbox=nodeType, arrowprops=arrow_args)

    def plotTree2(decision_tree,width,record,cur_depth,parentPt):
        # A recursive function 
        cur_num = record[cur_depth]
        max_width=np.max(width)*0.25
        node_num = width[cur_depth]
        step=max_width/(node_num-1)
        record[cur_depth]+=1

        change_height=0.3

        curPt=-max_width/2+step*cur_num,parentPt[1]-change_height
        if decision_tree["leaf"]==True:
            message='leaf:'+str(decision_tree["class"])
            plotNode(message,(parentPt[0],parentPt[1]-0.03),curPt,node)

        else:
            message='X'+str(decision_tree["attribute"])+'<'+str(decision_tree["value"])
            plotNode(message,(parentPt[0],parentPt[1]-0.03),curPt,node)
            plotTree2(decision_tree["left"],width,record,cur_depth+1,curPt)
            plotTree2(decision_tree["right"],width,record,cur_depth+1,curPt)

    # Draw a plot depending on maximum width
    print("Plotting decision tree v2 using width ...")
    fig = matplot.figure(1, facecolor = 'white')
    fig.clf()
    plot_decision_tree_v2.ax1 = matplot.subplot(111,frameon=False,xticks=[], yticks=[])
    message = 'X'+str(decision_tree["attribute"])+'<'+str(decision_tree["value"])
    plot_decision_tree_v2.ax1.annotate(message, xy = (0,1), xytext = (0,1), va = "center", ha = "center", bbox = node)
    max_width=np.max(np.array(width))*0.12
    record = np.zeros((len(width),))
    parentPt = (0,1)
    plotTree2(decision_tree["left"], np.array(width), record, 1, parentPt)
    plotTree2(decision_tree["right"], np.array(width), record, 1, parentPt)

    matplot.savefig(filename, bbox_inches = 'tight')
    print("Plotting completed. Plot saved to ", filename)
