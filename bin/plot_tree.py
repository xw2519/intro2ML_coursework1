import matplotlib.pyplot as plt
import numpy as np

#define style and colour of the node and arrow
node = dict(boxstyle="round4",fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt,parentPt,centerPt,nodeType):
    # plot a single node or leaf with array
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, \
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
    print("Plotting the tree v1 ...")
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111,frameon=False,xticks=[], yticks=[])
    message='X'+str(decision_tree["attribute"])+'<'+str(decision_tree["value"])
    createPlot.ax1.annotate(message,xy=(0,1),xytext=(0,1),va="center",ha="center",bbox=node)
    parentPt=(0,1)
    plotTree(decision_tree["left"],max_depth,1,parentPt,-1.0)
    plotTree(decision_tree["right"],max_depth,1,parentPt,1.0)

    plt.savefig(filename,bbox_inches='tight')
    print('Plot done, saved in',filename)


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
            #print("leaf here")
            #print(curPt)
            message='leaf:'+str(decision_tree["class"])
            plotNode(message,(parentPt[0],parentPt[1]-0.03),curPt,node)
            #print(message)
        else:
            message='X'+str(decision_tree["attribute"])+'<'+str(decision_tree["value"])
            plotNode(message,(parentPt[0],parentPt[1]-0.03),curPt,node)
            plotTree(decision_tree["left"],width,record,cur_depth+1,curPt)
            plotTree(decision_tree["right"],width,record,cur_depth+1,curPt)


    # Draw a plot depending on maximum width
    print("Plotting the tree v2 ...")
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111,frameon=False,xticks=[], yticks=[])
    message='X'+str(decision_tree["attribute"])+'<'+str(decision_tree["value"])
    createPlot.ax1.annotate(message,xy=(0,1),xytext=(0,1),va="center",ha="center",bbox=node)
    max_width=np.max(np.array(width))*0.12
    record=np.zeros((len(width),))
    parentPt=(0,1)
    plotTree2(decision_tree["left"],np.array(width),record,1,parentPt)
    plotTree2(decision_tree["right"],np.array(width),record,1,parentPt)
  

    plt.savefig(filename,bbox_inches='tight')
    print('Plot done, saved in',filename)

