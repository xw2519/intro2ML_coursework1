import matplotlib.pyplot as plt
 
#define style and colour of the node and arrow
node = dict(boxstyle="round4",fc="0.8")
arrow_args = dict(arrowstyle="<-")
 
def plotNode(nodeTxt,parentPt, centerPt,nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, \
    xycoords='axes fraction',
    xytext=centerPt, textcoords='axes fraction',\
    va="center",ha="center", bbox=nodeType, arrowprops=arrow_args)

def createPlot(decision_tree,max_depth):
    print("Plotting the tree ...")
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111,frameon=False,xticks=[], yticks=[])
    message='X'+str(decision_tree["attribute"])+'<'+str(decision_tree["value"])
    createPlot.ax1.annotate(message,xy=(0,1),xytext=(0,1),va="center",ha="center",bbox=node)
    #print(message)
    parentPt=(0,1)
    plotTree(decision_tree["left"],max_depth,1,parentPt,-1.0)
    plotTree(decision_tree["right"],max_depth,1,parentPt,1.0)
    # start to loop over 

    plt.savefig('tree.jpg',bbox_inches='tight')
    print('Plot done, saved in tree.jpg')

def plotTree(decision_tree,max_depth,cur_depth,parentPt,left):
    # left = -1.0, left branch, right =1.0, right branch 
    width=0.005
    param=max_depth-cur_depth
    change_height=1.5
    curPt=parentPt[0]+left*width*pow(2,param)/4,parentPt[1]-change_height
    if decision_tree["leaf"]==True:
        #print("leaf here")
        #print(curPt)
        message='leaf:'+str(decision_tree["class"])
        plotNode(message,(parentPt[0],parentPt[1]-0.03),curPt,node)
        #print(message)
    else:
        message='X'+str(decision_tree["attribute"])+'<'+str(decision_tree["value"])
        plotNode(message,(parentPt[0],parentPt[1]-0.03),curPt,node)
        plotTree(decision_tree["left"],max_depth,cur_depth+1,curPt,-1.0)
        plotTree(decision_tree["right"],max_depth,cur_depth+1,curPt,1.0)

'''
decision_tree={"attribute":0,"value":0,"left":{"attribute":3,"value":4,"left":{"class":15,"leaf":True},"right":{"class":14,"leaf":True},"leaf":False},"right":{"attribute":5,"value":8,"left":{"class":11,"leaf":True},"right":{"class":12,"leaf":True},"leaf":False},"leaf":False}
createPlot(decision_tree,3)
'''

