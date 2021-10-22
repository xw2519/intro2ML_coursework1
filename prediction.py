import numpy as np

def predict(dataset,decision_tree):
    '''
    ARGUMENT: 
    x: dataset with unknow labels
    decision_tree: decision tree use to predict the label
    RETURN:
    y: predicted labels
    '''
    print(dataset.shape,"check shape")
    y=np.empty(len(dataset),)
    for i in range(len(dataset)):
        print(i," i")
        y[i] = predict_1(dataset[i,:],decision_tree)

    return y
            
def predict_1(x,decision_tree):
    '''
    x is a [1,7] data,basically 1 line
    '''
    # if leaf, return result else keep predicting
    print(x,"x")
    if x[decision_tree["attribute"]] < decision_tree["value"]:
        print("smaller")
        if decision_tree["leaf"] == True:
            #print("hi")
            return decision_tree["left"]
        else:
            return predict_1(x,decision_tree["left"])
    else:
        print("bigger")
        if decision_tree["leaf"] == True:
            #print("hi")
            return decision_tree["right"]
        else:
            return predict_1(x,decision_tree["right"])

'''
dataset=np.array([[1,2,3,4,5,6,7],[2,3,4,5,6,7,8]])
decision_tree={"attribute":0,"value":0,"left":{"attribute":3,"value":4,"left":15,"right":16,"leaf":True},"right":{"attribute":5,"value":8,"left":11,"right":13,"leaf":True},"leaf":False}

y = predict(dataset,decision_tree)
print(y)
'''





            

        
