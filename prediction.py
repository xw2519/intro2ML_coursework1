import numpy as np

def predict(dataset,decision_tree):
    '''
    ARGUMENT: 
    x: dataset with unknow labels, shape=[n,7]
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
    x: single data, shape=[1,7]

    '''
    # if leaf, return result else keep predicting
    print(x,"x")
    if decision_tree["leaf"] == True:
            #print("leaf!")
            return decision_tree["class"]
    else:
        if x[decision_tree["attribute"]]<=decision_tree["value"]:
            print(str(x[decision_tree["attribute"]])+"<="+str(decision_tree["value"]))
            return predict_1(x,decision_tree["left"])
        else:
            print(str(x[decision_tree["attribute"]])+">"+str(decision_tree["value"]))
            return predict_1(x,decision_tree["right"])


dataset=np.array([[1,2,3,4,5,6,7],[2,3,4,5,6,7,8]])
decision_tree={"attribute":0,"value":0,"left":{"attribute":3,"value":4,"left":{"class":15,"leaf":True},"right":{"class":14,"leaf":True},"leaf":False},"right":{"attribute":5,"value":8,"left":{"class":11,"leaf":True},"right":{"class":12,"leaf":True},"leaf":False},"leaf":False}
y = predict(dataset,decision_tree)
print(y)






            

        
