import numpy as np 
import matplotlib as plt
import sys

np.set_printoptions(threshold=sys.maxsize)

def read_dataset(filepath): 
    x = np.loadtxt(filepath,usecols=[0,1,2,3,4,5,6]) 
    y = np.loadtxt(filepath,usecols=[7])
    return x, y

def find_split (x,y) :
    total_ig = np.zeros((7,2))
    for n in range(len(x[0])) :
        A=x[:,n]
        #print(A.max())
        
        order_A=np.argsort(A)
        A=A[order_A]
        label=y[order_A]
        #print(A)
        #print(label)

        prev_data=A[0]
        cut_prev = A[0]
        current=np.zeros(4)
        highest_ig=0
        cut=prev_data
        cutpoint = 0
        total = [(y==1).sum(),(y==2).sum(),(y==3).sum(),(y==4).sum()]

        for (i,value) in enumerate(A):
            #print(i)
            if i==(len(A)-1):
                break
            
            if label[i]==1:
                current[0]+=1
            elif label[i]==2:
                current[1]+=1
            elif label[i]==3:
                current[2]+=1
            elif label[i]==4:
                current[3]+=1
            else:
                print("ERROR")
                break

            if value==prev_data:
                prev_data=value
                continue
            else:
                #print(value," hi ")
            
                #print(current,"current")
                left_total = i+1
                right_total = len(A)-i-1

                #print(left_total)
                #print(right_total)
                
                prob_left=current/left_total
                prob_right=(total-current)/right_total
            
                #print(prob_left," prob_left")
                #print(prob_right," prob_right")

                entropy_left=0
                for prob in prob_left:
                    if prob == 0:
                        continue
                    else:
                        entropy_left+=-prob*np.log2(prob)
                #print(entropy_left,"entro_left")

                entropy_right=0
                for prob in prob_right:
                    if prob == 0:
                        continue
                    else:
                        entropy_right+=-prob*np.log2(prob)
                #print(entropy_right,"entro_right")

                rate1=(i+1)/float(len(A))
                rate2=(len(A)-i-1)/float(len(A))
                #print(rate1,"rate1")
                #print(rate2,"rate2")
                ig=2-rate1*entropy_left-rate2*entropy_right
                
                if ig > highest_ig:
                    highest_ig = ig
                    cut=(value+cut_prev)/2
                    cut_point = i
                prev_data=value 
                cut_prev = value
        total_ig[n]=[highest_ig,cut]
    attribute=np.argmax(total_ig[:,0],axis=0)
    value = total_ig[attribute,1]
    #print(total_ig)
    print('the attribute chosen is ' + str(attribute))
    print('the cut point is ' + str(value))
    return attribute,value,cut_point

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
nested_dict = {}
def DecisionTree(data,label,depth) :

    depth=0
    node_num=0
    [classes, label_unique] = np.unique(label, return_inverse=True) 
    print(len(classes))
    '''
    dict = {
        attribute :  
        value : 
        left :
        right :  
        leaf : True
    }
    '''
    if len(classes) == 1 :
        # 
        depth +=1
        node={"attribute":-1,"value":-1,"left":-1,"right":-1,"leaf":True}
        #return depth,node
    else:

        attribute,value,cut_point = find_split(data,label) 
        nested_dict[node_num]={"attribute":attribute,"value":value,"left":-1,"right":-1,"leaf":False}
        node_num+=1
        #split current data into left and right
        #tbc
        order=np.argsort(data[:,attribute])
        data=data[order]
        label=label[order]
        
        l_dataset=data[:cut_point] 
        l_label=label[:cut_point]
        r_dataset=data[cut_point:]
        r_label=data[cut_point:]
        print(len(l_dataset))
        print(len(r_dataset))
        print(len(l_label))
        print(len(r_label))
        #node <- a new decision tree with root as split value
        #l_depth=DecisionTree(l_dataset,l_label,depth+1)

        #r_depth=DecisionTree(r_dataset,r_label,depth+1)

        #return max(l_depth,r_depth)

def main() :
    data, label = read_dataset("./wifi_db/clean_dataset.txt")
    DecisionTree(data,label,depth = 0)

main()