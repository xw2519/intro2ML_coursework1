import numpy as np

def find_split (x,y) :
    #print("entering ")
    #print("shape is ", x.shape)
    total_ig = np.zeros((7,3))
    #print("len is ", len(x[0]))

    #if()
    
    for n in range(7) :
        A=x[:,n]
        #print(A.max())
        #print(len(x))
        order_A=np.argsort(A)
        A=A[order_A]
        label=y[order_A]
        #print(A)
        #print(label)
        #print(A.shape)
        #print(label.shape)

        '''
        if len(x) == 2:
            "entering here"
            attribute = n 
            value = np.mean(x)
            cutpoint = 1 
            break;        
        '''
        prev_data=A[0]
        cut_prev = A[0]
        current=np.zeros(4)
        highest_ig=0
        cut=prev_data
        cutpoint = 0
        total = [(y==1).sum(),(y==2).sum(),(y==3).sum(),(y==4).sum()]

        for (i,value) in enumerate(A):
            #print("hi", i)

            if i==(len(A)-1):
                break
            #print("my label is ", label[i])
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
                total_prob = np.array(total) / len(y)
                entropy_ori = 0
                for prob in total_prob:
                    if prob == 0:
                        continue
                    else:
                        entropy_ori+=-prob*np.log2(prob)
                ig=entropy_ori-rate1*entropy_left-rate2*entropy_right
                #print("ig is ", ig)
                if ig > highest_ig:
                    highest_ig = ig
                    cut=(value+cut_prev)/2
                    cutpoint = i
                prev_data=value 
                cut_prev = value
        total_ig[n]=[highest_ig,cut,cutpoint]
    attribute=np.argmax(total_ig[:,0],axis=0)
    value = total_ig[attribute,1]
    cutpoint = total_ig[attribute,2]
    #print(total_ig)
    print('the attribute chosen is ' + str(attribute))
    print('the cut value is ' + str(value))
    print('the cut point is ' + str(cutpoint))
    return attribute,value,int(cutpoint)