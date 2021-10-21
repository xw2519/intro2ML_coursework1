import numpy as np

def read_dataset(filepath): 
    x = np.loadtxt(filepath,usecols=[0,1,2,3,4,5,6]) 
    y = np.loadtxt(filepath,usecols=[7])
    return x, y

(x,y) = read_dataset("./wifi_db/clean_dataset.txt")

from numpy.random import default_rng
def split_dataset(x,y,test_prop,train_prop,random_generator):
    shuffled_order = random_generator.permutation(len(x))
    n_test = round(len(x) * test_prop)
    n_train = round(len(x) * train_prop)
    n_valid = len(x)-n_train-n_test
    x_train = x[shuffled_order[:n_train]]
    y_train = y[shuffled_order[:n_train]]
    x_test = x[shuffled_order[n_train:-n_valid]]
    y_test = y[shuffled_order[n_train:-n_valid]]
    x_valid = x[shuffled_order[-n_valid::]]
    y_valid = y[shuffled_order[-n_valid::]]

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print(x_valid.shape)
    print(y_valid.shape)

    return x_train,y_train,x_test,y_test,x_valid,y_valid

seed = 60012
rg = np.random.default_rng(seed)
a,b,c,d,e,f = split_dataset(x,y,test_prop=0.15,train_prop=0.7,random_generator=rg)


'''
nested_dictionary={}
for i in range(8):
    nested_dictionary[i]={'value':i}

print(nested_dictionary)
'''