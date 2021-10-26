import numpy as np
import collections

'''
the tree is pruned if a full iteration of the while loop completes without removing nodes

while tree is not pruned:
    for every node in tree:
        if node is connected to two leaves:
            prune and compare accuracy

'''
def prune_tree(training_set, validation_set, trained_tree):

    pruned         = False             # pruning is complete if a whole pass of the tree is made without removing nodes

    stack          = []                # stores current path in tree, used for traversing tree without recursion

    current_node   = trained_tree      # current node of interest

    explored_nodes = []                # contains all explored nodes


    while not pruned:

        pruned = True
        explored_nodes = []

        if ((current_node['left']['leaf']) and (current_node['right']['leaf'])):  # if the current node has 2 leaves

            # try to prune the tree

            pruned_node = current_node                                            # node to be removed

            unpruned_eval = evaluate(validation_set, trained_tree)                # evaluate before pruning


            search_set = np.copy(training_set)                                    # find the value of the leaf to replace the current node
            for i in range(len(stack)):
                if i < len(stack):
                    if stack[i]['left'] == stack[i+1]:
                        search_set = search_set[ search_set[node[:,'attribute']] <= node['value'] ]
                    else:
                        search_set = search_set[ search_set[node[:,'attribute']] >  node['value'] ]
                else:
                    if stack[i]['left'] == current_node:
                        search_set = search_set[ search_set[node[:,'attribute']] <= node['value'] ]
                    else:
                        search_set = search_set[ search_set[node[:,'attribute']] >  node['value'] ]

            value = collections.Counter(search_set[:,7]).most_common(1)[0]

            current_node = {"attribute" : 7, "value" : value, "left" : {} , "right" : {}, "leaf" : True}    # create leaf to replace current node

            pruned_eval = evaluate(validation_set, trained_tree)                  # evaluate after pruning


            # if pruning reduces performance, undo and continue traversing tree

            if pruned_eval <= unpruned_eval:
                current_node = pruned_node
                explored_nodes.append(current_node['left'])
                explored_nodes.append(current_node['right'])
            else:
                pruned = False

            current_node = stack.pop()

        else:                                                                     # if current node does NOT have 2 leaves, keep traversing tree

            if (current_node['left']['leaf']) and (current_node['left'] not in explored_nodes):
                explored_nodes.append(current_node['left'])
            if (current_node['right']['leaf']) and (current_node['right'] not in explored_nodes):
                explored_nodes.append(current_node['right'])

            if (current_node['left'] in explored_nodes) and (current_node['right'] in explored_nodes):
                current_node = stack.pop()

            if (not current_node['left']['leaf']) and (current_node['left'] not in explored_nodes):       # go to left node if it is not a leaf and has not been explored
               stack.append(current_node)
               current_node = current_node['left']
            else:                                      # if left node is a leaf, go to right node
                stack.append(current_node)
               current_node = current_node['right']

        explored_nodes.append(current_node)
