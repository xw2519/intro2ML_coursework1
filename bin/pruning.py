
'''
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

            pruned_node = current_node

            unpruned_eval = evaluate(validation_set, trained_tree)

            #value = ???

            current_node = {"attribute" : 7, "value" : value, "left" : {} , "right" : {}, "leaf" : True}

            pruned_eval = evaluate(validation_set, trained_tree)

            if pruned_eval < unpruned_eval:
                current_node = pruned_node
            else:
                pruned = False

            current_node = stack.pop()

        else:                                                                     # if current node does NOT have 2 leaves, keep traversing tree

            if (current_node['left'] in explored_nodes) and (current_node['right'] in explored_nodes):
                current_node = stack.pop()

            if (not current_node['left']['leaf']) and (current_node['left'] not in explored_nodes):       # go to left node if it is not a leaf and has not been explored
               stack.append(current_node)
               current_node = current_node['left']
            else:                                      # if left node is a leaf, go to right node
                stack.append(current_node)
               current_node = current_node['right']

        explored_nodes.append(current_node)
