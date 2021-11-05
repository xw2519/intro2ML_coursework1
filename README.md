# COMP97101/97151 - Introduction to ML: Coursework 1

A Python implementation of a decision tree algorithm that can determine one of the indoor locations based on recorded WIFI signal strengths.

Contents
========
 * [Contributors](#contributors)
 * [Features](#features)
 * [Usage](#usage)

### Contributors

- Anqi Qiu [CID: 01733465]
- Ebby Samson [CID: 01737449]
- Sijun You [CID: 01747112]
- Xin Wang [CID: 01735352]

### Features

- The program saves all plots of decision trees and confusion matrices in the [output](output) folder.
- All evaluation metrics are saved to the the [output](output) folder.
- Random seed is required to be specified by the user. This is done in order to allow the user to replicate a result if the random seed is known.
- The report is contained in the [docs](docs) folder. 

### Usage
To start the interactive program, run the Python file: [app.py](app.py). 

`app.py` is the central program that links all the sub-modules and allows the user to interact with the program. Every program feature is accessible from the terminal of `app.py`.

- The user will be prompt to enter the full file path to a dataset e.g. `wifi_db/clean_dataset.txt` for training and testing the decision tree.
- All the features of the program will be numbered and listed to the user to choose from. The user is expected to enter the number corresponding to the desired function.
- The program will execute the selected function and save all program output to the [output](output) folder. 
- The random seed needs to be specified if the user is performing `2: 10-fold cross validation` or `3: Nested 10-fold cross validation with post-pruning`.

An example output of program usage:
```
(.env) PS user> python3 app.py

Input filepath to dataset: ./wifi_db/clean_dataset.txt

Program functions: [Please enter the corresponding number]
 1: Plot Decision Tree
 2: 10-fold cross validation
 3: Nested 10-fold cross validation with post-pruning    

Choice: 2

-------------------- Program execution log --------------------
2: 10-fold cross validation

Please choose the random seed: 56

Loading dataset from:  wifi_db/clean_dataset.txt

Average depth of the unpruned tree:  13.5
---------------------------------------- Cross Validation Metrics ----------------------------------------------
Average Confusion Matrix:
[[49.7  0.   0.2  0.1]
 [ 0.  48.1  1.9  0. ]
 [ 0.3  2.1 47.4  0.2]
 [ 0.5  0.   0.2 49.3]]

Average Accuracy:              0.972
Accuracy Standard Deviation:   0.0127
Average Precision:            [0.985 0.958 0.954 0.993]
Average Recall:               [0.994 0.961 0.947 0.986]
Average F1 Score:             [0.989 0.959 0.95  0.989]

Saving cross validation result to 'output/cross_validation_results.txt'

Plotting confusion matrix...
Plotting completed. Plot saved to 'output/confusion_matrix.jpg'

To exit, type 'q'; Continue otherwise : 
```

