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

### Usage
To start the interactive program, run the Python file: [app.py](app.py). 

`app.py` is the central program that links all the sub-modules and allows the user to interact with the program. Every program feature is accessible from the terminal of `app.py`.

- The user will be prompt to enter the full file path to a dataset e.g. `datasets\clean_dataset.txt` for training and testing the decision tree.
- All the features of the program will be numbered and listed to the user to choose from. The user is expected to enter the number corresponding to the desired function.
- The program will execute the selected function and save all program output to the [output](output) folder. 
- The random seed needs to be specified if the user is performing `2: 10-fold cross validation` or `3: Nested 10-fold cross validation with post-pruning`.

An example output of program usage:
```
(.env) PS user> python3 app.py

Input path to dataset: /wifi_db/clean_dataset.txt

Program functions: [Please enter the corresponding number]
 1: Plot Decision Tree
 2: 10-fold cross validation
 3: Nested 10-fold cross validation with post-pruning    

Choice: 2

-------------------- Program execution log --------------------
2: 10-fold cross validation

Please choose the random seed: 56

Loading dataset from: /wifi_db/clean_dataset.txt

--------- Cross Validation Metrics ---------
Average Confusion Matrix:
[[38.5  2.6  3.8  4.1]   
 [ 2.5 40.3  4.3  2.6]   
 [ 3.   3.8 40.9  3.8]   
 [ 4.4  2.7  3.6 39.1]]

Average Accuracy:              0.794
Accuracy Standard Deviation:   0.01281
Average Precision:            [0.79712101 0.81247094 0.78128313 0.79145837]
Average Recall:               [0.7813252  0.81110196 0.79455889 0.78364575]
Average F1 Score:             [0.78608478 0.81014749 0.78598421 0.78480441]

Saving cross validation result to 'output/cross_validation_results.txt'
Plotting confusion matrix
Plotting completed. Plot saved to 'output/confusion_matrix.jpg'

To select other functions, press 'Enter'
```

