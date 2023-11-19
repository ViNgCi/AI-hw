'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Chris Clingerman
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score



def evaluatePerformance():
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
      
    ** Note that your implementation must follow this API**
    '''
    
    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n,d = X.shape

    # shuffle the data
    num_trials = 100
    num_folds = 10
    fold_accuracy = np.zeros((num_trials*num_folds,3))
    depths = [None, 1, 3] 
    for i in range(num_trials):
      idx = np.arange(n) 
      np.random.seed(13)
      np.random.shuffle(idx)
      X = X[idx]
      y = y[idx]
      for j in range(num_folds):
        # split the data
            test_indices = idx[j * (n // num_folds): (j + 1) * (n // num_folds)]
            train_indices = np.concatenate([idx[:j * (n // num_folds)], idx[(j + 1) * (n // num_folds):]])

            Xtrain, Xtest = X[train_indices], X[test_indices]
            ytrain, ytest = y[train_indices], y[test_indices]

            for depth_idx, depth in enumerate(depths):
                clf = tree.DecisionTreeClassifier(max_depth=depth)
                clf.fit(Xtrain, ytrain)
                y_pred = clf.predict(Xtest)
                # compute the training accuracy of the model
                meanDecisionTreeAccuracy = accuracy_score(ytest, y_pred)
                fold_accuracy[(i*num_folds+j, depth_idx)] = meanDecisionTreeAccuracy
    
    # output predictions on the remaining data
    y_pred = clf.predict(Xtest)

    # compute the training accuracy of the model
    meanDecisionTreeAccuracy = accuracy_score(ytest, y_pred)
    
    
    # TODO: update these statistics based on the results of your experiment
    stddevDecisionTreeAccuracy = 0
    meanDecisionStumpAccuracy = 0
    stddevDecisionStumpAccuracy = 0
    meanDT3Accuracy = 0
    stddevDT3Accuracy = 0

    # make certain that the return value matches the API specification
    stats = np.zeros((3,2))
    stats[0,0] = meanDecisionTreeAccuracy
    stats[0,1] = stddevDecisionTreeAccuracy
    stats[1,0] = meanDecisionStumpAccuracy
    stats[1,1] = stddevDecisionStumpAccuracy
    stats[2,0] = meanDT3Accuracy
    stats[2,1] = stddevDT3Accuracy
    return stats



# Do not modify from HERE...
if __name__ == "__main__":
    
    stats = evaluatePerformance()
    print( "Decision Tree Accuracy = ", stats[0,0], " (", stats[0,1], ")")
    print( "Decision Stump Accuracy = ", stats[1,0], " (", stats[1,1], ")")
    print( "3-level Decision Tree = ", stats[2,0], " (", stats[2,1], ")")
# ...to HERE.
