import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def learningCurve(X, y, clf, num_trials=100, num_folds=10):
    '''
    Generate a learning curve over the training data.
    
    Parameters:
        X: input features
        y: labels
        clf: classifier
        num_trials: number of trials for cross-validation
        num_folds: number of folds for cross-validation
    
    Returns:
        mean_accuracy: mean accuracy for each training subset size
        std_dev_accuracy: standard deviation of accuracy for each training subset size
    '''
    subset_sizes = np.arange(0.1, 1.1, 0.1)
    mean_accuracy = np.zeros(len(subset_sizes))
    std_dev_accuracy = np.zeros(len(subset_sizes))

    for i, subset_size in enumerate(subset_sizes):
        accuracies = []
        for _ in range(num_trials):
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-subset_size, random_state=42)

            # Train the classifier
            clf.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = clf.predict(X_test)

            # Compute accuracy and append to the list
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        # Calculate mean and standard deviation of accuracy for the current subset size
        mean_accuracy[i] = np.mean(accuracies)
        std_dev_accuracy[i] = np.std(accuracies)

    return mean_accuracy, std_dev_accuracy


def evaluatePerformance():
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross-validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump accuracy
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
    '''
    
    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n,d = X.shape

    # shuffle the data
    idx = np.arange(n)
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # Create classifiers
    clf_decision_tree = tree.DecisionTreeClassifier()
    clf_decision_stump = tree.DecisionTreeClassifier(max_depth=1)
    clf_3_level_tree = tree.DecisionTreeClassifier(max_depth=3)

    # Calculate learning curves
    mean_accuracy_dt, std_dev_accuracy_dt = learningCurve(X, y, clf_decision_tree)
    mean_accuracy_ds, std_dev_accuracy_ds = learningCurve(X, y, clf_decision_stump)
    mean_accuracy_dt3, std_dev_accuracy_dt3 = learningCurve(X, y, clf_3_level_tree)

    # TODO: update these statistics based on the results of your experiment
    stats = np.zeros((3, 2))
    stats[0, 0] = np.mean(mean_accuracy_dt)
    stats[0, 1] = np.mean(std_dev_accuracy_dt)
    stats[1, 0] = np.mean(mean_accuracy_ds)
    stats[1, 1] = np.mean(std_dev_accuracy_ds)
    stats[2, 0] = np.mean(mean_accuracy_dt3)
    stats[2, 1] = np.mean(std_dev_accuracy_dt3)

    # Plot learning curves
    plt.errorbar(np.arange(0.1, 1.1, 0.1), mean_accuracy_dt, yerr=std_dev_accuracy_dt, label='Decision Tree')
    plt.errorbar(np.arange(0.1, 1.1, 0.1), mean_accuracy_ds, yerr=std_dev_accuracy_ds, label='Decision Stump')
    plt.errorbar(np.arange(0.1, 1.1, 0.1), mean_accuracy_dt3, yerr=std_dev_accuracy_dt3, label='3-level Decision Tree')
    plt.xlabel('Training Subset Size')
    plt.ylabel('Mean Accuracy')
    plt.legend()
    plt.show()

    return stats

# Do not modify from HERE...
if __name__ == "__main__":
    stats = evaluatePerformance()
    print("Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")")
    print("Decision Stump Accuracy = ", stats[1, 0], " (", stats[1, 1], ")")
    print("3-level Decision Tree = ", stats[2, 0], " (", stats[2, 1], ")")
# ...to HERE.
