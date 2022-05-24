"""
Project
Identifying the flora or fauna from picture
#Goals
Measure efficiency of different algorithms by taking total training time, and the end test results
Random Forest, Ada boosting, Logistic regression, Bayesian DL(Gaussian), (NN Ensemble)
dataset and inspiration - https://www.kaggle.com/c/leaf-classification
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
import time


def leafclassification_dataprep():
    """
    This function loads and preps, and shapes the data of 99 different plants
    The function is an encoding function
    :return: X and Y for test and train data, pass the labels for later processing
    """
    # Read data
    train_data = pd.read_csv(r'kaggle-leaf-classification/train.csv')
    test_data = pd.read_csv(r'kaggle-leaf-classification/test.csv')
    # Share data as globals
    global X_test, y_test, X_train, y_train, X_std, test
    # Start to remove labels
    label_encoder = LabelEncoder().fit(train_data.species)
    labels = label_encoder.transform(train_data.species)
    # Grabbing the species names from data, so they can be added again later
    classes = list(label_encoder.classes_)
    # Dropping the names from the data
    train = train_data.drop(['id', 'species'], axis=1)
    test_ids = test_data.id
    test = test_data.drop('id', axis=1)
    X = train.values
    y = labels
    # Stratified data will be the majority of what is used
    # this random state integer was chosen for popularity, the integer doesn't matter if it is constant
    sss = StratifiedShuffleSplit(n_splits=10, test_size=.25, random_state=42)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


def alg_compare():
    """
    Trains and tests RF, AdaBoost, LR, and GNB
    it shows the accuracy and log loss for each function
    :return:
    """
    algorithms = [RandomForestClassifier(),
                  AdaBoostClassifier(),
                  LogisticRegression(),
                  GaussianNB()]
    # Preparing the plots to show accuracy, and log loss
    log_columns = ["Algorithm", "Accuracy", "Log Loss"]
    log = pd.DataFrame(columns=log_columns)
    for alg in algorithms:
        name = alg.__class__.__name__
        start_time = time.time()
        alg.fit(X_train, y_train)
        # Training
        train_predictions = alg.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        train_predictions = alg.predict_proba(X_test)
        ll = log_loss(y_test, train_predictions)
        log_entry = pd.DataFrame([[name, acc * 100, ll]], columns=log_columns)
        log = log.append(log_entry)
        print(name+" took %s seconds"%(time.time()-start_time))
        # Testing
        test_predictions = alg.predict_proba(test)


    sns.barplot(y='Algorithm', x='Accuracy', data=log, color='r')
    plt.title('Algorithm Accuracy(Training)')
    plt.xlabel('Accuracy %')
    plt.show()

    sns.barplot(y='Algorithm', x='Log Loss', data=log, color='r')
    plt.title('Algorithm Log Loss(Training)')
    plt.xlabel('Log Loss')
    plt.show()


if __name__ == "__main__":
    leafclassification_dataprep()
    alg_compare()
