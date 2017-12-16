import cPickle
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

def scale_descriptors(descriptors):
    scaler = StandardScaler().fit(descriptors)
    # Scale training data
    return scaler.transform(descriptors)

def train_knn(descriptors, labels, experiment_filename, save_model=False):
    descriptors = scale_descriptors(descriptors)
    myknn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    print 'Fitting KNN...'
    myknn.fit(descriptors, labels)
    print 'Done!'
    if save_model:
        print 'Saving KNN model...'
        cPickle.dump(myknn, open('./models/' + experiment_filename, 'w'))
        print 'KNN model saved!'
    return myknn


def train_random_forest(descriptors, labels, experiment_filename, save_model=False):
    descriptors = scale_descriptors(descriptors)
    myrf = RandomForestClassifier(n_estimators=10)
    print 'Fitting Random Forest...'
    myrf.fit(descriptors, labels)
    print 'Done!'
    if save_model:
        print 'Saving Random Forest model...'
        cPickle.dump(myrf, open('./models/' + experiment_filename, 'w'))
        print 'Model saved!'
    return myrf


def train_gaussian_naive_bayes(descriptors, labels, experiment_filename, save_model=False):
    descriptors = scale_descriptors(descriptors)
    mygnb = GaussianNB()
    print 'Fitting Gaussian Naive Bayes classifier...'
    mygnb.fit(descriptors, labels)
    print 'Done!'
    if save_model:
        print 'Saving GNB model...'
        cPickle.dump(mygnb, open('./models/' + experiment_filename, 'w'))
        print 'Model saved!'
    return mygnb


def train_svm(descriptors, labels, experiment_filename, save_model=False):
    descriptors = scale_descriptors(descriptors)
    mysvm = svm.SVC(C=5)
    print 'Fitting SVM with RBF kernel...'
    mysvm.fit(descriptors, labels)
    print 'Done!'
    if save_model:
        print 'Saving SVM with RBG kernel model...'
        cPickle.dump(mysvm, open('./models/' + experiment_filename, 'w'))
        print 'Model saved!'
    return mysvm


def train_logistic_regression(descriptors, labels, learning_rate=5e-5, num_steps=300000):

    def sigmoid(scores):
        return 1 / (1 + np.exp(-scores))

    def log_likelihood(features, target, w):
        scores = np.dot(features, w)
        ll = np.sum(target * scores - np.log(1 + np.exp(scores)))
        return ll
    print 'Fitting Logistic Regression Classifier...'
    weights = np.zeros(descriptors.shape[1])
    classes = ['Opencountry', 'coast', 'forest', 'highway', 'inside_city', 'mountain', 'street', 'tallbuilding']
    target = [float(classes.index(label)) for label in labels]

    for step in xrange(num_steps):
        scores = np.dot(descriptors, weights)
        predictions = sigmoid(scores)
        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(descriptors.T, output_error_signal)
        weights += learning_rate * gradient

        # Print log-likelihood every so often
        if step % 10000 == 0:
            print log_likelihood(descriptors, target, weights)
    print 'Done!'
    return weights



